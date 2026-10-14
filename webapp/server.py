"""FastAPI web demo: browser mic -> STT -> agent -> Gmail/Notion (MCP) -> TTS -> browser audio.

This wraps the same voice_notion_agent package used by the CLI in main.py.
It exists so the project can be demoed live from a shareable URL (e.g. on
Render) instead of only from a local terminal.

State is kept in a plain in-memory dict keyed by a client-generated session
id. That's fine for a single-instance demo deployment; it is not meant to
survive restarts or scale beyond one process.

SECURITY NOTE: the agent now has real Gmail (read/send) and Notion access
via MCP. If AUTH_PASSWORD is set, every route (including the page itself)
requires HTTP Basic Auth - the browser's native login prompt, checked
against AUTH_USERNAME/AUTH_PASSWORD. Set this before exposing the service
on a public URL, otherwise anyone with the link can read or send from your
real Gmail account.
"""
import asyncio
import base64
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from langchain.messages import HumanMessage
from openai import BadRequestError

from voice_notion_agent import config
from voice_notion_agent.agent import build_agent
from voice_notion_agent.logging_utils import log_stage
from voice_notion_agent.speech_to_text import transcribe_audio
from voice_notion_agent.text_to_speech import MAX_CHARS, synthesize_speech

config.validate()

_agent = None
_sessions: dict[str, list] = {}
_MAX_HISTORY_MESSAGES = 20

_AUTH_USERNAME = os.getenv("AUTH_USERNAME", "demo")
_AUTH_PASSWORD = os.getenv("AUTH_PASSWORD")
_basic_auth = HTTPBasic(auto_error=False)

_BUSY_REPLY = "The AI model is busy or out of quota right now. Please try again later."
_ERROR_REPLY = "Something went wrong on the server. Please check the logs."


def _require_auth(credentials: HTTPBasicCredentials | None = Depends(_basic_auth)) -> None:
    """No-op if AUTH_PASSWORD isn't set; otherwise requires matching HTTP Basic Auth.

    auto_error=False on HTTPBasic means a missing Authorization header comes
    back as None instead of an immediate 401 - that's what lets local dev
    (no AUTH_PASSWORD set) skip auth entirely rather than always prompting.
    """
    if not _AUTH_PASSWORD:
        return
    valid = credentials is not None and (
        secrets.compare_digest(credentials.username, _AUTH_USERNAME)
        and secrets.compare_digest(credentials.password, _AUTH_PASSWORD)
    )
    if not valid:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    _agent = await build_agent()
    yield


app = FastAPI(title="Voice Gmail/Notion Agent Demo", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"


@app.exception_handler(Exception)
async def unhandled_error(request, exc):
    """Last-resort handler: always answer with JSON so the browser can parse it."""
    print(f"[error] {request.url.path}: {exc!r}")
    return JSONResponse(
        {"transcript": "", "reply": _ERROR_REPLY, "audio_base64": None},
        status_code=500,
    )


@app.get("/", dependencies=[Depends(_require_auth)])
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


def _trim(messages: list) -> list:
    """Keep the last N messages, but always start on a human turn.

    Cutting in the middle of an assistant tool-call / tool-result pair
    leaves an orphan tool message, which Gemini rejects with a 400.
    """
    messages = messages[-_MAX_HISTORY_MESSAGES:]
    while messages and messages[0].type != "human":
        messages = messages[1:]
    return messages


def _spoken(text: str, limit: int = MAX_CHARS) -> str:
    """Shorten text to fit the TTS limit, cutting at a sentence end if possible."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: end + 1] if end > 40 else cut.rsplit(" ", 1)[0]


async def _run_agent_turn(session_id: str, transcript: str) -> str:
    log_stage("User -> Orchestrator", input=transcript)
    history = _sessions.get(session_id, [])
    result = await _agent.ainvoke({"messages": history + [HumanMessage(content=transcript)]})
    reply = result["messages"][-1].text  # .text is always a plain string
    print(f"[agent] model/tool messages this turn: {len(result['messages']) - len(history) - 1}")
    # Saved only after success, so a failed call never leaves a dangling message.
    _sessions[session_id] = _trim(result["messages"])
    log_stage("Orchestrator -> User", output=reply)
    return reply


async def _safe_turn(session_id: str, transcript: str) -> tuple[str, bool]:
    """Run a turn; on any model failure return a friendly message instead of a 500."""
    try:
        return await _run_agent_turn(session_id, transcript), True
    except Exception as exc:
        print(f"[agent] turn failed: {exc!r}")
        text = str(exc)
        busy = any(k in text for k in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
        return (_BUSY_REPLY if busy else _ERROR_REPLY), False


@app.post("/api/chat", dependencies=[Depends(_require_auth)])
async def chat(message: str = Form(...), session_id: str = Form(...)) -> JSONResponse:
    """Text-only turn, used by the web UI's fallback text box."""
    reply, _ = await _safe_turn(session_id, message)
    return JSONResponse({"transcript": message, "reply": reply})


@app.post("/api/voice", dependencies=[Depends(_require_auth)])
async def voice(audio: UploadFile = File(...), session_id: str = Form(...)) -> JSONResponse:
    """Voice turn: browser sends a recorded clip, we return text + spoken reply."""
    # Keep whatever extension the browser actually sent (webm/ogg/mp4) -
    # Whisper uses it as a hint for which container/codec to decode.
    suffix = Path(audio.filename or "command.webm").suffix or ".webm"
    tmp_path = f"/tmp/{uuid.uuid4()}{suffix}"
    audio_bytes = await audio.read()
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    print(
        f"[voice] received {len(audio_bytes)} bytes, "
        f"filename={audio.filename!r}, content_type={audio.content_type!r}, saved to {tmp_path}"
    )

    t0 = time.perf_counter()
    try:
        # transcribe_audio is a blocking network call. Calling it directly inside
        # this async handler froze the whole event loop (every other request
        # waited too), so run it in a worker thread.
        transcript = await asyncio.to_thread(transcribe_audio, tmp_path)
    except BadRequestError as exc:
        print(f"[voice] transcription failed: {exc}")
        return JSONResponse(
            {
                "transcript": "",
                "reply": "That clip was too short or couldn't be understood - press and hold while you speak.",
                "audio_base64": None,
            }
        )
    finally:
        try:
            os.remove(tmp_path)  # temp clips were never cleaned up before
        except OSError:
            pass
    t_stt = time.perf_counter() - t0

    print(f"[voice] transcript: {transcript!r}")

    if not transcript:
        return JSONResponse(
            {"transcript": "", "reply": "I didn't catch that, try again.", "audio_base64": None}
        )

    t1 = time.perf_counter()
    reply, ok = await _safe_turn(session_id, transcript)
    t_agent = time.perf_counter() - t1

    # synthesize_speech returns MP3 *bytes* (not a file path). If TTS fails or
    # the agent failed, still return the text reply so the UI shows something.
    audio_b64 = None
    t_tts = 0.0
    if ok:
        t2 = time.perf_counter()
        try:
            # Blocking Groq call + ffmpeg subprocess -> keep off the event loop.
            mp3_bytes = await asyncio.to_thread(synthesize_speech, _spoken(reply))
            audio_b64 = base64.b64encode(mp3_bytes).decode("ascii")
        except Exception as exc:
            print(f"[voice] TTS failed: {exc!r}")
        t_tts = time.perf_counter() - t2

    # One line per request so you can see exactly which stage is slow.
    print(f"[timing] stt={t_stt:.2f}s agent={t_agent:.2f}s tts={t_tts:.2f}s total={t_stt + t_agent + t_tts:.2f}s")

    return JSONResponse({"transcript": transcript, "reply": reply, "audio_base64": audio_b64})