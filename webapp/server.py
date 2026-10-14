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
import json
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from langchain.messages import HumanMessage
from openai import BadRequestError

from voice_notion_agent import config
from voice_notion_agent.agent import build_agent, build_specialists
from voice_notion_agent.live import run_live_session
from voice_notion_agent.logging_utils import log_stage
from voice_notion_agent.mcp_client import close_sessions
from voice_notion_agent.speech_to_text import transcribe_audio
from voice_notion_agent.text_to_speech import MAX_CHARS, synthesize_speech

config.validate()

_agent = None
_specialists = None
_sessions: dict[str, list] = {}
_MAX_HISTORY_MESSAGES = 20

_AUTH_USERNAME = os.getenv("AUTH_USERNAME", "demo")
_AUTH_PASSWORD = os.getenv("AUTH_PASSWORD")
_basic_auth = HTTPBasic(auto_error=False)

_BUSY_REPLY = "The AI model is busy or out of quota right now. Please try again later."
_ERROR_REPLY = "Something went wrong on the server. Please check the logs."

# Spoken the moment the Orchestrator hands off to a specialist, so the user
# hears something within ~1s instead of silence while Notion/Gmail/search run.
# Synthesized once at startup so playing one costs no TTS round-trip.
_ACK_TEXT = {
    "notion_expert": "Checking your Notion.",
    "email_expert": "Checking your email.",
    "researcher_expert": "Let me look that up.",
}
_ack_audio: dict[str, str] = {}


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
    global _agent, _specialists
    _specialists, _ = await asyncio.gather(build_specialists(), _preload_acks())
    _agent = await build_agent(_specialists)
    yield
    await close_sessions()


async def _preload_acks() -> None:
    """Best-effort: an ack whose TTS fails is still shown as text, just not spoken."""
    names = list(_ACK_TEXT)
    results = await asyncio.gather(
        *(asyncio.to_thread(synthesize_speech, _ACK_TEXT[n]) for n in names),
        return_exceptions=True,
    )
    for name, mp3 in zip(names, results):
        if isinstance(mp3, Exception):
            print(f"[voice] could not pre-synthesize ack for {name}: {mp3!r}")
        else:
            _ack_audio[name] = base64.b64encode(mp3).decode("ascii")


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


def _ws_authorized(ws: WebSocket) -> bool:
    """HTTP Basic check for the WebSocket handshake (HTTPBasic only handles plain requests).

    Browsers resend the Basic credentials they already used for the page on
    a same-origin WebSocket upgrade, so no extra login is needed.
    """
    if not _AUTH_PASSWORD:
        return True
    scheme, _, encoded = ws.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        username, _, password = base64.b64decode(encoded).decode().partition(":")
    except ValueError:
        return False
    return secrets.compare_digest(username, _AUTH_USERNAME) and secrets.compare_digest(password, _AUTH_PASSWORD)


@app.websocket("/api/live")
async def live(ws: WebSocket) -> None:
    """Real-time voice conversation: see voice_notion_agent/live.py for the protocol."""
    if not _ws_authorized(ws):
        await ws.close(code=1008)
        return
    await ws.accept()
    try:
        await run_live_session(ws, _specialists)
    except Exception as exc:
        print(f"[live] session error: {exc!r}")
        try:
            await ws.send_json({"type": "error", "text": "The live session ended unexpectedly."})
            await ws.close()
        except Exception:
            pass


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


async def _stream_turn(session_id: str, transcript: str):
    """Run one agent turn, yielding ("delegating", tool_name) each time the
    Orchestrator calls a specialist, then finally ("reply", text, ok).

    On any model failure the reply is a friendly message instead of a 500.
    """
    log_stage("User -> Orchestrator", input=transcript)
    history = _sessions.get(session_id, [])
    state = None
    try:
        async for mode, chunk in _agent.astream(
            {"messages": history + [HumanMessage(content=transcript)]},
            stream_mode=["updates", "values"],
        ):
            if mode == "values":
                state = chunk  # full state after each step; the last one is final
                continue
            for update in chunk.values():
                if not isinstance(update, dict):
                    continue
                for msg in update.get("messages", []):
                    for call in getattr(msg, "tool_calls", None) or []:
                        yield ("delegating", call["name"])
    except Exception as exc:
        print(f"[agent] turn failed: {exc!r}")
        text = str(exc)
        busy = any(k in text for k in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
        yield ("reply", _BUSY_REPLY if busy else _ERROR_REPLY, False)
        return

    reply = state["messages"][-1].text  # .text is always a plain string
    print(f"[agent] model/tool messages this turn: {len(state['messages']) - len(history) - 1}")
    # Saved only after success, so a failed call never leaves a dangling message.
    _sessions[session_id] = _trim(state["messages"])
    log_stage("Orchestrator -> User", output=reply)
    yield ("reply", reply, True)


@app.post("/api/chat", dependencies=[Depends(_require_auth)])
async def chat(message: str = Form(...), session_id: str = Form(...)) -> JSONResponse:
    """Text-only turn, used by the web UI's fallback text box."""
    reply = _ERROR_REPLY
    async for event in _stream_turn(session_id, message):
        if event[0] == "reply":
            reply = event[1]
    return JSONResponse({"transcript": message, "reply": reply})


def _event(type_: str, **fields) -> bytes:
    """One newline-delimited JSON event for the streaming voice response."""
    return (json.dumps({"type": type_, **fields}) + "\n").encode()


@app.post("/api/voice", dependencies=[Depends(_require_auth)])
async def voice(audio: UploadFile = File(...), session_id: str = Form(...)) -> StreamingResponse:
    """Voice turn, streamed so the browser can react before the agent finishes.

    Emits newline-delimited JSON events, in order:
      transcript -> ack* (spoken filler as specialists are called) -> reply -> audio
    """
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

    async def events():
        t0 = time.perf_counter()
        try:
            # transcribe_audio is a blocking network call - keep it off the event loop.
            transcript = await asyncio.to_thread(transcribe_audio, tmp_path)
        except BadRequestError as exc:
            print(f"[voice] transcription failed: {exc}")
            yield _event("reply", text="That clip was too short or couldn't be understood - press and hold while you speak.")
            return
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        t_stt = time.perf_counter() - t0

        print(f"[voice] transcript: {transcript!r}")
        if not transcript:
            yield _event("reply", text="I didn't catch that, try again.")
            return
        yield _event("transcript", text=transcript)

        t1 = time.perf_counter()
        t_first_ack = None
        acked: set[str] = set()
        reply, ok = _ERROR_REPLY, False
        async for event in _stream_turn(session_id, transcript):
            if event[0] == "delegating":
                name = event[1]
                if name in _ACK_TEXT and name not in acked:
                    acked.add(name)
                    t_first_ack = t_first_ack or time.perf_counter() - t1
                    yield _event("ack", text=_ACK_TEXT[name], audio_base64=_ack_audio.get(name))
            else:
                _, reply, ok = event
        t_agent = time.perf_counter() - t1

        # Send the text first so it shows up while TTS is still running.
        yield _event("reply", text=reply)

        t_tts = 0.0
        if ok:
            t2 = time.perf_counter()
            try:
                # Blocking Groq call + ffmpeg subprocess -> keep off the event loop.
                mp3_bytes = await asyncio.to_thread(synthesize_speech, _spoken(reply))
                yield _event("audio", audio_base64=base64.b64encode(mp3_bytes).decode("ascii"))
            except Exception as exc:
                print(f"[voice] TTS failed: {exc!r}")
            t_tts = time.perf_counter() - t2

        # One line per request so you can see exactly which stage is slow.
        ack = f"{t_stt + t_first_ack:.2f}s" if t_first_ack is not None else "-"
        print(
            f"[timing] stt={t_stt:.2f}s first_ack={ack} agent={t_agent:.2f}s "
            f"tts={t_tts:.2f}s total={t_stt + t_agent + t_tts:.2f}s"
        )

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        # Stop reverse proxies (e.g. Render/nginx) from buffering the stream.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
