"""Real-time voice via the Gemini Live API.

Browser mic audio streams straight into a Live session, which detects when
the user stops talking and starts answering within a fraction of a second -
no separate STT -> LLM -> TTS round trips. The Notion/Gmail/research
specialists are exposed as NON_BLOCKING function calls: the model says
something like "Let me check your Notion" right away, the specialist runs in
the background, and the model speaks the result as soon as it comes back.

Wire protocol with the browser (one WebSocket per conversation):
  browser -> server: binary = 16 kHz mono PCM16 mic audio
  server -> browser: binary = 24 kHz mono PCM16 speech
                     text   = JSON events: user / agent (transcript
                              fragments), tool, interrupted, turn_complete, error
"""
import asyncio
import os
from typing import Awaitable, Callable

from fastapi import WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

from . import config
from .agent import ORCHESTRATOR_SYSTEM_PROMPT, SPECIALIST_DESCRIPTIONS
from .logging_utils import log_stage

LIVE_MODEL = os.getenv("LIVE_MODEL", "gemini-3.8-live")
# How long a pause counts as "done talking". Lower = snappier, but a
# thinking pause mid-sentence is more likely to be taken as the end.
LIVE_SILENCE_MS = int(os.getenv("LIVE_SILENCE_MS", "500"))

_INPUT_MIME = "audio/pcm;rate=16000"

_LIVE_PROMPT = ORCHESTRATOR_SYSTEM_PROMPT + """
This is a live voice conversation, so never leave the user in silence.
Whenever you call a specialist, ALWAYS speak a short acknowledgement out loud
in the same turn (e.g. "Sure, let me check your Notion." or "One sec, looking
that up.") - specialists take several seconds, and the user must hear you
immediately. When a specialist's result arrives, tell the user the answer
briefly and naturally.
"""

_client = genai.Client(api_key=config.GOOGLE_API_KEY)

Specialists = dict[str, Callable[[str], Awaitable[str]]]


def _live_config(specialists: Specialists) -> types.LiveConnectConfig:
    declarations = [
        types.FunctionDeclaration(
            name=name,
            description=SPECIALIST_DESCRIPTIONS[name],
            parameters_json_schema={
                "type": "object",
                "properties": {"request": {"type": "string", "description": "What to do, in plain English."}},
                "required": ["request"],
            },
            # The model keeps talking while the specialist runs instead of
            # going silent until it returns.
            behavior=types.Behavior.NON_BLOCKING,
        )
        for name in specialists
    ]
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=_LIVE_PROMPT,
        tools=[types.Tool(function_declarations=declarations)],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                silence_duration_ms=LIVE_SILENCE_MS,
            ),
        ),
        # Audio sessions otherwise hit the context limit after ~15 minutes.
        context_window_compression=types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow(),
        ),
    )


async def run_live_session(ws: WebSocket, specialists: Specialists) -> None:
    """Bridge one browser WebSocket to one Gemini Live session until either side closes."""
    tool_tasks: dict[str, asyncio.Task] = {}

    async with _client.aio.live.connect(model=LIVE_MODEL, config=_live_config(specialists)) as session:
        await ws.send_json({"type": "ready"})

        async def run_tool(call: types.FunctionCall) -> None:
            request = (call.args or {}).get("request", "")
            await ws.send_json({"type": "tool", "name": call.name})
            try:
                result = await specialists[call.name](request)
            except Exception as exc:
                print(f"[live] {call.name} failed: {exc!r}")
                result = f"That failed: {exc}"
            await session.send_tool_response(function_responses=[types.FunctionResponse(
                id=call.id,
                name=call.name,
                response={"result": result},
                # Speak the result as soon as the model isn't mid-sentence.
                scheduling=types.FunctionResponseScheduling.WHEN_IDLE,
            )])

        async def browser_to_gemini() -> None:
            while True:
                chunk = await ws.receive_bytes()
                await session.send_realtime_input(audio=types.Blob(data=chunk, mime_type=_INPUT_MIME))

        async def gemini_to_browser() -> None:
            while True:
                got_any = False
                # receive() ends after each model turn, so loop around it.
                async for msg in session.receive():
                    got_any = True
                    if msg.tool_call:
                        for call in msg.tool_call.function_calls or []:
                            tool_tasks[call.id] = asyncio.create_task(run_tool(call))
                    if msg.tool_call_cancellation:
                        for call_id in msg.tool_call_cancellation.ids or []:
                            if task := tool_tasks.pop(call_id, None):
                                task.cancel()
                    if msg.go_away:
                        print(f"[live] server will close the session in {msg.go_away.time_left}")

                    sc = msg.server_content
                    if not sc:
                        continue
                    if sc.input_transcription and sc.input_transcription.text:
                        await ws.send_json({"type": "user", "text": sc.input_transcription.text})
                    if sc.output_transcription and sc.output_transcription.text:
                        await ws.send_json({"type": "agent", "text": sc.output_transcription.text})
                    if sc.model_turn:
                        for part in sc.model_turn.parts or []:
                            if part.inline_data and part.inline_data.data:
                                await ws.send_bytes(part.inline_data.data)
                    if sc.interrupted:
                        # User started talking over the reply: browser drops queued audio.
                        await ws.send_json({"type": "interrupted"})
                    if sc.turn_complete:
                        await ws.send_json({"type": "turn_complete"})
                if not got_any:
                    return  # Gemini closed the session

        tasks = [asyncio.create_task(browser_to_gemini()), asyncio.create_task(gemini_to_browser())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                exc = task.exception()
                if exc and not isinstance(exc, WebSocketDisconnect):
                    raise exc
        finally:
            for task in [*tasks, *tool_tasks.values()]:
                task.cancel()
            log_stage("live session", ended="browser or Gemini closed the connection")
