import os
import requests
import logging
from pathlib import Path

import requests
from openai import OpenAI

import config

_logger = logging.getLogger(__name__)


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=config.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )


def transcribe_audio(file_path: str) -> str:
    """Transcribe an audio file to text using Groq's Whisper model."""
    path = Path(file_path)
    try:
        with path.open("rb") as f:
            transcript = _get_client().audio.transcriptions.create(
                file=(path.name, f.read()),
                model=config.STT_MODEL,
            )
        return transcript.text
    except Exception:
        _logger.exception("Error transcribing audio")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample = Path("sample.wav")
    if not sample.exists():
        url = "https://cdn.openai.com/API/docs/audio/alloy.wav"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        sample.write_bytes(resp.content)

    print(transcribe_audio(str(sample)))