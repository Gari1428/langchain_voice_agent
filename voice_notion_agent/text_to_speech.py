import subprocess
from pathlib import Path

import imageio_ffmpeg
from openai import OpenAI

import config

MAX_CHARS = 200  # Groq's Orpheus input limit per request


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=config.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )


def _wav_to_mp3(wav_bytes: bytes) -> bytes:
    """Convert WAV bytes to MP3 bytes using the bundled ffmpeg."""
    try:
        result = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                "-i", "pipe:0",
                "-f", "mp3", "-codec:a", "libmp3lame", "-q:a", "2",
                "pipe:1",
            ],
            input=wav_bytes,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr.decode()}")
    return result.stdout


def synthesize_speech(text: str) -> bytes:
    """Convert text to spoken audio with Groq and return the MP3 bytes."""
    if len(text) > MAX_CHARS:
        raise ValueError(f"Text is {len(text)} characters; Groq allows at most {MAX_CHARS}.")

    response = _get_client().audio.speech.create(
        model=config.TTS_MODEL,
        voice=config.TTS_VOICE,
        input=text,
        response_format="wav",
    )
    return _wav_to_mp3(response.content)


if __name__ == "__main__":
    sample_text = "Hello, this is a test of the text-to-speech functionality."
    audio_data = synthesize_speech(sample_text)

    output_path = Path(__file__).parent / "output.mp3"
    output_path.write_bytes(audio_data)
    print(f"Audio saved to {output_path}")