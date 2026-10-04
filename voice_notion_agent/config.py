from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Only used by speech_to_text.py and text_to_speech.py
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.8-flash")
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
TTS_MODEL = os.getenv("TTS_MODEL", "canopylabs/orpheus-v1-english")
TTS_VOICE = os.getenv("TTS_VOICE", "autumn")


def validate() -> None:
    missing = [
        name
        for name, value in {
            "GOOGLE_API_KEY": GOOGLE_API_KEY,
            "SERPER_API_KEY": SERPER_API_KEY,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")