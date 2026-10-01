from dotenv import load_dotenv
import os
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVIL_API_KEY = os.gepythotenv("TAVIL_API_KEY")

CHAT_MODEL = os.getenv("CHAT_MODEL","openai/gpt-oss-120b")
STT_MODEL = os.getenv("STT_MODEL","whisper-large-v3-turbo")
TTS_MODEL = os.getenv("TTS_MODEL","canopylabs/orpheus-v1-english")
TTS_VOICE = os.getenv("TTS_VOICE","autumn")

def validate() -> None:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in the environment variable")
    if not TAVIL_API_KEY:
        raise ValueError("TAVIL_API_KEY is not set in the environment variable")