from langchain_google_genai import ChatGoogleGenerativeAI

from . import config


def get_chat_model() -> ChatGoogleGenerativeAI:
    """Return the Gemini chat model used by the agent and the research tool."""
    return ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.3,
    )