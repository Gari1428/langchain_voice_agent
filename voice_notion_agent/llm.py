from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from . import config


@lru_cache(maxsize=None)
def get_chat_model(temperature: float | None = None) -> BaseChatModel:
    """Return the shared Gemini chat model used by every agent and the research tool.

    thinking_level is set explicitly because Gemini 3 models think by default,
    which adds noticeable latency to every call - too slow for a voice reply.
    """
    kwargs = {"google_api_key": config.GOOGLE_API_KEY, "thinking_level": config.THINKING_LEVEL}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return init_chat_model(config.CHAT_MODEL, **kwargs)
