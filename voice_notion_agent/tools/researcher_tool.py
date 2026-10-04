import logging

import requests
from langchain_core.tools import tool

from .. import config
from ..llm import get_chat_model

_logger = logging.getLogger(__name__)

_llm = get_chat_model()


def _search(topic: str) -> str:
    """Search Google through Serper and return the result snippets as text."""
    resp = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": config.SERPER_API_KEY},
        json={"q": topic, "num": 5},
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json().get("organic", [])
    return "\n".join(r.get("snippet", "") for r in results)


def _as_text(content) -> str:
    """Gemini can return a list of parts instead of one string."""
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content


@tool
def research(topic: str) -> str:
    """
    Research a topic using a Serper web search and summarize the findings.

    Args:
        topic (str): The topic to research.

    Returns:
        str: A summary of the research results.
    """
    _logger.info(f"Researching topic: {topic}")

    results_text = _search(topic)
    _logger.info(f"Raw search results: {results_text}")

    prompt = f"""
    You are a research assistant. You will be given a topic of interest and the web search results.
    Your task is to summarize the information in a concise manner, highlighting the most important points.
    Topic: {topic}
    Search Results:
    {results_text}
    Please provide a summary of the research results in a clear and concise manner.
    """
    summary = _llm.invoke(prompt)
    return _as_text(summary.content)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(research.invoke("How smart is GPT-6?"))