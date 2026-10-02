import logging

import requests
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from .. import config

_logger = logging.getLogger(__name__)

_llm = ChatGroq(
    model=config.CHAT_MODEL,
    api_key=config.GROQ_API_KEY,
    temperature=0.3,
)


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
    return summary.content


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(research.invoke("How smart is GPT-6?"))