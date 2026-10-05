"""A web-research tool the agent can call before writing a Notion page."""
import sys

import requests
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from .. import config

_llm = ChatGoogleGenerativeAI(
    model=config.CHAT_MODEL,
    google_api_key=config.GOOGLE_API_KEY,
    temperature=0.3,
)

SERPER_URL = "https://google.serper.dev/search"


def _serper_search(query: str, num: int = 5) -> dict:
    """Call Serper.dev and return the raw JSON response as a dict."""
    response = requests.post(
        SERPER_URL,
        headers={
            "X-API-KEY": config.SERPER_API_KEY,
            "Content-Type": "application/json",
        },
        json={"q": query, "num": num},
        timeout=15,
    )
    response.raise_for_status()  # raises an error if the status is 401, 429, 500, etc.
    return response.json()


@tool
def research_topic(topic: str) -> str:
    """Research a topic on the web and return a concise markdown summary.

    Use this before creating a Notion page whenever the user asks you to
    "research", "look into", or "find out about" something. The summary
    should then be passed as the content when creating the Notion page.
    """
    raw_results = _serper_search(topic)
    results_text = "\n".join(
        f"- {r.get('title', '')}: {r.get('snippet', '')}"
        for r in raw_results.get("organic", [])
    )

    prompt = (
        "You are a research assistant. Using the raw search results below, "
        "write a concise, well-organized summary about the topic. Use short "
        "headings and bullet points. Keep it under 300 words.\n\n"
        f"Topic: {topic}\n\nRaw search results:\n{results_text}"
    )
    response = _llm.invoke(prompt)
    return response.text  # plain string, works even if the model returns content parts


if __name__ == "__main__":
    # Lets you test this file from the terminal
    config.validate()  # stops early with a clear message if a key is missing in .env
    topic = " ".join(sys.argv[1:]) or "How smart GPT-6 is?"
    print(research_topic.invoke({"topic": topic}))
