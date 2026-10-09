"""The web-research specialist sub-agent."""
import asyncio
import sys

from langchain.agents import create_agent
from voice_notion_agent import config
from voice_notion_agent.tools.researcher_tool import research_topic

SYSTEM_PROMPT = """You are the research specialist on a team of AI assistants.

Given a topic, use your web search tool to investigate it and produce a
concise, well-organized markdown summary: short headings, bullet points,
under 300 words. Search more than once if the first results are thin or
off-topic.

Return only the summary itself, ready to be handed to another assistant
(e.g. to be saved as a Notion page) - no preamble like "Here is a summary".
"""


async def build_researcher():
    """Assemble the research specialist agent."""
    return create_agent(
        config.CHAT_MODEL,
        tools=[research_topic],
        system_prompt=SYSTEM_PROMPT,
    )


async def main():
    topic = " ".join(sys.argv[1:]) or "latest GPT model"
    agent = await build_researcher()
    result = await agent.ainvoke({"messages": [{"role": "user", "content": topic}]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())