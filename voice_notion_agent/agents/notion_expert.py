import asyncio
import logging

from langchain.agents import create_agent
from langchain.messages import HumanMessage

from ..import config
from ..llm import get_chat_model
from ..mcp_client import mcp_client


_logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Notion expert agent that can perform various tasks using the tools.
You have access to the Notion tool that allow you to interact with your Notion account to retrieve and manage pages."""

async def build_notion_agent():
    """Build a Notion expert agent with provided tools.

    Returns:
        create_agent: An instance of the created Notion expert agent.
    """
    mcp_tools = await mcp_client.get_tools(server_name="notion")

    return create_agent(
    f"google_genai:{config.CHAT_MODEL}",
    tools=mcp_tools,
    system_prompt=SYSTEM_PROMPT,
)


if __name__ == "__main__":
    import asyncio

    async def main():
        config.validate()
        agent = await build_notion_agent()
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "What is the latest article saved in my Notion account?"}]}
        )
        print(result["messages"][-1].text)

    asyncio.run(main())