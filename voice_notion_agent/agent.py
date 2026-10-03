import asyncio
import logging

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_groq import ChatGroq

from . import config
from .mcp_client import mcp_client
from .tools.researcher_tool import research

_logger = logging.getLogger(__name__)

# Only these Notion tools are given to the model. Replace the names with the
# exact ones printed in your log on the first run.
ALLOWED_MCP_TOOLS = {"notion-search", "notion-fetch", "notion-create-pages","search_emails", "read_email"}


SYSTEM_PROMPT = """You are an orchestrator agent that can perform various tasks using the tools.
For tasks that require specific tools, you will use the provided tools to accomplish the task.
You have access to the following tools:
1. Research Tool: This tool searches the web with Serper and summarizes the findings.
2. Notion tools: This tool searches and reads pages, and creates new pages in the user's Notion workspace.
3. Gmail tools: This tool allows you to interact with my Gmail account to send and receive emails.

For all generic tasks you will use your own capabilities to accomplish the task."""


async def build_agent(tools=None):
    """Build an agent with the given tools plus a filtered set of Notion tools.

    Args:
        tools (list, optional): Tools for the agent. Defaults to [research].

    Returns:
        The created agent.
    """
    tools = list(tools) if tools else [research]

    all_mcp_tools = await mcp_client.get_tools()
    _logger.info("Available MCP tools: %s", [t.name for t in all_mcp_tools])

    mcp_tools = [t for t in all_mcp_tools if t.name in ALLOWED_MCP_TOOLS]
    if not mcp_tools:
        _logger.warning("No MCP tools matched ALLOWED_MCP_TOOLS; check the names in the log above.")
    tools.extend(mcp_tools)

    model = ChatGroq(
        model=config.CHAT_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.3,
    )

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )


async def main():
    agent = await build_agent()
    user_input = (
        "What's the latest email I received?"
    )
    response = await agent.ainvoke({"messages": [HumanMessage(content=user_input)]})
    print(response["messages"][-1].content)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config.validate()
    asyncio.run(main())