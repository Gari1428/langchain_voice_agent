import asyncio
import logging

from langchain.agents import create_agent
from langchain.messages import HumanMessage

from . import config
from .llm import get_chat_model
from .mcp_client import mcp_client
from .tools.researcher_tool import research

_logger = logging.getLogger(__name__)

ALLOWED_MCP_TOOLS = {
    "notion-search", "notion-fetch", "notion-create-pages",
    "search_emails", "read_email",
}

SYSTEM_PROMPT = """You are an orchestrator agent that can perform various tasks using the tools.
For tasks that require specific tools, you will use the provided tools to accomplish the task.
You have access to the following tools:
1. Research Tool: searches the web with Serper and summarizes the findings.
2. Notion tools: search and read pages, and create new pages in the user's Notion workspace.
3. Gmail tools: search and read emails in the user's Gmail account (read-only).

For all generic tasks you will use your own capabilities to accomplish the task."""


async def load_mcp_tools():
    loaded = []
    for server in ("notion", "gmail"):
        try:
            loaded.extend(await mcp_client.get_tools(server_name=server))
        except Exception as e:
            _logger.warning("Skipping MCP server '%s': %s", server, e)
    return loaded


def _message_text(message) -> str:
    content = message.content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content


async def build_agent(tools=None):
    tools = list(tools) if tools else [research]

    all_mcp_tools = await load_mcp_tools()
    _logger.info("Available MCP tools: %s", [t.name for t in all_mcp_tools])

    mcp_tools = [t for t in all_mcp_tools if t.name in ALLOWED_MCP_TOOLS]
    if not mcp_tools:
        _logger.warning("No MCP tools matched ALLOWED_MCP_TOOLS; check the names in the log above.")
    tools.extend(mcp_tools)

    return create_agent(
        model=get_chat_model(),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )


async def main():
    agent = await build_agent()
    user_input = "What's the latest email I received?"
    response = await agent.ainvoke({"messages": [HumanMessage(content=user_input)]})
    print(_message_text(response["messages"][-1]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config.validate()
    asyncio.run(main())