"""Builds the Orchestrator: a generalist agent that answers directly or
delegates to a specialist sub-agent (Notion, Gmail, or web research).

    Orchestrator -> notion_expert  (Notion MCP tools)
                -> email_expert   (Gmail MCP tools)
                -> researcher     (web search + summarize)

Each specialist is itself a full create_agent, wrapped as a plain tool so
the Orchestrator can call it like any other tool. This keeps each
specialist's tool access scoped to just the service it owns, and keeps
their system prompts focused on one job instead of one long prompt trying
to cover Gmail, Notion, and research at once.
"""
import asyncio
import logging

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_core.tools import tool

from . import config
from .agents.email_expert import build_email_expert
from .agents.notion_expert import build_notion_agent
from .llm import get_chat_model
from .logging_utils import log_stage
from .mcp_client import close_sessions
from .tools.researcher_tool import research_topic
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)


ORCHESTRATOR_SYSTEM_PROMPT = """You are Aria, a voice-controlled executive assistant.

Answer general questions directly yourself. For anything that needs real
access to the user's Notion workspace, Gmail inbox, or current web
information, delegate to the matching specialist tool instead of guessing:
- notion_expert: search, read, or create Notion pages.
- email_expert: check, search, draft, or send Gmail.
- researcher_expert: look something up on the web and summarize it.

Chain specialists when a request needs more than one step - e.g. ask
researcher_expert for a summary, then pass that summary to notion_expert to save
as a page.

To summarize or read a Notion page, send ONE request to notion_expert asking it to
find the page, read its full content, and return the summary. Never send a Notion
page title to researcher_expert. researcher_expert is only for information from the
public web.

Always give a short, spoken-friendly reply (1-2 sentences), since it may
be read aloud. Report the outcome naturally - never mention tools, agents,
or that you "delegated" something.
"""


_SPECIALIST_BUILD_TIMEOUT = 20.0


async def _try_build(build_fn, name: str):
    """Build a specialist, but never let its failure - or hang - take down the others.

    A specialist can fail to start for reasons outside our code - an
    expired/missing MCP auth token, a misconfigured OAuth app, npx unable
    to reach the network. Without the try/except, asyncio.gather would
    propagate the first exception and cancel every sibling task, so one bad
    token would crash the Orchestrator (and both other specialists)
    entirely. The timeout matters separately: mcp-remote with no cached
    Notion token doesn't raise, it opens an interactive OAuth URL and waits
    forever for a browser that will never visit it on a headless server -
    an exception handler alone would never fire.
    """
    try:
        return await asyncio.wait_for(build_fn(), timeout=_SPECIALIST_BUILD_TIMEOUT)
    except TimeoutError:
        print(
            f"[agent] {name} timed out after {_SPECIALIST_BUILD_TIMEOUT}s during initialization "
            "(likely stuck waiting on an interactive auth flow) - it will report itself unavailable"
        )
        return None
    except Exception as exc:  
        print(f"[agent] {name} failed to initialize, it will report itself unavailable: {exc}")
        return None


async def build_agent():
    """Build the three specialists, wrap them as tools, and assemble the Orchestrator."""
    notion_agent, email_agent = await asyncio.gather(
        _try_build(build_notion_agent, "notion_expert"),
        _try_build(build_email_expert, "email_expert"),
    )

    @tool
    async def notion_expert(request: str) -> str:
        """Delegate a Notion request (search, read, or create pages) to the Notion specialist."""
        if notion_agent is None:
            return "The Notion integration isn't available on this server right now."
        log_stage("Orchestrator -> notion_expert", input=request)
        result = await notion_agent.ainvoke({"messages": [HumanMessage(content=request)]})
        output = result["messages"][-1].text
        log_stage("notion_expert -> Orchestrator", output=output)
        return output

    @tool
    async def email_expert(request: str) -> str:
        """Delegate a Gmail request (check, search, draft, or send email) to the email specialist."""
        if email_agent is None:
            return "The Gmail integration isn't available on this server right now."
        log_stage("Orchestrator -> email_expert", input=request)
        result = await email_agent.ainvoke({"messages": [HumanMessage(content=request)]})
        output = result["messages"][-1].text
        log_stage("email_expert -> Orchestrator", output=output)
        return output

    @tool
    async def researcher_expert(topic: str) -> str:
        """Delegate a web research request to the research specialist; returns a markdown summary."""
        # Call the search+summarize tool directly. It already does Serper search
        # and an LLM summary, so wrapping it in a whole extra "researcher" agent
        # only added two more sequential LLM round-trips (~2-4s) for no benefit.
        log_stage("Orchestrator -> researcher", input=topic)
        output = await research_topic.ainvoke({"topic": topic})
        log_stage("researcher -> Orchestrator", output=output)
        return output

    tools = [notion_expert, email_expert, researcher_expert]
    return create_agent(get_chat_model(), tools=tools, system_prompt=ORCHESTRATOR_SYSTEM_PROMPT)

if __name__ == "__main__":
    async def main():
        config.validate()
        agent = await build_agent()
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="Give me latest means last saved notion article from my account and generate a summary of the article.")]}
        )
        print(result["messages"][-1].text)
        await close_sessions()

    asyncio.run(main())