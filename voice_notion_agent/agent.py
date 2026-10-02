from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_core.tools import tool
import config

import logging

_logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an orchestrator agent that can perform various tasks using the tools. 
For tasks that require specific tools, you will use the provided tools to accomplish the task.
For all generic tasks you will use your own capabilities to accomplish the task."""

def build_agent(tools):
    """Build an agent with provided tools.

    Args:
        tools (list): A list of tools to be used by the agent.

    Returns:
        create_agent: An instance of the created agent.
    """
    agent = create_agent(
        tools=[],
        system_prompt=SYSTEM_PROMPT,
        llm = config.LLM,
        verbose=True
    )

    return agent