from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from . import config
from .tools.researcher_tool import research

import logging

_logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an orchestrator agent that can perform various tasks using the tools. 
For tasks that require specific tools, you will use the provided tools to accomplish the task.
You have access to the following tools:
1. Research Tool: This tool allows you to search the topic using a Serper web search and summarize the findings. 


For all generic tasks you will use your own capabilities to accomplish the task."""

def build_agent(tools=None):
    """Build an agent with provided tools.

    Args:
        tools (list): A list of tools to be used by the agent.

    Returns:
        create_agent: An instance of the created agent.
    """

    tools = [research] or tools

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

if __name__ == "__main__":
    agent = build_agent()
    user_input = "Research about GPT -6."
    response = agent.invoke({"messages":([HumanMessage(content=user_input)])})
    print(response["messages"][-1].content)