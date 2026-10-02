from langchain_mcp_adapters import MultiserverMCPClient


mcp_client = MultiserverMCPClient({
    "gmail":{
        "transport":"stdio",
        "command":"npx",
        "args":["-y","@gongrzhe/server-gmail-autoauth-mcp"],

    },
    "notion":{
        "transport":"stdio",
        "command":"npx",
        "args":["-y","mcp-remote","https://mcp.notion.com/mcp"],
    }
})