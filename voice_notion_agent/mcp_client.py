"""MCP client with one long-lived session per server.

MultiServerMCPClient.get_tools() is stateless: every tool call opens a fresh
session, which for stdio servers means spawning a new `npx` process (and,
for Notion, re-doing the mcp-remote connect/auth handshake) on *every* call.
That was several seconds per tool call. Instead we open one session per
server at startup, keep it open, and bind the tools to it.
"""
import asyncio
import shutil

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


def _stdio(binary: str, package: str, *args: str) -> dict:
    """Use the globally installed binary if present (`npm i -g <package>`),
    otherwise fall back to `npx -y`, which re-resolves the package each launch."""
    path = shutil.which(binary)
    if path:
        return {"transport": "stdio", "command": path, "args": list(args)}
    return {"transport": "stdio", "command": "npx", "args": ["-y", package, *args]}


mcp_client = MultiServerMCPClient({
    "gmail": _stdio("gmail-mcp", "@gongrzhe/server-gmail-autoauth-mcp"),
    "notion": _stdio("mcp-remote", "mcp-remote", "https://mcp.notion.com/mcp"),
})

_stop = asyncio.Event()
_owners: list[asyncio.Task] = []


async def get_persistent_tools(server_name: str) -> list[BaseTool]:
    """Open a session to `server_name` that stays open until close_sessions().

    The session lives in its own task because the underlying stdio/anyio
    transport must be entered and exited from the same task - this way
    shutdown (or a timed-out startup) never trips over that.
    """
    ready = asyncio.get_running_loop().create_future()

    async def own_session():
        try:
            async with mcp_client.session(server_name) as session:
                tools = await load_mcp_tools(session)
                if not ready.done():
                    ready.set_result(tools)
                await _stop.wait()
        except Exception as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                print(f"[mcp] {server_name} session ended with error: {exc!r}")

    task = asyncio.create_task(own_session(), name=f"mcp-session:{server_name}")
    _owners.append(task)
    try:
        return await ready
    except BaseException:
        # Startup failed or timed out (e.g. stuck in an OAuth flow) - don't
        # leave the half-open session running in the background.
        task.cancel()
        raise


async def close_sessions() -> None:
    """Close every session opened by get_persistent_tools()."""
    _stop.set()
    await asyncio.gather(*_owners, return_exceptions=True)
    _owners.clear()
