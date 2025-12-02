"""Example: use chrome-devtools-mcp as a DeepAgents subagent to control Chrome.

This script shows how to:
- start an MCP client for the official Chrome DevTools MCP server
- convert MCP tools into LangChain tools via ``langchain-mcp-adapters``
- register those tools on a DeepAgents subagent
- invoke the subagent via the main DeepAgent to operate the browser

Usage (from repository root):

    uv run python libs/deepagents-cli/examples/chrome_mcp_subagent.py

Prerequisites:
- Node.js installed (so that ``npx chrome-devtools-mcp@latest`` works)
- Python deps installed, including ``langchain-mcp-adapters``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from deepagents import create_deep_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph.state import CompiledStateGraph

from deepagents_cli.config import create_model


async def build_chrome_mcp_agent() -> tuple[CompiledStateGraph, MultiServerMCPClient]:
    """Create a DeepAgent with a chrome-devtools MCP subagent attached.

    Returns:
        Tuple of (agent, mcp_client). The MCP client must remain alive while
        the agent is using MCP tools.
    """
    model = create_model()

    # Configure the chrome-devtools MCP server. This uses the official
    # npm package and starts it via `npx` so you don't need to install it globally.
    # NOTE: Newer versions of `langchain-mcp-adapters` expect the server mapping as
    # the first positional argument (no `servers=` keyword).
    mcp_client = MultiServerMCPClient(
        {
            "chrome-devtools": {
                "command": "npx",
                "args": ["chrome-devtools-mcp@latest", "--headless=false"],
                "transport": "stdio",
            },
        }
    )

    # Load tools exposed by the server. The Python MultiServerMCPClient implementation
    # does not expose explicit connect/close methods; it manages connections internally.
    chrome_tools = await mcp_client.get_tools()

    # Define a subagent that has access only to the Chrome DevTools MCP tools.
    chrome_subagent: dict[str, Any] = {
        "name": "chrome-browser-agent",
        "description": (
            "Use this agent to control a Chrome browser via Chrome DevTools MCP. "
            "It can open pages, run JavaScript, query the DOM, capture screenshots, "
            "and inspect network activity."
        ),
        "system_prompt": (
            "You are a specialized browser automation agent. "
            "Use the provided Chrome DevTools MCP tools to:\n"
            "- launch or connect to a Chrome instance\n"
            "- navigate to URLs\n"
            "- execute JavaScript in the page context\n"
            "- inspect and extract DOM content\n"
            "- capture screenshots and other diagnostics\n\n"
            "Always return concise, user-facing summaries of what you did and what you observed."
        ),
        "tools": chrome_tools,
    }

    # Create the main DeepAgent. It will automatically receive a `task` tool
    # via SubAgentMiddleware, which can be used to invoke `chrome-browser-agent`.
    agent = create_deep_agent(
        model=model,
        tools=[],
        subagents=[chrome_subagent],
        system_prompt=(
            "You are a coordinator agent. For any request that involves operating a web browser "
            "(opening pages, interacting with DOM, running JS, taking screenshots, debugging "
            "web apps), you SHOULD delegate the work to the `chrome-browser-agent` subagent "
            "via the `task` tool with subagent_type='chrome-browser-agent'. "
            "Describe clearly what the subagent should do in the browser and what information "
            "it should return."
        ),
    )

    return agent, mcp_client


async def demo_once() -> None:
    """Run a single demo interaction using the chrome-browser subagent."""
    agent, mcp_client = await build_chrome_mcp_agent()

    # Example user query that should cause the main agent to spawn the
    # chrome-browser-agent subagent via the `task` tool.
    input_state = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "请在浏览器里打开 https://baidu.com，等待页面加载完成，"
                    "然后读取页面标题和主内容的大致摘要，最后用中文告诉我。"
                ),
            },
        ],
    }

    # Use async streaming invoke to show that the constructed agent remains
    # a normal LangGraph compiled graph.
    async for chunk in agent.astream(input_state):
        messages = chunk.get("messages", [])
        if not messages:
            continue
        # Print only the latest message content for brevity.
        last_message = messages[-1]
        if hasattr(last_message, "content"):
            print(last_message.content)  # noqa: T201


def main() -> None:
    """Entry point for running the chrome MCP subagent demo."""
    asyncio.run(demo_once())


if __name__ == "__main__":
    main()


