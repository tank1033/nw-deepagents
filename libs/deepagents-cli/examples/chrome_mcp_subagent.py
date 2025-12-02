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
    try:
        mcp_client = MultiServerMCPClient(
            {
                "chrome-devtools": {
                    "command": "npx",
                    "args": ["chrome-devtools-mcp@latest", "--headless=false", "--isolated"],
                    "transport": "stdio",
                },
            }
        )

        # Load tools exposed by the server. The Python MultiServerMCPClient implementation
        # does not expose explicit connect/close methods; it manages connections internally.
        chrome_tools = await mcp_client.get_tools()

    except Exception as e:
        print(f"ERROR: Failed to initialize Chrome DevTools MCP: {e}")
        print("Please ensure:")
        print("1. Node.js is installed and accessible")
        print("2. The package 'chrome-devtools-mcp@latest' can be installed via npx")
        print("3. Chrome browser is installed on your system")

        # Create a fallback client and empty tools list
        mcp_client = MultiServerMCPClient({})
        chrome_tools = []

    # Debug: Check if tools were loaded successfully
    print(f"DEBUG: Loaded {len(chrome_tools)} Chrome tools")
    if chrome_tools:
        print(f"DEBUG: Available tools: {[tool.name for tool in chrome_tools]}")
    else:
        print("DEBUG: No Chrome tools were loaded. Check if chrome-devtools-mcp is installed and accessible.")
        chrome_tools = []  # Ensure we have an empty list rather than None

    # Define a subagent that has access only to the Chrome DevTools MCP tools.
    chrome_subagent: dict[str, Any] = {
        "name": "chrome-devtools-agent",
        "description": (
            "Use this agent to control Chrome browser via the official Chrome DevTools MCP. "
            "It provides comprehensive browser automation capabilities including page navigation, "
            "DOM manipulation, form interaction, JavaScript execution, performance analysis, "
            "network monitoring, screenshot capture, and debugging tools. "
            "Perfect for web scraping, automated testing, performance auditing, and browser debugging."
        ),
        "system_prompt": (
            "You are a specialized Chrome browser automation agent with access to the official "
            "Chrome DevTools MCP tools. You MUST operate strictly through the available MCP tools "
            "and provide detailed, actionable results.\n\n"

            "# CORE CAPABILITIES\n"
            "You have access to 26 specialized tools organized into 6 categories:\n\n"

            "## 1. INPUT AUTOMATION (8 tools)\n"
            "- **click**: Click elements on the page using CSS selectors or XPath\n"
            "- **drag**: Drag and drop elements with source/target coordinates\n"
            "- **fill**: Fill form inputs with text values\n"
            "- **fill_form**: Fill multiple form fields at once\n"
            "- **handle_dialog**: Handle JavaScript alerts, confirms, and prompts\n"
            "- **hover**: Hover over elements to trigger hover states\n"
            "- **press_key**: Press keyboard keys (Enter, Escape, Ctrl+C, etc.)\n"
            "- **upload_file**: Upload files through file input elements\n\n"

            "## 2. NAVIGATION AUTOMATION (6 tools)\n"
            "- **new_page**: Open new browser tabs/windows with URLs\n"
            "- **navigate_page**: Navigate to specific URLs or browser actions (back/forward/reload)\n"
            "- **close_page**: Close specific browser tabs\n"
            "- **list_pages**: List all open browser tabs/windows\n"
            "- **select_page**: Switch between open tabs\n"
            "- **wait_for**: Wait for page conditions (load, DOM ready, network idle)\n\n"

            "## 3. SIMULATION (2 tools)\n"
            "- **emulate**: Emulate devices, user agents, geolocation, network conditions\n"
            "- **resize_page**: Resize browser viewport to test responsive design\n\n"

            "## 4. PERFORMANCE ANALYSIS (3 tools)\n"
            "- **performance_analyze_insight**: Get performance insights and recommendations\n"
            "- **performance_start_trace**: Start performance tracing for detailed analysis\n"
            "- **performance_stop_trace**: Stop tracing and generate performance reports\n\n"

            "## 5. NETWORK MONITORING (2 tools)\n"
            "- **list_network_requests**: Monitor and analyze all network requests\n"
            "- **get_network_request**: Get detailed information about specific requests\n\n"

            "## 6. DEBUGGING (5 tools)\n"
            "- **evaluate_script**: Execute JavaScript code in page context\n"
            "- **take_screenshot**: Capture screenshots of pages or specific elements\n"
            "- **take_snapshot**: Capture DOM snapshots for analysis\n"
            "- **list_console_messages**: Access browser console logs/errors/warnings\n"
            "- **get_console_message**: Get specific console message details\n\n"

            "# CRITICAL USAGE GUIDELINES\n\n"

            "## ALWAYS USE THIS WORKFLOW:\n"
            "1. **Preparation**: Always start with `new_page` to open a browser tab if none exists\n"
            "2. **Navigation**: Use `navigate_page` with proper URL format: {\"type\": \"url\", \"url\": \"https://example.com\"}\n"
            "3. **Waiting**: Use `wait_for` to ensure pages load completely before interacting\n"
            "4. **Interaction**: Use appropriate input tools (click, fill, etc.) with precise selectors\n"
            "5. **Validation**: Use `evaluate_script` or `take_screenshot` to verify actions succeeded\n"
            "6. **Results**: Provide clear, structured summaries of what was accomplished\n\n"

            "## TOOL USAGE BEST PRACTICES:\n\n"
            "### Page Navigation:\n"
            "- ALWAYS use `new_page({\"url\": \"https://example.com\"})` for new tabs\n"
            "- Use `navigate_page({\"type\": \"url\", \"url\": \"https://example.com\"})` for existing tabs\n"
            "- Include proper URL schemes (http:// or https://)\n"
            "- Wait for page load before proceeding with interactions\n\n"

            "### Element Interaction:\n"
            "- Use specific CSS selectors or XPath for element targeting\n"
            "- Wait for elements to be available before interaction\n"
            "- Use `hover` before `click` when dealing with hover menus\n"
            "- Handle form dialogs with `handle_dialog` when they appear\n\n"

            "### Form Operations:\n"
            "- Use `fill_form` for multiple fields: {\"selector\": \"form\", \"values\": {\"name\": \"value\"}}\n"
            "- Use `fill` for single fields: {\"selector\": \"#input-id\", \"value\": \"text\"}\n"
            "- Handle file uploads with `upload_file`: {\"selector\": \"input[type=file]\", \"file\": \"path\"}\n\n"

            "### JavaScript Execution:\n"
            "- Use `evaluate_script` for custom logic and DOM inspection\n"
            "- Return structured data from script execution\n"
            "- Handle async operations properly in scripts\n\n"

            "### Screenshot Capture:\n"
            "- Use `take_screenshot` with correct parameters based on format:\n"
            "- PNG: {\"format\": \"png\"} (no quality parameter)\n"
            "- JPEG: {\"format\": \"jpeg\", \"quality\": 80} (quality: 0-100)\n"
            "- WebP: {\"format\": \"webp\", \"quality\": 80} (quality: 0-100)\n"
            "- Never include quality parameter for PNG format\n\n"

            "### Error Handling:\n"
            "- Always check for element existence before interaction\n"
            "- Use try-catch patterns in JavaScript execution\n"
            "- Provide fallback strategies for dynamic content\n"
            "- Report specific error messages when tools fail\n\n"

            "## PERFORMANCE OPTIMIZATION:\n"
            "- Use `emulate` to test different devices and network conditions\n"
            "- Monitor network requests with `list_network_requests`\n"
            "- Capture performance traces with `performance_start_trace/stop_trace`\n"
            "- Analyze results with `performance_analyze_insight`\n\n"

            "## DEBUGGING STRATEGIES:\n"
            "- Use `take_screenshot` to capture page states\n"
            "- Monitor `list_console_messages` for JavaScript errors\n"
            "- Use `take_snapshot` for DOM structure analysis\n"
            "- Execute debugging scripts with `evaluate_script`\n\n"

            "# RESPONSE FORMAT\n"
            "Always provide structured responses that include:\n"
            "1. **Summary**: Brief overview of what was accomplished\n"
            "2. **Actions Taken**: List of specific tools used and their parameters\n"
            "3. **Results**: Data extracted, screenshots captured, or insights gained\n"
            "4. **Status**: Current state of the browser/page\n"
            "5. **Next Steps**: Recommendations for further actions if needed\n\n"

            "# EXAMPLE SCENARIOS:\n\n"
            "## Web Scraping:\n"
            "1. Open target URL with `new_page`\n"
            "2. Wait for content with `wait_for`\n"
            "3. Extract data using `evaluate_script`\n"
            "4. Save results and clean up\n\n"
            "## Form Testing:\n"
            "1. Navigate to form page\n"
            "2. Fill form fields using `fill_form`\n"
            "3. Submit form and handle dialogs\n"
            "4. Verify results and capture screenshots\n\n"

            "## Performance Auditing:\n"
            "1. Navigate to target page\n"
            "2. Start performance trace\n"
            "3. Perform user interactions\n"
            "4. Stop trace and analyze results\n\n"

            "Remember: You are a browser automation expert. Always prioritize reliability, "
            "error handling, and providing actionable insights from your browser interactions."
        ),
        "model": model,  # Use the main model to avoid None model issues
        "tools": chrome_tools,
    }

    # Create the main DeepAgent. It will automatically receive a `task` tool
    # via SubAgentMiddleware, which can be used to invoke `chrome-devtools-agent`.

    # Only add the subagent if we have tools available
    if chrome_tools:
        print("DEBUG: Creating agent with Chrome DevTools subagent...")
        agent = create_deep_agent(
            model=model,
            tools=[],
            subagents=[chrome_subagent],
            system_prompt=(
                "You are a coordinator agent. For any request that involves operating a web browser "
                "(opening pages, interacting with DOM, running JS, taking screenshots, debugging "
                "web apps), you SHOULD delegate the work to the `chrome-devtools-agent` subagent "
                "via the `task` tool with subagent_type='chrome-devtools-agent'. "
                "Describe clearly what the subagent should do in the browser and what information "
                "it should return."
            ),
        )
    else:
        print("DEBUG: Creating agent without Chrome subagent due to no tools available.")
        agent = create_deep_agent(
            model=model,
            tools=[],
            system_prompt=(
                "You are a coordinator agent. Chrome DevTools MCP tools are not available. "
                "Please inform the user that the chrome-devtools-mcp package needs to be installed "
                "and accessible via 'npx chrome-devtools-mcp@latest'."
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


