import asyncio
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main() -> None:
    # 1. 启动 chrome-devtools-mcp 服务器（通过 npx）
    mcp_client = MultiServerMCPClient(
        {
            "chrome-devtools": {
                "command": "npx",
                "args": ["chrome-devtools-mcp@latest", "--headless=false"],
                "transport": "stdio",
            },
        }
    )

    # 2. 获取所有 MCP 工具
    tools = await mcp_client.get_tools()

    # 3. 找到 navigate_page 和 evaluate_script 这两个工具
    navigate_tool = None
    eval_tool = None
    for t in tools:
        name = getattr(t, "name", "")
        if name == "navigate_page":
            navigate_tool = t
        elif name == "evaluate_script":
            eval_tool = t

    if not navigate_tool:
        raise RuntimeError("navigate_page 工具未找到，请检查 chrome-devtools-mcp 版本。")
    if not eval_tool:
        raise RuntimeError("evaluate_script 工具未找到，请检查 chrome-devtools-mcp 版本。")

    # 4. 调用 navigate_page 打开百度
    print("Navigating to https://www.baidu.com ...")
    await navigate_tool.ainvoke(
        {
            "url": "https://www.baidu.com",
            # 可选字段：根据实际 schema 调整
            "wait_until": "load",
        }
    )

    # 5. 用 evaluate_script 读取 document.title
    result: Any = await eval_tool.ainvoke(
        {
            "expression": "document.title",
        }
    )
    # 不同实现返回结构可能略有差异，这里假设 result 本身就是标题或带有 value 字段
    title = result.get("value") if isinstance(result, dict) else str(result)
    print("Page title:", title)


if __name__ == "__main__":
    asyncio.run(main())