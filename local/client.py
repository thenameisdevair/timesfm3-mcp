import asyncio
import json
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = Path(__file__).with_name("server.py")


async def _print_tool(session, name, arguments):
    print(f"\n=== {name} {json.dumps(arguments)[:120]} ===")
    result = await session.call_tool(name, arguments=arguments)
    for item in result.content:
        print(item.text)


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=[str(SERVER)],
    )

    print("Booting the TimesFM-3 MCP server...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            history = [10.5, 12.1, 14.8, 15.2, 18.0]
            await _print_tool(
                session,
                "forecast",
                {"history": history, "horizon": 5},
            )

            sku_a = [10, 11, 13, 12, 14, 16, 15, 17]
            sku_b = [20, 21, 22, 24, 23, 26, 25, 28]
            promo = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # T=8, H=3
            await _print_tool(
                session,
                "forecast",
                {
                    "series": [sku_a, sku_b],
                    "series_ids": ["sku_a", "sku_b"],
                    "future_covariates": [promo],
                    "horizon": 3,
                },
            )


if __name__ == "__main__":
    asyncio.run(main())
