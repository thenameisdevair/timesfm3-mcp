import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = Path(__file__).with_name("server.py")


async def main():
    server_params = StdioServerParameters(
        command="python",
        args=[str(SERVER)],
    )

    print("Booting the TimesFM-3 MCP server (multivariate)...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            sku_a = [10, 11, 13, 12, 14, 16, 15, 17]
            sku_b = [20, 21, 22, 24, 23, 26, 25, 28]
            # T=8, H=3 so future covariate length must be 11
            promo = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

            print("Calling forecast on two SKUs + promo future covariate -> 3 steps")
            result = await session.call_tool(
                "forecast",
                arguments={
                    "series": [sku_a, sku_b],
                    "series_ids": ["sku_a", "sku_b"],
                    "future_covariates": [promo],
                    "horizon": 3,
                },
            )

            print("\n--- Model output ---")
            for item in result.content:
                print(item.text)

            print("\nCalling forecast with a bad future covariate length (should error)")
            bad = await session.call_tool(
                "forecast",
                arguments={
                    "series": [sku_a, sku_b],
                    "future_covariates": [[0, 1, 0]],
                    "horizon": 3,
                },
            )
            for item in bad.content:
                print(item.text)


if __name__ == "__main__":
    asyncio.run(main())
