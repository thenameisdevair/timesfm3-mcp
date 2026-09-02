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

    print("Booting the TimesFM-3 MCP server (univariate)...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            dummy_history = [10.5, 12.1, 14.8, 15.2, 18.0]
            horizon_steps = 5
            print(f"Calling forecast on {dummy_history} -> {horizon_steps} steps")

            result = await session.call_tool(
                "forecast",
                arguments={"history": dummy_history, "horizon": horizon_steps},
            )

            print("\n--- Model output ---")
            for item in result.content:
                print(item.text)


if __name__ == "__main__":
    asyncio.run(main())
