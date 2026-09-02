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

    print("Booting the TimesFM-3 MCP server (calendar labels)...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            history = [10, 11, 13, 12, 14, 16, 15, 17]
            print("Calling forecast with start=2026-08-25 freq=D -> 3 steps")
            result = await session.call_tool(
                "forecast",
                arguments={
                    "history": history,
                    "horizon": 3,
                    "start": "2026-08-25",
                    "freq": "D",
                },
            )
            print("\n--- Dated output ---")
            for item in result.content:
                print(item.text)

            print("\nCalling forecast with a gapped timestamp list (should error)")
            bad = await session.call_tool(
                "forecast",
                arguments={
                    "history": [1.0, 2.0, 3.0],
                    "horizon": 2,
                    "timestamps": ["2026-08-01", "2026-08-02", "2026-08-04"],
                },
            )
            for item in bad.content:
                print(item.text)


if __name__ == "__main__":
    asyncio.run(main())
