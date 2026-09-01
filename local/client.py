import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Point the client to your local server file
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"]
    )

    print("Booting up the MCP Server...")
    
    # Open the connection
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the handshake
            await session.initialize()
            
            # Prepare our dummy data
            dummy_history = [10.5, 12.1, 14.8, 15.2, 18.0]
            horizon_steps = 5
            
            print(f"Sending data to TimesFM: {dummy_history}")
            
            # Call the tool exactly as an AI agent would
            result = await session.call_tool(
                "forecast_demand",
                arguments={"history": dummy_history, "horizon": horizon_steps}
            )
            
            # Print the raw output from TimesFM
            print("\n--- Model Output ---")
            for item in result.content:
                print(item.text)

if __name__ == "__main__":
    asyncio.run(main())