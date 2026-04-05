import asyncio

async def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Load plugins first
    from mcp_servers import load_plugins
    load_plugins()
    
    # Initialize memory (which loads skills)
    from memory.retrieval import memory_retrieval
    await memory_retrieval.initialize()
    
    # Initialize external MCP tools
    from mcp_servers.external_mcp import initialize_external_mcp_tools
    await initialize_external_mcp_tools()
    
    # Print TCM
    from core.tool_manifest import build_capability_manifest
    print("\n--- TCM OUTPUT ---")
    print(build_capability_manifest())
    print("------------------\n")

if __name__ == "__main__":
    asyncio.run(main())
