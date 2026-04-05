"""
mcp_servers/external_mcp.py - Core integration for external MCP tools.
Reads configuration from mcp_config.json, initializes the MultiServerMCPClient,
and registers external tools into the existing GLOBAL_TOOL_REGISTRY.
"""

import os
import json
import logging
from pathlib import Path

# Note: We import GLOBAL_TOOL_REGISTRY and GLOBAL_TOOL_METADATA dynamically below to avoid circular imports.
from config.settings import settings

logger = logging.getLogger(__name__)

_mcp_client = None

def load_mcp_config() -> dict | None:
    """Read the MCP server config from mcp_config_path if it exists."""
    config_path = Path(settings.mcp_config_path)
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            full_config = json.load(f)
            # Support Claude Desktop format: {"mcpServers": { "server_name": {...} }}
            mcp_servers = full_config.get("mcpServers", {})
            # We pass the inner dict directly to MultiServerMCPClient
            return mcp_servers
    except Exception as e:
        logger.warning(f"Failed to read MCP config at {config_path}: {e}")
        return None

async def initialize_external_mcp_tools() -> int:
    """
    Initialize MCP tools if configured, and inject them into GLOBAL_TOOL_REGISTRY.
    """
    global _mcp_client
    config = load_mcp_config()
    if not config:
        logger.debug("No external MCP configuration found, skipping.")
        return 0
    
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        logger.warning("langchain-mcp-adapters is not installed.")
        return 0

    try:
        _mcp_client = MultiServerMCPClient(config)
        # Using get_tools() defaults to stateless mode where it retrieves tools from all servers
        # and wraps them so that they spawn a session per call
        tools = await _mcp_client.get_tools()
    except Exception as e:
        logger.error(f"Failed to initialize external MCP tools: {e}", exc_info=True)
        return 0

    if not tools:
        return 0

    # Lazy import to avoid circular dependency
    from mcp_servers import GLOBAL_TOOL_REGISTRY, GLOBAL_TOOL_METADATA

    registered_count = 0
    for tool in tools:
        # Prevent overriding of core local tools
        if tool.name in GLOBAL_TOOL_REGISTRY:
            logger.warning(
                f"MCP tool '{tool.name}' conflicts with a local core tool. Skipping MCP version."
            )
            continue
        
        # Add to the global registry
        GLOBAL_TOOL_REGISTRY[tool.name] = tool
        # Tag as external MCP tool, stateless
        GLOBAL_TOOL_METADATA[tool.name] = {
            "category": "mcp",
            "stateful": False,
            "observation_mode": "head"
        }
        registered_count += 1

    return registered_count

async def shutdown_external_mcp() -> None:
    """Shutdown active MCP background processes."""
    global _mcp_client
    if _mcp_client:
        try:
            # MultiServerMCPClient might implement an async close method.
            if hasattr(_mcp_client, "close") and callable(_mcp_client.close):
                await _mcp_client.close()
        except Exception as e:
            logger.error(f"Error shutting down MCP client: {e}")
        finally:
            _mcp_client = None
