"""
mcp_servers package — programmatic tool registry

Dynamically loads all plugins in the mcp_servers/ directory and aggregates their
TOOL_REGISTRY into a GLOBAL_TOOL_REGISTRY.
"""

import os
import importlib
import logging

logger = logging.getLogger(__name__)

GLOBAL_TOOL_REGISTRY = {}

def load_plugins():
    global GLOBAL_TOOL_REGISTRY
    plugin_dir = os.path.dirname(__file__)
    
    for filename in os.listdir(plugin_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = filename[:-3]
            try:
                module = importlib.import_module(f"mcp_servers.{module_name}")
                if hasattr(module, "TOOL_REGISTRY"):
                    registry = getattr(module, "TOOL_REGISTRY")
                    if isinstance(registry, dict):
                        GLOBAL_TOOL_REGISTRY.update(registry)
                        logger.info(f"✅ Loaded {len(registry)} tools from plugin: {module_name}")
            except Exception as e:
                logger.error(f"❌ Failed to load plugin {module_name}: {e}")

load_plugins()
