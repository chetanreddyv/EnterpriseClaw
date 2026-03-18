"""
mcp_servers package — programmatic tool registry

Dynamically loads all plugins in the mcp_servers/ directory and aggregates their
TOOL_REGISTRY into a GLOBAL_TOOL_REGISTRY.
"""

import os
import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

GLOBAL_TOOL_REGISTRY: dict[str, Any] = {}
GLOBAL_TOOL_METADATA: dict[str, dict[str, Any]] = {}


def _derive_category(module_name: str) -> str:
    """Derive a tool category from module filename stem."""
    if module_name.endswith("_tools"):
        return module_name[:-6]
    return module_name

def load_plugins():
    global GLOBAL_TOOL_REGISTRY, GLOBAL_TOOL_METADATA
    plugin_dir = os.path.dirname(__file__)

    GLOBAL_TOOL_REGISTRY.clear()
    GLOBAL_TOOL_METADATA.clear()
    
    for filename in sorted(os.listdir(plugin_dir)):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = filename[:-3]
            category = _derive_category(module_name)
            try:
                module = importlib.import_module(f"mcp_servers.{module_name}")
                if hasattr(module, "TOOL_REGISTRY"):
                    registry = getattr(module, "TOOL_REGISTRY")
                    if isinstance(registry, dict):
                        for tool_name, func in registry.items():
                            if tool_name in GLOBAL_TOOL_REGISTRY:
                                previous_module = GLOBAL_TOOL_METADATA.get(tool_name, {}).get("module", "unknown")
                                logger.warning(
                                    "Tool '%s' from plugin '%s' overrides previous definition from '%s'.",
                                    tool_name,
                                    module_name,
                                    previous_module,
                                )

                            tool_category = "browser" if tool_name == "batch_actions" else category

                            GLOBAL_TOOL_REGISTRY[tool_name] = func
                            GLOBAL_TOOL_METADATA[tool_name] = {
                                "module": module_name,
                                "category": tool_category,
                                # Default execution characteristics used by Worker runtime.
                                "stateful": (
                                    tool_name.startswith("browser_")
                                    or tool_name.startswith("exec_")
                                    or tool_name == "batch_actions"
                                ),
                                "observation_mode": "tail" if tool_name.startswith("exec_") else "head",
                            }

                        logger.info(
                            "✅ Loaded %d tools from plugin: %s [category=%s]",
                            len(registry),
                            module_name,
                            category,
                        )
            except Exception as e:
                logger.error(f"❌ Failed to load plugin {module_name}: {e}")

load_plugins()
