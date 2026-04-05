"""
core/tool_manifest.py — Auto-Generated Tool Capability Manifest (TCM).

Builds a compact capability catalog for the Supervisor prompt by scanning
the skill registry and MCP tool registry at runtime.
"""

import logging
from memory.retrieval import memory_retrieval
from mcp_servers import get_mcp_tool_summaries

logger = logging.getLogger(__name__)

def build_capability_manifest() -> str:
    """
    Build a compact, auto-generated capability catalog for the Supervisor prompt.
    
    Returns a markdown-formatted string listing all delegatable capabilities
    (core tools, MCP server tools, and registered skills).
    """
    lines = [
        "## Worker Capabilities (Available via Delegation)",
        "The following capabilities are available to the Worker agent.",
        "Delegate tasks matching these descriptions using `delegate_task(objective='...')`.",
        "",
        "- **exec_command** (core): Execute shell commands in the local terminal."
    ]
    
    # 1. Add MCP tools
    mcp_tools = get_mcp_tool_summaries()
    if mcp_tools:
        for tool in mcp_tools:
            name = tool.get("name", "")
            # Take just the first line of the description for compactness
            raw_desc = tool.get("description", "")
            desc = raw_desc.split("\n")[0].strip() if raw_desc else ""
            if name and desc:
                lines.append(f"- **{name}** (mcp): {desc}")
                
    # 2. Add Skills
    store = memory_retrieval.store
    if store and hasattr(store, "_skill_cache"):
        # The cache is populated by store.initialize_skills() which is called on startup.
        # We enforce deterministic ordering to keep the prompt stable.
        for skill_id in sorted(store._skill_cache.keys()):
            # Filter out internal/system skills that aren't valid targets for delegation
            if skill_id in ("identity", "onboarding"):
                continue
                
            cache_entry = store._skill_cache[skill_id]
            name = cache_entry.get("name", skill_id)
            desc = cache_entry.get("description", "")
            tools = cache_entry.get("tools", [])
            
            skill_line = f"- **{name}**"
            if desc:
                skill_line += f": {desc}"
            lines.append(skill_line)
            
            if tools:
                lines.append(f"  Tools: {', '.join(tools)}")

    return "\n".join(lines)
