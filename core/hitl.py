"""
core/hitl.py — Human-In-The-Loop Tiered Tool Approval.

Three-tier tool classification:
1. AUTONOMOUS — Always run freely, no approval needed.
    e.g., web_search, browser_snapshot, browser_get_text
2. ALLOWED (toggleable) — Run freely by default, but can be locked via /deny.
   e.g., delegate_task, browser_navigate
3. NOT_ALLOWED — Always require explicit HITL approval via interrupt().
   e.g., browser_click, browser_type, exec_command, browser_file_upload
"""

import logging
from langgraph.types import interrupt

logger = logging.getLogger(__name__)

DENY_PREFIX = "deny:"


# ── Tier Definitions ──────────────────────────────────────────

# Tier 1: AUTONOMOUS — no approval ever needed (safe, read-only)
AUTONOMOUS_TOOLS = {
    "web_search", "web_fetch",
    "save_to_long_term_memory",
    "browser_get_text", "browser_screenshot", "browser_snapshot",
    "browser_go_back", "browser_scroll", "browser_wait_for",
    "browser_tab_management",
    "escalate_to_supervisor",
}

# Tier 2: ALLOWED — auto-approved by default, user can /deny to gate them
ALLOWED_TOOLS = {
    "delegate_task",
    "browser_navigate",
}

# Tier 3: NOT_ALLOWED — always require explicit HITL approval
NOT_ALLOWED_TOOLS = {
    "browser_click", "browser_type", "browser_execute_js",
    "browser_select_option", "browser_press_key", "browser_hover",
    "browser_handle_dialog", "browser_file_upload",
    "exec_command", "write_file", "delete_file",
    "batch_actions",
}


def get_deny_marker(tool_name: str) -> str:
    """Return the persisted deny marker for a tool."""
    return f"{DENY_PREFIX}{tool_name}"


def _resolve_policy_sets(policy_entries: set[str] | None) -> tuple[set[str], set[str]]:
    """Split persisted policy directives into permitted and denied tool sets."""
    permitted: set[str] = set()
    denied: set[str] = set()

    for entry in policy_entries or set():
        if not isinstance(entry, str):
            continue
        value = entry.strip()
        if not value:
            continue

        if value.startswith(DENY_PREFIX):
            denied_tool = value[len(DENY_PREFIX):].strip()
            if denied_tool:
                denied.add(denied_tool)
        else:
            permitted.add(value)

    return permitted, denied


def get_tool_tier(tool_name: str) -> str:
    """Return the tier of a tool: 'autonomous', 'allowed', or 'not_allowed'."""
    if tool_name in AUTONOMOUS_TOOLS:
        return "autonomous"
    elif tool_name in ALLOWED_TOOLS:
        return "allowed"
    elif tool_name in NOT_ALLOWED_TOOLS:
        return "not_allowed"
    else:
        # Unknown tools default to not_allowed (safe by default)
        return "not_allowed"


def requires_approval(tool_name: str, approved_tools: set = None) -> bool:
    """
    Check if a tool requires HITL approval.
    
    - AUTONOMOUS tools: never need approval.
    - ALLOWED tools: need approval only if explicitly /deny'd.
    - NOT_ALLOWED tools: always need approval unless explicitly /permit'd (i.e., in approved_tools).
    """
    approved_tools = approved_tools or set()
    permitted_tools, denied_tools = _resolve_policy_sets(approved_tools)
    tier = get_tool_tier(tool_name)

    if tier == "autonomous":
        return False
    elif tier == "allowed":
        # Allowed tools are auto-approved unless explicitly denied.
        return tool_name in denied_tools
    else:
        # NOT_ALLOWED: require approval unless explicitly /permit'd
        return tool_name not in permitted_tools


def request_tool_approval(tool_name: str, tool_args: dict) -> dict:
    """
    Fire a LangGraph interrupt() to request HITL approval.
    Returns the user's decision (the resume value from app.py).
    """
    logger.info(f"🔐 HITL: Requesting approval for '{tool_name}' with args: {tool_args}")
    decision = interrupt({
        "action": tool_name,
        "tool_args": tool_args,
    })
    return decision
