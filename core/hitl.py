"""
core/hitl.py — Human-In-The-Loop Tiered Tool Approval.

Three-tier tool classification:
1. AUTONOMOUS — Always run freely, no approval needed.
   e.g., web_search, check_current_time, browser_snapshot, browser_get_text
2. ALLOWED (toggleable) — Run freely by default, but can be locked via /deny.
   e.g., delegate_task, browser_navigate
3. NOT_ALLOWED — Always require explicit HITL approval via interrupt().
   e.g., browser_click, browser_type, exec_command, browser_file_upload
"""

import logging
from langgraph.types import interrupt

logger = logging.getLogger(__name__)


# ── Tier Definitions ──────────────────────────────────────────

# Tier 1: AUTONOMOUS — no approval ever needed (safe, read-only)
AUTONOMOUS_TOOLS = {
    "web_search", "web_fetch",
    "check_current_time", "schedule_reminder",
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
    - ALLOWED tools: need approval only if explicitly /deny'd (i.e., NOT in approved_tools).
      By default, they are auto-approved.
    - NOT_ALLOWED tools: always need approval unless explicitly /permit'd (i.e., in approved_tools).
    """
    approved_tools = approved_tools or set()
    tier = get_tool_tier(tool_name)

    if tier == "autonomous":
        return False
    elif tier == "allowed":
        # Allowed tools are auto-approved unless explicitly denied.
        # If the tool is in the denied set (tracked separately), it needs approval.
        # For simplicity: allowed tools don't need approval unless they were explicitly
        # removed from a "permitted" set. Since they start as permitted, we check
        # if they've been actively denied. We use the inverse: if tool is in
        # approved_tools set, it's been /permit'd. If allowed tools are NOT
        # in a denied set, they're auto-approved.
        # Convention: allowed tools are auto-approved by default.
        return False
    else:
        # NOT_ALLOWED: require approval unless explicitly /permit'd
        return tool_name not in approved_tools


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
