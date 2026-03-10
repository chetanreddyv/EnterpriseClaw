"""
nodes/approval.py — Human-in-the-Loop approval node (Native LangGraph).

Uses LangGraph's interrupt() to pause the graph when a write action
is detected by the router. 

If approved, routes to execution.
If rejected, appends a ToolMessage representing the rejection back to the agent.
"""

import logging
from langgraph.types import interrupt
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)

async def human_approval_node(state: dict) -> dict:
    """
    Pause the graph and wait for human approval via Telegram buttons.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}
        
    last_message = messages[-1]
    
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {}

    # Gather tool calls that caused this interrupt
    # The router decided to send us here, so we review *all* tools in this step
    action_names = [call["name"] for call in last_message.tool_calls]
    
    # We display the first one for simplicity, or we can format all of them.
    # Let's format the interrupt payload
    first_call = last_message.tool_calls[0]
    action_name = first_call["name"]
    tool_args = first_call["args"]

    interrupt_payload = {
        "action": action_name,
        "skill": "unknown", # Skill mapping can be added back if needed
        "tool_args": tool_args,
        "details": f"Call **{action_name}** with args: {tool_args}",
    }

    # ── PAUSE ── LangGraph saves state; resumes when user responds ──
    decision = interrupt(interrupt_payload)

    if decision == "approve":
        # On approve, we just return an action state or let the router guide it
        # Actually, if approved, we want the graph to continue to `execute_tools`.
        # We can communicate this via a simple state update, or just return nothing since the edge will route it.
        # Wait, the edge from human_approval_node should be conditional!
        return {"approval_status": "approved"}

    elif isinstance(decision, str) and decision.startswith("edit:"):
        edit_instruction = decision[5:].strip()
        
        # Append ToolMessages for all interrupted tools indicating feedback
        tool_messages = []
        for call in last_message.tool_calls:
            tool_messages.append(
                ToolMessage(
                    content=f"User feedback/edit requested: {edit_instruction}. Please adjust your action.",
                    tool_call_id=call["id"]
                )
            )
        return {"messages": tool_messages, "approval_status": "rejected"}

    else:
        # Rejected 
        tool_messages = []
        for call in last_message.tool_calls:
            tool_messages.append(
                ToolMessage(
                    content=f"User rejected execution of this tool.",
                    tool_call_id=call["id"]
                )
            )
        return {"messages": tool_messages, "approval_status": "rejected"}
