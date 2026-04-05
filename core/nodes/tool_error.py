"""
core/nodes/tool_error.py — Explicit Error Recovery Node.

When the LLM hallucinates JSON or a tool crashes, this node intercepts
the error, increments the failure count, and prompts the LLM to auto-correct
its payload without crashing the entire graph turn.
"""

from typing import Dict, Any

def tool_error_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Called when a tool fails to execute or parse.
    Returns a ToolMessage containing the error to ask the LLM to fix it.
    """
    failures = state.get("tool_failure_count", 0) + 1
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    
    # Check if the error was triggered by a completely blank LLM generation
    if last_msg and getattr(last_msg, "type", "") == "ai":
        content = getattr(last_msg, "content", "")
        content_str = str(content).strip() if content else ""
        
        if not content_str and not getattr(last_msg, "tool_calls", []):
            # The LLM choked and output nothing. Inject a nudge.
            from langchain_core.messages import HumanMessage
            nudge = HumanMessage(content="You generated an empty response with no tool calls. Please try again and provide either a tool call or a text response.")
            return {
                "tool_failure_count": failures,
                "_retry": True, # Must be True so route_after_worker_error routes back to prompt_builder
                "messages": [nudge] 
            }
            
    # Standard tool execution or parsing error logic
    
    if failures >= 3:
        # Max retries hit, break the loop
        return {
            "tool_failure_count": 0,
            "_retry": False,
            "messages": [{"role": "system", "content": "Tool execution failed 3 times. Please apologize to the user and stop trying."}]
        }
        
    return {
        "tool_failure_count": failures,
        "_retry": True
    }
