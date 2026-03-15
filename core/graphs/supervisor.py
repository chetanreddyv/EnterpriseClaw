"""
core/graphs/supervisor.py — The Supervisor (MainGraph) Definition.

A lightweight conversational graph that handles user interaction and
delegates complex multi-step tasks to Worker SubGraphs.

Pipeline: Intent → PromptBuilder → Executor → [Tools | END]
                     ↑                           |
                     └───────────────────────────┘
"""

from langgraph.graph import StateGraph, END
from langgraph.types import RetryPolicy

from core.graphs.states import SupervisorState
from core.nodes.supervisor_nodes import (
    supervisor_intent_node,
    supervisor_prompt_builder_node,
    supervisor_executor_node,
    supervisor_compaction_node,
    supervisor_tools_node,
)
from core.nodes.tool_error import tool_error_node


def route_after_supervisor_executor(state: SupervisorState) -> str:
    """Determine where to go after the Supervisor LLM generates a response."""
    messages = state.get("messages", [])
    if not messages:
        return END

    last_msg = messages[-1]

    # Custom error recovery: blank generation from local models
    if state.get("_retry"):
        return "tool_error"

    if getattr(last_msg, "tool_calls", []):
        return "tools"

    return "compact"


def build_supervisor_graph(checkpointer=None):
    """Constructs the Supervisor graph — the user-facing conversational agent."""
    workflow = StateGraph(SupervisorState)

    # Retry policy for infrastructure resilience
    retry_policy = RetryPolicy(max_attempts=3)

    # Add Nodes
    workflow.add_node("intent", supervisor_intent_node, retry=retry_policy)
    workflow.add_node("prompt_builder", supervisor_prompt_builder_node, retry=retry_policy)
    workflow.add_node("executor", supervisor_executor_node, retry=retry_policy)
    workflow.add_node("tools", supervisor_tools_node, retry=retry_policy)
    workflow.add_node("compact", supervisor_compaction_node, retry=retry_policy)
    workflow.add_node("tool_error", tool_error_node)

    # Entry Point
    workflow.set_entry_point("intent")

    # Core flow
    workflow.add_edge("intent", "prompt_builder")
    workflow.add_edge("prompt_builder", "executor")

    # Conditional routing post-executor
    workflow.add_conditional_edges("executor", route_after_supervisor_executor, {
        "tools": "tools",
        "tool_error": "tool_error",
        "compact": "compact",
        END: END,
    })

    # Loop backs
    workflow.add_edge("tools", "prompt_builder")
    workflow.add_edge("tool_error", "prompt_builder")
    workflow.add_edge("compact", END)

    return workflow.compile(checkpointer=checkpointer)
