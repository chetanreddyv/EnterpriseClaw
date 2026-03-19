"""
core/graphs/worker.py — The Worker SubGraph Definition.

An ephemeral, specialized graph for multi-step task execution.
Operates in a strict Action-Observation loop with State Amnesia.

Pipeline: SkillContext → PromptBuilder → Executor → [Tools → PromptBuilder | Summarize → END]
                                                        ↑              |
                                                        └──────────────┘

Key properties:
- No persistent memory or conversation history.
- observation is REPLACED each turn (State Amnesia).
- messages contain only lightweight action summaries.
- Sequential execution for stateful domains (browser, exec).
- Hard step limit prevents infinite loops.
- escalate_to_supervisor allows graceful bailout.
"""

from langgraph.graph import StateGraph, END
from langgraph.types import RetryPolicy

from core.graphs.states import WorkerState
from config.settings import settings
from core.nodes.worker_nodes import (
    worker_skill_context_node,
    worker_prompt_builder_node,
    worker_executor_node,
    worker_tools_node,
    worker_summarize_node,
)
from core.nodes.tool_error import tool_error_node


def route_after_worker_executor(state: WorkerState) -> str:
    """Determine where to go after the Worker LLM generates a response."""
    # Blank generation / transport error recovery must take priority.
    if state.get("_retry"):
        return "tool_error"

    messages = state.get("messages", [])
    if not messages:
        return "summarize"

    last_msg = messages[-1]

    # If the LLM made tool calls, execute them
    if getattr(last_msg, "tool_calls", []):
        return "tools"

    # No tool calls = LLM is done, go to summarize
    return "summarize"


def route_after_worker_tools(state: WorkerState) -> str:
    """After tools finish, check if we should continue or stop."""
    # Check for escalation
    if state.get("status") == "escalated":
        return "summarize"

    # Check step limit
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", settings.worker_max_steps)
    if step_count >= max_steps:
        return "summarize"

    # Continue the loop
    return "prompt_builder"


def route_after_worker_error(state: WorkerState) -> str:
    """After error handling, check if we should retry or give up."""
    if not state.get("_retry"):
        return "summarize"

    # Check step limit even during retries
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", settings.worker_max_steps)
    if step_count >= max_steps:
        return "summarize"

    return "prompt_builder"


def build_worker_graph(checkpointer=None):
    """
    Constructs a generic Worker graph.

    The Worker receives only an objective and dynamically binds tools from
    matched skill frontmatter declarations.
    
    Args:
        checkpointer: Optional LangGraph checkpointer. When provided (e.g. for
            cron sessions), the Worker accumulates history across runs in the
            same thread_id, enabling stateful background tasks.
    """
    workflow = StateGraph(WorkerState)

    retry_policy = RetryPolicy(max_attempts=3)

    # Add Nodes
    workflow.add_node("skill_context", worker_skill_context_node, retry=retry_policy)
    workflow.add_node("prompt_builder", worker_prompt_builder_node, retry=retry_policy)
    workflow.add_node("executor", worker_executor_node, retry=retry_policy)
    workflow.add_node("tools", worker_tools_node, retry=retry_policy)
    workflow.add_node("tool_error", tool_error_node)
    workflow.add_node("summarize", worker_summarize_node)

    # Entry Point
    workflow.set_entry_point("skill_context")

    # Core flow
    workflow.add_edge("skill_context", "prompt_builder")
    workflow.add_edge("prompt_builder", "executor")

    # Conditional routing post-executor
    workflow.add_conditional_edges("executor", route_after_worker_executor, {
        "tools": "tools",
        "tool_error": "tool_error",
        "summarize": "summarize",
    })

    # Conditional routing post-tools (check escalation + step limit)
    workflow.add_conditional_edges("tools", route_after_worker_tools, {
        "prompt_builder": "prompt_builder",
        "summarize": "summarize",
    })

    # Error handling routes
    workflow.add_conditional_edges("tool_error", route_after_worker_error, {
        "prompt_builder": "prompt_builder",
        "summarize": "summarize",
    })

    # Summarize always ends
    workflow.add_edge("summarize", END)

    return workflow.compile(checkpointer=checkpointer)
