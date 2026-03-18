"""
core/graphs/states.py — State Schemas for the Supervisor-Worker Architecture.

Two completely separate schemas. The Supervisor never sees Worker internals.
The Worker's messages are a lightweight action ledger; heavy environment 
state lives only in `observation`, which is REPLACED (not appended) each turn.
"""

from typing import Annotated, TypedDict, NotRequired, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SupervisorState(TypedDict):
    """
    Lightweight state for the Supervisor (MainGraph).
    Holds the user's persistent conversation history and long-term memory context.
    """
    # Persistent chat history. Uses add_messages reducer.
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Core identifiers
    chat_id: str
    active_model: str
    
    # User input / query tracking
    user_input: str      # Raw tool/human message content (string or list of dicts)
    original_query: str  # Normalized text extracted from user_input for retrieval
    
    # Context injected by the supervisor intent node
    memory_context: str  # Relevant facts retrieved from long-term memory (DB)
    
    # Internal communication between PromptBuilder and Executor
    _formatted_prompt: list[BaseMessage] # Pruned and finalized message list for LLM call
    
    # Execution safety 
    tool_failure_count: int
    _retry: bool
    
    # HITL policy directives, e.g. tool names for /permit and deny markers (deny:<tool>)
    approved_tools: list[str]

    # Prevent repeated delegation loops on unresolved objectives within a turn.
    delegation_attempts: dict[str, int]
    escalated_objectives: list[str]


class WorkerState(TypedDict):
    """
    Ephemeral state for Worker SubGraphs.
    
    CRITICAL DESIGN DECISIONS:
    - `messages` is a lightweight ACTION LOG only ("Clicked X", "Typed Y").
      The heavy environment state lives ONLY in `observation`.
    - `observation` is REPLACED each turn, never appended.
    - This state is destroyed when the Worker finishes. It never persists
      to the Supervisor's CheckpointDB.
    """
    # Lightweight action ledger. ToolMessages contain tiny summaries only.
    messages: Annotated[list[BaseMessage], add_messages]
    
    # The task objective passed down from the Supervisor
    objective: str
    
    # REPLACED each turn. Current browser/exec environment state.
    # May be text-only or multimodal payload blocks.
    observation: str | list[dict[str, Any]]
    
    # JIT skill context fetched once from the Worker objective
    skill_prompts: str
    active_skills: list[str]
    active_skill_tools: list[str]  # Tool names declared by matched skills
    
    # Loop control
    step_count: int
    max_steps: int           # Hard cap (e.g., 25) to prevent infinite loops
    
    # Lifecycle
    status: str              # "running" | "completed" | "failed" | "escalated"
    result_summary: str      # Final summary returned to Supervisor
    
    # Active model (inherited from Supervisor)
    active_model: str
    
    # Execution safety (reused from V2 for blank-generation recovery)
    tool_failure_count: int
    _retry: bool
    
    # Internal communication between PromptBuilder and Executor
    _formatted_prompt: list[BaseMessage] # Minimal action-observation prompt (objective + observation)
    
    # HITL policy directives inherited from Supervisor
    approved_tools: list[str]

    # Execution context metadata (e.g., interactive vs cron job)
    execution_mode: NotRequired[str]     # "interactive" | "cron"
    execution_source: NotRequired[str]   # free-form source identifier for logging/debug
