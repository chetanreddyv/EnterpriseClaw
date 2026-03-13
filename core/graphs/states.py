"""
core/graphs/states.py — State Schemas for the Supervisor-Worker Architecture.

Two completely separate schemas. The Supervisor never sees Worker internals.
The Worker's messages are a lightweight action ledger; heavy environment 
state lives only in `observation`, which is REPLACED (not appended) each turn.
"""

from typing import Annotated, Optional, TypedDict
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
    user_input: str
    original_query: str
    
    # Context injected by the supervisor intent node
    memory_context: str
    skill_prompts: str
    active_skills: list[str]
    
    # Internal communication between PromptBuilder and Executor
    _formatted_prompt: list[BaseMessage]
    
    # Execution safety 
    tool_failure_count: int
    _retry: bool
    
    # HITL tiering: set of tool names that are /permit'd for auto-approval
    approved_tools: list[str]


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
    
    # REPLACED each turn. Current A11y tree / terminal output / diff.
    observation: str
    
    # "browser" | "exec" | "all" — determines which tools to bind
    # and whether to execute sequentially or concurrently
    tool_domain: str
    
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
    _formatted_prompt: list[BaseMessage]
    
    # HITL tiering: inherited from Supervisor's approved_tools
    approved_tools: list[str]
