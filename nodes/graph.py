"""
graph.py — LangGraph StateGraph definition.

Defines the AgentState, wires up all nodes, and compiles the graph
with SQLite checkpointing for persistent state across restarts.
"""

import logging
from typing import TypedDict, Optional, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from .agent import agent_node, _get_enabled_tools_and_write_actions
from .approval import human_approval_node
from .tools import execute_tools_node

logger = logging.getLogger(__name__)

# ==========================================================
# 1. State Schema
# ==========================================================

class AgentState(TypedDict):
    """Full state flowing through the LangGraph."""

    # ── Core ──────────────────────────────────────────────────
    chat_id: str                              # Telegram chat ID (= thread_id)
    user_input: str                           # Current user message
    messages: Annotated[list[BaseMessage], add_messages] # Context window

    # ── Agent ─────────────────────────────────────────────────
    agent_response: Optional[str]             # Final response text
    tool_failure_count: int                   # Self-correction counter
    approval_status: Optional[str]            # Track hitl output
    _retry: Optional[bool]                    # Self-correction retry flag (used by router)

    # ── Memory ────────────────────────────────────────────────
    memory_context: Optional[str]             # Retrieved long-term context
    context_data: Optional[dict]              # Additional system properties

    # ── Model ─────────────────────────────────────────────────
    active_model: Optional[str]               # Active LLM for this thread (e.g. "google_genai/gemini-3-flash-preview")
    approved_tools: Optional[list[str]]       # Tools that bypass HITL for this thread


# ==========================================================
# 2. Conditional Routing
# ==========================================================

def route_after_agent(state: AgentState) -> str:
    """Decide what happens after the agent node."""
    # If agent hit max failures, go straight to END
    if state.get("tool_failure_count", 0) >= 3:
        return END

    # If agent needs a retry (self-correction)
    if state.get("_retry"):
        return "agent"

    messages = state.get("messages", [])
    if not messages:
        return END

    last_message = messages[-1]

    # No tool calls? The agent is done.
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return END

    # Dynamically get the dangerous write actions
    _, write_actions = _get_enabled_tools_and_write_actions(load_all=True)
    approved_tools = state.get("approved_tools") or []

    # Check for dangerous tools requiring HITL
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        if tool_name in write_actions and tool_name not in approved_tools:
            return "human_approval"

    # Otherwise, it's a safe read tool. Execute immediately.
    return "execute_tools"

def route_after_approval(state: AgentState) -> str:
    """Decide what happens after human approval."""
    # If rejected or changed, loop back to agent with the feedback ToolMessage
    if state.get("approval_status") == "rejected":
        return "agent"
    
    # If approved, proceed to execute tools
    if state.get("approval_status") == "approved":
        return "execute_tools"
    
    # Fallback
    return "agent"


# ==========================================================
# 3. Graph Construction
# ==========================================================

def build_graph(checkpointer=None):
    """
    Build and compile the LangGraph StateGraph.
    
    Flow:
        START → agent → (conditional)
                           ├→ END
                           ├→ execute_tools → agent
                           └→ human_approval → (conditional)
                                                ├→ execute_tools
                                                └→ agent
    """
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("agent", agent_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("execute_tools", execute_tools_node)

    # Wire edges
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        ["human_approval", "execute_tools", "agent", END],
    )
    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        ["execute_tools", "agent"],
    )
    
    # Critical Loop: Execute tools always goes back to agent
    builder.add_edge("execute_tools", "agent")

    # Compile with checkpointer
    graph = builder.compile(checkpointer=checkpointer)
    logger.info("✅ LangGraph compiled successfully")
    return graph


# ==========================================================
# 4. Graph Instance (with SQLite checkpointer)
# ==========================================================

import os
from pathlib import Path

# Default path — can be overridden via the DB_PATH env var
_DEFAULT_DB_PATH = os.getenv("DB_PATH", "./data/checkpoints.db")

def checkpointer_context(db_path: str = _DEFAULT_DB_PATH):
    """
    Return an async context manager that opens an AsyncSqliteSaver.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"📂 SQLite checkpointer will use → {db_path}")
    return AsyncSqliteSaver.from_conn_string(db_path)
