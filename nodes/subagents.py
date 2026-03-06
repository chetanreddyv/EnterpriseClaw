"""
nodes/subagents.py — General-purpose autonomous sub-agent.

A powerful sub-agent with access to ALL tools. Designed for long-running,
multi-step tasks like browser automation, research, and data gathering.
Runs without HITL interruptions (browser tools execute directly).
Has a max_actions safety limit to prevent infinite loops.

The main agent delegates tasks here via `delegate_task`.
"""

import logging
from typing import Optional
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

from nodes.graph import AgentState
from nodes.tools import execute_tools_node
from mcp_servers import GLOBAL_TOOL_REGISTRY

logger = logging.getLogger(__name__)

# ── Safety ────────────────────────────────────────────────────
MAX_ACTIONS = 50  # Hard limit on tool calls per sub-agent invocation
# Tools that are NEVER available to sub-agents (too dangerous for autonomous use)
BLOCKED_TOOLS = {"exec_command", "delegate_task", "delegate_research"}


# ══════════════════════════════════════════════════════════════
# Sub-Agent Node
# ══════════════════════════════════════════════════════════════

async def general_subagent_node(state: AgentState) -> dict:
    """
    General-purpose autonomous sub-agent. Has access to all tools except
    shell commands. Loads relevant skill prompts via MemoryGate.
    """
    logger.info("--- [Node: General Sub-Agent] ---")

    messages = state.get("messages", [])
    user_input = state.get("user_input", "")
    tool_failure_count = state.get("tool_failure_count", 0)
    action_count = state.get("action_count", 0)

    # Safety: check max actions
    if action_count >= MAX_ACTIONS:
        logger.warning(f"  -> Sub-agent hit max_actions limit ({MAX_ACTIONS})")
        return {
            "messages": [AIMessage(content=(
                f"⚠️ I've reached the maximum action limit ({MAX_ACTIONS} tool calls). "
                "Here's what I've accomplished so far based on the conversation above. "
                "Please review and let me know if you'd like me to continue."
            ))],
            "tool_failure_count": tool_failure_count,
        }

    # ── Identity Prompt ───────────────────────────────────────
    identity = (
        "You are an **autonomous General Sub-Agent** — a jack of all trades.\n"
        "You have been delegated a task by the main agent. Complete it thoroughly.\n\n"
        "## Your Rules\n"
        "1. You have FULL access to browser tools and web tools. Use them freely.\n"
        "2. You are **persistent**. Do NOT give up after one failure. Try at least 3-5 approaches.\n"
        "3. After every browser action, ALWAYS inspect the result (use browser_get_text, browser_snapshot, or browser_screenshot).\n"
        "4. If a URL fails, go to the homepage and navigate manually.\n"
        "5. Think step-by-step like a human would. Scroll, look around, try alternatives.\n"
        "6. When you are FULLY done, output a clear final report summarizing what you accomplished.\n"
        "7. You do NOT have access to exec_command (shell). Do not attempt to use it.\n"
    )

    # ── Load relevant skill prompts ───────────────────────────
    skill_prompts = ""
    try:
        from memory import memorygate
        skill_prompts = await memorygate.get_relevant_skills(user_input)
    except Exception as e:
        logger.warning(f"  -> Skill retrieval failed for sub-agent: {e}")

    full_system_prompt = identity
    if skill_prompts:
        full_system_prompt += f"\n\n---\n\n## Relevant Skills\n{skill_prompts}"

    # ── Build Tools (all except blocked) ──────────────────────
    all_tools = []
    for name, func in GLOBAL_TOOL_REGISTRY.items():
        if name in BLOCKED_TOOLS:
            continue
        all_tools.append(func)

    logger.info(f"  -> Sub-agent has {len(all_tools)} tools available")

    # ── LLM (model-agnostic) ──────────────────────────────────
    _DEFAULT_MODEL = "google_genai/gemini-2.5-flash"
    model_string = state.get("active_model") or _DEFAULT_MODEL
    provider, actual_model = (
        model_string.split("/", 1) if "/" in model_string
        else (None, model_string)
    )

    try:
        llm = init_chat_model(actual_model, model_provider=provider)
    except Exception as e:
        logger.error(f"  -> Sub-agent failed to load model '{model_string}', falling back: {e}")
        llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

    llm_with_tools = llm.bind_tools(all_tools) if all_tools else llm

    # ── Invoke ────────────────────────────────────────────────
    try:
        invoke_messages = [SystemMessage(content=full_system_prompt)]
        invoke_messages.extend(messages)

        if user_input:
            human_msg = HumanMessage(content=user_input)
            invoke_messages.append(human_msg)

        result = await llm_with_tools.ainvoke(invoke_messages)

        new_messages = [human_msg, result] if user_input else [result]

        # Count tool calls for the action limit
        new_action_count = action_count
        if hasattr(result, "tool_calls") and result.tool_calls:
            new_action_count += len(result.tool_calls)

        return {
            "messages": new_messages,
            "tool_failure_count": 0,
            "agent_response": result.content,
            "user_input": "",
            "action_count": new_action_count,
        }

    except Exception as e:
        tool_failure_count += 1
        logger.error(f"  -> Sub-agent error (attempt {tool_failure_count}/3): {e}", exc_info=True)
        if tool_failure_count >= 3:
            return {
                "messages": [AIMessage(content=f"Sub-agent failed after 3 attempts: {str(e)}")],
                "tool_failure_count": tool_failure_count,
            }
        return {
            "tool_failure_count": tool_failure_count,
            "_retry": True,
        }


# ══════════════════════════════════════════════════════════════
# Routing
# ══════════════════════════════════════════════════════════════

def route_subagent(state: AgentState) -> str:
    """Route after sub-agent node. No HITL — all tools execute directly."""
    if state.get("tool_failure_count", 0) >= 3:
        return END

    if state.get("_retry"):
        return "subagent"

    if state.get("action_count", 0) >= MAX_ACTIONS:
        return END

    messages = state.get("messages", [])
    if not messages:
        return END

    last_message = messages[-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return END

    return "execute_tools"


# ══════════════════════════════════════════════════════════════
# Graph Builder
# ══════════════════════════════════════════════════════════════

def build_subagent_graph():
    """
    Build the general sub-agent graph. No HITL, no checkpointer.
    Runs ephemerally for delegated tasks.

    Flow: START → subagent ⇄ execute_tools → ... → END
    """
    builder = StateGraph(AgentState)
    builder.add_node("subagent", general_subagent_node)
    builder.add_node("execute_tools", execute_tools_node)

    builder.add_edge(START, "subagent")
    builder.add_conditional_edges(
        "subagent",
        route_subagent,
        ["execute_tools", "subagent", END],
    )
    builder.add_edge("execute_tools", "subagent")

    return builder.compile()


# Backward compatibility
build_researcher_graph = build_subagent_graph
