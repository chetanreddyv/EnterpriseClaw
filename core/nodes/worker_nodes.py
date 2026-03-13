"""
core/nodes/worker_nodes.py — Nodes for the Worker SubGraph (ephemeral task executor).

The Worker operates in a strict Action-Observation loop:
- `messages` is a lightweight action ledger (tiny strings only).
- `observation` holds the heavy environment state (A11y tree, terminal output).
- `observation` is REPLACED each turn, never appended.
- Sequential execution for stateful domains (browser, exec).
- Concurrent execution for read-only domains (all).
"""

import asyncio
import logging
import inspect
import json
from typing import Dict, Any

from langchain_core.messages import (
    SystemMessage, AIMessage, ToolMessage, HumanMessage,
)
from langchain_core.runnables import RunnableConfig

from core.graphs.states import WorkerState
from core.llm import init_agent_llm
from mcp_servers import GLOBAL_TOOL_REGISTRY

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
OBSERVATION_SEPARATOR = "\n===OBSERVATION===\n"
MAX_OBSERVATION_CHARS = 50_000  # ~12,500 tokens

# Tool sets per domain
BROWSER_TOOLS = {
    "browser_navigate", "browser_get_text", "browser_screenshot",
    "browser_go_back", "browser_scroll", "browser_wait_for",
    "browser_snapshot", "browser_tab_management",
    "browser_click", "browser_type", "browser_execute_js",
    "browser_select_option", "browser_press_key", "browser_hover",
    "browser_handle_dialog", "browser_file_upload",
    "batch_actions", "escalate_to_supervisor",
}

EXEC_TOOLS = {
    "exec_command", "batch_actions", "escalate_to_supervisor",
}

# "all" = everything available
ALL_TOOLS_BASE = {
    "web_search", "web_fetch", "check_current_time",
    "escalate_to_supervisor", "batch_actions",
}

# Stateful domains that require sequential execution
STATEFUL_DOMAINS = {"browser", "exec"}


def _get_worker_tools(tool_domain: str) -> list:
    """Resolve callable tools for a given domain."""
    if tool_domain == "browser":
        tool_names = BROWSER_TOOLS
    elif tool_domain == "exec":
        tool_names = EXEC_TOOLS
    else:
        # "all" = union of everything in the registry
        tool_names = ALL_TOOLS_BASE | BROWSER_TOOLS | EXEC_TOOLS

    tools = []
    for name in tool_names:
        func = GLOBAL_TOOL_REGISTRY.get(name)
        if func:
            tools.append(func)
    return tools


# ═══════════════════════════════════════════════════════════════
# 1. WORKER PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════

async def worker_prompt_builder_node(state: WorkerState) -> Dict[str, Any]:
    """
    Build a minimal system prompt: objective + current observation.
    No identity, no memory, no skills. Pure Action-Observation.
    """
    objective = state.get("objective", "Complete the assigned task.")
    observation = state.get("observation", "No environment state available yet.")
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 15)

    system_prompt = (
        f"## Task Executor\n"
        f"**Your objective:** {objective}\n\n"
        f"**Step {step_count + 1} of {max_steps}**\n\n"
        f"## Current Environment State\n"
        f"```\n{observation}\n```\n\n"
        f"## Instructions\n"
        f"- Take the NEXT action to accomplish your objective.\n"
        f"- If you need to fill multiple form fields, use the `batch_actions` tool.\n"
        f"- If you are stuck, confused, or the environment is not responding, "
        f"use `escalate_to_supervisor` immediately instead of looping.\n"
        f"- When the task is complete, respond with a summary of what you accomplished.\n"
    )

    sys_msg = SystemMessage(content=system_prompt, id="worker_system_msg")

    # Worker keeps a lightweight message tail — no complex pruning needed
    messages = state.get("messages", [])
    # Filter out old system messages
    history = [m for m in messages if not isinstance(m, SystemMessage)]
    # Keep only last 10 messages (action log is tiny)
    if len(history) > 10:
        history = history[-10:]

    logger.info("🌀"*80)
    logger.info("🌀 [WORKER TRACE: PROMPT]")
    logger.info(f"🌀 🎯 Objective: {objective}")
    logger.info(f"🌀 📍 Step: {step_count + 1}/{max_steps} | Model: {state.get('active_model') or 'default'}")
    logger.info(f"🌀 👁️ Observation: {len(observation)} chars")
    logger.info(f"🌀 📜 History: {len(history)} messages.")
    logger.info("🌀"*80)

    return {
        "_formatted_prompt": [sys_msg] + history,
        "step_count": step_count + 1,
    }


# ═══════════════════════════════════════════════════════════════
# 2. WORKER EXECUTOR
# ═══════════════════════════════════════════════════════════════

async def worker_executor_node(state: WorkerState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Invoke the LLM with domain-specific tools.
    Catches blank generations (local model quirk).
    """
    invoke_messages = state.get("_formatted_prompt")
    if not invoke_messages:
        logger.warning("WorkerExecutor: No `_formatted_prompt`. Exiting.")
        return {"status": "failed", "result_summary": "Internal error: no prompt."}

    tool_domain = state.get("tool_domain", "all")
    all_tools = _get_worker_tools(tool_domain)

    logger.info(f"WorkerExecutor: Domain '{tool_domain}', binding {len(all_tools)} tools.")

    llm = init_agent_llm(state.get("active_model", ""))
    llm_with_tools = llm.bind_tools(all_tools) if all_tools else llm

    try:
        result = await llm_with_tools.ainvoke(invoke_messages, config=config)

        # ── Catch Blank Generations ──
        content = getattr(result, "content", "")
        if isinstance(content, (list, dict)):
            content_str = str(content) if content else ""
        else:
            content_str = str(content).strip()

        # ── [LOGGING] WORKER OUTPUT BLOCK ──
        logger.info("⚡"*80)
        logger.info("⚡ [WORKER: ACTION]")
        if getattr(result, "tool_calls", []):
            logger.info(f"⚡ 🛠 Action: {[tc['name'] for tc in result.tool_calls]}")
        if content_str:
            logger.info(f"⚡ 📝 Response: {content_str}")
        logger.info("⚡"*80)

        if not content_str and not getattr(result, "tool_calls", []):
            logger.warning("WorkerExecutor: Blank generation detected.")
            return {"messages": [result], "_retry": True}

        if getattr(result, "tool_calls", []):
            logger.info(f"WorkerExecutor: Generated {len(result.tool_calls)} tool calls.")
        else:
            logger.info(f"WorkerExecutor: Generated text response ({len(content_str)} chars).")

        return {"messages": [result]}

    except Exception as e:
        logger.error(f"WorkerExecutor: LLM API Call Failed: {e}", exc_info=True)
        return {"_retry": True}


# ═══════════════════════════════════════════════════════════════
# 3. WORKER TOOLS NODE
# ═══════════════════════════════════════════════════════════════

async def _execute_single_tool(action_name: str, tool_args: dict, config: RunnableConfig):
    """Execute a single tool function from the registry."""
    func = GLOBAL_TOOL_REGISTRY.get(action_name)
    if not func:
        return f"Error: Tool '{action_name}' not found in registry."

    try:
        if hasattr(func, "ainvoke"):
            result = await func.ainvoke(tool_args, config=config)
        else:
            result = func(**tool_args)
            if inspect.isawaitable(result):
                result = await result

        # Multimodal arrays (screenshots)
        if isinstance(result, list):
            is_multimodal = len(result) > 0 and isinstance(result[0], dict) and "type" in result[0]
            if is_multimodal:
                return result
            try:
                return json.dumps(result)
            except Exception:
                return str(result)
        return str(result)

    except Exception as e:
        logger.error(f"  -> Worker tool {action_name} failed: {e}")
        return f"Error executing {action_name}: {e}"


def _split_observation(raw_output: str, tool_domain: str) -> tuple[str, str | None]:
    """
    Split tool output on OBSERVATION_SEPARATOR.
    Returns (lightweight_summary, heavy_observation_or_None).
    Applies truncation to the observation based on domain.
    """
    if OBSERVATION_SEPARATOR in raw_output:
        parts = raw_output.split(OBSERVATION_SEPARATOR, 1)
        summary = parts[0].strip()
        observation = parts[1].strip()

        # Truncate oversized observations
        if len(observation) > MAX_OBSERVATION_CHARS:
            if tool_domain == "exec":
                # Terminal: keep the TAIL (most recent output is most relevant)
                observation = "...[TRUNCATED]...\n" + observation[-MAX_OBSERVATION_CHARS:]
            else:
                # Browser: keep the HEAD (top of page / form fields are most relevant)
                observation = observation[:MAX_OBSERVATION_CHARS] + "\n...[TRUNCATED]..."

        return summary, observation
    else:
        # No separator — entire output is the summary (simple tool)
        return raw_output, None


async def worker_tools_node(state: WorkerState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Execute tool calls with domain-aware strategy:
    - Stateful domains (browser, exec): Sequential, halt on first failure.
    - Read-only domains (all): Concurrent via asyncio.gather.

    CRITICAL: Tool results are SPLIT:
    - Lightweight summary → ToolMessage (stays in messages)
    - Heavy observation → state["observation"] (REPLACED, never appended)
    
    HITL: Before executing any tool, checks tiered approval.
    NOT_ALLOWED tools fire interrupt() unless /permit'd.
    """
    from core.hitl import requires_approval, request_tool_approval

    messages = state.get("messages", [])
    if not messages:
        return {}

    last_message = messages[-1]
    if not getattr(last_message, "tool_calls", []):
        return {}

    tool_domain = state.get("tool_domain", "all")
    approved_tools = set(state.get("approved_tools") or [])
    tool_messages = []
    latest_observation = None  # Will hold the LAST observation (replaces, not appends)

    # ── Check for escalation sentinel ──
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "escalate_to_supervisor":
            reason = tool_call["args"].get("reason", "Worker escalated without providing a reason.")
            tool_messages.append(ToolMessage(
                content=f"Escalating: {reason}",
                tool_call_id=tool_call["id"],
            ))
            return {
                "messages": tool_messages,
                "status": "escalated",
                "result_summary": reason,
            }

    # ── Domain-aware execution ──
    if tool_domain in STATEFUL_DOMAINS:
        # SEQUENTIAL: Execute in exact LLM-specified order. Halt on first failure.
        for tool_call in last_message.tool_calls:
            action_name = tool_call["name"]
            call_id = tool_call["id"]

            # ── HITL TIERED APPROVAL ──
            if requires_approval(action_name, approved_tools):
                decision = request_tool_approval(action_name, tool_call["args"])
                if decision != "approve":
                    tool_messages.append(ToolMessage(
                        content=f"❌ Rejected by user: {action_name}. Reason: {decision}",
                        tool_call_id=call_id,
                    ))
                    # Halt remaining tools after rejection
                    idx = last_message.tool_calls.index(tool_call)
                    for remaining in last_message.tool_calls[idx + 1:]:
                        tool_messages.append(ToolMessage(
                            content=f"Skipped: previous action '{action_name}' was rejected.",
                            tool_call_id=remaining["id"],
                        ))
                    break

            logger.info(f"🛠️ Worker executing (sequential): {action_name}")

            result_content = await _execute_single_tool(action_name, tool_call["args"], config)

            # Handle exceptions as strings
            if isinstance(result_content, str) and result_content.startswith("Error"):
                tool_messages.append(ToolMessage(content=result_content, tool_call_id=call_id))
                # Halt: skip remaining tools
                idx = last_message.tool_calls.index(tool_call)
                for remaining in last_message.tool_calls[idx + 1:]:
                    tool_messages.append(ToolMessage(
                        content=f"Skipped: previous action '{action_name}' failed.",
                        tool_call_id=remaining["id"],
                    ))
                break

            # Split observation from summary
            if isinstance(result_content, str):
                summary, obs = _split_observation(result_content, tool_domain)
                tool_messages.append(ToolMessage(content=summary, tool_call_id=call_id))
                if obs is not None:
                    latest_observation = obs  # REPLACE, not append
            else:
                tool_messages.append(ToolMessage(content=str(result_content), tool_call_id=call_id))
    else:
        # CONCURRENT: Read-only tools can run in parallel
        # First, check HITL for any tool that requires approval (rare in read-only domain)
        for tool_call in last_message.tool_calls:
            if requires_approval(tool_call["name"], approved_tools):
                decision = request_tool_approval(tool_call["name"], tool_call["args"])
                if decision != "approve":
                    tool_messages.append(ToolMessage(
                        content=f"❌ Rejected by user: {tool_call['name']}.",
                        tool_call_id=tool_call["id"],
                    ))
                    return {"messages": tool_messages}

        tasks = [
            _execute_single_tool(tc["name"], tc["args"], config)
            for tc in last_message.tool_calls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for tool_call, result_content in zip(last_message.tool_calls, results):
            call_id = tool_call["id"]

            if isinstance(result_content, Exception):
                result_content = f"Error executing {tool_call['name']}: {str(result_content)}"

            if isinstance(result_content, str):
                summary, obs = _split_observation(result_content, tool_domain)
                tool_messages.append(ToolMessage(content=summary, tool_call_id=call_id))
                if obs is not None:
                    latest_observation = obs
            else:
                tool_messages.append(ToolMessage(content=str(result_content), tool_call_id=call_id))

    # Build update payload
    update_payload: Dict[str, Any] = {"messages": tool_messages}

    # REPLACE observation (State Amnesia) — only if a tool returned one
    if latest_observation is not None:
        update_payload["observation"] = latest_observation

    return update_payload


# ═══════════════════════════════════════════════════════════════
# 4. WORKER SUMMARIZE NODE
# ═══════════════════════════════════════════════════════════════

async def worker_summarize_node(state: WorkerState) -> Dict[str, Any]:
    """
    Called when the Worker reaches END.
    Extracts the last AI message as the result_summary.
    """
    messages = state.get("messages", [])

    # Find the last AI response
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return {
                    "status": "completed",
                    "result_summary": content.strip(),
                }

    return {
        "status": "completed",
        "result_summary": "Task completed but no summary was generated.",
    }
