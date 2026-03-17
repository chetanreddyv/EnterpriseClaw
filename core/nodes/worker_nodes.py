"""
core/nodes/worker_nodes.py — Nodes for the Worker SubGraph (ephemeral task executor).

The Worker operates in a strict Action-Observation loop:
- `messages` is a lightweight action ledger (tiny strings only).
- `observation` holds the heavy environment state (A11y tree, terminal output).
- `observation` is REPLACED each turn, never appended.
- Sequential execution for stateful tools.
- Concurrent execution for read-only tool sets.
"""

import asyncio
import logging
import inspect
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any

from langchain_core.messages import (
    SystemMessage, AIMessage, ToolMessage, HumanMessage,
)
from langchain_core.runnables import RunnableConfig

from core.graphs.states import WorkerState
from core.llm import init_agent_llm, is_llm_connection_error
from mcp_servers import (
    GLOBAL_TOOL_REGISTRY,
    GLOBAL_TOOL_METADATA,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
OBSERVATION_SEPARATOR = "\n===OBSERVATION===\n"
MAX_OBSERVATION_CHARS = 50_000  # ~12,500 tokens
MAX_SKILL_PROMPT_CHARS = 10_000


def _normalize_tool_names(raw_tool_names: list[str] | None) -> list[str]:
    """Normalize declared tool names from matched skill metadata."""
    normalized: list[str] = []
    for tool_name in raw_tool_names or []:
        if isinstance(tool_name, str):
            value = tool_name.strip()
            if value and value not in normalized:
                normalized.append(value)
    return normalized


def _resolve_skill_tools(active_skill_tools: list[str] | None) -> tuple[list, set[str]]:
    """Resolve callable tools strictly from matched skill declarations."""
    normalized_skill_tools = _normalize_tool_names(active_skill_tools)
    registry_names = set(GLOBAL_TOOL_REGISTRY.keys())

    unknown_tools = sorted(name for name in normalized_skill_tools if name not in registry_names)
    if unknown_tools:
        logger.warning(
            "WorkerToolBinding: Ignoring unknown skill-declared tools: %s",
            ", ".join(unknown_tools),
        )

    tool_names = {name for name in normalized_skill_tools if name in registry_names}
    tool_names.add("escalate_to_supervisor")

    if tool_names == {"escalate_to_supervisor"}:
        logger.warning(
            "WorkerToolBinding: No executable tools declared by matched skills. "
            "Worker will escalate for clarification."
        )

    tools = []
    for name in sorted(tool_names):
        func = GLOBAL_TOOL_REGISTRY.get(name)
        if func:
            tools.append(func)
        elif name == "escalate_to_supervisor":
            logger.warning("WorkerToolBinding: escalate_to_supervisor is not registered.")

    return tools, tool_names


def _is_stateful_tool(tool_name: str) -> bool:
    """Stateful tools must be executed sequentially to preserve environment integrity."""
    metadata = GLOBAL_TOOL_METADATA.get(tool_name, {})
    if bool(metadata.get("stateful")):
        return True

    category = str(metadata.get("category", "")).strip().lower()
    return category in {"browser", "exec"}


def _observation_mode_for_tool(tool_name: str) -> str:
    """Return observation truncation mode for a tool: 'head' or 'tail'."""
    metadata = GLOBAL_TOOL_METADATA.get(tool_name, {})
    mode = str(metadata.get("observation_mode", "head")).strip().lower()
    if mode in {"head", "tail"}:
        return mode
    return "tail" if tool_name.startswith("exec_") else "head"


# ═══════════════════════════════════════════════════════════════
# 1. WORKER SKILL CONTEXT NODE (JIT)
# ═══════════════════════════════════════════════════════════════

async def worker_skill_context_node(state: WorkerState) -> Dict[str, Any]:
    """Retrieve objective-matched skill prompts once at Worker start."""
    objective = state.get("objective", "").strip()
    fallback = "You are a specialized task executor. Proceed safely and verify actions before execution."

    # If scheduler pre-resolved skills/tools, keep those bindings deterministic.
    prebound_skill_tools = _normalize_tool_names(state.get("active_skill_tools") or [])
    prebound_skill_prompts = state.get("skill_prompts", "")
    prebound_skills = [s for s in (state.get("active_skills") or []) if isinstance(s, str) and s.strip()]
    if prebound_skill_tools:
        logger.info(
            "WorkerSkillContext: using prebound skills/tools from execution context (skills=%s, tools=%s)",
            prebound_skills,
            prebound_skill_tools,
        )
        return {
            "skill_prompts": prebound_skill_prompts or fallback,
            "active_skills": prebound_skills,
            "active_skill_tools": prebound_skill_tools,
        }

    if not objective:
        return {"skill_prompts": fallback, "active_skills": [], "active_skill_tools": []}

    try:
        from memory.retrieval import memory_retrieval

        skill_prompts, matched_skills, skill_tool_map = await memory_retrieval.get_relevant_skills_with_metadata(objective)
        if not skill_prompts:
            skill_prompts = fallback

        active_skill_tools: list[str] = []
        for skill_name in matched_skills or []:
            for tool_name in skill_tool_map.get(skill_name, []):
                if isinstance(tool_name, str):
                    value = tool_name.strip()
                    if value and value not in active_skill_tools:
                        active_skill_tools.append(value)

        if len(skill_prompts) > MAX_SKILL_PROMPT_CHARS:
            skill_prompts = (
                skill_prompts[:MAX_SKILL_PROMPT_CHARS]
                + "\n...[Skill context truncated to protect prompt budget]..."
            )

        logger.info("WorkerSkillContext: matched_skills=%s", matched_skills)
        if active_skill_tools:
            logger.info("WorkerSkillContext: matched_skill_tools=%s", active_skill_tools)
        return {
            "skill_prompts": skill_prompts,
            "active_skills": matched_skills or [],
            "active_skill_tools": active_skill_tools,
        }
    except Exception as e:
        logger.warning("WorkerSkillContext: retrieval failed: %s", e)
        return {
            "skill_prompts": fallback,
            "active_skills": [],
            "active_skill_tools": [],
        }


# ═══════════════════════════════════════════════════════════════
# 2. WORKER PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════

async def worker_prompt_builder_node(state: WorkerState) -> Dict[str, Any]:
    """
    Build the Worker prompt from objective + observation + JIT skill context.
    """
    objective = state.get("objective", "Complete the assigned task.")
    observation = state.get("observation", "No environment state available yet.")
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 15)
    active_skill_tools = state.get("active_skill_tools") or []
    active_skills = state.get("active_skills") or []
    skill_prompts = state.get("skill_prompts", "")

    _, resolved_tool_names = _resolve_skill_tools(active_skill_tools)
    available_tool_names = sorted(resolved_tool_names)
    tool_preview = ", ".join(available_tool_names[:24])
    if len(available_tool_names) > 24:
        tool_preview += ", ..."

    now_utc = datetime.now(timezone.utc)
    now_local = datetime.now().astimezone()

    batch_actions_hint = ""
    if "batch_actions" in available_tool_names:
        batch_actions_hint = "- If you need to fill multiple form fields, use the `batch_actions` tool.\n"

    escalation_only_hint = ""
    if available_tool_names == ["escalate_to_supervisor"]:
        escalation_only_hint = (
            "- No executable skills matched this objective. "
            "Call `escalate_to_supervisor` immediately and request missing context.\n"
        )

    system_prompt = (
        "## System Clock\n"
        f"- Current UTC Time: {now_utc.isoformat()}\n"
        f"- Local System Time: {now_local.isoformat()}\n"
        f"- Local Timezone: {now_local.tzname()}\n\n"
        "## Core Identity\n"
        "You are a specialized Worker for EnterpriseClaw. Execute the delegated objective safely and precisely.\n\n"
        f"## Operational Rules & Best Practices\n{skill_prompts}\n\n"
        "## Skill-Bound Tool Scope\n"
        f"- Active skills: {', '.join(active_skills) if active_skills else 'none'}\n"
        f"- Bound tools ({len(available_tool_names)}): {tool_preview}\n\n"
        "## Skill Composition Rule\n"
        "- Treat each `BEGIN SKILL` / `END SKILL` block as an independent SOP.\n"
        "- Combine skills only when the objective requires a cross-domain handoff.\n\n"
        f"## Your Objective\n{objective}\n\n"
        f"## Progress\n- Step {step_count + 1} of {max_steps}\n\n"
        f"## Current Environment State\n"
        f"```\n{observation}\n```\n\n"
        f"## Rules\n"
        f"- Take the NEXT action to accomplish your objective.\n"
        f"{batch_actions_hint}"
        f"{escalation_only_hint}"
        f"- Never invent tool names. Use only the exact tool names listed in `Bound tools` above.\n"
        f"- If the exact tool you want is unavailable, use `escalate_to_supervisor` instead of guessing.\n"
        f"- If you are stuck, confused, or the environment is not responding, "
        f"use `escalate_to_supervisor` immediately instead of looping.\n"
        f"- When the task is complete, respond with a summary of what you accomplished.\n"
    )

    sys_msg = SystemMessage(content=system_prompt, id="worker_system_msg")
    objective_msg = HumanMessage(content=f"Objective: {objective}", id="worker_objective_msg")

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
    logger.info(f"🌀 🧩 Active Skills: {state.get('active_skills') or 'None Matched'}")
    logger.info(f"🌀 👁️ Observation: {len(observation)} chars")
    logger.info(f"🌀 📜 History: {len(history)} messages.")
    logger.info("🌀"*80)

    return {
        "_formatted_prompt": [sys_msg, objective_msg] + history,
        "step_count": step_count + 1,
    }


# ═══════════════════════════════════════════════════════════════
# 3. WORKER EXECUTOR
# ═══════════════════════════════════════════════════════════════

async def worker_executor_node(state: WorkerState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Invoke the LLM with skill-bound tools.
    Catches blank generations (local model quirk).
    """
    invoke_messages = state.get("_formatted_prompt")
    if not invoke_messages:
        logger.warning("WorkerExecutor: No `_formatted_prompt`. Exiting.")
        return {"status": "failed", "result_summary": "Internal error: no prompt."}

    active_skill_tools = state.get("active_skill_tools") or []
    all_tools, resolved_tool_names = _resolve_skill_tools(active_skill_tools)

    logger.info(
        "WorkerExecutor: Binding %d tools (skill tools=%d).",
        len(all_tools),
        len(_normalize_tool_names(active_skill_tools)),
    )
    logger.debug("WorkerExecutor: Bound tools => %s", sorted(resolved_tool_names))

    if resolved_tool_names == {"escalate_to_supervisor"}:
        reason = (
            "No executable skills matched the delegated objective. "
            "Need user clarification or additional skills."
        )
        logger.warning("WorkerExecutor: Escalating immediately because no usable skill tools were resolved.")
        return {
            "messages": [AIMessage(content=reason)],
            "status": "escalated",
            "result_summary": reason,
            "_retry": False,
        }

    llm = init_agent_llm(state.get("active_model", ""))
    llm_with_tools = llm.bind_tools(all_tools) if all_tools else llm

    try:
        invoke_start = time.perf_counter()
        result = await llm_with_tools.ainvoke(invoke_messages, config=config)
        invoke_ms = int((time.perf_counter() - invoke_start) * 1000)
        logger.info(
            "WorkerExecutor: Model invoke latency=%dms (bound_tools=%d)",
            invoke_ms,
            len(all_tools),
        )

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
        if is_llm_connection_error(e):
            active_model = state.get("active_model") or "default"
            failure_summary = (
                f"Worker failed because the model backend for `{active_model}` was unreachable "
                "(connection error)."
            )
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I cannot continue this delegated task right now because the model backend is unreachable. "
                            "Escalating control to the supervisor."
                        )
                    )
                ],
                "status": "failed",
                "result_summary": failure_summary,
                "_retry": False,
                "tool_failure_count": 0,
            }

        return {"_retry": True}


# ═══════════════════════════════════════════════════════════════
# 4. WORKER TOOLS NODE
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


def _split_observation(raw_output: str, tool_name: str) -> tuple[str, str | None]:
    """
    Split tool output on OBSERVATION_SEPARATOR.
    Returns (lightweight_summary, heavy_observation_or_None).
    Applies truncation strategy based on the producing tool.
    """
    if OBSERVATION_SEPARATOR in raw_output:
        parts = raw_output.split(OBSERVATION_SEPARATOR, 1)
        summary = parts[0].strip()
        observation = parts[1].strip()

        # Truncate oversized observations
        if len(observation) > MAX_OBSERVATION_CHARS:
            if _observation_mode_for_tool(tool_name) == "tail":
                # Terminal: keep the TAIL (most recent output is most relevant)
                observation = "...[TRUNCATED]...\n" + observation[-MAX_OBSERVATION_CHARS:]
            else:
                # Browser: keep the HEAD (top of page / form fields are most relevant)
                observation = observation[:MAX_OBSERVATION_CHARS] + "\n...[TRUNCATED]..."

        return summary, observation
    else:
        # No separator — entire output is the summary (simple tool)
        return raw_output, None


def _execution_mode(state: WorkerState, config: RunnableConfig) -> str:
    """Resolve execution mode from state/config. Defaults to interactive."""
    state_mode = str(state.get("execution_mode") or "").strip().lower()
    if state_mode in {"interactive", "cron"}:
        return state_mode

    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    config_mode = str(configurable.get("execution_mode") or "").strip().lower()
    if config_mode in {"interactive", "cron"}:
        return config_mode

    return "interactive"


def _cron_auto_approved_tools(state: WorkerState) -> set[str]:
    """Tools that cron runs may auto-approve: only skill-declared tools."""
    declared = set(_normalize_tool_names(state.get("active_skill_tools") or []))
    # Escalation sentinel should always stay callable.
    declared.add("escalate_to_supervisor")
    return declared


async def worker_tools_node(state: WorkerState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Execute tool calls with metadata-aware strategy:
    - Stateful tools: Sequential, halt on first failure.
    - Read-only tools: Concurrent via asyncio.gather.

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

    execution_mode = _execution_mode(state, config)
    is_cron_mode = execution_mode == "cron"

    approved_tools = set(state.get("approved_tools") or [])
    if is_cron_mode:
        approved_tools.update(_cron_auto_approved_tools(state))
        logger.info(
            "WorkerHITL: cron mode detected; auto-approved %d skill-declared tools.",
            len(approved_tools),
        )

    tool_messages = []
    latest_observation = None  # Will hold the LAST observation (replaces, not appends)
    tool_calls = list(last_message.tool_calls)

    # ── Check for escalation sentinel ──
    for tool_call in tool_calls:
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
    run_sequential = any(_is_stateful_tool(tc["name"]) for tc in tool_calls)

    if run_sequential:
        # SEQUENTIAL: Execute in exact LLM-specified order. Halt on first failure.
        for idx, tool_call in enumerate(tool_calls):
            action_name = tool_call["name"]
            call_id = tool_call["id"]

            if action_name not in GLOBAL_TOOL_REGISTRY:
                tool_messages.append(ToolMessage(
                    content=f"Error: Tool '{action_name}' not found in registry.",
                    tool_call_id=call_id,
                ))
                for remaining in tool_calls[idx + 1:]:
                    tool_messages.append(ToolMessage(
                        content=f"Skipped: previous action '{action_name}' failed.",
                        tool_call_id=remaining["id"],
                    ))
                break

            # ── HITL TIERED APPROVAL ──
            if requires_approval(action_name, approved_tools):
                if is_cron_mode:
                    tool_messages.append(ToolMessage(
                        content=(
                            f"❌ Rejected in cron mode: '{action_name}' requires interactive approval "
                            "and was not declared by the bound skill set for this job."
                        ),
                        tool_call_id=call_id,
                    ))
                    # Halt remaining tools after deterministic cron rejection.
                    for remaining in tool_calls[idx + 1:]:
                        tool_messages.append(ToolMessage(
                            content=(
                                f"Skipped: previous action '{action_name}' was rejected by cron policy."
                            ),
                            tool_call_id=remaining["id"],
                        ))
                    break

                decision = request_tool_approval(action_name, tool_call["args"])
                if decision != "approve":
                    tool_messages.append(ToolMessage(
                        content=f"❌ Rejected by user: {action_name}. Reason: {decision}",
                        tool_call_id=call_id,
                    ))
                    # Halt remaining tools after rejection
                    for remaining in tool_calls[idx + 1:]:
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
                for remaining in tool_calls[idx + 1:]:
                    tool_messages.append(ToolMessage(
                        content=f"Skipped: previous action '{action_name}' failed.",
                        tool_call_id=remaining["id"],
                    ))
                break

            # Split observation from summary
            if isinstance(result_content, str):
                summary, obs = _split_observation(result_content, action_name)
                tool_messages.append(ToolMessage(content=summary, tool_call_id=call_id))
                if obs is not None:
                    latest_observation = obs  # REPLACE, not append
            else:
                tool_messages.append(ToolMessage(content=str(result_content), tool_call_id=call_id))
    else:
        # CONCURRENT: Read-only tools can run in parallel
        # First, check HITL for any tool that requires approval (rare in read-only domain)
        executable_tool_calls = []
        for tool_call in tool_calls:
            action_name = tool_call["name"]

            if action_name not in GLOBAL_TOOL_REGISTRY:
                tool_messages.append(ToolMessage(
                    content=f"Error: Tool '{action_name}' not found in registry.",
                    tool_call_id=tool_call["id"],
                ))
                continue

            if requires_approval(action_name, approved_tools):
                if is_cron_mode:
                    tool_messages.append(ToolMessage(
                        content=(
                            f"❌ Rejected in cron mode: '{action_name}' requires interactive approval "
                            "and was not declared by the bound skill set for this job."
                        ),
                        tool_call_id=tool_call["id"],
                    ))
                    continue

                decision = request_tool_approval(action_name, tool_call["args"])
                if decision != "approve":
                    tool_messages.append(ToolMessage(
                        content=f"❌ Rejected by user: {action_name}.",
                        tool_call_id=tool_call["id"],
                    ))
                    return {"messages": tool_messages}
            executable_tool_calls.append(tool_call)

        if not executable_tool_calls:
            return {"messages": tool_messages}

        tasks = [
            _execute_single_tool(tc["name"], tc["args"], config)
            for tc in executable_tool_calls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for tool_call, result_content in zip(executable_tool_calls, results):
            call_id = tool_call["id"]

            if isinstance(result_content, Exception):
                result_content = f"Error executing {tool_call['name']}: {str(result_content)}"

            if isinstance(result_content, str):
                summary, obs = _split_observation(result_content, tool_call["name"])
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
# 5. WORKER SUMMARIZE NODE
# ═══════════════════════════════════════════════════════════════

async def worker_summarize_node(state: WorkerState) -> Dict[str, Any]:
    """
    Called when the Worker reaches END.
    Extracts the last AI message as the result_summary.
    """
    existing_status = state.get("status")
    existing_summary = (state.get("result_summary") or "").strip()
    if existing_status in {"failed", "escalated"} and existing_summary:
        return {
            "status": existing_status,
            "result_summary": existing_summary,
        }

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
