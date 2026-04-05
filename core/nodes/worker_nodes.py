"""
core/nodes/worker_nodes.py — Nodes for the Worker SubGraph (ephemeral task executor).

The Worker operates in a strict Action-Observation loop:
- `messages` is a lightweight action ledger (tiny strings only).
- `environment_snapshot` holds the heavy environment state (A11y tree, terminal output).
- `environment_snapshot` is REPLACED each turn, never appended.
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
from core.observers import get_browser_environment_state, get_exec_environment_state
from core.errors import InfrastructureError
from mcp_servers import (
    GLOBAL_TOOL_REGISTRY,
    GLOBAL_TOOL_METADATA,
)
from config.settings import settings

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
MAX_OBSERVATION_CHARS = settings.worker_max_observation_chars
MAX_SKILL_PROMPT_CHARS = settings.worker_max_skill_prompt_chars

TOOL_ERROR_PREFIXES = (
    "error",
    "failed",
    "❌",
    "action failed",
)


def _build_infrastructure_failure_summary(error: InfrastructureError) -> str:
    """Normalize fatal infrastructure summaries returned to the Supervisor."""
    return (
        "System Infrastructure Failure: "
        f"{str(error).strip()}. "
        "The task was aborted to prevent token burn."
    )


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
    tool_names.add("complete_task")
    tool_names.add("exec_command")  # Core tool: always available for terminal access

    if tool_names == {"escalate_to_supervisor", "complete_task", "exec_command"}:
        logger.warning(
            "WorkerToolBinding: No skill-specific tools declared beyond core tools. "
            "Worker will operate with core tools only."
        )

    tools = []
    for name in sorted(tool_names):
        func = GLOBAL_TOOL_REGISTRY.get(name)
        if func:
            tools.append(func)
        elif name in ("escalate_to_supervisor", "complete_task"):
            logger.warning(f"WorkerToolBinding: {name} is not registered in GLOBAL_TOOL_REGISTRY.")

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


def _truncate_observation(observation: str, tool_name: str) -> str:
    """Apply metadata-aware truncation to oversized observations."""
    if len(observation) <= MAX_OBSERVATION_CHARS:
        return observation

    if _observation_mode_for_tool(tool_name) == "tail":
        # Terminal output: latest lines tend to be most useful.
        return "...[TRUNCATED]...\n" + observation[-MAX_OBSERVATION_CHARS:]

    # Browser output: page top and current form region are often near the head.
    return observation[:MAX_OBSERVATION_CHARS] + "\n...[TRUNCATED]..."


def _observation_to_prompt_text(observation: Any) -> str:
    """Convert string or multimodal observation payload into prompt-safe text."""
    if isinstance(observation, str):
        return observation

    if isinstance(observation, list):
        parts: list[str] = []
        has_image = False
        for item in observation:
            if isinstance(item, dict):
                item_type = str(item.get("type", "")).strip().lower()
                if item_type == "text":
                    text = str(item.get("text", "")).strip()
                    if text:
                        parts.append(text)
                elif item_type == "image_url":
                    has_image = True
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())

        if has_image:
            parts.append("[Visual context image attached in observation payload]")

        if parts:
            return "\n\n".join(parts)
        return "No environment state available yet."

    text = str(observation).strip()
    return text or "No environment state available yet."


def _extract_text_from_content(content: Any) -> str:
    """
    Extract text from AIMessage.content in any format (string or multimodal list).
    Used by worker_summarize_node to capture actual response text.
    Returns empty string if content is empty or None, not a generic fallback message.
    """
    if not content:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = str(item.get("type", "")).strip().lower()
                if item_type == "text":
                    text = str(item.get("text", "")).strip()
                    if text:
                        parts.append(text)
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())

        if parts:
            return "\n\n".join(parts)
        return ""

    # Fallback for dict or other types
    if isinstance(content, dict):
        text = str(content.get("text", "")).strip()
        return text if text else ""

    return ""


def _extract_tool_call_ids(tool_calls: list[Any]) -> list[str]:
    """Return normalized tool_call ids from AIMessage.tool_calls payloads."""
    ids: list[str] = []
    for tool_call in tool_calls:
        call_id = None
        if isinstance(tool_call, dict):
            call_id = tool_call.get("id")
        else:
            call_id = getattr(tool_call, "id", None)

        if isinstance(call_id, str):
            value = call_id.strip()
            if value and value not in ids:
                ids.append(value)
    return ids


def _coerce_valid_worker_history(messages: list[Any]) -> list[Any]:
    """
    Drop malformed tool-history patterns before an LLM call.

    Guarantees:
    - No orphan ToolMessage entries.
    - Any AI tool_calls block is kept only if all expected tool_call_ids are present.
    """
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    valid: list[Any] = []

    i = 0
    while i < len(non_system):
        msg = non_system[i]

        if isinstance(msg, ToolMessage):
            i += 1
            continue

        if isinstance(msg, AIMessage):
            tool_calls = list(getattr(msg, "tool_calls", []) or [])
            if tool_calls:
                j = i + 1
                following_tools: list[ToolMessage] = []
                while j < len(non_system) and isinstance(non_system[j], ToolMessage):
                    following_tools.append(non_system[j])
                    j += 1

                expected_ids = _extract_tool_call_ids(tool_calls)
                matched_tools: list[ToolMessage] = []
                matched_ids: set[str] = set()

                for tool_msg in following_tools:
                    tool_call_id = getattr(tool_msg, "tool_call_id", None)
                    if (
                        isinstance(tool_call_id, str)
                        and tool_call_id in expected_ids
                        and tool_call_id not in matched_ids
                    ):
                        matched_tools.append(tool_msg)
                        matched_ids.add(tool_call_id)

                if expected_ids and all(call_id in matched_ids for call_id in expected_ids):
                    valid.append(msg)
                    valid.extend(matched_tools)
                elif not expected_ids and str(getattr(msg, "content", "")).strip():
                    # Defensive fallback: retain text response if tool_call ids are absent.
                    valid.append(msg)

                i = j
                continue

        valid.append(msg)
        i += 1

    return valid


def _build_worker_history_tail(messages: list[Any], max_messages: int = 10) -> list[Any]:
    """Build a pair-safe worker history tail that never starts with a ToolMessage."""
    valid_history = _coerce_valid_worker_history(messages)
    if not valid_history:
        return []

    segments: list[list[Any]] = []
    i = 0
    while i < len(valid_history):
        msg = valid_history[i]
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", []):
            segment: list[Any] = [msg]
            i += 1
            while i < len(valid_history) and isinstance(valid_history[i], ToolMessage):
                segment.append(valid_history[i])
                i += 1
            segments.append(segment)
            continue

        segments.append([msg])
        i += 1

    selected: list[Any] = []
    count = 0
    for segment in reversed(segments):
        if count + len(segment) > max_messages and selected:
            break
        if len(segment) > max_messages and not selected:
            selected = segment[:]
            count = len(segment)
            break

        selected = segment + selected
        count += len(segment)

    while selected and isinstance(selected[0], ToolMessage):
        selected = selected[1:]

    return selected


def _tool_category(tool_name: str) -> str:
    """Resolve normalized runtime category for a tool."""
    metadata = GLOBAL_TOOL_METADATA.get(tool_name, {})
    category = str(metadata.get("category", "")).strip().lower()
    if category:
        return category
    if tool_name.startswith("browser_") or tool_name == "batch_actions":
        return "browser"
    if tool_name.startswith("exec_"):
        return "exec"
    return ""


def _observation_categories_for_tools(tool_names: list[str]) -> set[str]:
    """Determine which observer pipelines should run after tool execution."""
    categories: set[str] = set()
    for tool_name in tool_names:
        category = _tool_category(tool_name)
        if category == "browser":
            categories.add("browser")
        elif category == "exec":
            categories.add("exec")
    return categories


def _looks_like_tool_error(text: str) -> bool:
    normalized = str(text).strip().lower()
    if not normalized:
        return True
    return normalized.startswith(TOOL_ERROR_PREFIXES)


def _summarize_tool_result(tool_name: str, result_content: str) -> str:
    """Keep action ledger lightweight while preserving error diagnostics."""
    summary = str(result_content).strip()
    if _looks_like_tool_error(summary):
        return summary

    category = _tool_category(tool_name)
    if category in {"browser", "exec"}:
        return (
            f"Action dispatched successfully via {tool_name}. "
            "See environment observation for current state."
        )

    return summary or f"Action successful: {tool_name}."


async def _refresh_environment_snapshot(categories: set[str], config: RunnableConfig) -> Any | None:
    """Fetch current environment anchor for requested categories."""
    if not categories:
        return None

    browser_state: Any | None = None
    exec_state: Any | None = None

    if "browser" in categories:
        browser_state = await get_browser_environment_state(config)
    if "exec" in categories:
        exec_state = await get_exec_environment_state(config)

    if "exec" in categories and "browser" not in categories:
        return _truncate_observation(_observation_to_prompt_text(exec_state), "exec_command")

    if isinstance(browser_state, list):
        final_payload: list[dict[str, Any]] = list(browser_state)
        if exec_state:
            exec_text = _truncate_observation(_observation_to_prompt_text(exec_state), "exec_command")
            final_payload.append({"type": "text", "text": f"\n\n💻 Terminal Output:\n{exec_text}"})
        return final_payload

    sections: list[str] = []
    if browser_state is not None:
        sections.append(_truncate_observation(_observation_to_prompt_text(browser_state), "browser_snapshot"))
    if exec_state is not None:
        sections.append(_truncate_observation(_observation_to_prompt_text(exec_state), "exec_command"))

    if not sections:
        return None

    combined = "\n\n".join(section for section in sections if section.strip()).strip()
    if not combined:
        return None

    if len(combined) > MAX_OBSERVATION_CHARS:
        combined = combined[:MAX_OBSERVATION_CHARS] + "\n...[TRUNCATED]..."

    return combined


async def worker_refresh_environment_node(state: WorkerState, config: RunnableConfig) -> Dict[str, Any]:
    """Refresh replace-only ephemeral environment anchor before next model turn."""
    raw_categories = state.get("_refresh_categories") or []
    categories = {
        str(value).strip().lower()
        for value in raw_categories
        if str(value).strip().lower() in {"browser", "exec"}
    }

    update: Dict[str, Any] = {"_refresh_categories": []}
    if not categories:
        return update

    latest_observation = await _refresh_environment_snapshot(categories, config)
    if latest_observation is not None:
        update["environment_snapshot"] = latest_observation

    return update


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
    Includes deterministic content passthrough when user_content is present.
    """
    objective_raw = state.get("objective")
    if isinstance(objective_raw, str):
        objective = objective_raw.strip()
    elif objective_raw is None:
        objective = ""
    else:
        objective = str(objective_raw).strip()
    if not objective:
        objective = "Complete the assigned task."

    snapshot_raw = state.get("environment_snapshot", "No environment state available yet.")
    observation = _observation_to_prompt_text(snapshot_raw)
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", settings.worker_max_steps)
    active_skill_tools = state.get("active_skill_tools") or []
    active_skills = state.get("active_skills") or []
    skill_prompts = state.get("skill_prompts", "")

    # Content passthrough: raw user message for verbatim reproduction tasks.
    user_content = (state.get("user_content") or "").strip()

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
    if available_tool_names == ["complete_task", "escalate_to_supervisor"] or available_tool_names == ["escalate_to_supervisor"]:
        escalation_only_hint = (
            "- No executable skills matched this objective. "
            "Call `escalate_to_supervisor` immediately and request missing context.\n"
        )

    # Content-fidelity escalation rule: only relevant when user_content is absent
    content_fidelity_hint = ""
    if not user_content:
        content_fidelity_hint = (
            "- CRITICAL: If the objective requires writing specific file contents, running an exact command, "
            "or reproducing verbatim text that is NOT provided in your context, you MUST call "
            "`escalate_to_supervisor` with reason 'Missing verbatim content — cannot reproduce from "
            "summarized objective.' Do NOT hallucinate or invent placeholder content.\n"
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
        f"## Progress\n- Step {step_count + 1} of {max_steps}\n\n"
        f"## Rules\n"
        f"- Take the NEXT action to accomplish your objective.\n"
        f"- Only perform ONE action per turn. Attempting multiple actions at once is strictly forbidden.\n"
        f"- Never guess or invent internal URLs (e.g. going directly to `/cart`) if there are visible UI controls to click instead.\n"
        f"{batch_actions_hint}"
        f"{escalation_only_hint}"
        f"{content_fidelity_hint}"
        f"- The `Current Environment State` below updates automatically after every action you take. "
        f"Do not use tools only to 'look' at the screen or terminal immediately after an action; "
        f"read the provided state first.\n"
        f"- Never invent tool names. Use only the exact tool names listed in `Bound tools` above.\n"
        f"- If the exact tool you want is unavailable, use `escalate_to_supervisor` instead of guessing.\n"
        f"- If you are stuck, confused, or the environment is not responding, "
        f"use `escalate_to_supervisor` immediately instead of looping.\n"
        f"- When the task is complete, you MUST call the `complete_task` tool with a summary of what you accomplished. Do not just reply with text.\n"
    )

    sys_msg = SystemMessage(content=system_prompt, id="worker_system_msg")
    observation_header = "## Current Environment State\n"
    if isinstance(snapshot_raw, list):
        # Multimodal observations must be sent in a HumanMessage, not a SystemMessage.
        observation_payload: list[dict[str, Any] | str] = [{"type": "text", "text": observation_header}]
        observation_payload.extend(snapshot_raw)
        observation_msg = HumanMessage(content=observation_payload, id="worker_observation_msg")
    else:
        observation_msg = HumanMessage(
            content=f"{observation_header}```\n{observation}\n```",
            id="worker_observation_msg",
        )

    # Worker keeps a lightweight, pair-safe message tail.
    messages = state.get("messages", [])
    history = _build_worker_history_tail(messages, max_messages=10)

    # ── Objective Anchor Pattern ──────────────────────────────────────────────
    # The Objective is always presented as a persistent HumanMessage right after
    # the System prompt. This ensures Gemini (and all other LLMs) see a valid
    # turn order: [System] → [Human: Objective] → [History] → [Human: Observation]
    # The objective is NOT stored in messages (keeping them lightweight); instead
    # it is reconstructed each turn from state.objective since it never changes
    # during a Worker session.
    objective_anchor = HumanMessage(
        content=f"Objective: {objective}",
        id="worker_objective_anchor",
    )

    # ── Content Passthrough Block ─────────────────────────────────────────────
    # When user_content is present, inject it as a dedicated HumanMessage between
    # the objective and the action history. This ensures the Worker has the raw
    # user message available for verbatim reproduction tasks (heredocs, file
    # writes, exact commands) without relying on the LLM to reproduce content
    # from a summarized objective.
    content_block: list = []
    if user_content:
        content_msg = HumanMessage(
            content=(
                "## Original User Message (use this for exact content reproduction)\n"
                "The following is the EXACT, UNMODIFIED user message. When your objective "
                "requires writing files, running commands, or reproducing content, use the "
                "text below VERBATIM — do NOT paraphrase or summarize it.\n\n"
                f"```\n{user_content}\n```"
            ),
            id="worker_user_content_block",
        )
        content_block = [content_msg]

    logger.info("🌀")
    logger.info("🌀 [WORKER TRACE: PROMPT]")
    logger.info(f"🌀 🎯 Objective: {objective}")
    logger.info(f"🌀 📍 Step: {step_count + 1}/{max_steps} | Model: {state.get('active_model') or 'default'}")
    logger.info(f"🌀 🧩 Active Skills: {state.get('active_skills') or 'None Matched'}")
    logger.info(f"🌀 👁️ Observation: {len(observation)} chars")
    logger.info(f"🌀 📎 User Content: {len(user_content)} chars" if user_content else "🌀 📎 User Content: none")
    logger.info(f"🌀 📜 History: {len(history)} messages.")
    logger.info("🌀 📋 Prompt Structure: [System] → [Objective]%s → [History] → [Observation]",
                " → [UserContent]" if content_block else "")
    logger.info("🌀")

    return {
        "_formatted_prompt": [sys_msg] + [objective_anchor] + content_block + history + [observation_msg],
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
    if all_tools:
        # Prevent parallel tool calls context poisoning for single-action models where supported.
        kwargs = {}
        active_model = state.get("active_model", "")
        if settings.strict_action_loop and active_model.startswith("openai/"):
            kwargs["parallel_tool_calls"] = False
        llm_with_tools = llm.bind_tools(all_tools, **kwargs)
    else:
        llm_with_tools = llm

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
        logger.info("⚡")
        logger.info("⚡ [WORKER: ACTION]")
        if getattr(result, "tool_calls", []):
            logger.info(f"⚡ 🛠 Action: {[tc['name'] for tc in result.tool_calls]}")
        if content_str:
            logger.info(f"⚡ 📝 Response: {content_str}")
        logger.info("⚡")

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
                return "Action successful: multimodal output captured."
            try:
                return json.dumps(result)
            except Exception:
                return str(result)
        return str(result)

    except InfrastructureError:
        raise
    except Exception as e:
        logger.error(f"  -> Worker tool {action_name} failed: {e}")
        return f"Error executing {action_name}: {e}"


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
    declared.add("complete_task")
    return declared


async def worker_tools_node(state: WorkerState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Execute tool calls with metadata-aware strategy:
    - Stateful tools: Sequential, halt on first failure.
    - Read-only tools: Concurrent via asyncio.gather.

    CRITICAL: Tool results are action summaries only.
    Observation is refreshed centrally after actions via environment observers.
    
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
    latest_observation = None  # Will hold the LAST observation (replace-only ephemeral anchor)
    executed_tool_names: list[str] = []
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
        if tool_call["name"] == "complete_task":
            summary = tool_call["args"].get("summary", "Task completed via tool call.")
            tool_messages.append(ToolMessage(
                content=f"Task completed: {summary}",
                tool_call_id=tool_call["id"],
            ))
            return {
                "messages": tool_messages,
                "status": "completed",
                "result_summary": summary,
            }

    # ── Enforce Cardinality Guard ──
    if settings.strict_action_loop:
        stateful_count = 0
        allowed_tool_calls = []
        truncated_tool_calls = []
        for tc in tool_calls:
            if _is_stateful_tool(tc["name"]) and tc["name"] != "batch_actions":
                if stateful_count == 0:
                    allowed_tool_calls.append(tc)
                    stateful_count += 1
                else:
                    truncated_tool_calls.append(tc)
            else:
                allowed_tool_calls.append(tc)

        if truncated_tool_calls:
            logger.warning(
                "WorkerHITL: Cardinality guard dropped %d tool(s): %s", 
                len(truncated_tool_calls), 
                [t['name'] for t in truncated_tool_calls]
            )
            # Synthesize fake failed tool messages to prevent LLM hallucination
            for tc in truncated_tool_calls:
                tool_messages.append(ToolMessage(
                    content="[SYSTEM ERROR]: Tool execution blocked. You violated the strict one-action-per-turn rule. Observe the current state and try again.",
                    tool_call_id=tc["id"]
                ))
            tool_calls = allowed_tool_calls

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
            executed_tool_names.append(action_name)

            try:
                result_content = await _execute_single_tool(action_name, tool_call["args"], config)
            except InfrastructureError as fatal_err:
                logger.error(
                    "WorkerCircuitBreaker: fatal infrastructure error in tool %s: %s",
                    action_name,
                    fatal_err,
                )
                return {
                    "status": "failed",
                    "result_summary": _build_infrastructure_failure_summary(fatal_err),
                    "tool_failure_count": 0,
                    "_retry": False,
                }

            # Handle exceptions as strings
            if isinstance(result_content, str) and _looks_like_tool_error(result_content):
                tool_messages.append(ToolMessage(content=result_content, tool_call_id=call_id))
                # Halt: skip remaining tools
                for remaining in tool_calls[idx + 1:]:
                    tool_messages.append(ToolMessage(
                        content=f"Skipped: previous action '{action_name}' failed.",
                        tool_call_id=remaining["id"],
                    ))
                break

            summary = _summarize_tool_result(action_name, str(result_content))
            tool_messages.append(ToolMessage(content=summary, tool_call_id=call_id))
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
        executed_tool_names.extend(tc["name"] for tc in executable_tool_calls)

        for result_content in results:
            if isinstance(result_content, InfrastructureError):
                logger.error(
                    "WorkerCircuitBreaker: fatal infrastructure error during concurrent tools: %s",
                    result_content,
                )
                return {
                    "status": "failed",
                    "result_summary": _build_infrastructure_failure_summary(result_content),
                    "tool_failure_count": 0,
                    "_retry": False,
                }

        for tool_call, result_content in zip(executable_tool_calls, results):
            call_id = tool_call["id"]
            action_name = tool_call["name"]

            if isinstance(result_content, Exception):
                result_content = f"Error executing {action_name}: {str(result_content)}"

            summary = _summarize_tool_result(action_name, str(result_content))
            tool_messages.append(ToolMessage(content=summary, tool_call_id=call_id))

    # Build update payload: action ledger updates only.
    update_payload: Dict[str, Any] = {"messages": tool_messages}
    update_payload["_refresh_categories"] = sorted(_observation_categories_for_tools(executed_tool_names))

    return update_payload


# ═══════════════════════════════════════════════════════════════
# 5. WORKER SUMMARIZE NODE
# ═══════════════════════════════════════════════════════════════

async def worker_summarize_node(state: WorkerState) -> Dict[str, Any]:
    """
    Called when the Worker reaches END.
    Extracts the last AI message as the result_summary.
    Handles both string and multimodal (list) content formats.
    """
    existing_status = state.get("status")
    existing_summary = (state.get("result_summary") or "").strip()
    if existing_status in {"failed", "escalated", "completed"} and existing_summary:
        return {
            "status": existing_status,
            "result_summary": existing_summary,
        }

    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", settings.worker_max_steps)
    if step_count >= max_steps:
        return {
            "status": "failed",
            "result_summary": f"Worker reached maximum step limit ({max_steps}) without calling complete_task.",
        }

    messages = state.get("messages", [])

    # Find the last AI response and extract its text (handles string and multimodal formats)
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            text_summary = _extract_text_from_content(content)
            if text_summary:
                return {
                    "status": "completed",
                    "result_summary": text_summary,
                }

    return {
        "status": "completed",
        "result_summary": "Task completed but no summary was generated.",
    }
