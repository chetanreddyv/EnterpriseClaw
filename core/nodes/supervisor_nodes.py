"""
core/nodes/supervisor_nodes.py — Nodes for the Supervisor (MainGraph).

The Supervisor is lightweight. It handles:
1. Intent: Fetch memory context, normalize user input.
2. Prompt Building: Identity + memory + GC + token pruning.
3. Execution: Fixed tool set (web_search, save memory, delegate_task).
4. Tool Execution: Runs supervisor-level tools.
"""

import logging
from typing import Dict, Any
from pathlib import Path
from datetime import datetime, timezone

from langgraph.errors import GraphBubbleUp
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage,
    RemoveMessage, trim_messages,
)
from langchain_core.runnables import RunnableConfig

from core.graphs.states import SupervisorState
from core.llm import init_agent_llm, is_llm_connection_error
from mcp_servers import GLOBAL_TOOL_REGISTRY

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
IDENTITY_FILE = Path(__file__).parent.parent.parent / "skills" / "identity" / "skill.md"

# The Supervisor's fixed tool set — simple, read-only + delegation.
SUPERVISOR_TOOLS = {
    "web_search", "web_fetch",
    "save_to_long_term_memory",
    "list_scheduled_tasks",
    "cancel_task",
    "cancel_all_scheduled_tasks",
    "delegate_task",
}

DIAGNOSTIC_TOOL_MARKERS = (
    "error",
    "rejected",
    "escalated",
    "failed",
    "skipped",
)


def _load_identity_prompt() -> str:
    """Load the core identity prompt from identity.md."""
    if IDENTITY_FILE.exists():
        return IDENTITY_FILE.read_text()
    return ""


# ═══════════════════════════════════════════════════════════════
# 1. INTENT NODE
# ═══════════════════════════════════════════════════════════════

async def supervisor_intent_node(state: SupervisorState) -> Dict[str, Any]:
    """
    Normalize inputs, fetch long-term memory, and prepare for prompt building.
    Simplified from V2: no dynamic skill routing needed for the Supervisor.
    """
    user_input = state.get("user_input", "")
    original_query = state.get("original_query", "")

    # 1. Normalize Intent
    if user_input:
        if isinstance(user_input, list):
            text_parts = [item.get("text", "") for item in user_input if item.get("type") == "text"]
            original_query = " ".join(text_parts).strip()
        else:
            original_query = str(user_input)

    thread_id = state.get("chat_id", "default_thread")

    # 2. Retrieve Memory
    memory_context = ""
    try:
        from memory.retrieval import memory_retrieval
        memory_context = await memory_retrieval.get_context(thread_id=thread_id)
    except Exception as e:
        logger.warning(f"SupervisorIntent: Memory retrieval failed: {e}")

    logger.info(f"SupervisorIntent: Retrieved {len(memory_context)} bytes of memory context.")

    # 4. Inject user message
    messages_update = []
    if user_input:
        messages_update.append(HumanMessage(content=user_input))

    return {
        "user_input": user_input,
        "original_query": original_query,
        "memory_context": memory_context,
        "messages": messages_update,
        "tool_failure_count": 0,
        "_retry": False,
        "delegation_attempts": {},
        "escalated_objectives": [],
    }


# ═══════════════════════════════════════════════════════════════
# 2. PROMPT BUILDER NODE
# ═══════════════════════════════════════════════════════════════

async def supervisor_prompt_builder_node(state: SupervisorState) -> Dict[str, Any]:
    """
    Build the system prompt and prune history.
    Reuses the V2 token-aware pruning and GC logic.
    No transient_tool_output handling — Workers manage their own ephemeral state.
    """
    # 1. ── Construct modular system prompt ───────────────────────
    prompt_parts = []

    # Section 1: System Clock
    now_utc = datetime.now(timezone.utc)
    now_local = datetime.now().astimezone()
    prompt_parts.append(
        "## System Clock\n"
        f"- Current UTC Time: {now_utc.isoformat()}\n"
        f"- Local System Time: {now_local.isoformat()}\n"
        f"- Local Timezone: {now_local.tzname()}\n"
        "- Treat this clock as authoritative for temporal reasoning."
    )

    # Section 2: Core Identity
    identity_prompt = _load_identity_prompt()
    if identity_prompt:
        prompt_parts.append(f"## Core Identity\n{identity_prompt}")

    # Section 3: User Context
    memory_context = state.get("memory_context")
    if memory_context:
        prompt_parts.append(f"## User Context\n{memory_context}")

    # Section 4: Core Capabilities
    capabilities_prompt = (
        "## Core Capabilities\n"
        "- Use `save_to_long_term_memory` to persist durable user facts and preferences.\n"
        "- Use the System Clock section above for time-sensitive decisions.\n"
        "- Use `web_search` and `web_fetch` for factual lookups.\n"
        f"- Exact supervisor tools: {', '.join(sorted(SUPERVISOR_TOOLS))}.\n"
        "- Use `delegate_task(objective=...)` for complex multi-step execution. "
        "The Worker will select tools dynamically from matched skills."
    )
    prompt_parts.append(capabilities_prompt)

    # Section 5: Rules
    rules_lines = [
        "## Rules",
        "- You are in LIVE mode and TOOL CALLING IS ENABLED.",
        "- If the user asks to list cron jobs, scheduled jobs, or scheduled tasks, call `list_scheduled_tasks` directly.",
        "- If the user asks to cancel all cron/scheduled jobs, call `cancel_all_scheduled_tasks` directly.",
        "- If the user asks to cancel one scheduled job and provides a job id, call `cancel_task(job_id=...)`.",
        "- Delegate complex tasks via `delegate_task(objective='...')` with a focused objective.",
        "- Example: delegate_task(objective='Navigate LinkedIn, find the hiring manager, and email my resume summary').",
        "- If Worker escalation occurs, do not re-delegate the same unchanged objective.",
        "- Ask the user for clarification, constraints, or missing context after escalation.",
        "- Never invent tool names. Use only the exact supervisor tools listed above.",
        "- Never mention delegation mechanics to the user. Return only outcome-focused responses.",
    ]
    prompt_parts.append("\n".join(rules_lines))

    full_system_prompt = "\n\n---\n\n".join(prompt_parts) + (
        "\n\nALWAYS format your output using standard Markdown. Do NOT use HTML tags. "
        "Respond directly to the user.\n\n"
        "CRITICAL: Be concise. Keep conversational responses under 2 sentences "
        "unless the user explicitly asks for detail."
    )

    sys_msg = SystemMessage(content=full_system_prompt, id="supervisor_system_msg")

    # 2. ── Prune the Message History ───────────────────────────
    messages = state.get("messages", [])

    def _rough_token_counter(msgs: list) -> int:
        count = 0
        for m in msgs:
            content = m.content
            if isinstance(content, str):
                count += len(content) // 4
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            count += len(str(item.get("text", ""))) // 4
                        elif item.get("type") == "image_url":
                            count += 250
            else:
                count += len(str(content)) // 4
            count += 4
        return count

    # Pre-emptively truncate massive AI/Tool outputs (ephemeral copies)
    processed_history = []
    for m in messages:
        if isinstance(m, SystemMessage):
            continue
        if isinstance(m, (AIMessage, ToolMessage)):
            content = m.content
            if isinstance(content, str) and len(content) > 3000:
                truncated = content[:3000] + "\n...[Content truncated to preserve context window]..."
                if hasattr(m, "model_copy"):
                    m = m.model_copy(update={"content": truncated})
                else:
                    m = m.copy(update={"content": truncated})
        processed_history.append(m)

    # 1. Identify the most recent HumanMessage index
    last_human_idx = next(
        (i for i in reversed(range(len(processed_history)))
         if isinstance(processed_history[i], HumanMessage)),
        0
    )

    # 2. Slice the history to start EXACTLY at the last Human message,
    # capturing the current active loop without dropping the prompt.
    active_turn_history = processed_history[last_human_idx:]

    temp_messages = [sys_msg] + active_turn_history

    trimmed_messages = trim_messages(
        temp_messages,
        max_tokens=6000,
        strategy="last",
        token_counter=_rough_token_counter,
        include_system=True,
        start_on="human",
        allow_partial=False,
    )

    retained_history = trimmed_messages[1:] if trimmed_messages else []
    retained_ids = {m.id for m in retained_history if hasattr(m, "id") and m.id}

    # 3. ── GC stale messages from SqliteCheckpointDB ────────────
    all_old_ids = {m.id for m in messages if not isinstance(m, SystemMessage) and hasattr(m, "id") and m.id}
    stale_ids = all_old_ids - retained_ids
    delete_cmds = [RemoveMessage(id=msg_id) for msg_id in stale_ids if msg_id]

    logger.info("═"*80)
    logger.info("═ [SUPERVISOR TRACE: PROMPT]")
    logger.info(f"═ 🧠 Active Model: {state.get('active_model') or 'DEFAULT (lmstudio/qwen)'}")
    logger.info(f"═ 📊 System Prompt: {len(full_system_prompt)} chars")
    
    # Context summary
    mem_len = len(state.get("memory_context") or "")
    logger.info(f"═ 📜 Memory Context: {mem_len} bytes")

    # History summary
    logger.info(f"═ 💬 History: {len(retained_history)} messages retained.")
    for i, m in enumerate(retained_history):
        role = "AI" if isinstance(m, AIMessage) else "User" if isinstance(m, HumanMessage) else "Tool" if isinstance(m, ToolMessage) else "Sys"
        logger.info(f"   ═ [{i}] {role}: {str(m.content)[:100]}...")
    logger.info("═"*80)

    return {
        "messages": delete_cmds,
        "_formatted_prompt": [sys_msg] + retained_history,
    }


def _has_diagnostic_signal(content: str) -> bool:
    text = str(content).strip().lower()
    if not text:
        return False
    return any(marker in text for marker in DIAGNOSTIC_TOOL_MARKERS)


async def supervisor_compaction_node(state: SupervisorState) -> Dict[str, Any]:
    """
    Remove intermediate tool-call artifacts from the latest completed turn.
    Keeps Human + final AI response while preserving diagnostic traces.
    """
    if state.get("_retry"):
        return {}

    messages = state.get("messages", [])
    if len(messages) < 3:
        return {}

    final_ai_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", []):
            content = msg.content
            if isinstance(content, str) and content.strip():
                final_ai_idx = idx
                break

    if final_ai_idx <= 0:
        return {}

    human_idx = -1
    for idx in range(final_ai_idx - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            human_idx = idx
            break

    if human_idx < 0:
        return {}

    candidate_segment = messages[human_idx + 1:final_ai_idx]
    if not candidate_segment:
        return {}

    # Preserve full trace for problematic turns.
    for msg in candidate_segment:
        if isinstance(msg, ToolMessage) and _has_diagnostic_signal(msg.content):
            return {}

    remove_ids: list[str] = []
    for msg in candidate_segment:
        msg_id = getattr(msg, "id", None)
        if not msg_id:
            continue
        if isinstance(msg, ToolMessage):
            remove_ids.append(msg_id)
        elif isinstance(msg, AIMessage) and getattr(msg, "tool_calls", []):
            remove_ids.append(msg_id)

    if not remove_ids:
        return {}

    logger.info(
        "SupervisorCompaction: Removing %d intermediate tool messages from latest turn.",
        len(remove_ids),
    )
    return {"messages": [RemoveMessage(id=msg_id) for msg_id in remove_ids]}


# ═══════════════════════════════════════════════════════════════
# 3. EXECUTOR NODE
# ═══════════════════════════════════════════════════════════════

async def supervisor_executor_node(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Invoke the LLM with the fixed Supervisor tool set.
    """
    invoke_messages = state.get("_formatted_prompt")
    if not invoke_messages:
        return {}

    # Compile the fixed Supervisor tool set
    all_tools = []
    for tool_name in SUPERVISOR_TOOLS:
        func = GLOBAL_TOOL_REGISTRY.get(tool_name)
        if func:
            all_tools.append(func)

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

        # ── [LOGGING] LLM OUTPUT BLOCK ──
        logger.info("✨")
        logger.info("✨ [SUPERVISOR: LLM OUTPUT]")
        if getattr(result, "tool_calls", []):
            logger.info(f"✨ 🛠 Tool Calls: {[tc['name'] for tc in result.tool_calls]}")
        if content_str:
            logger.info(f"✨ 📝 Response: {content_str}")
        logger.info("✨")

        if not content_str and not getattr(result, "tool_calls", []):
            logger.warning("SupervisorExecutor: Blank generation detected.")
            return {"messages": [result], "_retry": True}

        return {"messages": [result]}

    except Exception as e:
        logger.error(f"SupervisorExecutor: LLM API Call Failed: {e}", exc_info=True)
        if is_llm_connection_error(e):
            active_model = state.get("active_model") or "default"
            guidance = "Please check your configured model backend and try again."
            if str(active_model).startswith("lmstudio/"):
                guidance = (
                    "LM Studio appears unreachable at the configured local endpoint. "
                    "Please ensure LM Studio is running and the server is started (default: http://localhost:1234/v1)."
                )

            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"I could not reach the model backend for `{active_model}` due to a connection error. "
                            f"{guidance}"
                        )
                    )
                ],
                "_retry": False,
                "tool_failure_count": 0,
            }

        return {"_retry": True}


# ═══════════════════════════════════════════════════════════════
# 4. TOOL EXECUTION NODE
# ═══════════════════════════════════════════════════════════════

async def supervisor_tools_node(state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Execute Supervisor-level tool calls.
    Supervisor tools are simple (web_search, delegate_task) so no 
    observation splitting or bloat protection is needed here.
    The delegate_task tool blocks until the Worker finishes.
    """
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_message = messages[-1]
    if not getattr(last_message, "tool_calls", []):
        return {}

    tool_messages = []
    delegation_attempts = dict(state.get("delegation_attempts") or {})
    escalated_objectives = set(state.get("escalated_objectives") or [])

    for tool_call in last_message.tool_calls:
        action_name = tool_call["name"]
        tool_args = dict(tool_call.get("args") or {})
        call_id = tool_call["id"]

        logger.info(f"🛠️ Supervisor executing tool: {action_name}")

        objective_key = ""
        if action_name == "delegate_task":
            objective_key = str(tool_args.get("objective", "")).strip()

            if objective_key and objective_key in escalated_objectives:
                tool_messages.append(ToolMessage(
                    content=(
                        "⚠️ Delegation blocked: this objective already escalated in the current turn. "
                        "Ask the user for clarification or adjust the objective before retrying."
                    ),
                    tool_call_id=call_id,
                ))
                continue

            if objective_key:
                delegation_attempts[objective_key] = delegation_attempts.get(objective_key, 0) + 1

        func = GLOBAL_TOOL_REGISTRY.get(action_name)
        if not func:
            tool_messages.append(ToolMessage(
                content=f"Error: Tool '{action_name}' not found.",
                tool_call_id=call_id,
            ))
            continue

        try:
            import inspect
            
            # Inject active_model into delegate_task if available
            if action_name == "delegate_task":
                tool_args["active_model"] = state.get("active_model", "")
                tool_args["approved_tools"] = state.get("approved_tools", [])

            if hasattr(func, "ainvoke"):
                result = await func.ainvoke(tool_args, config=config)
            else:
                result = func(**tool_args)
                if inspect.isawaitable(result):
                    result = await result

            result_text = str(result)
            if action_name == "delegate_task" and objective_key and result_text.startswith("⚠️ WORKER ESCALATED:"):
                escalated_objectives.add(objective_key)

            tool_messages.append(ToolMessage(content=result_text, tool_call_id=call_id))
        except GraphBubbleUp:
            logger.info("Supervisor tool %s raised GraphBubbleUp; propagating for HITL flow.", action_name)
            raise
        except Exception as e:
            logger.error(f"  -> Supervisor tool {action_name} failed: {e}")
            tool_messages.append(ToolMessage(
                content=f"Error executing {action_name}: {str(e)}",
                tool_call_id=call_id,
            ))

    return {
        "messages": tool_messages,
        "delegation_attempts": delegation_attempts,
        "escalated_objectives": sorted(escalated_objectives),
    }
