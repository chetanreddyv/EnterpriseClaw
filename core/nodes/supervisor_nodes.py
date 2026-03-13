"""
core/nodes/supervisor_nodes.py — Nodes for the Supervisor (MainGraph).

The Supervisor is lightweight. It handles:
1. Intent: Fetch memory context, normalize user input.
2. Prompt Building: Identity + memory + GC + token pruning.
3. Execution: Fixed tool set (web_search, check_time, delegate_task).
4. Tool Execution: Runs supervisor-level tools.
"""

import logging
from typing import Dict, Any
from pathlib import Path

from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage,
    RemoveMessage, trim_messages,
)
from langchain_core.runnables import RunnableConfig

from core.graphs.states import SupervisorState
from core.llm import init_agent_llm
from mcp_servers import GLOBAL_TOOL_REGISTRY

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────
IDENTITY_FILE = Path(__file__).parent.parent.parent / "skills" / "identity" / "skill.md"

# The Supervisor's fixed tool set — simple, read-only + delegation.
SUPERVISOR_TOOLS = {
    "web_search", "web_fetch",
    "check_current_time", "save_to_long_term_memory",
    "delegate_task",
}


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

    # 3. Resolve Skills (for prompt enrichment, not tool binding)
    skill_prompts = state.get("skill_prompts", "")
    matched_skill_names = state.get("active_skills", [])
    
    try:
        from memory.retrieval import memory_retrieval
        new_prompts, new_matched = await memory_retrieval.get_relevant_skills(original_query)
        logger.info(f"SupervisorIntent: Matched skills: {new_matched}")
        
        # Sticky routing: retain active skill on short follow-ups
        if not new_matched and matched_skill_names:
            pass
        else:
            skill_prompts = new_prompts
            matched_skill_names = new_matched
    except Exception as e:
        logger.warning(f"SupervisorIntent: Skill retrieval failed: {e}")
        if not skill_prompts:
            skill_prompts = "You are a helpful personal assistant. Be concise and accurate."

    logger.info(f"SupervisorIntent: Retrieved {len(memory_context)} bytes of memory context.")
    logger.info(f"SupervisorIntent: Matched skills: {matched_skill_names}")

    # 4. Inject user message
    messages_update = []
    if user_input:
        messages_update.append(HumanMessage(content=user_input))

    return {
        "user_input": user_input,
        "original_query": original_query,
        "memory_context": memory_context,
        "skill_prompts": skill_prompts,
        "active_skills": matched_skill_names,
        "messages": messages_update,
        "tool_failure_count": 0,
        "_retry": False,
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
    # 1. ── Construct system prompt: Identity → Memory → Skills ────
    prompt_parts = []

    identity_prompt = _load_identity_prompt()
    if identity_prompt:
        prompt_parts.append(identity_prompt)

    memory_context = state.get("memory_context")
    if memory_context:
        prompt_parts.append(f"## User Context (from long-term memory)\n{memory_context}")

    skill_prompts = state.get("skill_prompts")
    if skill_prompts:
        prompt_parts.append(skill_prompts)

    # Active Memory + Delegation instructions
    delegation_prompt = (
        "## Active Memory & Task Delegation\n"
        "You have direct control over your long-term memory and can delegate complex tasks.\n"
        "- To remember facts, use the `save_to_long_term_memory` tool.\n"
        "- For time awareness, use `check_current_time`.\n"
        "- For quick factual lookups, use `web_search` or `web_fetch`.\n"
        "- For complex multi-step tasks (browsing the web, running code, filling forms), "
        "use `delegate_task` with an objective and tool_domain ('browser', 'exec', or 'all').\n"
        "  Example: delegate_task(objective='Go to example.com and describe the page', tool_domain='browser')\n"
    )
    prompt_parts.append(delegation_prompt)

    status_prompt = "## Operational Status\n- You are in LIVE mode.\n- TOOL CALLING IS ENABLED."
    prompt_parts.append(status_prompt)

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

    # Hard cap at 6 messages
    if len(processed_history) > 6:
        processed_history = processed_history[-6:]

    temp_messages = [sys_msg] + processed_history

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
    skill_list = state.get("active_skills") or []
    logger.info(f"═ 📜 Memory Context: {mem_len} bytes | 🛠 Matched Skills: {skill_list}")

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
    import asyncio

    messages = state.get("messages", [])
    if not messages:
        return {}

    last_message = messages[-1]
    if not getattr(last_message, "tool_calls", []):
        return {}

    tool_messages = []

    for tool_call in last_message.tool_calls:
        action_name = tool_call["name"]
        tool_args = tool_call["args"]
        call_id = tool_call["id"]

        logger.info(f"🛠️ Supervisor executing tool: {action_name}")

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

            if hasattr(func, "ainvoke"):
                result = await func.ainvoke(tool_args, config=config)
            else:
                result = func(**tool_args)
                if inspect.isawaitable(result):
                    result = await result
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=call_id))
        except Exception as e:
            logger.error(f"  -> Supervisor tool {action_name} failed: {e}")
            tool_messages.append(ToolMessage(
                content=f"Error executing {action_name}: {str(e)}",
                tool_call_id=call_id,
            ))

    return {"messages": tool_messages}
