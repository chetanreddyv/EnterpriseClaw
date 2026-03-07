"""
nodes/agent.py — Core LangChain executor node.

Loads the relevant skill prompt, attaches tools dynamically,
and executes the agent via LangChain. 
"""

import json
import logging
from pathlib import Path
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from core.llm import init_agent_llm

logger = logging.getLogger(__name__)

# Paths
SKILLS_DIR = Path(__file__).parent.parent / "skills"
IDENTITY_FILE = Path(__file__).parent.parent / "skills" / "identity" / "skill.md"

def _load_identity_prompt() -> str:
    """Load the core identity prompt from identity.md."""
    if IDENTITY_FILE.exists():
        return IDENTITY_FILE.read_text()
    return ""



import re

def _parse_skill_frontmatter(skill_path: Path) -> dict:
    """Extracts YAML frontmatter from a Markdown file, including JSON metadata blocks."""
    content = skill_path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return {}

    frontmatter = {}
    for line in match.group(1).strip().split('\n'):
        if ':' in line:
            key, val = line.split(':', 1)
            val = val.strip().strip("'\"")
            
            # Attempt to parse inline JSON (used by Nanobot/OpenClaw metadata)
            if val.startswith('{') and val.endswith('}'):
                try:
                    val = json.loads(val)
                except json.JSONDecodeError:
                    pass
            frontmatter[key.strip()] = val
            
    return frontmatter

def _get_enabled_tools_and_write_actions() -> tuple[list[str], set[str]]:
    """
    Returns enabled tools and write actions.
    Auto-loads from SKILL.md files.
    """
    # 0. The Standard Library (Core Capabilities available to all skills)
    # We remove delegate_task and let the agent browse autonomously
    enabled_tools = {"exec_command", "web_search", "web_fetch", "check_current_time", "schedule_reminder", "save_to_long_term_memory"}
    
    # Tiered Autonomy: Only OS-level destructive and active browsing actions require HITL
    dangerous_actions = {
        "exec_command", "write_file", "delete_file",
        "browser_click", "browser_type", "browser_execute_js", 
        "browser_select_option", "browser_press_key", "browser_hover", 
        "browser_handle_dialog", "browser_file_upload"
    }
    write_actions = set(dangerous_actions)
    
    # 1. Universal Auto-Loader: Scan all skill.md files
    # Case-insensitive match for skill.md or SKILL.md
    for skill_file in SKILLS_DIR.rglob("*"):
        if skill_file.name.lower() != "skill.md":
            continue
            
        skill_name = skill_file.parent.name
        frontmatter = _parse_skill_frontmatter(skill_file)
        requested_tools = []

        # A. Support EnterpriseClaw Native Frontmatter (tools: tool_1, tool_2)
        raw_tools = frontmatter.get("tools", "")
        if raw_tools:
            requested_tools.extend([t.strip() for t in raw_tools.split(",") if t.strip()])

        # B. Support Nanobot/OpenClaw Bridge
        meta_json = frontmatter.get("metadata", {})
        if isinstance(meta_json, dict):
            # If the skill requires CLI binaries, it inherently needs the 'exec_command' tool
            framework_meta = meta_json.get("nanobot", meta_json.get("openclaw", {}))
            if framework_meta.get("requires", {}).get("bins"):
                if "exec_command" not in requested_tools:
                    requested_tools.append("exec_command")

        # 2. Register Discovered Tools
        for action in requested_tools:
            enabled_tools.add(action)
            
            # Safety Fallback: Automatically treat dangerous actions as requiring HITL
            if action in dangerous_actions:
                write_actions.add(action)

    return list(enabled_tools), write_actions

async def agent_node(state: dict) -> dict:
    """
    Execute the LangChain agent with dynamically loaded skills + tools.
    """
    logger.info("--- [Node: Agent] ---")
    
    messages = state.get("messages", [])
    user_input = state.get("user_input", "")
    
    if not messages and not user_input:
        logger.debug("  -> Empty state and no user input, skipping agent loop.")
        return {}
    tool_failure_count = state.get("tool_failure_count", 0)

    # ── Load core identity ──────────────────────────────────────────
    identity_prompt = _load_identity_prompt()

    # ── Fetch long-term memory context and dynamic skill context ────
    memory_context = ""
    skill_prompts = "You are a helpful personal assistant. Be concise and accurate."
    thread_id = state.get("chat_id", "default_thread")
    try:
        from memory.gate import memorygate
        memory_context = await memorygate.get_context(thread_id=thread_id)
        if memory_context != "No established context.":
            logger.info(f"  -> Successfully retrieved memory context ({len(memory_context)} chars)")
        skill_prompts = await memorygate.get_relevant_skills(user_input)
        logger.info(f"  -> Skill prompts loaded ({len(skill_prompts)} chars): {skill_prompts[:200]}...")
    except Exception as e:
        logger.warning(f"  -> Context/Skill retrieval skipped/failed: {e}")

    # ── Construct full system prompt: Identity → Memory → Skill ────
    prompt_parts = []
    if identity_prompt:
        prompt_parts.append(identity_prompt)
    if memory_context:
        prompt_parts.append(f"## User Context (from long-term memory)\\n{memory_context}")
    prompt_parts.append(skill_prompts)
    
    # ── Inject Active Memory Control ──────────────────────────────
    active_memory_prompt = (
        "## Active Memory & Temporal Awareness\n"
        "You have direct control over your long-term memory and schedule.\n"
        "- If you learn an important fact, preference, or rule about the user, ALWAYS use the `save_to_long_term_memory` tool to remember it permanently.\n"
        "- If you need to perform actions in the future (e.g., 'remind me tonight', 'check email in 1 hour'), use the `schedule_reminder` tool.\n"
        "- If you need to orient yourself chronologically, use the `check_current_time` tool."
    )
    prompt_parts.append(active_memory_prompt)
    
    # We append extra formatting rules if we want
    full_system_prompt = "\\n\\n---\\n\\n".join(prompt_parts) + "\\n\\nALWAYS format your output using standard Markdown (use *, _, `, ```, lists). Do NOT use HTML tags. Respond directly to the user."

    # ── Build LangChain Tools ─────────────────────────────────────
    enabled_tool_names, _ = _get_enabled_tools_and_write_actions()
    
    from mcp_servers import GLOBAL_TOOL_REGISTRY
    
    all_tools = []
    for action_name in enabled_tool_names:
        real_func = GLOBAL_TOOL_REGISTRY.get(action_name)
        if not real_func:
            logger.warning(f"  -> Tool {action_name} enabled but missing.")
            continue
        # In LangChain, tools can be raw functions. `bind_tools` converts them.
        all_tools.append(real_func)

    llm = init_agent_llm(state.get("active_model"))

    # Bind tools
    llm_with_tools = llm.bind_tools(all_tools) if all_tools else llm

    try:
        # Prepend the system message
        invoke_messages = [SystemMessage(content=full_system_prompt)]
        
        # Then append the persistent history
        invoke_messages.extend(messages)
        
        # Then append the current user input as a HumanMessage
        if user_input:
            human_msg = HumanMessage(content=user_input)
            invoke_messages.append(human_msg)

        result = await llm_with_tools.ainvoke(invoke_messages)
        logger.info(f"  -> Agent response generated ({len(str(result.content))} chars)")
        
        # Return both the human message (so it saves to state) and the AI response
        new_messages = [human_msg, result] if user_input else [result]
        
        return {
            "messages": new_messages, 
            "tool_failure_count": 0, 
            "agent_response": result.content,
            "user_input": "" # Clear user input so we don't duplicate it on loopbacks
        }

    except Exception as e:
        tool_failure_count += 1
        logger.error(f"  -> Agent error (attempt {tool_failure_count}/3): {e}", exc_info=True)

        if tool_failure_count >= 3:
            logger.error("  -> Max failures reached, returning fallback response")
            return {
                "messages": [AIMessage(content=f"I'm sorry, I encountered repeated errors trying to process your request. Error: {str(e)}\\n\\nPlease try again.")],
                "tool_failure_count": tool_failure_count,
            }

        return {
            "tool_failure_count": tool_failure_count,
            "_retry": True,
        }
