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

def _get_enabled_tools_and_write_actions(active_skills: list[str] = None, load_all: bool = False) -> tuple[list[str], set[str]]:
    """
    Returns enabled tools and write actions dynamically based on active skills.
    Defaults to Standard Library if no skills are active.
    """
    enabled_tools = {"exec_command", "web_search", "web_fetch", "check_current_time", "schedule_reminder", "save_to_long_term_memory"}
    
    # Tiered Autonomy: Only OS-level destructive and active browsing actions require HITL
    dangerous_actions = {
        "exec_command", "write_file", "delete_file",
        "browser_click", "browser_type", "browser_execute_js", 
        "browser_select_option", "browser_press_key", "browser_hover", 
        "browser_handle_dialog", "browser_file_upload"
    }
    write_actions = set(dangerous_actions)
    
    skill_files_to_check = []
    
    if load_all:
        for skill_file in SKILLS_DIR.rglob("*"):
            if skill_file.name.lower() == "skill.md":
                skill_files_to_check.append(skill_file)
    elif active_skills:
        for skill_name in active_skills:
            skill_dir = SKILLS_DIR / skill_name
            if skill_dir.exists():
                for f in skill_dir.iterdir():
                    if f.name.lower() == "skill.md":
                        skill_files_to_check.append(f)
                        break

    for skill_file in skill_files_to_check:
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
    original_query = state.get("original_query", "")
    
    # ALWAYS update intent when the user speaks. 
    if user_input:
        original_query = user_input
        
    if not messages and not user_input and not original_query:
        logger.debug("  -> Empty state and no user input, skipping agent loop.")
        return {}
        
    tool_failure_count = state.get("tool_failure_count", 0)

    # ── Load core identity ──────────────────────────────────────────
    identity_prompt = _load_identity_prompt()

    # ── Fetch long-term memory context and dynamic skill context ────
    memory_context = ""
    skill_prompts = state.get("skill_prompts")
    matched_skill_names = state.get("active_skills")
    
    thread_id = state.get("chat_id", "default_thread")
    try:
        from memory.retrieval import memory_retrieval
        memory_context = await memory_retrieval.get_context(thread_id=thread_id)
        if memory_context != "No established context.":
            logger.info(f"  -> Successfully retrieved memory context ({len(memory_context)} chars)")
            
        if matched_skill_names is None or skill_prompts is None:
            skill_prompts, matched_skill_names = await memory_retrieval.get_relevant_skills(original_query)
            logger.info(f"  -> Skill prompts loaded ({len(skill_prompts)} chars): {skill_prompts[:200]}...")
        else:
            logger.info(f"  -> Using frozen active skills from state: {matched_skill_names}")
    except Exception as e:
        logger.warning(f"  -> Context/Skill retrieval skipped/failed: {e}")
        if matched_skill_names is None:
            matched_skill_names = []
        if skill_prompts is None:
            skill_prompts = "You are a helpful personal assistant. Be concise and accurate."

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
    enabled_tool_names, _ = _get_enabled_tools_and_write_actions(active_skills=matched_skill_names)
    
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
        # ── 2. Compile & Truncate Messages ───────────────────────────────────
        invoke_messages = [SystemMessage(content=full_system_prompt)]
        
        # Sliding window truncation (last 5 turns) to prevent context bloat
        MAX_TURNS = 5
        human_indices = [i for i, m in enumerate(messages) if getattr(m, "type", "") == "human"]
        recent_messages = messages[human_indices[-MAX_TURNS]:] if len(human_indices) > MAX_TURNS else messages

        human_msg = HumanMessage(content=user_input) if user_input else None
        
        # Create the raw sequence of messages meant for this invocation
        raw_sequence = recent_messages + ([human_msg] if human_msg else [])

        filtered_messages = []
        for msg in raw_sequence:
            msg_copy = msg.copy()
            
            # A. Safely handle multimodal lists WITHOUT destroying images
            if isinstance(msg_copy.content, list):
                new_content = []
                for item in msg_copy.content:
                    if isinstance(item, dict):
                        if item.get("type") == "text" and item.get("text"):
                            new_content.append({"type": "text", "text": item["text"]})
                        elif item.get("type") == "image_url":
                            new_content.append(item) 
                    elif isinstance(item, str) and item.strip():
                        new_content.append({"type": "text", "text": item})
                msg_copy.content = new_content if new_content else " "

            # B. Ensure string content is never null or purely whitespace
            elif not msg_copy.content or (isinstance(msg_copy.content, str) and not msg_copy.content.strip()):
                msg_copy.content = " " 

            # C. Strip poisoned fallback errors to prevent hallucinatory loops
            if getattr(msg_copy, "type", "") == "ai" and isinstance(msg_copy.content, str):
                if any(err in msg_copy.content for err in ["I encountered an error", "Error code:", "I'm sorry"]):
                    continue  
                    
            # D. Strip completely empty AI messages (that have no tool calls)
            if getattr(msg_copy, "type", "") == "ai" and msg_copy.content == " " and not getattr(msg_copy, "tool_calls", []):
                continue

            filtered_messages.append(msg_copy)

        # ── Consolidate Adjacent Roles (Fixes Qwen Chat Template Crash) ────
        consolidated_messages = []
        for msg in filtered_messages:
            if not consolidated_messages:
                consolidated_messages.append(msg)
                continue
                
            prev_msg = consolidated_messages[-1]
            prev_type = getattr(prev_msg, "type", "")
            curr_type = getattr(msg, "type", "")
            
            # If two messages of the same type are adjacent, merge them
            if prev_type == curr_type and prev_type in ("human", "ai"):
                # Merge contents safely to preserve multimodal lists
                if isinstance(prev_msg.content, list) or isinstance(msg.content, list):
                    prev_list = prev_msg.content if isinstance(prev_msg.content, list) else [{"type": "text", "text": str(prev_msg.content)}]
                    curr_list = msg.content if isinstance(msg.content, list) else [{"type": "text", "text": str(msg.content)}]
                    merged_content = prev_list + [{"type": "text", "text": "\n\n[Follow-up]: "}] + curr_list
                else:
                    merged_content = f"{prev_msg.content}\n\n[Follow-up]: {msg.content}".strip()
                    
                if prev_type == "human":
                    consolidated_messages[-1] = HumanMessage(content=merged_content)
                else:
                    msg.content = merged_content if merged_content else " "
                    consolidated_messages[-1] = msg
            else:
                consolidated_messages.append(msg)

        invoke_messages.extend(consolidated_messages)

        # ── 3. Execute & Return ──────────────────────────────────────────────
        result = await llm_with_tools.ainvoke(invoke_messages)
        logger.info(f"  -> Agent response generated ({len(str(result.content))} chars)")
        
        # Note: We return the raw un-merged human_msg to LangGraph state so DB checkpoints remain pure
        new_messages = [human_msg, result] if human_msg else [result]
        
        return {
            "messages": new_messages, 
            "tool_failure_count": 0, 
            "agent_response": result.content,
            "user_input": "", 
            "original_query": original_query,
            "active_skills": matched_skill_names,
            "skill_prompts": skill_prompts
        }

    except Exception as e:
        tool_failure_count += 1
        logger.error(f"  -> Agent error (attempt {tool_failure_count}/3): {e}", exc_info=True)

        if tool_failure_count >= 3:
            logger.error("  -> Max failures reached, returning fallback response")
            fallback_msg = AIMessage(content="I encountered an error processing your request. Please try again.")
            new_msgs = [human_msg, fallback_msg] if human_msg else [fallback_msg]
            
            return {
                "messages": new_msgs, 
                "tool_failure_count": 0, 
                "user_input": "",
                "original_query": original_query
            }

        return {
            "tool_failure_count": tool_failure_count,
            "_retry": True,
        }
