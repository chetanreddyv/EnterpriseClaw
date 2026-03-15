import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)

REMINDERS_FILE = Path("data/reminders.json")

# Separator protocol for splitting tool output into summary + observation
OBSERVATION_SEPARATOR = "\n===OBSERVATION===\n"

@tool
def schedule_reminder(message: str, time_str: str) -> str:
    """
    Schedule a reminder or future task. The heartbeat system will check this schedule
    every 15 minutes and inject the message back into the conversation context.
    
    Args:
        message (str): The reminder text (e.g. "Check the stock price of AAPL", "Reply to Bob's email")
        time_str (str): When you want to be reminded. You can use ISO format (2025-10-15T14:30:00Z) or plain English (e.g., "in 30 mins", "tomorrow at 9am").
    """
    try:
        REMINDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        reminders = []
        if REMINDERS_FILE.exists():
            with open(REMINDERS_FILE, "r") as f:
                reminders = json.load(f)
                
        reminders.append({
            "message": message,
            "target_time": time_str,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending"
        })
        
        with open(REMINDERS_FILE, "w") as f:
            json.dump(reminders, f, indent=2)
            
        return f"Successfully scheduled reminder '{message}' for '{time_str}'."
    except Exception as e:
        logger.error(f"Failed to schedule reminder: {e}")
        return f"Failed to schedule reminder: {e}"

@tool
async def save_to_long_term_memory(fact: str) -> str:
    """
    Save an important fact, preference, or rule about the user into your long-term 
    semantic memory vector store. Use this whenever the user tells you something 
    you need to remember for future conversations. DO NOT save temporary/ephemeral context.
    
    Args:
        fact (str): The specific, detailed fact to remember (e.g., "User's favorite programming language is Python.", "User prefers concise answers without apologies.", "User lives in New York.")
    """
    try:
        from memory.retrieval import memory_retrieval
        
        class TempMemoryObj:
            facts = [fact]
            preferences = []
            corrections = []
            updates = []
            obsolete_items = []
            important = True
            
        mem_obj = TempMemoryObj()
        # Default to a generic source thread since active control is manual
        await memory_retrieval.store.apply_updates(mem_obj, source_thread_id="agent_manual_memory")
        return f"Successfully saved to long-term memory: '{fact}'."
    except Exception as e:
        logger.error(f"Failed to save memory: {e}")
        return f"Error saving memory: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# Supervisor-Worker Architecture Tools
# ═══════════════════════════════════════════════════════════════

@tool
async def delegate_task(
    objective: str,
    domain: str,
    max_steps: int = 15,
    active_model: str = "",
    approved_tools: list[str] | None = None,
) -> str:
    """
    Delegate a complex, multi-step task to a specialized Worker agent.
    The Worker will execute the task autonomously and return a summary.
    
    Use this for tasks that require multiple interactions with an environment:
    - Browsing the web, filling forms, clicking buttons
    - Running code and inspecting outputs
    - Any task requiring more than one tool call in sequence
    
    Args:
        objective (str): A clear, specific description of what the Worker should accomplish.
            Example: "Navigate to greenhouse.io and apply for the Software Engineer role."
        domain (str): Worker execution domain. One of:
            - "browser": web navigation and form interaction workflows.
            - "exec": terminal/code execution workflows.
            - "all": broad multi-tool reasoning workflows.
        max_steps (int): Maximum number of action steps before the Worker gives up. Default 15.
        active_model (str): Internal use only.
        approved_tools (list[str] | None): Internal HITL policy directives inherited from Supervisor.
    """
    from core.graphs.worker import build_worker_graph

    normalized_domain = str(domain or "").strip().lower()
    domain_to_categories = {
        "browser": ["browser"],
        "exec": ["exec"],
        "all": ["all"],
    }

    if normalized_domain not in domain_to_categories:
        return "Error: `domain` must be one of: browser, exec, all."

    normalized_categories = domain_to_categories[normalized_domain]

    logger.info(
        "🚀 Delegating task to Worker [domain=%s, model=%s]: %s",
        normalized_domain,
        active_model or "default",
        objective,
    )

    worker_graph = build_worker_graph()
    result = await worker_graph.ainvoke({
        "objective": objective,
        "required_tool_categories": normalized_categories,
        "skill_prompts": "",
        "active_skills": [],
        "active_skill_tools": [],
        "max_steps": max_steps,
        "step_count": 0,
        "status": "running",
        "messages": [],
        "observation": "No environment state yet. Take the first action to observe the environment.",
        "result_summary": "",
        "tool_failure_count": 0,
        "_retry": False,
        "_formatted_prompt": [],
        "active_model": active_model,
        "approved_tools": approved_tools or [],
    })

    status = result.get("status", "completed")
    summary = result.get("result_summary", "Task completed but no summary was generated.")

    if status == "escalated":
        logger.warning(f"⚠️ Worker escalated: {summary}")
        return f"⚠️ WORKER ESCALATED: {summary}"
    elif status == "failed":
        logger.error(f"❌ Worker failed: {summary}")
        return f"❌ Task failed: {summary}"

    logger.info(f"✅ Worker completed: {summary[:100]}...")
    return summary


@tool
def escalate_to_supervisor(reason: str) -> str:
    """
    Use this when you are stuck, confused, or the environment is not responding to your actions.
    Immediately stops the current task and returns control to the Supervisor with an explanation.
    
    Args:
        reason (str): A clear description of why you are stuck and what you need.
            Example: "A CAPTCHA appeared that I cannot solve. Need human intervention."
    """
    # This is a sentinel tool. The worker_tools_node detects its name
    # and sets status="escalated" + result_summary=reason in the state.
    return reason


@tool
async def batch_actions(actions: list, config: RunnableConfig = None) -> str:
    """
    Execute multiple actions sequentially in one turn for efficiency.
    Only takes the expensive observation snapshot at the very end (or at the point of failure).
    
    Args:
        actions (list): A list of action dictionaries. Each action has:
            - "type": "click" | "type" | "select" | "press_key"
            - "selector": CSS selector for the target element
            - "text": Text to type (for "type" and "select" actions)
            - "key": Key to press (for "press_key" action)
        
        Example: [
            {"type": "type", "selector": "#name", "text": "Alice"},
            {"type": "type", "selector": "#email", "text": "alice@example.com"},
            {"type": "click", "selector": "#submit"}
        ]
    """
    from mcp_servers.browser_tools import BrowserSessionManager

    # Extract thread_id from config
    thread_id = "default"
    if config and "configurable" in config:
        thread_id = config["configurable"].get("thread_id", "default")

    results = []
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        
        for i, action in enumerate(actions):
            action_type = action.get("type", "")
            selector = action.get("selector", "")

            if action_type == "click":
                await page.click(selector)
                results.append(f"click '{selector}'")
            elif action_type == "type":
                text = action.get("text", "")
                await page.fill(selector, text)
                results.append(f"type '{selector}' = '{text}'")
            elif action_type == "select":
                text = action.get("text", "")
                await page.select_option(selector, text)
                results.append(f"select '{selector}' = '{text}'")
            elif action_type == "press_key":
                key = action.get("key", "Enter")
                await page.keyboard.press(key)
                results.append(f"press_key '{key}'")
            else:
                results.append(f"unknown action type '{action_type}'")

            # Small delay between actions for DOM stability
            await page.wait_for_timeout(200)

    except Exception as e:
        # Partial failure: snapshot at point of crash
        try:
            from mcp_servers.browser_tools import browser_snapshot
            snapshot = await browser_snapshot.ainvoke({}, config=config)
        except Exception:
            snapshot = "Failed to capture snapshot after error."

        summary = (
            f"Batch failed at step {len(results) + 1}/{len(actions)} "
            f"({action.get('type', '?')} '{action.get('selector', '?')}')\n"
            f"Error: {str(e)}\n"
            f"Previous {len(results)} actions completed: {', '.join(results)}"
        )
        return f"{summary}{OBSERVATION_SEPARATOR}{snapshot}"

    # Full success: snapshot at the end
    try:
        from mcp_servers.browser_tools import browser_snapshot
        await page.wait_for_load_state("networkidle", timeout=3000)
        snapshot = await browser_snapshot.ainvoke({}, config=config)
    except Exception:
        snapshot = "Page state captured but snapshot failed."

    summary = "Batch completed: " + ", ".join(results)
    return f"{summary}{OBSERVATION_SEPARATOR}{snapshot}"


# Register tools so they are dynamically loaded by the GLOBAL_TOOL_REGISTRY in __init__.py
TOOL_REGISTRY = {
    "schedule_reminder": schedule_reminder,
    "save_to_long_term_memory": save_to_long_term_memory,
    "delegate_task": delegate_task,
    "escalate_to_supervisor": escalate_to_supervisor,
    "batch_actions": batch_actions,
}

