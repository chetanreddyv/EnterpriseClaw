import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from config.settings import settings
from core.exec_sandbox import exec_command

logger = logging.getLogger(__name__)

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
    max_steps: int = settings.worker_max_steps,
    active_model: str = "",
    approved_tools: Optional[List[str]] = None,
    user_content: str = "",
    config: RunnableConfig = None,
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
        max_steps (int): Maximum number of action steps before the Worker gives up. Default 15.
        active_model (str): Internal use only.
        approved_tools (list[str] | None): Internal HITL policy directives inherited from Supervisor.
        user_content (str): Internal use only. Auto-injected raw user message for content fidelity.
    """
    from core.graphs.worker import build_worker_graph

    if max_steps < 1:
        max_steps = 1

    # LangGraph recursion counts graph traversals, not just worker "steps".
    # Worker loop can consume ~3 traversals per step (prompt -> executor -> tools),
    # so provision recursion budget from max_steps to avoid premature aborts.
    worker_recursion_limit = max(50, (max_steps * 4) + 10)

    logger.info(
        "🚀 Delegating task to Worker [model=%s, max_steps=%d, recursion_limit=%d]: %s",
        active_model or "default",
        max_steps,
        worker_recursion_limit,
        objective,
    )

    worker_graph = build_worker_graph()

    worker_init: Dict[str, Any] = {
        "objective": objective,
        "skill_prompts": "",
        "active_skills": [],
        "active_skill_tools": [],
        "max_steps": max_steps,
        "step_count": 0,
        "status": "running",
        "messages": [],
        "environment_snapshot": "No environment state yet. Take the first action to observe the environment.",
        "result_summary": "",
        "tool_failure_count": 0,
        "_retry": False,
        "_formatted_prompt": [],
        "active_model": active_model,
        "approved_tools": approved_tools or [],
    }

    # Content passthrough: attach raw user message for verbatim reproduction tasks.
    if user_content:
        worker_init["user_content"] = user_content
        logger.info(
            "🔗 Content passthrough: attaching %d chars of raw user content to Worker.",
            len(user_content),
        )

    result = await worker_graph.ainvoke(
        worker_init,
        config={
            "recursion_limit": worker_recursion_limit,
            "configurable": config.get("configurable", {}) if config else {}
        },
    )

    status = result.get("status", "completed")
    summary = result.get("result_summary", "Task completed but no summary was generated.")

    if status == "escalated":
        logger.warning(f"⚠️ Worker escalated: {summary}")
        return f"⚠️ WORKER ESCALATED: {summary}"
    elif status == "failed":
        if str(summary).strip().lower().startswith("system infrastructure failure:"):
            logger.error(f"🛑 Worker infrastructure failure: {summary}")
            return f"🛑 WORKER INFRASTRUCTURE FAILURE: {summary}"
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
def complete_task(summary: str) -> str:
    """
    Use this tool ONLY when you have fully accomplished your objective.
    Immediately stops the current task and returns the summary to the Supervisor.
    
    Args:
        summary (str): A detailed explanation of exactly what you accomplished.
            Example: "Successfully navigated to YouTube, searched for the trailer, and clicked play. The video is currently playing."
    """
    # This is a sentinel tool. The worker_tools_node detects its name
    # and sets status="completed" + result_summary=summary in the state.
    return summary


@tool
async def batch_actions(actions: List[Dict[str, Any]], config: RunnableConfig = None) -> str:
    """
    Execute multiple browser actions sequentially in one turn for efficiency.
    Returns only an action summary; environment state is refreshed by the Worker observer loop.
    
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
    from core.browser_session import BrowserSessionManager

    results = []
    try:
        session = await BrowserSessionManager.get_session()
        page = await session.get_current_page()
        
        for action in actions:
            action_type = action.get("type", "")
            index = action.get("index")
            selector = action.get("selector", "")

            if action_type == "click":
                if index is not None:
                    element = await BrowserSessionManager.get_element_by_index(index)
                    if element:
                        xpath = element.xpath
                        clicked = await BrowserSessionManager.evaluate_js(
                            f"""
                            (() => {{
                                const el = document.evaluate("{xpath}", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                if (!el) return false;
                                el.click();
                                return true;
                            }})()
                            """,
                            page=page,
                        )
                        results.append(f"click [{index}]" if clicked else f"click [{index}] FAILED: xpath not found")
                    else:
                        results.append(f"click [{index}] FAILED: not found")
                elif selector:
                    escaped = selector.replace("'", "\\'")
                    await BrowserSessionManager.evaluate_js(
                        f"document.querySelector('{escaped}').click()",
                        page=page,
                    )
                    results.append(f"click '{selector}'")
            elif action_type == "type":
                text = action.get("text", "")
                escaped_text = text.replace("\\", "\\\\").replace("'", "\\'")
                if index is not None:
                    element = await BrowserSessionManager.get_element_by_index(index)
                    if element:
                        xpath = element.xpath
                        await BrowserSessionManager.evaluate_js(
                            f"""
                            (() => {{
                                const el = document.evaluate("{xpath}", document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                if (el) {{ el.focus(); el.value = '{escaped_text}'; el.dispatchEvent(new Event('input', {{bubbles:true}})); }}
                            }})()
                            """,
                            page=page,
                        )
                        results.append(f"type [{index}] = '{text}'")
                    else:
                        results.append(f"type [{index}] FAILED: not found")
                elif selector:
                    escaped_sel = selector.replace("'", "\\'")
                    await BrowserSessionManager.evaluate_js(
                        f"""
                        (() => {{
                            const el = document.querySelector('{escaped_sel}');
                            if (el) {{ el.focus(); el.value = '{escaped_text}'; el.dispatchEvent(new Event('input', {{bubbles:true}})); }}
                        }})()
                        """,
                        page=page,
                    )
                    results.append(f"type '{selector}' = '{text}'")
            elif action_type == "select":
                text = action.get("text", "")
                escaped_sel = selector.replace("'", "\\'")
                escaped_text = text.replace("'", "\\'")
                await BrowserSessionManager.evaluate_js(
                    f"""
                    (() => {{
                        const el = document.querySelector('{escaped_sel}');
                        if (el) {{
                            const opt = Array.from(el.options).find(o => o.text === '{escaped_text}' || o.value === '{escaped_text}');
                            if (opt) {{ el.value = opt.value; el.dispatchEvent(new Event('change', {{bubbles:true}})); }}
                        }}
                    }})()
                    """,
                    page=page,
                )
                results.append(f"select '{selector}' = '{text}'")
            elif action_type == "press_key":
                key = action.get("key", "Enter")
                await page.press(key)
                results.append(f"press_key '{key}'")
            else:
                results.append(f"unknown action type '{action_type}'")

            # Small delay between actions for DOM stability
            await asyncio.sleep(0.2)

    except Exception as e:
        summary = (
            f"Batch failed at step {len(results) + 1}/{len(actions)} "
            f"({action.get('type', '?')} '{action.get('selector', '?')}')\n"
            f"Error: {str(e)}\n"
            f"Previous {len(results)} actions completed: {', '.join(results)}"
        )
        return f"Action failed: {summary}"

    summary = "Batch completed: " + ", ".join(results)
    return f"Action successful: {summary}"


# ═══════════════════════════════════════════════════════════════
# Scheduler Tools (System-Level Scheduling)
# ═══════════════════════════════════════════════════════════════

@tool
async def schedule_background_task(
    objective: str,
    schedule_type: str,
    schedule_value: str,
    deliver_mode: str = "silent",
    target: str = "isolated",
    timezone: str = None,
    skill_name: str = None,
    retry_max_retries: int = None,
    retry_initial_backoff_seconds: int = None,
    retry_max_backoff_seconds: int = None,
    config: RunnableConfig = None,
) -> str:
    """
    Schedule a background task to run on a specific schedule.
    
    The system scheduler will execute this task in an isolated environment
    at the specified time(s). You do NOT need to manage the clock—the backend does.
    
    Args:
        objective (str): What the agent should do (e.g., "Search for new job postings in Python")
        schedule_type (str): One of "cron" (e.g., "0 9 * * *"), "every" (milliseconds), or "at" (timestamp)
        schedule_value (str): The schedule expression
            - For "cron": Standard cron expression (e.g., "0 9 * * 1-5" = 9am weekdays)
            - For "every": Milliseconds as string (e.g., "900000" = 15 minutes)
            - For "at": Unix timestamp in milliseconds or ISO 8601 timestamp
        deliver_mode (str): "silent", "reply", or "announce"
        target (str): "isolated" (run separately, default), "main" (run in user session), or "session:<id>"
        timezone (str): Timezone for cron expressions (e.g., "America/New_York")
        skill_name (str): Optional skill to activate explicitly (e.g., "job-finder", "browser-use").
            Use this when you know which skill the task needs. Bypasses semantic search for
            more reliable tool binding.
        retry_max_retries (int): Optional per-job retry max attempts.
        retry_initial_backoff_seconds (int): Optional per-job initial retry delay.
        retry_max_backoff_seconds (int): Optional per-job max retry delay.
    
    Returns:
        str: Job ID if successful, error message otherwise
    """
    try:
        from core.scheduler import get_scheduler
        
        scheduler = await get_scheduler()
        
        # Generate a descriptive job name
        name = f"Background Task: {objective[:50]}"
        
        cfg = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
        origin_thread_id = cfg.get("thread_id")
        origin_platform = cfg.get("platform")

        job_id = scheduler.add_job(
            name=name,
            objective=objective,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
            deliver_mode=deliver_mode,
            target=target,
            tz=timezone,
            skill_name=skill_name,
            channel=str(origin_thread_id) if origin_thread_id is not None else None,
            platform=str(origin_platform) if origin_platform is not None else "system",
            retry_max_retries=retry_max_retries,
            retry_initial_backoff_seconds=retry_initial_backoff_seconds,
            retry_max_backoff_seconds=retry_max_backoff_seconds,
        )
        
        return f"✅ Scheduled task (ID: {job_id}). Will execute on schedule: {schedule_type}={schedule_value}"
    except Exception as e:
        logger.error(f"schedule_background_task failed: {e}")
        return f"❌ Failed to schedule task: {str(e)}"


@tool
async def send_user_notification(message: str, channel: str = "telegram", thread_id: str = None) -> str:
    """
    Send a notification to the user from a background task.
    
    Use this when a scheduled job needs to notify the user of results.
    The message will be delivered asynchronously via the specified channel.
    
    Args:
        message (str): The message to send to the user
        channel (str): "telegram", "web", or "all"
        thread_id (str): Specific thread/conversation ID (optional; uses last known thread if not specified)
    
    Returns:
        str: Confirmation or error message
    """
    try:
        from core.channel_manager import channel_manager
        
        if not thread_id:
            # Default to a system thread or the user's last known thread
            thread_id = "system_notification"
        
        await channel_manager.send_message(
            platform=channel,
            thread_id=thread_id,
            content=f"📬 Background Task Update:\n{message}",
        )
        
        return f"✅ Notification sent to {channel}"
    except Exception as e:
        logger.error(f"send_user_notification failed: {e}")
        return f"❌ Failed to send notification: {str(e)}"


@tool
async def list_scheduled_tasks() -> str:
    """
    View all currently scheduled background tasks.
    
    Returns a summary of all active and pending tasks with their schedules and next run times.
    """
    try:
        from core.scheduler import get_scheduler
        
        scheduler = await get_scheduler()
        jobs = scheduler.list_jobs()
        
        if not jobs:
            return "No scheduled tasks currently."
        
        lines = []
        for job in jobs:
            job_id = job.get("id")
            name = job.get("name")
            enabled = job.get("enabled", True)
            schedule = job.get("schedule", {})
            state = job.get("state", {})
            
            status = "✅ Active" if enabled else "⏸️ Paused"
            next_run = state.get("nextRunAtMs")
            
            next_run_str = ""
            if next_run:
                from datetime import datetime, timezone
                next_dt = datetime.fromtimestamp(next_run / 1000, tz=timezone.utc)
                next_run_str = f" | Next: {next_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}"
            
            schedule_kind = schedule.get("kind", "unknown")
            schedule_expr = schedule.get("expr") or schedule.get("everyMs") or schedule.get("atMs", "N/A")
            
            lines.append(f"- `{job_id}` [{status}] {name}\n  Schedule: {schedule_kind}={schedule_expr}{next_run_str}")
        
        return "📅 **Scheduled Tasks:**\n" + "\n".join(lines)
    except Exception as e:
        logger.error(f"list_scheduled_tasks failed: {e}")
        return f"❌ Failed to list tasks: {str(e)}"


@tool
async def get_scheduler_health() -> str:
    """Return scheduler health snapshot (in-flight, due lag, retries)."""
    try:
        from core.scheduler import get_scheduler

        scheduler = await get_scheduler()
        health = scheduler.get_health_snapshot()
        return "🩺 Scheduler Health\n" + json.dumps(health, indent=2, sort_keys=True)
    except Exception as e:
        logger.error(f"get_scheduler_health failed: {e}")
        return f"❌ Failed to get scheduler health: {str(e)}"


@tool
async def cancel_task(job_id: str) -> str:
    """
    Cancel and remove a scheduled background task.
    
    Args:
        job_id (str): The ID of the task to cancel (from list_scheduled_tasks)
    
    Returns:
        str: Confirmation or error message
    """
    try:
        from core.scheduler import get_scheduler
        
        scheduler = await get_scheduler()
        success = scheduler.cancel_job(job_id)
        
        if success:
            return f"✅ Task {job_id} has been cancelled and removed."
        else:
            return f"❌ Task {job_id} not found."
    except Exception as e:
        logger.error(f"cancel_task failed: {e}")
        return f"❌ Failed to cancel task: {str(e)}"


@tool
async def cancel_scheduled_task(job_id: str) -> str:
    """
    Cancel and remove one scheduled background task.

    Alias of cancel_task with a scheduler-specific name for clearer routing.

    Args:
        job_id (str): The ID of the task to cancel (from list_scheduled_tasks)

    Returns:
        str: Confirmation or error message
    """
    try:
        from core.scheduler import get_scheduler

        scheduler = await get_scheduler()
        success = scheduler.cancel_job(job_id)

        if success:
            return f"✅ Task {job_id} has been cancelled and removed."
        return f"❌ Task {job_id} not found."
    except Exception as e:
        logger.error(f"cancel_scheduled_task failed: {e}")
        return f"❌ Failed to cancel scheduled task: {str(e)}"


@tool
async def cancel_all_scheduled_tasks() -> str:
    """
    Cancel and remove all scheduled background tasks.

    Returns:
        str: Summary of cancelled jobs and any failures.
    """
    try:
        from core.scheduler import get_scheduler

        scheduler = await get_scheduler()
        jobs = scheduler.list_jobs()

        if not jobs:
            return "No scheduled tasks currently."

        cancelled: list[str] = []
        failed: list[str] = []

        for job in jobs:
            job_id = str(job.get("id", "")).strip()
            if not job_id:
                continue

            try:
                if scheduler.cancel_job(job_id):
                    cancelled.append(job_id)
                else:
                    failed.append(job_id)
            except Exception as cancel_error:
                failed.append(f"{job_id} ({cancel_error})")

        if not cancelled and failed:
            return "❌ Failed to cancel scheduled tasks:\n" + "\n".join([f"- {item}" for item in failed])

        response_lines = [
            f"✅ Cancelled {len(cancelled)} scheduled task(s).",
        ]
        if cancelled:
            response_lines.append("Cancelled IDs: " + ", ".join(cancelled))
        if failed:
            response_lines.append("⚠️ Failed IDs: " + ", ".join(failed))

        return "\n".join(response_lines)
    except Exception as e:
        logger.error(f"cancel_all_scheduled_tasks failed: {e}")
        return f"❌ Failed to cancel all scheduled tasks: {str(e)}"


# Register tools so they are dynamically loaded by the GLOBAL_TOOL_REGISTRY in __init__.py
TOOL_REGISTRY = {
    "save_to_long_term_memory": save_to_long_term_memory,
    "delegate_task": delegate_task,
    "escalate_to_supervisor": escalate_to_supervisor,
    "complete_task": complete_task,
    "batch_actions": batch_actions,
    "schedule_background_task": schedule_background_task,
    "send_user_notification": send_user_notification,
    "list_scheduled_tasks": list_scheduled_tasks,
    "get_scheduler_health": get_scheduler_health,
    "cancel_scheduled_task": cancel_scheduled_task,
    "cancel_task": cancel_task,
    "cancel_all_scheduled_tasks": cancel_all_scheduled_tasks,
    "exec_command": exec_command,
}

