"""
mcp_servers/background_tools.py

Provides tools for asynchronous background execution (sub-agents).
The main agent delegates long-running tasks here via `delegate_task`.
"""

import asyncio
import logging
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import httpx

logger = logging.getLogger(__name__)

async def run_subagent_task(query: str, thread_id: str, platform: str):
    """
    Background task that executes a delegated task via the General Sub-Agent.
    Runs ephemerally (no checkpointer) and notifies the user when done.
    """
    logger.info(f"[SubAgent] Starting task: '{query[:80]}' (thread={thread_id}, platform={platform})")

    try:
        from nodes.subagents import build_subagent_graph
        subagent_graph = build_subagent_graph()

        sub_state = {
            "messages": [],
            "user_input": query,
            "chat_id": f"subagent_{thread_id}",
            "tool_failure_count": 0,
            "action_count": 0,
        }

        # Ephemeral execution (no checkpointer)
        result = await subagent_graph.ainvoke(
            sub_state,
            config={"recursion_limit": 100}
        )

        # Safely parse the final response
        summary = "Task completed but no summary was produced."
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, "content"):
                if isinstance(last_msg.content, str) and last_msg.content:
                    summary = last_msg.content
                elif isinstance(last_msg.content, list):
                    texts = [
                        item.get("text", "")
                        for item in last_msg.content
                        if isinstance(item, dict) and "text" in item
                    ]
                    if texts:
                        summary = "\n".join(texts)

        action_count = result.get("action_count", 0)
        logger.info(f"[SubAgent] Task completed ({action_count} actions): '{query[:50]}'")

        # Notify user via the Universal Gateway using the original thread_id
        async with httpx.AsyncClient() as client:
            await client.post(
                f"http://localhost:8000/api/v1/system/{thread_id}/notify",
                json={
                    "message": f"🔔 **[Sub-Agent Report]** ({action_count} actions)\n\n{summary}",
                    "platform": platform
                },
                timeout=30.0
            )

    except Exception as e:
        logger.error(f"[SubAgent] Task failed: {e}", exc_info=True)
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"http://localhost:8000/api/v1/system/{thread_id}/notify",
                    json={
                        "message": f"🔔 **[Sub-Agent Failed]**\n\n❌ {str(e)[:500]}",
                        "platform": platform
                    },
                    timeout=10.0
                )
        except Exception:
            pass


@tool
async def delegate_task(query: str, config: RunnableConfig) -> str:
    """
    Delegate a task to the autonomous sub-agent for background execution.
    Use this for long-running tasks like browser automation, job applications,
    multi-step research, data gathering, or any task that takes many steps.

    The sub-agent has access to ALL tools (browser, web search, etc.) and
    will work autonomously until the task is complete, then report back.

    Args:
        query: A detailed description of the task to accomplish. Be specific
               about what you want the sub-agent to do, including any URLs,
               credentials to use, or steps to follow.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "default_thread")
    platform = config.get("configurable", {}).get("platform", "telegram")
    logger.info(f"Delegating task: {query[:80]} (thread: {thread_id} via {platform})")

    # Fire and forget
    asyncio.create_task(run_subagent_task(query, thread_id, platform))

    return (
        f"✅ Task delegated to sub-agent: '{query[:100]}'\n\n"
        "The sub-agent is now working autonomously in the background. "
        "It will notify you with a report when the task is complete."
    )


# Backward compatibility alias (Python-level only, not registered as separate tool)
delegate_research = delegate_task

TOOL_REGISTRY = {
    "delegate_task": delegate_task,
}
