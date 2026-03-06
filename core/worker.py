import logging
import asyncio

from core.messaging import IncomingMessageEvent, ResumeEvent, SystemEvent
from core.channel_manager import channel_manager
from langgraph.types import Command

logger = logging.getLogger("core.worker")

async def agent_daemon(graph, event: IncomingMessageEvent) -> dict:
    """
    Unified LangGraph runner for the Event Bus.
    """
    config = {"configurable": {"thread_id": event.user_id, "platform": event.platform}}

    try:
        async for graph_event in graph.astream(
            {
                "chat_id": event.user_id,
                "user_input": event.text,
                "tool_failure_count": 0,
            },
            config=config,
            stream_mode="updates",
        ):
            logger.debug(f"Graph event ({event.platform}): {graph_event}")

        state = await graph.aget_state(config)

        if state.next:
            interrupted = state.tasks[0].interrupts[0].value
            tool_args = interrupted.get("tool_args", {})
            
            # Send abstract approval request to the ChannelManager
            await channel_manager.request_approval(
                platform=event.platform,
                thread_id=event.user_id,
                tool_name=interrupted.get('action', 'unknown'),
                args=tool_args
            )
            logger.info(f"🔐 HITL: Sent approval request to {event.platform} user {event.user_id}")
            
            return {
                "approval_required": True,
                "action": interrupted.get("action", "unknown"),
            }
        else:
            messages = state.values.get("messages", [])
            response = "Done!"
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    if isinstance(last_msg.content, str) and last_msg.content:
                        response = last_msg.content
                    elif isinstance(last_msg.content, list):
                        texts = [item.get("text", "") for item in last_msg.content if isinstance(item, dict) and "text" in item]
                        if texts:
                            response = "\n".join(texts)

            if response.strip() != "HEARTBEAT_OK" and response != "Done!":
                await channel_manager.send_message(
                    platform=event.platform,
                    thread_id=event.user_id,
                    content=response
                )
                logger.info(f"💬 Sent response to {event.platform} user {event.user_id}")
            elif response.strip() == "HEARTBEAT_OK":
                logger.info(f"🔇 Suppressed HEARTBEAT_OK from {event.platform} user {event.user_id}")

            asyncio.create_task(_run_memorygate(event.user_id, event.text, response))
            
            return {"response": response}

    except Exception as e:
        logger.error(f"❌ Graph execution error ({event.platform}): {e}", exc_info=True)
        error_text = f"❌ Sorry, I encountered an error:\n{str(e)[:500]}"
        try:
            await channel_manager.send_message(event.platform, event.user_id, error_text)
        except Exception:
            # The channel adapter is dead. Inject the error into the state 
            # so it's not lost forever.
            logger.critical(f"FATAL: Channel manager failed to deliver error for thread {event.user_id}")
            # (Optional: push to a Redis dead-letter queue here)
        return {"error": str(e)[:500]}


async def resume_daemon(graph, event: ResumeEvent) -> dict:
    """
    Unified LangGraph resumer for the Event Bus.
    """
    config = {"configurable": {"thread_id": event.user_id, "platform": event.platform}}
    try:
        async for graph_event in graph.astream(
            Command(resume=event.decision),
            config=config,
            stream_mode="updates",
        ):
            logger.debug(f"Resume event ({event.platform}): {graph_event}")

        state = await graph.aget_state(config)
        if not state.next:
            response = state.values.get("agent_response", "Done!")
            if response == "Done!":
                messages = state.values.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, "content") and isinstance(last_msg.content, str):
                        response = last_msg.content

            await channel_manager.send_message(event.platform, event.user_id, response)
            return {"response": response}
        else:
            interrupted = state.tasks[0].interrupts[0].value
            tool_args = interrupted.get("tool_args", {})
            # Loopback HITL
            await channel_manager.request_approval(
                platform=event.platform,
                thread_id=event.user_id,
                tool_name=interrupted.get('action', 'unknown'),
                args=tool_args
            )
            return {"status": "paused_again"}

    except Exception as e:
        logger.error(f"❌ Resume error ({event.platform}): {e}", exc_info=True)
        await channel_manager.send_message(event.platform, event.user_id, f"❌ Error resuming action:\n`{str(e)[:500]}`")
        return {"error": str(e)[:500]}

async def system_daemon(graph, event: SystemEvent) -> dict:
    """
    Unified LangGraph runner for Headless System Events (e.g., cron jobs).
    Executes in the background using the specified thread_id context.
    Optionally delivers the response via the ChannelManager.
    """
    config = {"configurable": {"thread_id": event.user_id, "platform": event.platform}}

    try:
        async for graph_event in graph.astream(
            {
                "chat_id": event.user_id,
                "user_input": event.text,
                "tool_failure_count": 0,
            },
            config=config,
            stream_mode="updates",
        ):
            logger.debug(f"Graph system event ({event.platform}): {graph_event}")

        state = await graph.aget_state(config)

        if state.next:
            interrupted = state.tasks[0].interrupts[0].value
            tool_args = interrupted.get("tool_args", {})
            
            if event.deliver:
                await channel_manager.request_approval(
                    platform=event.platform,
                    thread_id=event.user_id,
                    tool_name=interrupted.get('action', 'unknown'),
                    args=tool_args
                )
                logger.info(f"🔐 HITL: Sent system approval request to {event.platform} user {event.user_id}")
            
            return {
                "approval_required": True,
                "action": interrupted.get("action", "unknown"),
            }
        else:
            messages = state.values.get("messages", [])
            response = "Done!"
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    if isinstance(last_msg.content, str) and last_msg.content:
                        response = last_msg.content
                    elif isinstance(last_msg.content, list):
                        texts = [item.get("text", "") for item in last_msg.content if isinstance(item, dict) and "text" in item]
                        if texts:
                            response = "\n".join(texts)

            if event.deliver and response.strip() != "HEARTBEAT_OK" and response != "Done!":
                await channel_manager.send_message(
                    platform=event.platform,
                    thread_id=event.user_id,
                    content=response
                )
                logger.info(f"💬 Sent system response to {event.platform} user {event.user_id}")

            asyncio.create_task(_run_memorygate(event.user_id, event.text, response))
            
            return {"response": response}

    except Exception as e:
        logger.error(f"❌ System graph execution error ({event.platform}): {e}", exc_info=True)
        if event.deliver:
            try:
                await channel_manager.send_message(
                    event.platform, 
                    event.user_id, 
                    f"❌ Background task error:\n{str(e)[:500]}"
                )
            except Exception:
                # The channel adapter is dead. Inject the error into the state 
                # so it's not lost forever.
                logger.critical(f"FATAL: Channel manager failed to deliver system error for thread {event.user_id}")
                # (Optional: push to a Redis dead-letter queue here)
        return {"error": str(e)[:500]}

async def _run_memorygate(thread_id: str, user_input: str, agent_response: str):
    """
    Background task: extract context from the conversation
    and store in long-term memory.
    """
    try:
        from memory import memorygate
        await memorygate.process(thread_id, user_input, agent_response)
        # Background sync Zvec vector index AFTER SQLite writes are done
        await memorygate.store.sync_pending_memories()
    except Exception as e:
        logger.error(f"Memorygate error: {e}", exc_info=True)
