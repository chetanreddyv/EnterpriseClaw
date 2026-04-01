"""
app.py — FastAPI application.

Handles Telegram webhooks, security validation, and bridges
the messaging layer with the LangGraph agentic loop.
"""

import asyncio
import logging
from typing import Optional
from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from langgraph.types import Command
from config.settings import settings
from interfaces.telegram import TelegramClient
from interfaces.whatsapp import WhatsAppClient
from core.graphs.supervisor import build_supervisor_graph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from core.channel_manager import channel_manager
from core.llm import extract_response

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
# Suppress high-frequency reload file-change chatter from watchfiles.
logging.getLogger("watchfiles").setLevel(logging.WARNING)
logging.getLogger("watchfiles.main").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ── Globals (initialized at startup) ────────────────────────
telegram_client: TelegramClient = None
whatsapp_client: Optional[WhatsAppClient] = None
graph = None
checkpointer = None


# ==========================================================
# 0. Shared Logic
# ==========================================================

async def _handle_commands(chat_id: str, text: str, platform: str) -> bool:
    """Handle slash commands (e.g. /model, /permit). Returns True if handled."""
    if not isinstance(text, str) or not text.startswith("/"): return False
    parts = text.split()
    cmd = parts[0].lower()

    async def _invoke_registry_tool(tool_name: str, args: dict | None = None) -> str:
        from mcp_servers import GLOBAL_TOOL_REGISTRY
        import inspect

        func = GLOBAL_TOOL_REGISTRY.get(tool_name)
        if not func:
            return f"❌ Tool `{tool_name}` is not available."

        payload = args or {}
        if hasattr(func, "ainvoke"):
            result = await func.ainvoke(
                payload,
                config={"configurable": {"thread_id": str(chat_id), "platform": platform}},
            )
        else:
            result = func(**payload)
            if inspect.isawaitable(result):
                result = await result

    if cmd in ("/new", "/clear"):
        try:
            from memory.retrieval import memory_retrieval
            from langchain_core.messages import RemoveMessage
            
            # 1. Clear relational DB history
            await memory_retrieval.db.clear_history(str(chat_id))
            
            # 2. Clear graph state messages
            config = {"configurable": {"thread_id": str(chat_id)}}
            state = await graph.aget_state(config)
            
            messages_to_remove = [
                RemoveMessage(id=m.id) for m in state.values.get("messages", []) if getattr(m, "id", None)
            ]
            if messages_to_remove:
                await graph.aupdate_state(config, {"messages": messages_to_remove})
                
            await channel_manager.send_message(platform, chat_id, "🧹 Conversation history cleared. New chat started!")
        except Exception as e:
            logger.error(f"Failed to clear history: {e}", exc_info=True)
            await channel_manager.send_message(platform, chat_id, f"❌ Failed to clear history: {e}")
        return True

    if cmd == "/model" and len(parts) > 1:
        await graph.aupdate_state({"configurable": {"thread_id": chat_id}}, {"active_model": parts[1]})
        await channel_manager.send_message(platform, chat_id, f"🔄 Brain swapped! Now using: `{parts[1]}`")
        return True

    if cmd in ("/permit", "/deny") and len(parts) > 1:
        from core.hitl import get_deny_marker

        state = await graph.aget_state({"configurable": {"thread_id": chat_id}})
        approved = set(state.values.get("approved_tools") or [])
        target_tool = parts[1].strip()
        deny_marker = get_deny_marker(target_tool)

        if cmd == "/permit":
            approved.discard(deny_marker)
            approved.add(target_tool)
            msg = f"🔓 `{target_tool}` is now AUTO-APPROVED"
        else:
            approved.discard(target_tool)
            approved.add(deny_marker)
            msg = f"🔒 `{target_tool}` now REQUIRES approval"

        await graph.aupdate_state(
            {"configurable": {"thread_id": chat_id}},
            {"approved_tools": sorted(approved)},
        )
        await channel_manager.send_message(platform, chat_id, msg)
        return True

    if cmd == "/tools":
        from core.hitl import get_tool_tier
        from mcp_servers import GLOBAL_TOOL_REGISTRY

        policy_entries = set((await graph.aget_state({"configurable": {"thread_id": chat_id}})).values.get("approved_tools") or [])
        approved = {
            p for p in policy_entries
            if isinstance(p, str) and p and not p.startswith("deny:")
        }
        denied = {
            p[len("deny:"):]
            for p in policy_entries
            if isinstance(p, str) and p.startswith("deny:") and p[len("deny:"):]
        }

        all_tools = sorted(GLOBAL_TOOL_REGISTRY.keys())
        lines = []
        for t in all_tools:
            tier = get_tool_tier(t)
            if tier == "autonomous":
                lines.append(f"🟢 `{t}`")
            elif tier == "allowed":
                if t in denied:
                    lines.append(f"🔒 `{t}` (denied, requires approval)")
                elif t in approved:
                    lines.append(f"🔓 `{t}` (permitted)")
                else:
                    lines.append(f"🔵 `{t}` (auto-allowed)")
            elif t in approved:
                lines.append(f"🔓 `{t}` (permitted)")
            else:
                lines.append(f"🔒 `{t}` (requires approval)")
        await channel_manager.send_message(platform, chat_id, "🛠️ **Tool Permissions (This Thread)**\n🟢 Autonomous | 🔵 Allowed | 🔓 Permitted | 🔒 Requires Approval\n" + "\n".join(lines))
        return True

    if cmd == "/models":
        configured = []
        if settings.openai_api_key:
            configured.append("openai")
        if settings.google_api_key:
            configured.append("google")
        if settings.claude_api_key:
            configured.append("claude")
        if settings.lm_studio_base_url:
            configured.append("lm_studio")

        if configured:
            final_msg = "🌐 **Configured Model Providers:**\n" + "\n".join([f"- `{m}`" for m in configured])
        else:
            final_msg = "No models available."

        await channel_manager.send_message(platform, chat_id, final_msg)
        return True

    if cmd == "/cron":
        subcommand = parts[1].lower() if len(parts) > 1 else "list"

        if subcommand in {"help", "h", "?"}:
            help_text = (
                "📅 **Cron Commands**\n"
                "- `/cron` or `/cron list` — list scheduled tasks\n"
                "- `/cron cancel <job_id>` — cancel one scheduled task\n\n"
                "- `/cron cancel all` or `/cron cancel-all` — cancel all scheduled tasks\n\n"
                "To create tasks, use natural language, for example:\n"
                "\"Schedule a background task to check AI jobs daily at 9 AM.\""
            )
            await channel_manager.send_message(platform, chat_id, help_text)
            return True

        if subcommand in {"list", "ls"}:
            try:
                result = await _invoke_registry_tool("list_scheduled_tasks")
                await channel_manager.send_message(platform, chat_id, result)
            except Exception as e:
                await channel_manager.send_message(platform, chat_id, f"❌ Failed to list cron jobs: {e}")
            return True

        if subcommand in {"cancel-all", "clear", "wipe", "purge"}:
            try:
                result = await _invoke_registry_tool("cancel_all_scheduled_tasks")
                await channel_manager.send_message(platform, chat_id, result)
            except Exception as e:
                await channel_manager.send_message(platform, chat_id, f"❌ Failed to cancel all cron jobs: {e}")
            return True

        if subcommand in {"cancel", "remove", "rm", "delete"}:
            if len(parts) < 3:
                await channel_manager.send_message(
                    platform,
                    chat_id,
                    "Usage: `/cron cancel <job_id>` or `/cron cancel all`",
                )
                return True

            job_id = parts[2].strip()
            if job_id.lower() in {"all", "*"}:
                try:
                    result = await _invoke_registry_tool("cancel_all_scheduled_tasks")
                    await channel_manager.send_message(platform, chat_id, result)
                except Exception as e:
                    await channel_manager.send_message(platform, chat_id, f"❌ Failed to cancel all cron jobs: {e}")
                return True

            try:
                result = await _invoke_registry_tool("cancel_scheduled_task", {"job_id": job_id})
                await channel_manager.send_message(platform, chat_id, result)
            except Exception as e:
                await channel_manager.send_message(platform, chat_id, f"❌ Failed to cancel cron job: {e}")
            return True

        await channel_manager.send_message(
            platform,
            chat_id,
            "Unknown `/cron` command. Use `/cron help`. To create a task, send a normal message like: Schedule a background task to ...",
        )
        return True

    if cmd == "/settings":
        subcommand = parts[1].lower() if len(parts) > 1 else "list"

        # ── /settings help ───────────────────────────────────
        if subcommand in {"help", "h", "?"}:
            help_text = (
                "⚙️ **Settings Commands**\n"
                "- `/settings` or `/settings list` — show all current settings\n"
                "- `/settings set <key> <value>` — change a setting (takes effect immediately)\n"
                "- `/settings reset` — reload settings from .env file\n\n"
                "**Toggleable Keys:**\n"
                "- `hitl_enabled` — enable/disable HITL approval gates\n"
                "- `default_model` — default LLM model string\n"
                "- `worker_max_steps` — max steps per Worker task\n"
                "- `worker_max_observation_chars` — observation payload size limit\n"
                "- `worker_max_skill_prompt_chars` — skill prompt context limit\n"
                "- `supervisor_token_budget` — supervisor history trimming budget\n"
                "- `supervisor_content_truncation` — AI/Tool content truncation chars\n"
                "- `scheduler_max_concurrent_jobs` — max concurrent cron jobs\n"
                "- `scheduler_heartbeat_seconds` — seconds between scheduler checks\n"
                "- `scheduler_run_history_max` — run history records per job\n\n"
                "**Example:**\n"
                "`/settings set hitl_enabled false`\n"
                "`/settings set default_model google_genai/gemini-2.0-flash`\n"
                "`/settings set worker_max_steps 25`"
            )
            await channel_manager.send_message(platform, chat_id, help_text)
            return True

        # ── /settings list ───────────────────────────────────
        if subcommand in {"list", "ls", "show", "get"}:
            hitl_icon = "✅" if settings.hitl_enabled else "❌"
            lines = [
                "⚙️ **Current Settings**\n",
                f"**🤖 Model**",
                f"  `default_model` = `{settings.default_model}`\n",
                f"**🔐 HITL**",
                f"  `hitl_enabled` = {hitl_icon} `{settings.hitl_enabled}`\n",
                f"**🌀 Worker**",
                f"  `worker_max_steps` = `{settings.worker_max_steps}`",
                f"  `worker_max_observation_chars` = `{settings.worker_max_observation_chars:,}`",
                f"  `worker_max_skill_prompt_chars` = `{settings.worker_max_skill_prompt_chars:,}`\n",
                f"**🧠 Supervisor**",
                f"  `supervisor_token_budget` = `{settings.supervisor_token_budget:,}`",
                f"  `supervisor_content_truncation` = `{settings.supervisor_content_truncation:,}`\n",
                f"**📅 Scheduler**",
                f"  `scheduler_max_concurrent_jobs` = `{settings.scheduler_max_concurrent_jobs}`",
                f"  `scheduler_heartbeat_seconds` = `{settings.scheduler_heartbeat_seconds}`",
                f"  `scheduler_run_history_max` = `{settings.scheduler_run_history_max}`",
            ]
            await channel_manager.send_message(platform, chat_id, "\n".join(lines))
            return True

        # ── /settings set <key> <value> ──────────────────────
        if subcommand == "set":
            if len(parts) < 4:
                await channel_manager.send_message(
                    platform, chat_id,
                    "Usage: `/settings set <key> <value>`\nExample: `/settings set hitl_enabled false`"
                )
                return True

            key = parts[2].lower().strip()
            raw_value = parts[3].strip()

            # Allowed mutable keys and their types
            MUTABLE_SETTINGS = {
                "hitl_enabled": bool,
                "default_model": str,
                "worker_max_steps": int,
                "worker_max_observation_chars": int,
                "worker_max_skill_prompt_chars": int,
                "supervisor_token_budget": int,
                "supervisor_content_truncation": int,
                "scheduler_max_concurrent_jobs": int,
                "scheduler_heartbeat_seconds": int,
                "scheduler_run_history_max": int,
            }

            if key not in MUTABLE_SETTINGS:
                known = ", ".join(f"`{k}`" for k in sorted(MUTABLE_SETTINGS))
                await channel_manager.send_message(
                    platform, chat_id,
                    f"❌ Unknown setting `{key}`.\nKnown keys: {known}\n\nTip: use `/settings help` for details."
                )
                return True

            expected_type = MUTABLE_SETTINGS[key]
            try:
                if expected_type is bool:
                    if raw_value.lower() in {"true", "1", "yes", "on"}:
                        coerced = True
                    elif raw_value.lower() in {"false", "0", "no", "off"}:
                        coerced = False
                    else:
                        raise ValueError(f"Expected true/false, got '{raw_value}'")
                elif expected_type is int:
                    coerced = int(raw_value)
                    if coerced < 0:
                        raise ValueError("Value must be a non-negative integer")
                else:
                    coerced = raw_value

                # Mutate the live settings singleton
                object.__setattr__(settings, key, coerced)
                icon = "✅" if coerced is True else ("❌" if coerced is False else "✏️")
                await channel_manager.send_message(
                    platform, chat_id,
                    f"{icon} Setting updated: `{key}` → `{coerced}`\n\n"
                    "⚠️ *This change is live immediately but not persisted. "
                    "To make it permanent, add it to your `.env` file.*"
                )
            except ValueError as e:
                await channel_manager.send_message(
                    platform, chat_id,
                    f"❌ Invalid value for `{key}`: {e}"
                )
            return True

        # ── /settings reset ──────────────────────────────────
        if subcommand == "reset":
            try:
                from config.settings import Settings
                new_settings = Settings()
                # Copy all mutable fields from freshly loaded settings
                for field_name in new_settings.model_fields:
                    object.__setattr__(settings, field_name, getattr(new_settings, field_name))
                await channel_manager.send_message(
                    platform, chat_id,
                    "🔄 Settings reloaded from `.env` file successfully."
                )
            except Exception as e:
                await channel_manager.send_message(
                    platform, chat_id,
                    f"❌ Failed to reload settings: {e}"
                )
            return True

        await channel_manager.send_message(
            platform, chat_id,
            "Unknown `/settings` command. Use `/settings help`.",
        )
        return True

    return False


def _parse_whatsapp_hitl_decision(text: str) -> str | None:
    """Parse a text message into a HITL resume decision when explicitly provided."""
    normalized = (text or "").strip().lower()
    if normalized in {"/approve", "approve"}:
        return "approve"
    if normalized in {"/reject", "reject"}:
        return "reject"
    if normalized in {"/edit", "edit"}:
        return "edit"
    return None


# ==========================================================
# 1. Lifespan (startup / shutdown)
# ==========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global telegram_client, whatsapp_client, graph, checkpointer

    logger.info("🚀 Starting Personal Assistant...")

    # ── Onboarding check ─────────────────────────────────────
    if settings.needs_onboarding:
        logger.warning("  ⚠️  EnterpriseClaw is not configured yet! Run the setup wizard:")
        logger.warning("  make setup")
        logger.warning("  Or use: uv run onboard")
        # Yield to keep FastAPI alive but don't initialize anything
        yield
        return

    # Initialize Telegram client
    telegram_client = TelegramClient(settings.telegram_bot_token)
    channel_manager.register_client("telegram", telegram_client)
    logger.info("✅ Telegram client initialized and registered")

    # Initialize WhatsApp client (optional)
    whatsapp_bridge_task = None
    whatsapp_poll_task = None
    if settings.whatsapp_enabled:
        whatsapp_client = WhatsAppClient(
            bridge_url=settings.whatsapp_bridge_url,
            bridge_token=settings.whatsapp_bridge_token,
            allow_from=settings.whatsapp_allow_from_list,
        )
        channel_manager.register_client("whatsapp", whatsapp_client)
        logger.info("✅ WhatsApp client initialized and registered")
    else:
        whatsapp_client = None
        logger.info("ℹ️ WhatsApp channel is disabled")
    
    # Register Web client
    from interfaces.web_chat import web_client
    channel_manager.register_client("web", web_client)
    logger.info("✅ Web client registered")

    # Initialize MemoryRetrieval
    try:
        from memory.retrieval import memory_retrieval
        await memory_retrieval.initialize()
        logger.info("✅ MemoryRetrieval initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize MemoryRetrieval: {e}")

    logger.info(f"✅ Allowed chat IDs: {settings.allowed_chat_id_list}")

    # Open the SQLite checkpointer for the full lifespan of the app
    import os
    from pathlib import Path
    db_path = os.getenv("DB_PATH", "./data/checkpoints_v2.db")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    async with AsyncSqliteSaver.from_conn_string(db_path) as cp:
        checkpointer = cp
        # Build and compile the LangGraph with the persistent checkpointer
        graph = build_supervisor_graph(checkpointer=checkpointer)
        logger.info("✅ Supervisor-Worker graph compiled with SQLite checkpointer")

        from core.scheduler import initialize_scheduler, shutdown_scheduler
        await initialize_scheduler(checkpointer=checkpointer)
        logger.info("💓 System Scheduler initialized (manages background tasks and heartbeat)")

        # Delete any existing webhook and start polling for local dev
        await telegram_client.delete_webhook()
        logger.info("✅ Webhook cleared — using polling mode")

        # Start polling in background
        polling_task = asyncio.create_task(_poll_telegram())
        if whatsapp_client:
            whatsapp_bridge_task = asyncio.create_task(whatsapp_client.start())
            whatsapp_poll_task = asyncio.create_task(_poll_whatsapp())

        logger.info("🟢 Personal Assistant is ready! (polling mode)")

        yield

        # Shutdown
        logger.info("🔴 Shutting down...")
        background_tasks = [task for task in (polling_task, whatsapp_poll_task, whatsapp_bridge_task) if task]
        for task in background_tasks:
            task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)

        if whatsapp_client:
            await whatsapp_client.stop()

        await shutdown_scheduler()
        logger.info("✅ System Scheduler shut down")
        await telegram_client.close()
        try:
            from core.browser_session import BrowserSessionManager
            await BrowserSessionManager.shutdown()
            logger.info("✅ Browser sessions closed")
        except Exception as e:
            logger.error(f"❌ Error closing browser sessions: {e}")
        try:
            from memory.retrieval import memory_retrieval
            await memory_retrieval.store.close()
            logger.info("✅ Zvec memory stores flushed and closed")
        except Exception as e:
            logger.error(f"❌ Error closing memory stores: {e}")
            
    # SQLite connection is closed automatically when the async with block exits
    logger.info("✅ SQLite checkpointer closed")


# ==========================================================
# 2. Direct Graph Executions 
# ==========================================================

async def direct_agent_invoke(chat_id: str, text: str, platform: str, user_name: str = "User"):
    """
    Core entrypoint for processing a message from a user.
    """
    logger.info("📥")
    logger.info("📥 [NEW USER REQUEST]")
    logger.info(f"📥 From: '{user_name}' on {platform} (Thread: {chat_id})")
    logger.info(f"📥 Message: {text}")
    logger.info("📥")
    
    # 1. Handle Commands (e.g. /permit, /model, /clear)
    if await _handle_commands(chat_id, text, platform):
        return
    from memory.retrieval import memory_retrieval
    from config.settings import settings
    
    config = {
        "configurable": {
            "thread_id": str(chat_id),
            "platform": platform,
            "enable_multimodal_observation": settings.enable_multimodal_observation,
        }
    }

    try:
        # 🚨 Intercept messages while graph is interrupted (HITL Deadlock Fix)
        current_state = await graph.aget_state(config)
        if current_state.next and current_state.next[0] == "human_approval":
            logger.info(f"Buffered pending user feedback from {chat_id}: {str(text)[:50]}")
            
            # Append to the pending feedback state instead of crashing
            existing_feedback = current_state.values.get("pending_user_feedback")
            new_feedback = f"{existing_feedback}\n{text}" if existing_feedback else text
            
            await graph.aupdate_state(config, {"pending_user_feedback": new_feedback})
            
            await channel_manager.send_message(
                platform, 
                str(chat_id), 
                "📥 *Message buffered.* I will read this as soon as you **Approve** or **Reject** the pending action."
            )
            return {"status": "buffered"}

        async for graph_event in graph.astream(
            {
                "chat_id": str(chat_id),
                "user_input": text,
                "original_query": text,
            },
            config=config,
            stream_mode="updates",
        ):
            pass

        state = await graph.aget_state(config)

        if state.next:
            interrupted = state.tasks[0].interrupts[0].value
            tool_args = interrupted.get("tool_args", {})

            if platform == "system":
                logger.warning(
                    "HITL interrupt occurred in system context for action '%s'; "
                    "system runs cannot request interactive approval.",
                    interrupted.get("action", "unknown"),
                )
                return {
                    "approval_required": False,
                    "error": "System run requested interactive approval. Scheduler policy should avoid this path.",
                    "action": interrupted.get("action", "unknown"),
                }
            
            # Request explicit HITL approval for dangerous tools
            await channel_manager.request_approval(
                platform=platform,
                thread_id=str(chat_id),
                tool_name=interrupted.get('action', 'unknown'),
                args=tool_args
            )
            logger.info(f"🔐 HITL: Sent approval request to {platform} user {chat_id}")
            
            return {
                "approval_required": True,
                "action": interrupted.get("action", "unknown"),
            }
        else:
            response = extract_response(state.values)

            # Send response to user (unless running in silent background mode)
            if platform != "system":
                await channel_manager.send_message(
                    platform=platform,
                    thread_id=str(chat_id),
                    content=response
                )

            # 1. Fast History Write ONLY. Semantic extraction is now governed directly by the agent via `@tool`.
            
            # Safely handle multimodal list (images) for DB text insertion
            if isinstance(text, list):
                db_text = " ".join([i.get("text", "") for i in text if isinstance(i, dict) and i.get("type", "") == "text"]).strip()
                if not db_text:
                    db_text = "[Media Payload]"
            else:
                db_text = str(text)

            await memory_retrieval.db.add_history(str(chat_id), "user", db_text)
            await memory_retrieval.db.add_history(str(chat_id), "assistant", response)
            
            return {"response": response}

    except Exception as e:
        logger.error(f"❌ Graph execution error ({platform}): {e}", exc_info=True)
        error_text = f"❌ Sorry, I encountered an error:\n{str(e)[:500]}"
        if platform != "system":
            try:
                await channel_manager.send_message(platform, str(chat_id), error_text)
            except Exception:
                logger.critical(f"FATAL: Channel manager failed to deliver error for thread {chat_id}")
        return {"error": str(e)[:500]}


async def direct_resume_invoke(chat_id: str, decision: str, platform: str) -> dict:
    """Resumes the LangGraph state machine after a HITL approval."""
    from memory.retrieval import memory_retrieval
    from config.settings import settings
    config = {
        "configurable": {
            "thread_id": str(chat_id),
            "platform": platform,
            "enable_multimodal_observation": settings.enable_multimodal_observation,
        }
    }
    
    try:
        async for graph_event in graph.astream(
            Command(resume=decision),
            config=config,
            stream_mode="updates",
        ):
            pass

        state = await graph.aget_state(config)
        if not state.next:
            response = extract_response(state.values)

            await channel_manager.send_message(platform, str(chat_id), response)
            
            # Fast history write after resumption
            await memory_retrieval.db.add_history(str(chat_id), "assistant", response)
            
            return {"response": response}
        else:
            interrupted = state.tasks[0].interrupts[0].value
            tool_args = interrupted.get("tool_args", {})

            if platform == "system":
                logger.warning(
                    "HITL re-interrupt occurred in system context for action '%s'; cannot route approval.",
                    interrupted.get("action", "unknown"),
                )
                return {
                    "status": "paused_again",
                    "error": "System run requested interactive approval during resume.",
                    "action": interrupted.get("action", "unknown"),
                }

            # Loopback HITL
            await channel_manager.request_approval(
                platform=platform,
                thread_id=str(chat_id),
                tool_name=interrupted.get('action', 'unknown'),
                args=tool_args
            )
            return {"status": "paused_again"}

    except Exception as e:
        logger.error(f"❌ Resume error ({platform}): {e}", exc_info=True)
        await channel_manager.send_message(platform, str(chat_id), f"❌ Error resuming action:\n`{str(e)[:500]}`")
        return {"error": str(e)[:500]}


# ==========================================================
# 3. Telegram Polling (for local dev)
# ==========================================================

async def _poll_telegram():
    """
    Background task: long-poll Telegram for updates.
    This replaces webhooks for local development.
    """
    offset = 0
    logger.info("📡 Polling Telegram for updates...")

    while True:
        try:
            updates = await telegram_client.get_updates(offset=offset, timeout=30)
            for update in updates:
                offset = update["update_id"] + 1

                # Handle callback queries (button clicks)
                if "callback_query" in update:
                    await _handle_callback_query(update["callback_query"])
                    continue

                # Handle incoming messages (text or media)
                message = update.get("message", {})
                chat_id = message.get("chat", {}).get("id")
                
                # Extract text and handle incoming images for multimodal vision models
                text = message.get("text", "") or message.get("caption", "")
                photo = message.get("photo")
                
                if photo:
                    # Telegram sends multiple thumbnails. We want the highest resolution (last item).
                    try:
                        highest_res = sorted(photo, key=lambda p: p["file_size"])[-1]
                        file_info = await telegram_client.get_file(highest_res["file_id"])
                        if file_info.get("ok"):
                            import base64
                            file_path = file_info["result"]["file_path"]
                            img_bytes = await telegram_client.download_file(file_path)
                            b64_string = base64.b64encode(img_bytes).decode("utf-8")
                            
                            # Build LangChain-compatible multimodal list map
                            content = []
                            if text:
                                content.append({"type": "text", "text": text})
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_string}"}
                            })
                            text = content  # Reassign text argument to our new multimodal list
                    except Exception as e:
                        logger.error(f"Failed to download image from Telegram: {e}")

                if not chat_id or not text:
                    continue

                # Check chat_id allowlist
                allowed = settings.allowed_chat_id_list
                if allowed and chat_id not in allowed:
                    logger.warning(f"⚠️ Ignored message from unauthorized chat_id: {chat_id}")
                    continue

                if isinstance(text, list):
                    text_preview = "[multimodal payload]"
                else:
                    text_preview = str(text).replace("\n", " ")[:120]
                logger.info("📨 Telegram inbound message chat_id=%s text=%s", chat_id, text_preview)

                # ── Handle Commands (e.g. /model) ──────────────────────────
                if await _handle_commands(str(chat_id), text, "telegram"):
                    logger.info("📨 Telegram command handled for chat_id=%s", chat_id)
                    continue

                # Show typing indicator
                await telegram_client.send_typing_action(chat_id)

                # Run the graph directly in the background
                asyncio.create_task(direct_agent_invoke(str(chat_id), text, "telegram"))

        except asyncio.CancelledError:
            logger.info("📡 Polling stopped")
            break
        except Exception as e:
            logger.error(f"Polling error: {e}")
            await asyncio.sleep(2)


async def _poll_whatsapp():
    """
    Background task: consume normalized events from the WhatsApp bridge client.
    """
    if not whatsapp_client:
        return

    logger.info("📡 Listening for WhatsApp bridge events...")

    while True:
        try:
            event = await whatsapp_client.next_event()

            chat_id = str(event.get("chat_id", "") or "").strip()
            text = str(event.get("content", "") or "").strip()
            sender_id = str(event.get("sender_id", "User") or "User")

            if not chat_id or not text:
                continue

            text_preview = text.replace("\n", " ")[:120]
            logger.info("📨 WhatsApp inbound message chat_id=%s text=%s", chat_id, text_preview)

            decision = _parse_whatsapp_hitl_decision(text)
            if decision:
                config = {"configurable": {"thread_id": chat_id, "platform": "whatsapp"}}
                state = await graph.aget_state(config)

                if state.next and state.next[0] == "human_approval":
                    asyncio.create_task(direct_resume_invoke(chat_id, decision, "whatsapp"))
                else:
                    await channel_manager.send_message(
                        "whatsapp",
                        chat_id,
                        "No action is currently waiting for approval in this chat.",
                    )
                continue

            if await _handle_commands(chat_id, text, "whatsapp"):
                continue

            asyncio.create_task(direct_agent_invoke(chat_id, text, "whatsapp", user_name=sender_id))

        except asyncio.CancelledError:
            logger.info("📡 WhatsApp polling stopped")
            break
        except Exception as e:
            logger.error("WhatsApp polling error: %s", e)
            await asyncio.sleep(2)


# ==========================================================
# 3. FastAPI App
# ==========================================================

app = FastAPI(
    title="Personal AI Assistant",
    description="Agentic personal assistant via Telegram, WhatsApp & Web",
    version="0.1.0",
    lifespan=lifespan,
)

from fastapi.staticfiles import StaticFiles
from interfaces.web_chat import router as chat_router

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(chat_router)


# ==========================================================
# 3. Security Dependencies
# ==========================================================

async def verify_telegram_secret(request: Request):
    """Validate the X-Telegram-Bot-Api-Secret-Token header."""
    if not settings.telegram_secret_token:
        return  # No secret configured, skip validation

    secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if secret != settings.telegram_secret_token:
        logger.warning(f"⚠️ Rejected request: invalid secret token")
        raise HTTPException(status_code=401, detail="Invalid secret token")


def verify_chat_id(chat_id: int):
    """Ensure the chat_id is in the allowlist."""
    allowed = settings.allowed_chat_id_list
    if allowed and chat_id not in allowed:
        logger.warning(f"⚠️ Rejected request from unauthorized chat_id: {chat_id}")
        raise HTTPException(status_code=403, detail="Unauthorized chat")


# ==========================================================
# 4. Endpoints
# ==========================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "graph_ready": graph is not None,
        "telegram_ready": telegram_client is not None,
        "whatsapp_enabled": settings.whatsapp_enabled,
        "whatsapp_ready": whatsapp_client is not None,
        "whatsapp_connected": bool(whatsapp_client and whatsapp_client.is_connected),
    }


@app.post("/webhook", dependencies=[Depends(verify_telegram_secret)])
async def webhook(request: Request):
    """
    Receive Telegram webhook updates.
    
    Handles two types:
    1. message — User sent a text message → run the graph
    2. callback_query — User clicked an inline button → resume the graph
    """
    body = await request.json()

    # ── Handle callback queries (button clicks) ──────────────
    if "callback_query" in body:
        return await _handle_callback_query(body["callback_query"])

    # ── Handle text and media messages ─────────────────────────────────
    if "message" in body:
        message = body["message"]
        chat_id = message.get("chat", {}).get("id")
        text = message.get("text", "") or message.get("caption", "")
        photo = message.get("photo")
        
        if photo:
            try:
                highest_res = sorted(photo, key=lambda p: p["file_size"])[-1]
                file_info = await telegram_client.get_file(highest_res["file_id"])
                if file_info.get("ok"):
                    import base64
                    file_path = file_info["result"]["file_path"]
                    img_bytes = await telegram_client.download_file(file_path)
                    b64_string = base64.b64encode(img_bytes).decode("utf-8")
                    
                    content = []
                    if text:
                        content.append({"type": "text", "text": text})
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_string}"}
                    })
                    text = content
            except Exception as e:
                logger.error(f"Webhook failed to download image: {e}")

        if not chat_id or not text:
            return {"status": "ignored", "reason": "no chat_id or text"}

        verify_chat_id(chat_id)

        # ── Handle Commands (e.g. /model) ────────────────────────────
        if await _handle_commands(str(chat_id), text, "telegram"):
            return {"status": "command_handled"}

        # Show typing indicator
        await telegram_client.send_typing_action(chat_id)

        # Run the graph directly (async without blocking the webhook response)
        asyncio.create_task(direct_agent_invoke(str(chat_id), text, "telegram"))

        return {"status": "processing"}

    return {"status": "ignored", "reason": "unhandled update type"}


# ==========================================================
# ==========================================================
# 5. Background Handlers
# ==========================================================

async def _handle_callback_query(callback_query: dict):
    """
    Handle inline keyboard button clicks (Approve/Reject/Edit).
    Resumes the paused LangGraph.
    """
    callback_id = callback_query.get("id")
    data = callback_query.get("data", "")
    chat_id = callback_query.get("message", {}).get("chat", {}).get("id")
    message_id = callback_query.get("message", {}).get("message_id")

    if not data or not chat_id:
        return {"status": "ignored"}

    # Parse callback data: "approve:thread_id", "reject:thread_id", "edit:thread_id"
    parts = data.split(":", 1)
    decision = parts[0]
    thread_id = parts[1] if len(parts) > 1 else str(chat_id)

    # Acknowledge the button press immediately
    await telegram_client.answer_callback_query(
        callback_id, text=f"{'✅ Approved' if decision == 'approve' else '❌ Rejected' if decision == 'reject' else '✏️ Editing'}..."
    )

    # Update the original message to show the decision
    decision_text = {
        "approve": "✅ *Approved* — executing...",
        "reject": "❌ *Rejected*",
        "edit": "✏️ *Editing* — please send your modifications...",
    }.get(decision, "Unknown action")

    await telegram_client.edit_message_text(
        chat_id=chat_id,
        message_id=message_id,
        text=decision_text,
    )

    if decision == "edit":
        # For edit, we wait for the user's next message (it'll come through /webhook)
        # For now, just inform the user
        await telegram_client.send_message(
            chat_id=chat_id,
            text="Please send me the changes you'd like to make.",
        )
        return {"status": "awaiting_edit"}

    # Resume the graph directly
    asyncio.create_task(direct_resume_invoke(thread_id, decision, "telegram"))

    return {"status": "resumed"}





@app.post("/api/v1/chat/{thread_id}")
async def chat_endpoint(thread_id: str, request: Request):
    """
    Standardized Gateway API for chat clients.
    Validates payload, pushes to the Event Bus (LaneManager), and returns instantly.
    """
    if not graph:
        return JSONResponse({"error": "Graph not initialized"}, status_code=503)

    body = await request.json()
    user_input = body.get("user_input", "").strip()
    if not user_input:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    # ── Handle Commands (e.g. /model) ────────────────────────────
    if await _handle_commands(thread_id, user_input, "web"):
        return {"status": "command_handled"}

    # Execute instantly and directly in background
    asyncio.create_task(direct_agent_invoke(thread_id, user_input, "web"))
    
    return {"status": "queued"}


@app.post("/api/v1/chat/{thread_id}/resume")
async def resume_hitl_endpoint(thread_id: str, request: Request):
    """
    Standardized Gateway API for resuming paused execution.
    Pushes ResumeEvent to Event Bus and returns instantly.
    """
    if not graph:
        return JSONResponse({"error": "Graph not initialized"}, status_code=503)

    body = await request.json()
    decision = body.get("action", body.get("decision", "reject"))

    # Resume directly
    asyncio.create_task(direct_resume_invoke(thread_id, decision, "web"))
    
    return {"status": "queued"}


@app.post("/api/v1/system/{thread_id}/notify")
async def system_notify_endpoint(thread_id: str, request: Request):
    """
    Standardized Gateway API for external tools to inject 
    asynchronous notifications into the graph state and deliver them to the user.
    """
    if not graph:
        return JSONResponse({"error": "Graph not initialized"}, status_code=503)

    body = await request.json()
    message = body.get("message", "")
    platform = body.get("platform", "web")

    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    # Inject into LangGraph state directly so the agent remembers it
    from langchain_core.messages import AIMessage
    await graph.aupdate_state(
        {"configurable": {"thread_id": thread_id}},
        {"messages": [AIMessage(content=message)]}
    )

    # Deliver to the user's UI
    await channel_manager.send_message(platform, thread_id, message)
    
    return {"status": "delivered"}


# ==========================================================
# 8. Entrypoint
# ==========================================================

async def run_cli():
    """Terminal UI loop for the agent."""
    global graph, checkpointer
    
    if settings.needs_onboarding:
        print("⚠️ EnterpriseClaw is not configured yet! Run the setup wizard: uv run python onboarding.py --cli")
        return

    from memory.retrieval import memory_retrieval
    from interfaces.cli import CLIClient
    
    await memory_retrieval.initialize()
    channel_manager.register_client("cli", CLIClient())
    
    async with AsyncSqliteSaver.from_conn_string("./data/checkpoints_v2.db") as cp:
        checkpointer = cp
        graph = build_supervisor_graph()
        graph.checkpointer = checkpointer
        chat_id = "cli_user"
        print("\n🦅 EnterpriseClaw CLI is ready! Type 'exit' to quit.\n")

        while True:
            try:
                user_input = await asyncio.to_thread(input, "You: ")
                if user_input.strip().lower() in ("exit", "quit"):
                    break
                if not user_input.strip():
                    continue

                if await _handle_commands(chat_id, user_input, "cli"):
                    continue

                result = await direct_agent_invoke(chat_id, user_input, "cli")

                # Handle continuous loopback for chained HITL actions
                while result.get("approval_required") or result.get("status") == "paused_again":
                    decision_input = await asyncio.to_thread(input, "\n[y/n/e] > ")
                    decision_map = {"y": "approve", "n": "reject", "e": "edit"}
                    decision = decision_map.get(decision_input.strip().lower(), "reject")
                    result = await direct_resume_invoke(chat_id, decision, "cli")

            except EOFError:
                break
            except Exception as e:
                logger.error(f"❌ CLI Error: {e}")

if __name__ == "__main__":
    import sys
    if "--cli" in sys.argv:
        asyncio.run(run_cli())
    else:
        import uvicorn
        reload_enabled = "--reload" in sys.argv
        logger.info("Starting API server (reload=%s)", reload_enabled)
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=reload_enabled)
