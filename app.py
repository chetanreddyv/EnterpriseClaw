"""
app.py — FastAPI application.

Handles Telegram webhooks, security validation, and bridges
the messaging layer with the LangGraph agentic loop.
"""

import asyncio
import logging
from dotenv import load_dotenv

# Load .env into os.environ BEFORE any other imports that read env vars
load_dotenv()
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from langgraph.types import Command

from config.settings import settings
from interfaces.telegram import TelegramClient
from nodes.graph import build_graph, checkpointer_context
from core.channel_manager import channel_manager
from core.llm import extract_response

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress verbose Pydantic-to-Gemini schema warnings
logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)

# ── Globals (initialized at startup) ────────────────────────
telegram_client: TelegramClient = None
graph = None
checkpointer = None


# ==========================================================
# 0. Shared Logic
# ==========================================================

async def _handle_commands(chat_id: str, text: str, platform: str) -> bool:
    """Handle slash commands (e.g. /model, /permit). Returns True if handled."""
    if not text.startswith("/"): return False
    parts = text.split()
    cmd = parts[0].lower()

    if cmd == "/model" and len(parts) > 1:
        await graph.aupdate_state({"configurable": {"thread_id": chat_id}}, {"active_model": parts[1]})
        await channel_manager.send_message(platform, chat_id, f"🔄 Brain swapped! Now using: `{parts[1]}`")
        return True

    if cmd in ("/permit", "/deny") and len(parts) > 1:
        state = await graph.aget_state({"configurable": {"thread_id": chat_id}})
        approved = set(state.values.get("approved_tools") or [])
        if cmd == "/permit":
            approved.add(parts[1])
            msg = f"🔓 `{parts[1]}` is now AUTO-APPROVED"
        else:
            approved.discard(parts[1])
            msg = f"🔒 `{parts[1]}` now REQUIRES approval"
        await graph.aupdate_state({"configurable": {"thread_id": chat_id}}, {"approved_tools": list(approved)})
        await channel_manager.send_message(platform, chat_id, msg)
        return True

    if cmd == "/tools":
        from nodes.agent import _get_enabled_tools_and_write_actions
        enabled, write_actions = _get_enabled_tools_and_write_actions(load_all=True)
        approved = set((await graph.aget_state({"configurable": {"thread_id": chat_id}})).values.get("approved_tools") or [])
        lines = [f"🔓 `{t}`" if t in approved else f" `{t}`" if t in write_actions else f"🟢 `{t}`" for t in sorted(enabled)]
        await channel_manager.send_message(platform, chat_id, "🛠️ **Tool Permissions (This Thread)**\n" + "\n".join(lines))
        return True

    if cmd == "/models":
        import httpx
        from config.settings import settings
        base_url = settings.lm_studio_base_url
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/models", timeout=3.0)
                response.raise_for_status()
                data = response.json()
                models = [model.get("id") for model in data.get("data", [])]
                if models:
                    msg = "🧠 **Available Local Models (LM Studio)**:\n" + "\n".join([f"- `lmstudio/{m}`" for m in models])
                else:
                    msg = "🧠 No local models currently loaded in LM Studio."
        except Exception as e:
            msg = f"❌ Could not connect to LM Studio at `{base_url}`.\nEnsure it is running and the local server is started.\nError: `{e}`"
        await channel_manager.send_message(platform, chat_id, msg)
        return True

    return False


# ==========================================================
# 1. Lifespan (startup / shutdown)
# ==========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global telegram_client, graph, checkpointer

    logger.info("🚀 Starting Personal Assistant...")

    # ── Onboarding check ─────────────────────────────────────
    if settings.needs_onboarding:
        logger.warning("  ⚠️  EnterpriseClaw is not configured yet! Run the setup wizard:")
        logger.warning("    uv run python onboarding.py")
        logger.warning("  Or use CLI mode: uv run python onboarding.py --cli")
        # Yield to keep FastAPI alive but don't initialize anything
        yield
        return

    # Initialize Telegram client
    telegram_client = TelegramClient(settings.telegram_bot_token)
    channel_manager.register_client("telegram", telegram_client)
    logger.info("✅ Telegram client initialized and registered")
    
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
    async with checkpointer_context() as checkpointer:
        # Build and compile the LangGraph with the persistent checkpointer
        graph = build_graph(checkpointer=checkpointer)
        logger.info("✅ LangGraph compiled with SQLite checkpointer")

        # Start Heartbeat Daemon
        heartbeat_task = asyncio.create_task(system_heartbeat())
        logger.info("💓 System Heartbeat loop started")

        # Delete any existing webhook and start polling for local dev
        await telegram_client.delete_webhook()
        logger.info("✅ Webhook cleared — using polling mode")

        # Start polling in background
        polling_task = asyncio.create_task(_poll_telegram())
        logger.info("🟢 Personal Assistant is ready! (polling mode)")

        yield

        # Shutdown
        logger.info("🔴 Shutting down...")
        heartbeat_task.cancel()
        polling_task.cancel()
        await telegram_client.close()
        try:
            from mcp_servers.browser_tools import BrowserSessionManager
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



async def system_heartbeat():
    """
    Background pulse: Wakes up the agent every 15 minutes to perform any time-based tasks.
    It checks for reminders or scheduled tasks via its internal tools.
    """
    pulse_delay = 900  # 15 minutes
    logger.info(f"💓 Heartbeat daemon initialized. Pulse every {pulse_delay}s.")
    
    # Wait initially to allow startup to finish
    await asyncio.sleep(10)
    
    while True:
        try:
            await asyncio.sleep(pulse_delay)
            logger.info("💓 Sending System Heartbeat to Agent...")
            
            # The system thread ID
            thread_id = "system_cron_thread"
            
            # Inject a silent trigger into the graph using direct_agent_invoke
            await direct_agent_invoke(
                chat_id=thread_id,
                text="[SYSTEM EVENT: HEARTBEAT. Check your schedule, read unread emails, and perform any necessary background tasks. If nothing needs doing, reply 'IDLE'.]",
                platform="system"
            )

        except asyncio.CancelledError:
            logger.info("💓 Heartbeat daemon stopped")
            break
        except Exception as e:
            logger.error(f"💓 Heartbeat error: {e}", exc_info=True)


# ==========================================================
# 2. Direct Graph Executions (Replacing worker.py / lane_manager queues)
# ==========================================================

async def direct_agent_invoke(chat_id: str, text: str, platform: str) -> dict:
    """Invokes the LangGraph state machine directly from a webhook route."""
    from memory.retrieval import memory_retrieval
    
    config = {"configurable": {"thread_id": str(chat_id), "platform": platform}}

    try:
        # 🚨 Intercept messages while graph is interrupted (HITL Edit Fix)
        current_state = await graph.aget_state(config)
        if current_state.next and current_state.next[0] == "human_approval":
            logger.info(f"Intercepted HITL edit instruction from {chat_id}: {text[:50]}")
            from langchain_core.messages import HumanMessage
            # Inject the user's instructions into the state
            await graph.aupdate_state(config, {"messages": [HumanMessage(content=f"User requested edits to the pending action: {text}")]})
            # Explicitly resume the graph with the 'edit' action 
            return await direct_resume_invoke(chat_id, "edit", platform)

        async for graph_event in graph.astream(
            {
                "chat_id": str(chat_id),
                "user_input": text,
                "tool_failure_count": 0,
                "active_skills": None,
                "skill_prompts": None,
            },
            config=config,
            stream_mode="updates",
        ):
            logger.debug(f"Graph event ({platform}): {graph_event}")

        state = await graph.aget_state(config)

        if state.next:
            interrupted = state.tasks[0].interrupts[0].value
            tool_args = interrupted.get("tool_args", {})
            
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

            # Prevent chatbot spam for background/internal silent responses
            if platform != "system" and response.strip() != "HEARTBEAT_OK" and response != "Done!" and response != "IDLE":
                await channel_manager.send_message(
                    platform=platform,
                    thread_id=str(chat_id),
                    content=response
                )
                logger.info(f"💬 Sent response to {platform} user {chat_id}")
            elif response.strip() == "HEARTBEAT_OK" or response == "IDLE":
                logger.info(f"🔇 Suppressed internal heartbeat string from {platform} thread {chat_id}")

            # 1. Fast History Write ONLY. Semantic extraction is now governed directly by the agent via `@tool`.
            await memory_retrieval.db.add_history(str(chat_id), "user", text)
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
    config = {"configurable": {"thread_id": str(chat_id), "platform": platform}}
    
    try:
        async for graph_event in graph.astream(
            Command(resume=decision),
            config=config,
            stream_mode="updates",
        ):
            logger.debug(f"Resume event ({platform}): {graph_event}")

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

                # Handle text messages
                message = update.get("message", {})
                chat_id = message.get("chat", {}).get("id")
                text = message.get("text", "")

                if not chat_id or not text:
                    continue

                # Check chat_id allowlist
                allowed = settings.allowed_chat_id_list
                if allowed and chat_id not in allowed:
                    logger.warning(f"⚠️ Ignored message from unauthorized chat_id: {chat_id}")
                    continue

                logger.info(f"📩 Message from {chat_id}: {text[:100]}")

                # ── Handle Commands (e.g. /model) ──────────────────────────
                if await _handle_commands(str(chat_id), text, "telegram"):
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


# ==========================================================
# 3. FastAPI App
# ==========================================================

app = FastAPI(
    title="Personal AI Assistant",
    description="Agentic personal assistant via Telegram & Web",
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
    logger.info(f"📩 Webhook received")

    # ── Handle callback queries (button clicks) ──────────────
    if "callback_query" in body:
        return await _handle_callback_query(body["callback_query"])

    # ── Handle text messages ─────────────────────────────────
    if "message" in body:
        message = body["message"]
        chat_id = message.get("chat", {}).get("id")
        text = message.get("text", "")

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

    logger.info(f"🔘 Callback: {decision} from {chat_id} (thread: {thread_id})")

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

    logger.info(f"🌐 Gateway API request queued from {thread_id}: {user_input[:100]}")

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

    logger.info(f"🌐 Gateway API resume queued: {decision} for thread {thread_id}")

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

    logger.info(f"🌐 Gateway API notification received for {thread_id} via {platform}")

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
    
    async with checkpointer_context() as cp:
        checkpointer = cp
        graph = build_graph(checkpointer=checkpointer)
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
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
