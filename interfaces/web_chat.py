"""
web_chat.py — Localhost web chat interface for EnterpriseClaw.

Provides a beautiful single-page chat UI served at /chat that
communicates with the same LangGraph pipeline used by Telegram.

API endpoints are defined in app.py — this module only serves the HTML.
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/chat", response_class=HTMLResponse)
async def serve_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

from interfaces.base import ClientInterface
from typing import Dict, Any

class WebClient(ClientInterface):
    """
    Adapter for the local Web GUI.
    Currently just logs output so the frontend can poll it, 
    but designed to be extended with WebSockets or SSEs.
    """
    async def send_message(self, thread_id: str, content: str) -> None:
        # TODO: Upgrade to use FastAPI's StreamingResponse (Server-Sent Events) or WebSockets
        # to instantly push the message (e.g. 🔔 **[Subagent Report]**) to the browser.
        # In a real app, this would push via WebSocket to `thread_id`
        logger.info(f"🖥️ [WEB OUT] {thread_id}: {content[:100]}")

    async def request_approval(self, thread_id: str, tool_name: str, args: Dict[str, Any]) -> None:
        logger.info(f"🖥️ [WEB HITL] {thread_id} needs approval for {tool_name}")

web_client = WebClient()


