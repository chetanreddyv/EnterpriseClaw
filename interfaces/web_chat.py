"""
web_chat.py — Localhost web chat interface for EnterpriseClaw.

Provides a beautiful single-page chat UI served at /chat that
communicates with the same LangGraph pipeline used by Telegram.

Serves HTML and a lightweight polling endpoint for web events.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from interfaces.base import ClientInterface

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/chat", response_class=HTMLResponse)
async def serve_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

class WebClient(ClientInterface):
    """
    Adapter for the local Web GUI.
    Uses a lightweight in-memory event queue polled by the frontend.
    """

    def __init__(self) -> None:
        self._events: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def _enqueue(self, thread_id: str, event: dict[str, Any]) -> None:
        async with self._lock:
            self._events[thread_id].append(event)

    async def drain_events(self, thread_id: str) -> list[dict[str, Any]]:
        async with self._lock:
            events = self._events.get(thread_id, [])
            self._events[thread_id] = []
            return events

    async def send_message(self, thread_id: str, content: str) -> None:
        await self._enqueue(
            thread_id,
            {
                "id": f"msg-{int(time.time() * 1000)}",
                "type": "message",
                "content": content,
            },
        )

    async def request_approval(self, thread_id: str, tool_name: str, args: Dict[str, Any]) -> None:
        safe_args = json.loads(json.dumps(args or {}, default=str))
        await self._enqueue(
            thread_id,
            {
                "id": f"approval-{int(time.time() * 1000)}",
                "type": "approval",
                "tool_name": tool_name,
                "args": safe_args,
            },
        )

web_client = WebClient()


@router.get("/api/v1/chat/{thread_id}/events", response_class=JSONResponse)
async def web_chat_events(thread_id: str):
    events = await web_client.drain_events(thread_id)
    return {"events": events}


