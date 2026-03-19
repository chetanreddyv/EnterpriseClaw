"""
whatsapp.py — WhatsApp bridge client.

Connects to a Node.js WebSocket bridge (for example, Baileys-based)
and exposes EnterpriseClaw's ClientInterface contract.
"""

import asyncio
import json
import logging
import mimetypes
from collections import OrderedDict
from typing import Any, Dict

from interfaces.base import ClientInterface

logger = logging.getLogger(__name__)


class WhatsAppClient(ClientInterface):
    """
    WhatsApp adapter backed by a Node.js WebSocket bridge.

    Inbound messages are normalized into events consumed by app.py.
    Outbound messages are sent to the bridge with payload:
    {"type": "send", "to": "<jid>", "text": "..."}
    """

    def __init__(
        self,
        bridge_url: str,
        bridge_token: str = "",
        allow_from: list[str] | None = None,
    ):
        self.bridge_url = bridge_url
        self.bridge_token = bridge_token
        self.allow_from = {
            str(item).strip()
            for item in (allow_from or [])
            if str(item).strip()
        }

        self._ws = None
        self._running = False
        self._connected = False
        self._events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def start(self) -> None:
        """Connect to the WebSocket bridge and stream inbound events."""
        import websockets

        logger.info("Connecting to WhatsApp bridge at %s", self.bridge_url)
        self._running = True

        while self._running:
            try:
                async with websockets.connect(
                    self.bridge_url,
                    ping_interval=20,
                    ping_timeout=20,
                    max_size=2 * 1024 * 1024,
                ) as ws:
                    self._ws = ws
                    self._connected = True

                    if self.bridge_token:
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "auth",
                                    "token": self.bridge_token,
                                },
                                ensure_ascii=False,
                            )
                        )

                    logger.info("Connected to WhatsApp bridge")

                    async for raw_message in ws:
                        try:
                            await self._handle_bridge_message(raw_message)
                        except Exception as e:
                            logger.error("Error handling WhatsApp bridge message: %s", e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("WhatsApp bridge connection error: %s", e)
                if self._running:
                    logger.info("Reconnecting to WhatsApp bridge in 5 seconds...")
                    await asyncio.sleep(5)
            finally:
                self._connected = False
                self._ws = None

    async def stop(self) -> None:
        """Stop the bridge loop and close the active websocket."""
        self._running = False
        self._connected = False

        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug("WhatsApp bridge close error: %s", e)
            finally:
                self._ws = None

    async def next_event(self) -> dict[str, Any]:
        """Wait for the next normalized inbound WhatsApp event."""
        return await self._events.get()

    async def send_message(self, thread_id: str, content: str) -> None:
        """Send a text message to a WhatsApp JID via the bridge."""
        await self._send_payload(
            {
                "type": "send",
                "to": thread_id,
                "text": str(content),
            }
        )

    async def request_approval(self, thread_id: str, tool_name: str, args: Dict[str, Any]) -> None:
        """Send HITL approval prompt over WhatsApp (text-only)."""
        if args:
            args_lines = [f"- {k}: {v}" for k, v in args.items()]
            args_text = "\n".join(args_lines)
        else:
            args_text = "- (No arguments)"

        prompt = (
            "Action requires approval\n\n"
            f"Tool: {tool_name}\n"
            f"Arguments:\n{args_text}\n\n"
            "Reply with one of these commands:\n"
            "/approve\n"
            "/reject\n"
            "/edit"
        )

        await self.send_message(thread_id, prompt)

    async def _send_payload(self, payload: dict[str, Any]) -> None:
        if not self._ws or not self._connected:
            logger.warning("WhatsApp bridge is not connected; dropping outbound payload")
            return

        try:
            await self._ws.send(json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            logger.error("Error sending WhatsApp bridge payload: %s", e)

    def _sender_allowed(self, sender: str, user_id: str, sender_id: str, chat_id: str) -> bool:
        if not self.allow_from:
            return True

        return any(
            candidate in self.allow_from
            for candidate in {sender, user_id, sender_id, chat_id}
            if candidate
        )

    def _remember_message_id(self, message_id: str) -> bool:
        """Returns True for a new message id, False for duplicates."""
        if not message_id:
            return True

        if message_id in self._processed_message_ids:
            return False

        self._processed_message_ids[message_id] = None
        while len(self._processed_message_ids) > 1000:
            self._processed_message_ids.popitem(last=False)
        return True

    @staticmethod
    def _append_media_tags(content: str, media_paths: list[str]) -> str:
        tagged_content = content
        for media_path in media_paths:
            if not isinstance(media_path, str):
                continue
            mime, _ = mimetypes.guess_type(media_path)
            media_type = "image" if mime and mime.startswith("image/") else "file"
            media_tag = f"[{media_type}: {media_path}]"
            tagged_content = f"{tagged_content}\n{media_tag}" if tagged_content else media_tag
        return tagged_content

    async def _handle_bridge_message(self, raw: str) -> None:
        """Parse bridge messages and enqueue user-originated chat events."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from WhatsApp bridge: %s", str(raw)[:200])
            return

        msg_type = data.get("type")

        if msg_type == "message":
            if data.get("fromMe"):
                return

            sender = str(data.get("sender", "") or "")
            pn = str(data.get("pn", "") or "")
            chat_id = str(data.get("chat", "") or sender or pn)
            user_id = pn or sender
            sender_id = user_id.split("@")[0] if "@" in user_id else user_id
            content = str(data.get("content", "") or "").strip()
            message_id = str(data.get("id", "") or "")

            if not chat_id:
                logger.warning("WhatsApp bridge message missing sender/chat id")
                return

            if not self._remember_message_id(message_id):
                return

            if not self._sender_allowed(sender=sender, user_id=user_id, sender_id=sender_id, chat_id=chat_id):
                logger.warning("Ignored WhatsApp message from unauthorized sender: %s", sender or pn or chat_id)
                return

            if content == "[Voice Message]":
                content = "[Voice Message: Transcription not available for WhatsApp yet]"

            raw_media = data.get("media") or []
            media_paths = [m for m in raw_media if isinstance(m, str)] if isinstance(raw_media, list) else []
            content = self._append_media_tags(content, media_paths)

            if not content:
                content = "[Unsupported or empty WhatsApp message]"

            await self._events.put(
                {
                    "chat_id": chat_id,
                    "sender_id": sender_id,
                    "content": content,
                    "media": media_paths,
                    "metadata": {
                        "message_id": message_id,
                        "timestamp": data.get("timestamp"),
                        "is_group": bool(data.get("isGroup", False)),
                    },
                }
            )
            return

        if msg_type == "status":
            status = str(data.get("status", "") or "")
            logger.info("WhatsApp status: %s", status)
            self._connected = status == "connected"
            return

        if msg_type == "qr":
            logger.info("WhatsApp bridge requested QR auth. Scan it in the bridge terminal.")
            return

        if msg_type == "error":
            logger.error("WhatsApp bridge error: %s", data.get("error"))
            return

        logger.debug("Unhandled WhatsApp bridge message type: %s", msg_type)
