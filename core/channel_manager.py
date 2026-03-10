import logging
from typing import Dict
from interfaces.base import ClientInterface

logger = logging.getLogger("core.channel_manager")

class ChannelManager:
    """
    Central registry for output channels (e.g., Telegram, Web, CLI).
    The LangGraph worker pushes abstract outgoing messages to this manager,
    which routes them to the correct platform adapter.
    """
    def __init__(self):
        self.clients: Dict[str, ClientInterface] = {}

    def register_client(self, platform_name: str, client: ClientInterface):
        """Register a client adapter for a specific platform."""
        self.clients[platform_name] = client

    async def send_message(self, platform: str, thread_id: str, content: str):
        """Route a standard text message to the appropriate platform."""
        client = self.clients.get(platform)
        if client:
            await client.send_message(thread_id, content)
        else:
            logger.error(f"❌ Unknown platform '{platform}', cannot send message to thread {thread_id}")

    async def request_approval(self, platform: str, thread_id: str, tool_name: str, args: dict):
        """Route an approval request (HITL buttons) to the appropriate platform."""
        client = self.clients.get(platform)
        if client:
            await client.request_approval(thread_id, tool_name, args)
        else:
            logger.error(f"❌ Unknown platform '{platform}', cannot request approval for thread {thread_id}")

# Singleton instance
channel_manager = ChannelManager()
