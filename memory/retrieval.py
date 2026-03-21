"""
memory/gate.py — MemoryGate: High-level orchestrator for memory extraction and retrieval.

This is the main interface used by app.py and worker.py.
"""

import asyncio
import logging

from .db import DatabaseClient
from .vectorstore import ZvecMemoryStore

logger = logging.getLogger(__name__)


class MemoryRetrieval:
    """Read-only semantic retrieval and vector search."""

    def __init__(self):
        self.db = DatabaseClient()
        self.store = ZvecMemoryStore(db_client=self.db)
        self._initialized = False
        self._initialize_lock: asyncio.Lock | None = None

    async def initialize(self):
        """Must be called at application startup."""
        if self._initialized:
            logger.debug("MemoryRetrieval: initialize() skipped; already initialized.")
            return

        if self._initialize_lock is None:
            self._initialize_lock = asyncio.Lock()

        async with self._initialize_lock:
            if self._initialized:
                logger.debug("MemoryRetrieval: initialize() skipped inside lock; already initialized.")
                return

            self.db.initialize()
            # CRITICAL FIX: store.initialize() is now properly awaited
            # to prevent blocking the event loop on startup.
            await self.store.initialize()
            # If corruption was detected, rebuild the vector index from durable SQLite
            if self.store._needs_rebuild:
                await self.store.rebuild_from_sqlite()
            await self.store.initialize_skills()
            self._initialized = True

    # (process() method removed to replace background extraction with active tools)

    async def get_context(self, thread_id: str) -> str:
        """Retrieve relevant context for the agent node before generation."""
        # 1. Fetch recent fast history
        recent_rows = await self.db.get_recent_history(thread_id, limit=6)

        parts = []
        if recent_rows:
            history_str = "\n".join(f"- **{row['role'].title()}**: {row['content']}" for row in recent_rows)
            # 2. Hybrid semantic+BM25 context based on user's latest message
            latest_user = next((row['content'] for row in reversed(recent_rows) if row['role'] == 'user'), "")

            semantic_context = ""
            if latest_user:
                semantic_context = await self.store.get_relevant_context(latest_user, top_k=3)

            if semantic_context:
                parts.append("## Semantic Memory Context")
                parts.append(semantic_context)

            parts.append("## Recent Conversation History")
            parts.append(history_str)

        return "\n".join(parts) if parts else "No established context."

    async def get_relevant_skills(self, user_input: str) -> tuple[str, list[str]]:
        """Fetch dynamic skill prompts to inject into system prompt."""
        return await self.store.get_relevant_skills(user_input, top_k=2)

    async def get_relevant_skills_with_metadata(
        self,
        user_input: str,
    ) -> tuple[str, list[str], dict[str, list[str]]]:
        """Fetch matched skill prompts plus frontmatter tool declarations."""
        return await self.store.get_relevant_skills_with_metadata(user_input, top_k=3)

    async def get_all_memories(self):
        """Return all active memory items for debugging/admin."""
        return await self.db.get_all_active_items()

    async def get_stats(self):
        """Return memory statistics."""
        return await self.db.get_stats()


# Singleton
memory_retrieval = MemoryRetrieval()
