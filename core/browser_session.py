"""
core/browser_session.py — browser-use BrowserSession wrapper.

Manages a singleton BrowserSession with per-thread page tracking.
Replaces the custom BrowserSessionManager from browser_tools.py.
"""

import asyncio
import logging
import os
import time
from typing import Dict

from browser_use.browser.session import BrowserSession
from browser_use.dom.views import EnhancedDOMTreeNode

logger = logging.getLogger("core.browser_session")

USER_DATA_DIR = "./data/browser_profile"


class BrowserSessionManager:
    """
    Manages a singleton browser-use BrowserSession with per-thread page tracking.

    - Lazy-initializes BrowserSession on first use.
    - Per-thread page isolation (tab per thread).
    - Idle-timeout GC for stale threads.
    - Exposes browser-use state engine (get_state, selector_map) for the observer.
    """

    _session: BrowserSession | None = None
    _threads: Dict[str, dict] = {}  # thread_id -> {"page_idx": int, "last_accessed": float, ...}
    _gc_task: asyncio.Task | None = None
    _lock = asyncio.Lock()

    GC_INTERVAL_SECONDS = 600
    IDLE_TIMEOUT_SECONDS = 1800

    @classmethod
    async def _ensure_session(cls):
        """Lazy-init: start BrowserSession if not already running."""
        if cls._session is None:
            is_headless = os.environ.get("BROWSER_HEADLESS", "false").lower() == "true"
            os.makedirs(USER_DATA_DIR, exist_ok=True)

            cls._session = BrowserSession(
                headless=is_headless,
                user_data_dir=USER_DATA_DIR,
                viewport={"width": 1280, "height": 720},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/121.0.0.0 Safari/537.36"
                ),
                disable_security=True,
                highlight_elements=True,
            )
            await cls._session.start()

            if cls._gc_task is None:
                cls._gc_task = asyncio.create_task(cls._gc_loop())

    @classmethod
    async def get_session(cls) -> BrowserSession:
        """Return the active BrowserSession, creating if needed."""
        async with cls._lock:
            await cls._ensure_session()
        assert cls._session is not None
        return cls._session

    @classmethod
    async def get_page(cls, thread_id: str):
        """Return the active Playwright Page for a thread."""
        async with cls._lock:
            await cls._ensure_session()
            assert cls._session is not None

            if thread_id in cls._threads:
                cls._threads[thread_id]["last_accessed"] = time.time()
                # Use the session's current page
                return await cls._session.get_current_page()

            # First call for this thread — use the session's current page
            page = await cls._session.get_current_page()
            cls._threads[thread_id] = {
                "last_accessed": time.time(),
                "just_opened_new_tab": False,
            }
            return page

    @classmethod
    async def get_browser_state(cls):
        """Return browser-use's BrowserStateSummary (DOM tree + optional screenshot)."""
        session = await cls.get_session()
        return await session.get_browser_state_summary(include_screenshot=True)

    @classmethod
    async def get_state_text(cls) -> str:
        """Return browser-use's LLM-optimized DOM string (numbered interactive elements)."""
        session = await cls.get_session()
        return await session.get_state_as_text()

    @classmethod
    async def get_selector_map(cls) -> dict[int, EnhancedDOMTreeNode]:
        """Return browser-use's index→DOMNode map for element targeting."""
        session = await cls.get_session()
        return await session.get_selector_map()

    @classmethod
    async def get_element_by_index(cls, index: int) -> EnhancedDOMTreeNode | None:
        """Look up a DOM element by its Set-of-Marks index."""
        session = await cls.get_session()
        return await session.get_element_by_index(index)

    @classmethod
    async def take_screenshot(cls, quality: int = 50) -> bytes:
        """Take a JPEG screenshot via browser-use."""
        session = await cls.get_session()
        return await session.take_screenshot(format="jpeg", quality=quality)

    @classmethod
    async def pop_new_tab_notice(cls, thread_id: str) -> bool:
        """Return and reset per-thread new-tab notice flag."""
        async with cls._lock:
            entry = cls._threads.get(thread_id)
            if not entry:
                return False
            opened = bool(entry.get("just_opened_new_tab"))
            entry["just_opened_new_tab"] = False
            return opened

    @classmethod
    async def close_context(cls, thread_id: str):
        """Remove a thread's tracking entry."""
        async with cls._lock:
            cls._threads.pop(thread_id, None)

    @classmethod
    async def shutdown(cls):
        """Close everything — called from FastAPI lifespan shutdown."""
        if cls._gc_task:
            cls._gc_task.cancel()
            cls._gc_task = None

        cls._threads.clear()

        if cls._session:
            try:
                await cls._session.stop()
            except Exception as e:
                logger.warning("Error stopping BrowserSession: %s", e)
            cls._session = None

    @classmethod
    async def _gc_loop(cls):
        """Background task: sweep idle thread entries."""
        try:
            while True:
                await asyncio.sleep(cls.GC_INTERVAL_SECONDS)
                now = time.time()
                stale = [
                    tid for tid, entry in cls._threads.items()
                    if (now - entry["last_accessed"]) > cls.IDLE_TIMEOUT_SECONDS
                ]
                for tid in stale:
                    await cls.close_context(tid)
        except asyncio.CancelledError:
            pass


def get_active_session_manager() -> type[BrowserSessionManager]:
    """Module-level accessor for the session manager class."""
    return BrowserSessionManager
