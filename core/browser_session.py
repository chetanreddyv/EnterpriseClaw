"""
core/browser_session.py — browser-use BrowserSession wrapper.

Manages a singleton BrowserSession with per-thread page tracking.
Replaces the custom BrowserSessionManager from browser_tools.py.
"""

import asyncio
import logging
import os
import re
import time
from typing import Any, Dict

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
    _EMPTY_DOM_MARKER = "empty dom tree"

    @staticmethod
    def _unwrap_iife(script: str) -> str:
        """Convert `(callable)()` payloads into callable forms expected by browser-use."""
        source = script.strip()
        iife_match = re.match(r"^\(\s*(?P<body>[\s\S]+?)\s*\)\(\s*\)\s*$", source)
        if iife_match:
            return iife_match.group("body").strip()
        return source

    @staticmethod
    def _looks_like_js_callable(script: str) -> bool:
        """Return True when the script is already a JS callable payload."""
        source = script.lstrip()

        if source.startswith("function") or source.startswith("(function"):
            return True

        if re.match(r"^\(?\s*(async\s+)?\([^)]*\)\s*=>", source):
            return True

        if re.match(r"^\(?\s*(async\s+)?[A-Za-z_$][\w$]*\s*=>", source):
            return True

        return False

    @classmethod
    def wrap_js_task(cls, script: str) -> str:
        """
        Normalize any raw JS snippet into browser-use compatible evaluate payload.

        - Existing callable payloads are preserved.
        - Invocation-style IIFEs are unwrapped to callable form.
        - Raw script statements are wrapped in a callable arrow function.
        """
        source = str(script or "").strip()
        if not source:
            return "() => null"

        source = cls._unwrap_iife(source)

        if cls._looks_like_js_callable(source):
            return source

        return f"() => {{ {source} }}"

    @classmethod
    async def evaluate_js(cls, script: str, page: Any = None) -> Any:
        """Run JS through a centralized wrapper to avoid evaluate API mismatches."""
        session = await cls.get_session()
        target_page = page or await session.get_current_page()
        return await target_page.evaluate(cls.wrap_js_task(script))

    @classmethod
    async def wait_for_perception_settle(cls, seconds: float = 0.3) -> None:
        """Bounded settle delay for browser-use perception refresh on SPA pages."""
        await asyncio.sleep(min(max(seconds, 0.05), 1.0))

    @classmethod
    def _is_empty_perception_text(cls, state_text: str) -> bool:
        """Detect known browser-use empty perception payloads."""
        text = str(state_text or "").strip()
        if not text:
            return True
        return cls._EMPTY_DOM_MARKER in text.lower()

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
        """Return the active browser-use page for a thread."""
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
    async def get_state_text(cls, settle_seconds: float = 0.3, retries: int = 2) -> str:
        """Return browser-use's LLM-optimized DOM string with warmup/retry safeguards."""
        session = await cls.get_session()
        total_attempts = max(1, retries + 1)
        last_text = ""

        for attempt in range(total_attempts):
            if settle_seconds > 0:
                await cls.wait_for_perception_settle(settle_seconds)

            try:
                await session.get_browser_state_summary()
            except Exception as warmup_error:
                logger.debug(
                    "Perception warmup failed (attempt %d/%d): %s",
                    attempt + 1,
                    total_attempts,
                    warmup_error,
                )

            last_text = await session.get_state_as_text()
            if not cls._is_empty_perception_text(last_text):
                return last_text

        return str(last_text or "").strip()

    @classmethod
    async def get_selector_map(
        cls,
        settle_seconds: float = 0.3,
        retries: int = 2,
    ) -> dict[int, EnhancedDOMTreeNode]:
        """Return browser-use index map with perception warmup/retry safeguards."""
        session = await cls.get_session()
        total_attempts = max(1, retries + 1)
        last_map: dict[int, EnhancedDOMTreeNode] = {}

        for attempt in range(total_attempts):
            if settle_seconds > 0:
                await cls.wait_for_perception_settle(settle_seconds)

            try:
                await session.get_browser_state_summary()
            except Exception as warmup_error:
                logger.debug(
                    "Selector-map warmup failed (attempt %d/%d): %s",
                    attempt + 1,
                    total_attempts,
                    warmup_error,
                )

            selector_map = await session.get_selector_map()
            if selector_map:
                return selector_map
            last_map = selector_map or {}

        return last_map

    @classmethod
    async def get_element_by_index(cls, index: int) -> EnhancedDOMTreeNode | None:
        """Look up a DOM element by its Set-of-Marks index."""
        session = await cls.get_session()
        element = await session.get_element_by_index(index)
        if element is not None:
            return element

        # One warmup retry helps when perception lags right after navigation.
        await cls.get_selector_map(settle_seconds=0.2, retries=1)
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
