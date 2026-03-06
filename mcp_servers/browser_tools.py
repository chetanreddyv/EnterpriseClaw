"""
mcp_servers/browser_tools.py — Playwright-based browser automation tools.

Provides the agent with full browser control: navigate, click, type,
screenshot, extract text, and execute JavaScript. Each conversation
thread gets an isolated BrowserContext with its own cookies/storage.

Write-type tools (click, type, execute_js) are HITL-gated by the
agent router. Read-type tools (navigate, get_text, screenshot) run freely.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any

from pydantic import Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger("mcp.browser_tools")

SCREENSHOT_DIR = Path("./data/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# Browser Session Manager (Singleton)
# ══════════════════════════════════════════════════════════════

class BrowserSessionManager:
    """
    Manages a single Chromium browser instance with per-thread BrowserContexts.
    Lazy-initializes Playwright on first use. Includes idle-timeout GC.
    """

    _playwright = None
    _browser = None
    _contexts: Dict[str, dict] = {}   # thread_id -> {"context": BrowserContext, "page": Page, "last_accessed": float}
    _gc_task = None
    _lock = asyncio.Lock()

    GC_INTERVAL_SECONDS = 600   # Sweep every 10 minutes
    IDLE_TIMEOUT_SECONDS = 1800  # Close contexts idle > 30 minutes

    @classmethod
    async def _ensure_browser(cls):
        """Lazy-init: start Playwright + Chromium if not already running."""
        if cls._browser is None:
            from playwright.async_api import async_playwright
            cls._playwright = await async_playwright().start()
            cls._browser = await cls._playwright.chromium.launch(headless=True)
            logger.info("🌐 Playwright Chromium launched (headless)")

            # Start GC sweep
            if cls._gc_task is None:
                cls._gc_task = asyncio.create_task(cls._gc_loop())

    @classmethod
    async def get_page(cls, thread_id: str):
        """
        Return the Playwright Page for a thread, creating a new
        BrowserContext if one doesn't exist yet.
        """
        async with cls._lock:
            await cls._ensure_browser()

            if thread_id in cls._contexts:
                entry = cls._contexts[thread_id]
                entry["last_accessed"] = time.time()
                return entry["page"]

            # Create isolated context + page
            context = await cls._browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            )
            page = await context.new_page()
            cls._contexts[thread_id] = {
                "context": context,
                "page": page,
                "last_accessed": time.time(),
            }
            logger.info(f"🌐 New browser context created for thread {thread_id}")
            return page

    @classmethod
    async def close_context(cls, thread_id: str):
        """Close and remove a thread's browser context."""
        async with cls._lock:
            entry = cls._contexts.pop(thread_id, None)
            if entry:
                try:
                    await entry["context"].close()
                    logger.info(f"🌐 Browser context closed for thread {thread_id}")
                except Exception as e:
                    logger.warning(f"Error closing context {thread_id}: {e}")

    @classmethod
    async def shutdown(cls):
        """Close everything — called from FastAPI lifespan shutdown."""
        if cls._gc_task:
            cls._gc_task.cancel()
            cls._gc_task = None

        # Close all contexts
        for thread_id in list(cls._contexts.keys()):
            await cls.close_context(thread_id)

        # Close browser + playwright
        if cls._browser:
            await cls._browser.close()
            cls._browser = None
        if cls._playwright:
            await cls._playwright.stop()
            cls._playwright = None
        logger.info("🌐 Playwright shut down cleanly")

    @classmethod
    async def _gc_loop(cls):
        """Background task: sweep idle contexts every GC_INTERVAL_SECONDS."""
        try:
            while True:
                await asyncio.sleep(cls.GC_INTERVAL_SECONDS)
                now = time.time()
                stale = [
                    tid for tid, entry in cls._contexts.items()
                    if (now - entry["last_accessed"]) > cls.IDLE_TIMEOUT_SECONDS
                ]
                for tid in stale:
                    logger.info(f"🧹 GC: closing idle browser context for thread {tid}")
                    await cls.close_context(tid)
        except asyncio.CancelledError:
            pass


def _smart_truncate(text: str, max_chars: int = 12000) -> str:
    """Truncate text at the nearest newline boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars * 0.8:
        truncated = truncated[:last_nl]
    return truncated + "\n\n... [Content truncated]"


# ══════════════════════════════════════════════════════════════
# Tool Implementations
# ══════════════════════════════════════════════════════════════

@tool
async def browser_navigate(
    url: str = Field(..., description="The URL to navigate to."),
    config: RunnableConfig = None,
) -> str:
    """
    Navigate the browser to a URL and return the page title and visible text content.
    Use this to open websites for reading, research, or preparing for interaction.

    Args:
        url: The full URL to navigate to (e.g. https://example.com).
    """
    thread_id = config.get("configurable", {}).get("thread_id", "default") if config else "default"
    logger.info(f"🌐 browser_navigate(url='{url}', thread={thread_id})")

    try:
        page = await BrowserSessionManager.get_page(thread_id)
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        title = await page.title()

        # Extract visible text via accessibility snapshot
        text_content = await page.inner_text("body")
        text_content = _smart_truncate(text_content)

        logger.info(f"✅ browser_navigate: loaded '{title}' ({len(text_content)} chars)")
        return f"## Page Loaded: {title}\n**URL**: {url}\n\n{text_content}"
    except Exception as e:
        logger.error(f"❌ browser_navigate failed: {e}")
        return f"Failed to navigate to {url}: {type(e).__name__} - {str(e)}"


@tool
async def browser_get_text(
    config: RunnableConfig = None,
) -> str:
    """
    Extract all visible text from the current page.
    Use this after navigating to read the full page content.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "default") if config else "default"
    logger.info(f"🌐 browser_get_text(thread={thread_id})")

    try:
        page = await BrowserSessionManager.get_page(thread_id)
        current_url = page.url
        title = await page.title()

        if current_url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        text_content = await page.inner_text("body")
        text_content = _smart_truncate(text_content)

        return f"## Current Page: {title}\n**URL**: {current_url}\n\n{text_content}"
    except Exception as e:
        logger.error(f"❌ browser_get_text failed: {e}")
        return f"Failed to extract text: {type(e).__name__} - {str(e)}"


@tool
async def browser_screenshot(
    config: RunnableConfig = None,
) -> str:
    """
    Take a screenshot of the current page and save it to disk.
    Returns the file path of the saved screenshot.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "default") if config else "default"
    logger.info(f"🌐 browser_screenshot(thread={thread_id})")

    try:
        page = await BrowserSessionManager.get_page(thread_id)
        current_url = page.url

        if current_url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        timestamp = int(time.time())
        filename = f"{thread_id}_{timestamp}.png"
        filepath = SCREENSHOT_DIR / filename

        await page.screenshot(path=str(filepath), full_page=False)
        logger.info(f"✅ browser_screenshot saved to {filepath}")
        return f"Screenshot saved to `{filepath}`. Current page: {current_url}"
    except Exception as e:
        logger.error(f"❌ browser_screenshot failed: {e}")
        return f"Failed to take screenshot: {type(e).__name__} - {str(e)}"


@tool
async def browser_click(
    selector: str = Field(
        ...,
        description="CSS selector OR visible text of the element to click. "
                    "Examples: '#submit-btn', 'button.login', 'Sign In', 'Accept Cookies'."
    ),
    config: RunnableConfig = None,
) -> str:
    """
    Click an element on the current page. Supports CSS selectors and text matching.
    This is a WRITE action and requires human approval.

    Args:
        selector: CSS selector or visible text of the element to click.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "default") if config else "default"
    logger.info(f"🌐 browser_click(selector='{selector}', thread={thread_id})")

    try:
        page = await BrowserSessionManager.get_page(thread_id)

        if page.url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        # Strategy 1: Try as CSS selector
        try:
            await page.click(selector, timeout=5000)
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
            new_url = page.url
            title = await page.title()
            return f"✅ Clicked `{selector}`. Page is now: **{title}** ({new_url})"
        except Exception:
            pass

        # Strategy 2: Try as visible text (case-insensitive)
        try:
            locator = page.get_by_text(selector, exact=False).first
            await locator.click(timeout=5000)
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
            new_url = page.url
            title = await page.title()
            return f"✅ Clicked text '{selector}'. Page is now: **{title}** ({new_url})"
        except Exception:
            pass

        # Strategy 3: Try as role-based locator
        try:
            locator = page.get_by_role("link", name=selector).first
            await locator.click(timeout=5000)
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
            new_url = page.url
            title = await page.title()
            return f"✅ Clicked link '{selector}'. Page is now: **{title}** ({new_url})"
        except Exception:
            pass

        return f"❌ Could not find element matching '{selector}' on the page. Try using browser_get_text to inspect the page content first."

    except Exception as e:
        logger.error(f"❌ browser_click failed: {e}")
        return f"Failed to click '{selector}': {type(e).__name__} - {str(e)}"


@tool
async def browser_type(
    selector: str = Field(
        ...,
        description="CSS selector of the input field. Examples: '#search', 'input[name=email]', '#username'."
    ),
    text: str = Field(
        ...,
        description="The text to type into the field."
    ),
    config: RunnableConfig = None,
) -> str:
    """
    Type text into an input field on the current page.
    This is a WRITE action and requires human approval.

    Args:
        selector: CSS selector of the input field.
        text: The text to type.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "default") if config else "default"
    logger.info(f"🌐 browser_type(selector='{selector}', text='{text[:50]}', thread={thread_id})")

    try:
        page = await BrowserSessionManager.get_page(thread_id)

        if page.url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        # Try CSS selector first
        try:
            await page.fill(selector, text, timeout=5000)
            return f"✅ Typed '{text}' into `{selector}`."
        except Exception:
            pass

        # Fallback: try placeholder text matching
        try:
            locator = page.get_by_placeholder(selector, exact=False).first
            await locator.fill(text, timeout=5000)
            return f"✅ Typed '{text}' into field with placeholder '{selector}'."
        except Exception:
            pass

        # Fallback: try label matching
        try:
            locator = page.get_by_label(selector, exact=False).first
            await locator.fill(text, timeout=5000)
            return f"✅ Typed '{text}' into field labeled '{selector}'."
        except Exception:
            pass

        return f"❌ Could not find input matching '{selector}'. Try using browser_get_text to inspect the page."

    except Exception as e:
        logger.error(f"❌ browser_type failed: {e}")
        return f"Failed to type into '{selector}': {type(e).__name__} - {str(e)}"


@tool
async def browser_execute_js(
    script: str = Field(
        ...,
        description="JavaScript code to execute on the current page."
    ),
    config: RunnableConfig = None,
) -> str:
    """
    Execute arbitrary JavaScript on the current page and return the result.
    This is a WRITE action and requires human approval.

    Args:
        script: The JavaScript code to run.
    """
    thread_id = config.get("configurable", {}).get("thread_id", "default") if config else "default"
    logger.info(f"🌐 browser_execute_js(thread={thread_id})")

    try:
        page = await BrowserSessionManager.get_page(thread_id)

        if page.url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        result = await page.evaluate(script)
        result_str = str(result) if result is not None else "(no return value)"

        if len(result_str) > 5000:
            result_str = result_str[:5000] + "\n... [Output truncated]"

        return f"✅ JavaScript executed.\n\nResult:\n```\n{result_str}\n```"
    except Exception as e:
        logger.error(f"❌ browser_execute_js failed: {e}")
        return f"JavaScript execution failed: {type(e).__name__} - {str(e)}"


# ══════════════════════════════════════════════════════════════
# Tool Registry
# ══════════════════════════════════════════════════════════════

TOOL_REGISTRY: Dict[str, Any] = {
    "browser_navigate": browser_navigate,
    "browser_get_text": browser_get_text,
    "browser_screenshot": browser_screenshot,
    "browser_click": browser_click,
    "browser_type": browser_type,
    "browser_execute_js": browser_execute_js,
}
