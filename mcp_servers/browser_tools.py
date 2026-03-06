"""
mcp_servers/browser_tools.py — Playwright-based browser automation tools.

Provides the agent with full browser control: navigate, click, type,
screenshot, extract text, scroll, hover, select dropdowns, manage tabs,
press keys, handle dialogs, and upload files.

Each conversation thread gets an isolated BrowserContext with its own cookies/storage.

Write-type tools are HITL-gated by the agent router.
Read-type tools run freely.
"""

import asyncio
import base64
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

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
    Supports multi-tab workflows per thread.
    """

    _playwright = None
    _browser = None
    _contexts: Dict[str, dict] = {}   # thread_id -> {"context", "pages", "active_page_idx", "last_accessed"}
    _gc_task = None
    _lock = asyncio.Lock()

    GC_INTERVAL_SECONDS = 600   # Sweep every 10 minutes
    IDLE_TIMEOUT_SECONDS = 1800  # Close contexts idle > 30 minutes

    @classmethod
    async def _ensure_browser(cls):
        """Lazy-init: start Playwright + Chromium if not already running."""
        if cls._browser is None:
            import os
            from playwright.async_api import async_playwright
            cls._playwright = await async_playwright().start()
            
            # Default to visible browser unless explicitly set to headless
            is_headless = os.environ.get("BROWSER_HEADLESS", "false").lower() == "true"
            
            cls._browser = await cls._playwright.chromium.launch(headless=is_headless)
            logger.info(f"🌐 Playwright Chromium launched (headless={is_headless})")

            # Start GC sweep
            if cls._gc_task is None:
                cls._gc_task = asyncio.create_task(cls._gc_loop())

    @classmethod
    async def get_page(cls, thread_id: str):
        """Return the active Playwright Page for a thread, creating context if needed."""
        async with cls._lock:
            await cls._ensure_browser()

            if thread_id in cls._contexts:
                entry = cls._contexts[thread_id]
                entry["last_accessed"] = time.time()
                return entry["pages"][entry["active_page_idx"]]

            # Create isolated context + page
            context = await cls._browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            )
            # Auto-handle dialogs (accept by default) unless explicitly managed
            context.on("dialog", lambda dialog: asyncio.create_task(dialog.accept()))

            page = await context.new_page()
            cls._contexts[thread_id] = {
                "context": context,
                "pages": [page],
                "active_page_idx": 0,
                "last_accessed": time.time(),
            }
            logger.info(f"🌐 New browser context created for thread {thread_id}")
            return page

    @classmethod
    async def get_entry(cls, thread_id: str) -> dict:
        """Return the full session entry for tab management."""
        await cls.get_page(thread_id)  # Ensure context exists
        return cls._contexts[thread_id]

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

        for thread_id in list(cls._contexts.keys()):
            await cls.close_context(thread_id)

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


def _get_thread_id(config) -> str:
    return config.get("configurable", {}).get("thread_id", "default") if config else "default"


# ══════════════════════════════════════════════════════════════
# READ TOOLS (Autonomous)
# ══════════════════════════════════════════════════════════════

@tool
async def browser_navigate(
    url: str = Field(..., description="The URL to navigate to."),
    config: RunnableConfig = None,
) -> str:
    """
    Navigate the browser to a URL and return the page title and visible text content.
    Use this to open websites for reading, research, or preparing for interaction.
    """
    thread_id = _get_thread_id(config)
    logger.info(f"🌐 browser_navigate(url='{url}', thread={thread_id})")
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        title = await page.title()
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
    """Extract all visible text from the current page."""
    thread_id = _get_thread_id(config)
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        if page.url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."
        title = await page.title()
        text_content = await page.inner_text("body")
        text_content = _smart_truncate(text_content)
        return f"## Current Page: {title}\n**URL**: {page.url}\n\n{text_content}"
    except Exception as e:
        return f"Failed to extract text: {type(e).__name__} - {str(e)}"


@tool
async def browser_screenshot(
    config: RunnableConfig = None,
) -> list:
    """
    Take a screenshot of the current page. Returns the image as a multimodal
    content block so you can visually inspect the page layout and content.
    Also saves the PNG to disk for user reference.
    """
    thread_id = _get_thread_id(config)
    logger.info(f"🌐 browser_screenshot(thread={thread_id})")
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        if page.url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        # Save to disk
        timestamp = int(time.time())
        filename = f"{thread_id}_{timestamp}.png"
        filepath = SCREENSHOT_DIR / filename
        screenshot_bytes = await page.screenshot(full_page=False)
        filepath.write_bytes(screenshot_bytes)

        # Return multimodal content block for LLM vision
        b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        logger.info(f"✅ browser_screenshot saved to {filepath}")
        return [
            {"type": "text", "text": f"Screenshot of {page.url} (saved to {filepath}):"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
    except Exception as e:
        logger.error(f"❌ browser_screenshot failed: {e}")
        return f"Failed to take screenshot: {type(e).__name__} - {str(e)}"


@tool
async def browser_go_back(
    config: RunnableConfig = None,
) -> str:
    """Go back to the previous page in history. Use this if you navigated to a wrong page."""
    thread_id = _get_thread_id(config)
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        await page.go_back(wait_until="domcontentloaded", timeout=15000)
        title = await page.title()
        return f"✅ Went back. Now on: **{title}** ({page.url})"
    except Exception as e:
        return f"Failed to go back: {type(e).__name__} - {str(e)}"


@tool
async def browser_scroll(
    direction: str = Field("down", description="Scroll direction: 'up' or 'down'."),
    amount: int = Field(3, description="Number of viewport-heights to scroll (1-10). Default 3."),
    config: RunnableConfig = None,
) -> str:
    """Scroll the page up or down to see more content. Use after navigating to see content below the fold or in infinite-scroll pages."""
    thread_id = _get_thread_id(config)
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        if page.url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."
        pixels = min(max(amount, 1), 10) * 720  # viewport height = 720
        delta = pixels if direction == "down" else -pixels
        await page.mouse.wheel(0, delta)
        await asyncio.sleep(0.5)  # Let content load
        return f"✅ Scrolled {direction} by {pixels}px. Use browser_get_text or browser_screenshot to see the new content."
    except Exception as e:
        return f"Failed to scroll: {type(e).__name__} - {str(e)}"


@tool
async def browser_wait_for(
    seconds: float = Field(2.0, description="Seconds to wait (max 30)."),
    text: Optional[str] = Field(None, description="Optional: wait for this text to appear on the page instead of a fixed delay."),
    config: RunnableConfig = None,
) -> str:
    """Wait for a specified duration or for specific text to appear on the page. Use this for pages with async/AJAX loading."""
    thread_id = _get_thread_id(config)
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        if text:
            await page.wait_for_selector(f"text={text}", timeout=min(seconds, 30) * 1000)
            return f"✅ Text '{text}' appeared on the page."
        else:
            await asyncio.sleep(min(seconds, 30))
            return f"✅ Waited {seconds} seconds."
    except Exception as e:
        return f"Wait failed: {type(e).__name__} - {str(e)}"


@tool
async def browser_snapshot(
    config: RunnableConfig = None,
) -> str:
    """
    Capture a structured snapshot of all interactive elements on the current page.
    Returns roles, names, and values of all buttons, links, inputs, selects, etc.
    Much more useful than raw text for finding the right selectors to click or type into.
    """
    thread_id = _get_thread_id(config)
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        if page.url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        # Extract interactive elements via JavaScript
        snapshot_js = """
        () => {
            const elements = [];
            const interactiveSelectors = 'a, button, input, select, textarea, [role], [aria-label], [onclick], summary, details';
            document.querySelectorAll(interactiveSelectors).forEach((el, idx) => {
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 && rect.height === 0) return; // Skip hidden
                const role = el.getAttribute('role') || el.tagName.toLowerCase();
                const name = el.getAttribute('aria-label')
                    || el.getAttribute('name')
                    || el.getAttribute('placeholder')
                    || el.textContent?.trim().substring(0, 80)
                    || '';
                const type = el.getAttribute('type') || '';
                const value = el.value || '';
                const href = el.getAttribute('href') || '';
                let entry = `[${idx}] <${role}>`;
                if (type) entry += ` type="${type}"`;
                if (name) entry += ` "${name}"`;
                if (value) entry += ` value="${value}"`;
                if (href) entry += ` href="${href}"`;
                const id = el.id ? ` id="${el.id}"` : '';
                const cls = el.className ? ` class="${String(el.className).substring(0, 50)}"` : '';
                entry += id + cls;
                elements.push(entry);
            });
            return elements.join('\\n');
        }
        """
        tree_text = await page.evaluate(snapshot_js)
        if not tree_text:
            return "No interactive elements found on this page. Try browser_get_text instead."

        tree_text = _smart_truncate(tree_text)
        title = await page.title()
        return f"## Interactive Elements: {title}\n**URL**: {page.url}\n\n```\n{tree_text}\n```"
    except Exception as e:
        return f"Failed to capture snapshot: {type(e).__name__} - {str(e)}"


@tool
async def browser_tab_management(
    action: str = Field(..., description="Action: 'list', 'new', 'switch', or 'close'."),
    index: Optional[int] = Field(None, description="Tab index for 'switch' and 'close'. 0-based."),
    config: RunnableConfig = None,
) -> str:
    """
    Manage browser tabs. 'list' shows all tabs, 'new' opens a blank tab,
    'switch' activates a tab by index, 'close' closes a tab by index (defaults to current).
    """
    thread_id = _get_thread_id(config)
    try:
        entry = await BrowserSessionManager.get_entry(thread_id)
        pages = entry["pages"]

        if action == "list":
            lines = []
            for i, p in enumerate(pages):
                marker = "→ " if i == entry["active_page_idx"] else "  "
                try:
                    title = await p.title()
                except:
                    title = "(closed)"
                lines.append(f"{marker}[{i}] {title} — {p.url}")
            return "## Open Tabs\n" + "\n".join(lines)

        elif action == "new":
            new_page = await entry["context"].new_page()
            pages.append(new_page)
            entry["active_page_idx"] = len(pages) - 1
            return f"✅ Opened new tab (index {len(pages) - 1}). Use browser_navigate to load a page."

        elif action == "switch":
            if index is None or index < 0 or index >= len(pages):
                return f"❌ Invalid index. You have {len(pages)} tabs (0-{len(pages)-1})."
            entry["active_page_idx"] = index
            page = pages[index]
            title = await page.title()
            return f"✅ Switched to tab [{index}]: **{title}** ({page.url})"

        elif action == "close":
            idx = index if index is not None else entry["active_page_idx"]
            if idx < 0 or idx >= len(pages):
                return f"❌ Invalid index. You have {len(pages)} tabs (0-{len(pages)-1})."
            if len(pages) == 1:
                return "❌ Cannot close the last tab. Use browser_navigate instead."
            closed_page = pages.pop(idx)
            await closed_page.close()
            entry["active_page_idx"] = min(entry["active_page_idx"], len(pages) - 1)
            return f"✅ Closed tab [{idx}]. Active tab is now [{entry['active_page_idx']}]."

        return f"❌ Unknown action '{action}'. Use: list, new, switch, close."
    except Exception as e:
        return f"Tab management failed: {type(e).__name__} - {str(e)}"


# ══════════════════════════════════════════════════════════════
# WRITE TOOLS (HITL-gated)
# ══════════════════════════════════════════════════════════════

@tool
async def browser_click(
    selector: str = Field(
        ...,
        description="CSS selector OR visible text of the element to click. "
                    "Examples: '#submit-btn', 'button.login', 'Sign In', 'Accept Cookies'."
    ),
    double_click: bool = Field(False, description="Whether to double-click instead of single click."),
    config: RunnableConfig = None,
) -> str:
    """
    Click an element on the current page. Supports CSS selectors and text matching.
    This is a WRITE action and requires human approval.
    """
    thread_id = _get_thread_id(config)
    logger.info(f"🌐 browser_click(selector='{selector}', thread={thread_id})")
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        if page.url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        click_kwargs = {"timeout": 5000}
        if double_click:
            click_kwargs["click_count"] = 2

        # Strategy 1: CSS selector
        try:
            await page.click(selector, **click_kwargs)
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
            return f"✅ Clicked `{selector}`. Page: **{await page.title()}** ({page.url})"
        except Exception:
            pass

        # Strategy 2: Visible text
        try:
            locator = page.get_by_text(selector, exact=False).first
            await locator.click(**click_kwargs)
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
            return f"✅ Clicked text '{selector}'. Page: **{await page.title()}** ({page.url})"
        except Exception:
            pass

        # Strategy 3: Role-based
        for role in ["link", "button", "menuitem"]:
            try:
                locator = page.get_by_role(role, name=selector).first
                await locator.click(**click_kwargs)
                await page.wait_for_load_state("domcontentloaded", timeout=10000)
                return f"✅ Clicked {role} '{selector}'. Page: **{await page.title()}** ({page.url})"
            except Exception:
                continue

        return f"❌ Could not find element matching '{selector}'. Try browser_snapshot to inspect the page."
    except Exception as e:
        return f"Failed to click '{selector}': {type(e).__name__} - {str(e)}"


@tool
async def browser_type(
    selector: str = Field(
        ...,
        description="CSS selector, placeholder text, or label of the input field."
    ),
    text: str = Field(..., description="The text to type into the field."),
    submit: bool = Field(False, description="Press Enter after typing to submit the form."),
    config: RunnableConfig = None,
) -> str:
    """
    Type text into an input field on the current page.
    Set submit=True to press Enter after typing (e.g. for search boxes).
    This is a WRITE action and requires human approval.
    """
    thread_id = _get_thread_id(config)
    logger.info(f"🌐 browser_type(selector='{selector}', text='{text[:50]}', thread={thread_id})")
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        if page.url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        filled = False

        # Try CSS selector
        try:
            await page.fill(selector, text, timeout=5000)
            filled = True
        except Exception:
            pass

        # Fallback: placeholder
        if not filled:
            try:
                locator = page.get_by_placeholder(selector, exact=False).first
                await locator.fill(text, timeout=5000)
                filled = True
            except Exception:
                pass

        # Fallback: label
        if not filled:
            try:
                locator = page.get_by_label(selector, exact=False).first
                await locator.fill(text, timeout=5000)
                filled = True
            except Exception:
                pass

        if not filled:
            return f"❌ Could not find input matching '{selector}'. Try browser_snapshot to inspect the page."

        result = f"✅ Typed '{text}' into `{selector}`."
        if submit:
            await page.keyboard.press("Enter")
            await page.wait_for_load_state("domcontentloaded", timeout=10000)
            result += f" Submitted. Page: **{await page.title()}** ({page.url})"
        return result
    except Exception as e:
        return f"Failed to type into '{selector}': {type(e).__name__} - {str(e)}"


@tool
async def browser_execute_js(
    script: str = Field(..., description="JavaScript code to execute on the current page."),
    config: RunnableConfig = None,
) -> str:
    """
    Execute arbitrary JavaScript on the current page and return the result.
    This is a WRITE action and requires human approval.
    """
    thread_id = _get_thread_id(config)
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
        return f"JavaScript execution failed: {type(e).__name__} - {str(e)}"


@tool
async def browser_select_option(
    selector: str = Field(..., description="CSS selector of the <select> dropdown element."),
    value: str = Field(..., description="The visible text or value of the option to select."),
    config: RunnableConfig = None,
) -> str:
    """
    Select an option from a <select> dropdown menu.
    This is a WRITE action and requires human approval.
    """
    thread_id = _get_thread_id(config)
    logger.info(f"🌐 browser_select_option(selector='{selector}', value='{value}', thread={thread_id})")
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        if page.url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        # Try by label (visible text) first, then by value attribute
        try:
            await page.select_option(selector, label=value, timeout=5000)
            return f"✅ Selected '{value}' from `{selector}`."
        except Exception:
            pass

        try:
            await page.select_option(selector, value=value, timeout=5000)
            return f"✅ Selected value='{value}' from `{selector}`."
        except Exception:
            pass

        return f"❌ Could not select '{value}' in `{selector}`. Verify the selector and available options."
    except Exception as e:
        return f"Failed to select option: {type(e).__name__} - {str(e)}"


@tool
async def browser_press_key(
    key: str = Field(
        ...,
        description="Key to press. Examples: 'Enter', 'Escape', 'Tab', 'ArrowDown', 'Backspace', 'Control+a'."
    ),
    config: RunnableConfig = None,
) -> str:
    """
    Press a keyboard key. Use for submitting forms (Enter), closing modals (Escape),
    navigating fields (Tab), or keyboard shortcuts (Control+a).
    This is a WRITE action and requires human approval.
    """
    thread_id = _get_thread_id(config)
    logger.info(f"🌐 browser_press_key(key='{key}', thread={thread_id})")
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        await page.keyboard.press(key)
        await asyncio.sleep(0.3)
        return f"✅ Pressed '{key}'."
    except Exception as e:
        return f"Failed to press key '{key}': {type(e).__name__} - {str(e)}"


@tool
async def browser_hover(
    selector: str = Field(
        ...,
        description="CSS selector or visible text of the element to hover over."
    ),
    config: RunnableConfig = None,
) -> str:
    """
    Hover over an element to trigger dropdown menus, tooltips, or mega-navigation.
    This is a WRITE action and requires human approval.
    """
    thread_id = _get_thread_id(config)
    logger.info(f"🌐 browser_hover(selector='{selector}', thread={thread_id})")
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        if page.url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        # Try CSS selector
        try:
            await page.hover(selector, timeout=5000)
            return f"✅ Hovering over `{selector}`. Use browser_get_text or browser_screenshot to see the result."
        except Exception:
            pass

        # Try text matching
        try:
            locator = page.get_by_text(selector, exact=False).first
            await locator.hover(timeout=5000)
            return f"✅ Hovering over text '{selector}'. Use browser_get_text or browser_screenshot to see the result."
        except Exception:
            pass

        return f"❌ Could not find element matching '{selector}'. Try browser_snapshot to inspect the page."
    except Exception as e:
        return f"Failed to hover: {type(e).__name__} - {str(e)}"


@tool
async def browser_handle_dialog(
    action: str = Field("accept", description="How to handle the dialog: 'accept' or 'dismiss'."),
    prompt_text: Optional[str] = Field(None, description="Text to enter in a prompt dialog (if applicable)."),
    config: RunnableConfig = None,
) -> str:
    """
    Change how browser dialogs (alert, confirm, prompt) are handled.
    By default dialogs are auto-accepted. Use this to dismiss them or enter prompt text.
    This is a WRITE action and requires human approval.
    """
    thread_id = _get_thread_id(config)
    try:
        page = await BrowserSessionManager.get_page(thread_id)

        # Remove previous handler and set new one
        entry = await BrowserSessionManager.get_entry(thread_id)
        context = entry["context"]

        # Remove all existing dialog handlers
        context.remove_listener("dialog", context.listeners("dialog")[0] if context.listeners("dialog") else None)

        if action == "accept":
            async def _accept_dialog(dialog):
                if prompt_text:
                    await dialog.accept(prompt_text)
                else:
                    await dialog.accept()
            context.on("dialog", lambda d: asyncio.create_task(_accept_dialog(d)))
            msg = f"✅ Dialogs will be accepted"
            if prompt_text:
                msg += f" with text '{prompt_text}'"
        else:
            context.on("dialog", lambda d: asyncio.create_task(d.dismiss()))
            msg = "✅ Dialogs will be dismissed"

        return msg + ". This applies to all future dialogs on this page."
    except Exception as e:
        return f"Failed to configure dialog handling: {type(e).__name__} - {str(e)}"


@tool
async def browser_file_upload(
    selector: str = Field(..., description="CSS selector of the file input element (e.g. 'input[type=file]')."),
    file_paths: str = Field(
        ...,
        description="Comma-separated absolute paths to files to upload. Example: '/path/to/resume.pdf'"
    ),
    config: RunnableConfig = None,
) -> str:
    """
    Upload one or more files to a file input element on the page.
    This is a WRITE action and requires human approval.
    """
    thread_id = _get_thread_id(config)
    logger.info(f"🌐 browser_file_upload(selector='{selector}', thread={thread_id})")
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        if page.url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        paths = [p.strip() for p in file_paths.split(",") if p.strip()]
        # Validate paths exist
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            return f"❌ Files not found: {', '.join(missing)}"

        await page.set_input_files(selector, paths, timeout=5000)
        return f"✅ Uploaded {len(paths)} file(s) to `{selector}`: {', '.join(Path(p).name for p in paths)}"
    except Exception as e:
        return f"Failed to upload files: {type(e).__name__} - {str(e)}"


# ══════════════════════════════════════════════════════════════
# Tool Registry — 16 tools total
# ══════════════════════════════════════════════════════════════

TOOL_REGISTRY: Dict[str, Any] = {
    # Read Tools (Autonomous)
    "browser_navigate": browser_navigate,
    "browser_get_text": browser_get_text,
    "browser_screenshot": browser_screenshot,
    "browser_go_back": browser_go_back,
    "browser_scroll": browser_scroll,
    "browser_wait_for": browser_wait_for,
    "browser_snapshot": browser_snapshot,
    "browser_tab_management": browser_tab_management,
    # Write Tools (HITL-gated)
    "browser_click": browser_click,
    "browser_type": browser_type,
    "browser_execute_js": browser_execute_js,
    "browser_select_option": browser_select_option,
    "browser_press_key": browser_press_key,
    "browser_hover": browser_hover,
    "browser_handle_dialog": browser_handle_dialog,
    "browser_file_upload": browser_file_upload,
}
