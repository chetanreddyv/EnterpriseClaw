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

from core.text_utils import get_thread_id, smart_truncate

logger = logging.getLogger("mcp.browser_tools")

SCREENSHOT_DIR = Path("./data/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

SOM_ID_ATTR = "data-claw-id"
SOM_PREV_OUTLINE_ATTR = "data-claw-prev-outline"
SOM_OVERLAY_ID = "claw-som-overlay"


def _log_strategy_failure(tool_name: str, strategy: str, selector: str, error: Exception) -> None:
    """Debug-level visibility for fallback strategy failures."""
    logger.debug(
        "%s fallback failed [%s] selector='%s': %s: %s",
        tool_name,
        strategy,
        selector,
        type(error).__name__,
        error,
    )


async def _wait_for_page_settle(page, include_networkidle: bool = True, settle_seconds: float = 0.3) -> None:
    """Best-effort wait for SPA updates to settle before reading page state."""
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=1500)
    except Exception:
        pass

    if include_networkidle:
        try:
            await page.wait_for_load_state("networkidle", timeout=1500)
        except Exception:
            pass

    if settle_seconds > 0:
        await asyncio.sleep(min(max(settle_seconds, 0.05), 1.0))


async def get_current_page_context(config: RunnableConfig = None) -> dict[str, str]:
    """Return URL and title breadcrumbs for the active page."""
    thread_id = _get_thread_id(config)
    page = await BrowserSessionManager.get_page(thread_id)
    if page.url == "about:blank":
        return {"url": "about:blank", "title": "(blank page)"}

    title = await page.title()
    return {"url": page.url, "title": title}


async def get_interactive_element_index(
    config: RunnableConfig = None,
    max_elements: int = 80,
    annotate_dom: bool = False,
) -> dict[str, Any]:
    """Return a compressed index of visible interactive elements."""
    thread_id = _get_thread_id(config)
    page = await BrowserSessionManager.get_page(thread_id)
    if page.url == "about:blank":
        return {
            "url": "about:blank",
            "title": "(blank page)",
            "elements": [],
        }

    snapshot_js = r"""
    ([maxElements, annotateDom, idAttr]) => {
        const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
        const viewportWidth = window.innerWidth || document.documentElement.clientWidth;

        const interactiveRoles = new Set([
            'button',
            'link',
            'menuitem',
            'tab',
            'checkbox',
            'radio',
            'switch',
            'textbox',
            'combobox',
            'option',
            'spinbutton',
            'slider',
        ]);

        const parseOpacity = (value) => {
            const parsed = Number.parseFloat(value);
            return Number.isFinite(parsed) ? parsed : 1;
        };

        const pickText = (el) => {
            const aria = el.getAttribute('aria-label') || '';
            const placeholder = el.getAttribute('placeholder') || '';
            const text = (el.innerText || el.textContent || '').replace(/\s+/g, ' ').trim();
            return aria || placeholder || text || '';
        };

        const isVisible = (el) => {
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden' || parseOpacity(style.opacity) <= 0.05) {
                return false;
            }
            const rect = el.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0) {
                return false;
            }
            if (
                rect.bottom < 0 ||
                rect.top > viewportHeight ||
                rect.right < 0 ||
                rect.left > viewportWidth
            ) {
                return false;
            }
            return true;
        };

        const isInteractive = (el) => {
            const tag = el.tagName.toLowerCase();
            if (['a', 'button', 'input', 'select', 'textarea', 'summary'].includes(tag)) {
                return true;
            }
            const role = (el.getAttribute('role') || '').trim().toLowerCase();
            if (role && interactiveRoles.has(role)) {
                return true;
            }

            const tabindex = el.getAttribute('tabindex');
            if (tabindex !== null) {
                const tabIndexNum = Number.parseInt(tabindex, 10);
                if (!Number.isNaN(tabIndexNum) && tabIndexNum >= 0) {
                    return true;
                }
            }

            if (el.hasAttribute('contenteditable')) {
                const contentEditable = (el.getAttribute('contenteditable') || '').trim().toLowerCase();
                if (contentEditable === '' || contentEditable === 'true') {
                    return true;
                }
            }

            if (el.getAttribute('aria-haspopup')) {
                return true;
            }

            if (el.hasAttribute('onclick') || el.hasAttribute('href')) {
                return true;
            }
            return false;
        };

        const scoreCandidate = (tag, role, typeAttr, text, rect) => {
            let score = 0;
            if (tag === 'button') score += 6;
            if (tag === 'a') score += 4;
            if (tag === 'input') {
                if (['submit', 'button', 'image', 'checkbox', 'radio'].includes(typeAttr)) {
                    score += 5;
                } else {
                    score += 2;
                }
            }
            if (['button', 'link', 'menuitem', 'tab'].includes(role)) score += 3;
            if (text) score += text.length <= 48 ? 2 : 1;

            const area = Math.max(1, rect.width * rect.height);
            if (area >= 12000) score += 2;
            else if (area >= 3000) score += 1;
            return score;
        };

        const attrsToKeep = ['href', 'type', 'name', 'placeholder', 'id', 'role'];
        const selectors = [
            'a',
            'button',
            'input',
            'select',
            'textarea',
            'summary',
            '[role]',
            '[onclick]',
            '[href]',
            '[tabindex]:not([tabindex="-1"])',
            '[contenteditable="true"]',
            '[aria-haspopup]'
        ].join(', ');
        const nodes = Array.from(document.querySelectorAll(selectors));

        if (annotateDom) {
            document.querySelectorAll(`[${idAttr}]`).forEach((el) => el.removeAttribute(idAttr));
        }

        const seen = new Set();
        const candidates = [];

        for (const el of nodes) {
            if (!isInteractive(el) || !isVisible(el)) continue;

            const tag = el.tagName.toLowerCase();
            const role = (el.getAttribute('role') || '').trim().toLowerCase();
            const typeAttr = (el.getAttribute('type') || '').trim().toLowerCase();
            const rect = el.getBoundingClientRect();
            const attrs = [];
            for (const attrName of attrsToKeep) {
                const attrValue = el.getAttribute(attrName);
                if (!attrValue) continue;
                const normalized = String(attrValue).replace(/\s+/g, ' ').trim().slice(0, 80);
                attrs.push(`${attrName}="${normalized}"`);
            }

            const text = pickText(el).slice(0, 90);
            const signature = `${tag}|${role}|${attrs.join('|')}|${text}|${Math.round(rect.left)}|${Math.round(rect.top)}|${Math.round(rect.width)}|${Math.round(rect.height)}`;
            if (seen.has(signature)) continue;
            seen.add(signature);

            const centerX = rect.left + rect.width / 2;
            let bucket = 'center';
            if (centerX < viewportWidth * 0.34) {
                bucket = 'left';
            } else if (centerX > viewportWidth * 0.66) {
                bucket = 'right';
            }

            candidates.push({
                el,
                tag,
                attrs,
                text,
                bucket,
                score: scoreCandidate(tag, role, typeAttr, text, rect),
                top: rect.top,
                left: rect.left,
            });
        }

        const targetCount = Math.min(maxElements, candidates.length);
        if (!targetCount) {
            return [];
        }

        const buckets = { right: [], center: [], left: [] };
        for (const candidate of candidates) {
            buckets[candidate.bucket].push(candidate);
        }

        for (const name of Object.keys(buckets)) {
            buckets[name].sort((a, b) => {
                if (b.score !== a.score) return b.score - a.score;
                if (a.top !== b.top) return a.top - b.top;
                return a.left - b.left;
            });
        }

        const selected = [];
        const cursors = { right: 0, center: 0, left: 0 };
        const bucketOrder = ['right', 'center', 'left'];

        while (selected.length < targetCount) {
            let progressed = false;
            for (const bucketName of bucketOrder) {
                const cursor = cursors[bucketName];
                const bucketItems = buckets[bucketName];
                if (cursor >= bucketItems.length) continue;
                selected.push(bucketItems[cursor]);
                cursors[bucketName] = cursor + 1;
                progressed = true;
                if (selected.length >= targetCount) break;
            }
            if (!progressed) break;
        }

        const elements = [];
        let nextId = 1;
        for (const candidate of selected) {
            if (annotateDom) {
                candidate.el.setAttribute(idAttr, String(nextId));
            }

            elements.push({
                index: nextId,
                tag: candidate.tag,
                attrs: candidate.attrs,
                text: candidate.text,
            });
            nextId += 1;
        }

        return elements;
    }
    """

    elements = await page.evaluate(
        snapshot_js,
        [
            max(10, min(max_elements, 200)),
            bool(annotate_dom),
            SOM_ID_ATTR,
        ],
    )
    context = await get_current_page_context(config)
    return {
        "url": context["url"],
        "title": context["title"],
        "elements": elements,
    }


async def _inject_som_overlay(page) -> int:
    """Draw temporary numbered overlays for currently annotated interactive elements."""
    overlay_js = r"""
    ([idAttr, overlayId, prevOutlineAttr]) => {
        const staleOverlay = document.getElementById(overlayId);
        if (staleOverlay) staleOverlay.remove();
        document.querySelectorAll('.claw-som-label').forEach((node) => node.remove());

        const elements = Array.from(document.querySelectorAll(`[${idAttr}]`));
        if (!elements.length) return 0;

        const root = document.createElement('div');
        root.id = overlayId;
        root.setAttribute('aria-hidden', 'true');
        root.style.position = 'fixed';
        root.style.left = '0';
        root.style.top = '0';
        root.style.width = '100vw';
        root.style.height = '100vh';
        root.style.pointerEvents = 'none';
        root.style.zIndex = '2147483647';
        (document.body || document.documentElement).appendChild(root);

        let drawn = 0;
        for (const el of elements) {
            const rect = el.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0) continue;
            if (rect.bottom < 0 || rect.top > window.innerHeight || rect.right < 0 || rect.left > window.innerWidth) {
                continue;
            }

            const id = el.getAttribute(idAttr);
            if (!id) continue;

            if (!el.hasAttribute(prevOutlineAttr)) {
                el.setAttribute(prevOutlineAttr, el.style.outline || '');
            }
            el.style.outline = '2px solid #ff3b30';
            el.style.outlineOffset = '1px';

            const label = document.createElement('div');
            label.className = 'claw-som-label';
            label.textContent = id;
            label.style.position = 'fixed';
            label.style.left = `${Math.max(0, Math.round(rect.left))}px`;
            label.style.top = `${Math.max(0, Math.round(rect.top - 14))}px`;
            label.style.background = '#ff3b30';
            label.style.color = '#ffffff';
            label.style.fontFamily = 'monospace';
            label.style.fontSize = '11px';
            label.style.fontWeight = '700';
            label.style.lineHeight = '1';
            label.style.padding = '2px 4px';
            label.style.borderRadius = '3px';
            label.style.boxShadow = '0 1px 2px rgba(0, 0, 0, 0.35)';
            label.style.pointerEvents = 'none';
            label.style.zIndex = '2147483647';
            root.appendChild(label);

            drawn += 1;
        }

        return drawn;
    }
    """

    return int(
        await page.evaluate(
            overlay_js,
            [SOM_ID_ATTR, SOM_OVERLAY_ID, SOM_PREV_OUTLINE_ATTR],
        )
    )


async def _cleanup_som_overlay(page) -> None:
    """Remove temporary Set-of-Marks overlays and attributes from the DOM."""
    cleanup_js = r"""
    ([idAttr, overlayId, prevOutlineAttr]) => {
        const overlay = document.getElementById(overlayId);
        if (overlay) overlay.remove();
        document.querySelectorAll('.claw-som-label').forEach((node) => node.remove());

        document.querySelectorAll(`[${idAttr}]`).forEach((el) => {
            if (el.hasAttribute(prevOutlineAttr)) {
                el.style.outline = el.getAttribute(prevOutlineAttr) || '';
                el.removeAttribute(prevOutlineAttr);
            } else {
                el.style.outline = '';
            }
            el.style.outlineOffset = '';
            el.removeAttribute(idAttr);
        });
    }
    """

    await page.evaluate(
        cleanup_js,
        [SOM_ID_ATTR, SOM_OVERLAY_ID, SOM_PREV_OUTLINE_ATTR],
    )


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
    _page_listener_registered = False

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
            
            USER_DATA_DIR = "./data/browser_profile"
            os.makedirs(USER_DATA_DIR, exist_ok=True)
            
            # This launches Chrome and saves all cookies/logins to the folder
            cls._browser = await cls._playwright.chromium.launch_persistent_context(
                user_data_dir=USER_DATA_DIR,
                headless=is_headless,
                channel="chrome", # Optional: uses your actual installed Chrome browser
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                # Anti-detection flags to allow Google SSO logins
                ignore_default_args=["--enable-automation"],
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            )
            # Auto-handle dialogs (accept by default) unless explicitly managed
            cls._browser.on("dialog", lambda dialog: asyncio.create_task(dialog.accept()))
            if not cls._page_listener_registered:
                cls._browser.on("page", lambda page: asyncio.create_task(cls._handle_new_page_event(page)))
                cls._page_listener_registered = True
            

            # Start GC sweep
            if cls._gc_task is None:
                cls._gc_task = asyncio.create_task(cls._gc_loop())

    @classmethod
    def _find_entry_by_page(cls, target_page):
        for entry in cls._contexts.values():
            for page in entry.get("pages", []):
                if page is target_page:
                    return entry
        return None

    @classmethod
    async def _handle_new_page_event(cls, new_page):
        """Auto-focus new tabs opened from an existing tracked page."""
        try:
            opener_page = await new_page.opener()
        except Exception as e:
            logger.debug("Failed to resolve opener for new page: %s", e)
            opener_page = None

        # Ignore pages without an opener to avoid cross-thread mis-assignment.
        if opener_page is None:
            return

        try:
            await new_page.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception:
            # Some pages never reach this state quickly; still track/focus them.
            pass

        async with cls._lock:
            entry = cls._find_entry_by_page(opener_page)
            if not entry:
                return

            pages = entry.get("pages", [])
            if not any(page is new_page for page in pages):
                pages.append(new_page)

            for idx, page in enumerate(pages):
                if page is new_page:
                    entry["active_page_idx"] = idx
                    break

            entry["last_accessed"] = time.time()
            entry["just_opened_new_tab"] = True

    @classmethod
    async def pop_new_tab_notice(cls, thread_id: str) -> bool:
        """Return and reset per-thread new-tab notice flag."""
        async with cls._lock:
            entry = cls._contexts.get(thread_id)
            if not entry:
                return False
            opened = bool(entry.get("just_opened_new_tab"))
            entry["just_opened_new_tab"] = False
            return opened

    @classmethod
    async def get_page(cls, thread_id: str):
        """Return the active Playwright Page for a thread, creating context if needed."""
        async with cls._lock:
            await cls._ensure_browser()

            if thread_id in cls._contexts:
                entry = cls._contexts[thread_id]
                entry["last_accessed"] = time.time()
                return entry["pages"][entry["active_page_idx"]]

            # Create a new page within the shared persistent context
            # The first page is created automatically by launch_persistent_context
            if not getattr(cls, "_default_page_used", False) and cls._browser.pages:
                page = cls._browser.pages[0]
                cls._default_page_used = True
            else:
                page = await cls._browser.new_page()

            cls._contexts[thread_id] = {
                "context": cls._browser,
                "pages": [page],
                "active_page_idx": 0,
                "last_accessed": time.time(),
                "just_opened_new_tab": False,
            }
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
                    for p in entry["pages"]:
                        await p.close()
                except Exception as e:
                    logger.warning(f"Error closing pages {thread_id}: {e}")

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
        cls._default_page_used = False
        cls._page_listener_registered = False

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
                    await cls.close_context(tid)
        except asyncio.CancelledError:
            pass

def _get_thread_id(config) -> str:
    return get_thread_id(config, default="default")


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
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await _wait_for_page_settle(page, include_networkidle=True, settle_seconds=0.15)
        title = await page.title()
        return f"Action successful: navigated to {url}. Current page: {title} ({page.url})"
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
        text_content = smart_truncate(text_content, max_chars=12000)
        return f"## Current Page: {title}\n**URL**: {page.url}\n\n{text_content}"
    except Exception as e:
        return f"Failed to extract text: {type(e).__name__} - {str(e)}"


@tool
async def browser_screenshot(
    config: RunnableConfig = None,
) -> list:
    """
    Take a compressed screenshot of the current page and return a multimodal payload.
    Uses JPEG quality compression to reduce token and latency overhead.
    """
    thread_id = _get_thread_id(config)
    page = None
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        if page.url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        await _wait_for_page_settle(page, include_networkidle=True, settle_seconds=0.25)

        marker_count = await page.evaluate(
            "(idAttr) => document.querySelectorAll(`[${idAttr}]`).length",
            SOM_ID_ATTR,
        )
        if int(marker_count) == 0:
            await get_interactive_element_index(config=config, max_elements=80, annotate_dom=True)

        await _inject_som_overlay(page)

        # Save to disk
        timestamp = int(time.time())
        filename = f"{thread_id}_{timestamp}.jpg"
        filepath = SCREENSHOT_DIR / filename
        screenshot_bytes = await page.screenshot(type="jpeg", quality=50, full_page=False)
        filepath.write_bytes(screenshot_bytes)

        # Return multimodal content block for LLM vision
        b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        return [
            {"type": "text", "text": f"Set-of-Marks screenshot of {page.url} (saved to {filepath})."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
    except Exception as e:
        logger.error(f"❌ browser_screenshot failed: {e}")
        return f"Failed to take screenshot: {type(e).__name__} - {str(e)}"
    finally:
        if page is not None:
            try:
                await _cleanup_som_overlay(page)
            except Exception as cleanup_error:
                logger.debug("Failed to cleanup Set-of-Marks overlay: %s", cleanup_error)


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
        return f"Action successful: scrolled {direction} by {pixels}px."
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
    max_elements: int = Field(
        80,
        description="Maximum number of visible interactive elements to include (10-200).",
    ),
    config: RunnableConfig = None,
) -> str:
    """
    Capture a structured snapshot of interactive elements on the current page.
    Default output is a compressed, indexed map of visible interactables.
    """
    try:
        page = await BrowserSessionManager.get_page(_get_thread_id(config))
        if page.url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        await _wait_for_page_settle(page, include_networkidle=True, settle_seconds=0.2)

        snapshot = await get_interactive_element_index(config=config, max_elements=max_elements)
        elements = snapshot.get("elements", [])
        if not elements:
            return "No interactive elements found on this page."

        lines: list[str] = []
        for item in elements:
            idx = item.get("index", "?")
            tag = str(item.get("tag", "element"))
            attrs = item.get("attrs", []) or []
            attrs_str = f" {' '.join(attrs)}" if attrs else ""
            text = str(item.get("text", "")).strip()
            if tag == "input":
                line = f"[{idx}] <{tag}{attrs_str}>"
            else:
                line = f"[{idx}] <{tag}{attrs_str}>{text}</{tag}>" if text else f"[{idx}] <{tag}{attrs_str}></{tag}>"
            lines.append(line)

        element_map = smart_truncate("\n".join(lines), max_chars=5000)
        return (
            f"URL: {snapshot.get('url', page.url)}\n"
            f"Title: {snapshot.get('title', '(unknown)')}\n\n"
            f"Interactive Elements:\n{element_map}"
        )
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
                except Exception:
                    title = "(closed)"
                lines.append(f"{marker}[{i}] {title} — {p.url}")
            return "## Open Tabs\n" + "\n".join(lines)

        elif action == "new":
            new_page = await entry["context"].new_page()
            if not any(p is new_page for p in pages):
                pages.append(new_page)
            entry["active_page_idx"] = next(i for i, p in enumerate(pages) if p is new_page)
            entry["just_opened_new_tab"] = False
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
            entry["active_page_idx"] = max(0, idx - 1)
            return f"✅ Closed tab [{idx}]. Active tab is now [{entry['active_page_idx']}]."

        return f"❌ Unknown action '{action}'. Use: list, new, switch, close."
    except Exception as e:
        return f"Tab management failed: {type(e).__name__} - {str(e)}"


@tool
async def browser_close_current_tab(
    config: RunnableConfig = None,
) -> str:
    """
    Close the active tab and return focus to the previous tab in the stack.
    """
    thread_id = _get_thread_id(config)
    try:
        entry = await BrowserSessionManager.get_entry(thread_id)
        pages = entry["pages"]

        if len(pages) <= 1:
            return "Error: Cannot close the only open tab."

        idx = entry["active_page_idx"]
        current_page = pages.pop(idx)
        await current_page.close()

        entry["active_page_idx"] = max(0, idx - 1)
        entry["just_opened_new_tab"] = False
        active_page = pages[entry["active_page_idx"]]
        title = await active_page.title()
        return (
            "Action successful: Closed current tab. "
            f"Focus returned to previous page ({title})."
        )
    except Exception as e:
        return f"Failed to close current tab: {type(e).__name__} - {str(e)}"


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
    try:
        entry = await BrowserSessionManager.get_entry(thread_id)
        page = entry["pages"][entry["active_page_idx"]]
        if page.url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        click_kwargs = {"timeout": 5000}
        if double_click:
            click_kwargs["click_count"] = 2

        summary = None

        async def _click_css() -> str:
            await page.click(selector, **click_kwargs)
            await _wait_for_page_settle(page, include_networkidle=True, settle_seconds=0.25)
            return f"clicked `{selector}`."

        async def _click_text() -> str:
            locator = page.get_by_text(selector, exact=False).first
            await locator.click(**click_kwargs)
            await _wait_for_page_settle(page, include_networkidle=True, settle_seconds=0.25)
            return f"clicked text '{selector}'."

        async def _click_role(role: str) -> str:
            locator = page.get_by_role(role, name=selector).first
            await locator.click(**click_kwargs)
            await _wait_for_page_settle(page, include_networkidle=True, settle_seconds=0.25)
            return f"clicked {role} '{selector}'."

        strategies = [
            ("css", _click_css),
            ("text", _click_text),
            ("role:link", lambda: _click_role("link")),
            ("role:button", lambda: _click_role("button")),
            ("role:menuitem", lambda: _click_role("menuitem")),
        ]

        for strategy_name, strategy in strategies:
            try:
                summary = await strategy()
                break
            except Exception as e:
                _log_strategy_failure("browser_click", strategy_name, selector, e)

        if not summary:
            return f"❌ Could not find element matching '{selector}'."

        # Allow context-level "page" event to register and focus the new tab.
        await asyncio.sleep(1.0)
        if await BrowserSessionManager.pop_new_tab_notice(thread_id):
            return (
                f"Action successful: {summary} "
                "[SYSTEM NOTICE: A new tab opened and was automatically focused.]"
            )

        return f"Action successful: {summary}"
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
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        if page.url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        filled = False
        strategies = [
            ("css", lambda: page.fill(selector, text, timeout=5000)),
            ("placeholder", lambda: page.get_by_placeholder(selector, exact=False).first.fill(text, timeout=5000)),
            ("label", lambda: page.get_by_label(selector, exact=False).first.fill(text, timeout=5000)),
        ]

        for strategy_name, strategy in strategies:
            try:
                await strategy()
                filled = True
                break
            except Exception as e:
                _log_strategy_failure("browser_type", strategy_name, selector, e)

        if not filled:
            return f"❌ Could not find input matching '{selector}'."

        summary = f"typed into `{selector}`"
        if submit:
            await page.keyboard.press("Enter")
            await _wait_for_page_settle(page, include_networkidle=True, settle_seconds=0.25)
            summary += " and submitted"

        return f"Action successful: {summary}."
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
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        if page.url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."
        result = await page.evaluate(script)
        result_preview = str(result) if result is not None else "(no return value)"
        if len(result_preview) > 180:
            result_preview = result_preview[:180] + "..."
        return f"Action successful: JavaScript executed. Result preview: {result_preview}"
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
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        if page.url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        summary = None
        strategies = [
            (
                "label",
                lambda: page.select_option(selector, label=value, timeout=5000),
                f"✅ Selected '{value}' from `{selector}`.",
            ),
            (
                "value",
                lambda: page.select_option(selector, value=value, timeout=5000),
                f"✅ Selected value='{value}' from `{selector}`.",
            ),
        ]

        for strategy_name, strategy, strategy_summary in strategies:
            try:
                await strategy()
                summary = strategy_summary
                break
            except Exception as e:
                _log_strategy_failure("browser_select_option", strategy_name, selector, e)

        if not summary:
            return f"❌ Could not select '{value}' in `{selector}`. Verify the selector and available options."

        await _wait_for_page_settle(page, include_networkidle=True, settle_seconds=0.2)

        return f"Action successful: {summary}"
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
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        await page.keyboard.press(key)
        await asyncio.sleep(0.3)
        summary = f"Action successful: pressed '{key}'."
        return summary
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
    try:
        page = await BrowserSessionManager.get_page(thread_id)
        if page.url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        summary = None
        strategies = [
            (
                "css",
                lambda: page.hover(selector, timeout=5000),
                f"✅ Hovering over `{selector}`.",
            ),
            (
                "text",
                lambda: page.get_by_text(selector, exact=False).first.hover(timeout=5000),
                f"✅ Hovering over text '{selector}'.",
            ),
        ]

        for strategy_name, strategy, strategy_summary in strategies:
            try:
                await strategy()
                summary = strategy_summary
                break
            except Exception as e:
                _log_strategy_failure("browser_hover", strategy_name, selector, e)

        if not summary:
            return f"❌ Could not find element matching '{selector}'."

        return f"Action successful: {summary}"
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
# Tool Registry — 17 tools total
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
    "browser_close_current_tab": browser_close_current_tab,
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
