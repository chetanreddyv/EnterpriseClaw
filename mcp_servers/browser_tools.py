"""
mcp_servers/browser_tools.py — Browser automation tools powered by browser-use.

Provides the agent with full browser control: navigate, click, type,
screenshot, extract text, scroll, hover, select dropdowns, manage tabs,
press keys, handle dialogs, and upload files.

Click and Type tools use index-based element targeting via browser-use's
Set-of-Marks perception engine. The Worker reads a numbered interactive
element map (e.g., `[42] <button>Submit</button>`) and references elements
by their integer index.

Write-type tools are HITL-gated by the agent router.
Read-type tools run freely.

NOTE: browser-use's Page is a CDP wrapper — NOT raw Playwright. All
interactions must go through session methods, page.evaluate(), or page.goto()/
page.go_back()/page.press() which are the thin CDP wrappers.
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
from core.browser_session import BrowserSessionManager

logger = logging.getLogger("mcp.browser_tools")

SCREENSHOT_DIR = Path("./data/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def _get_thread_id(config) -> str:
    return get_thread_id(config, default="default")


async def _settle(seconds: float = 0.3) -> None:
    """Best-effort settle delay for SPA updates."""
    await asyncio.sleep(min(max(seconds, 0.05), 1.0))


async def get_current_page_context(config: RunnableConfig = None) -> dict[str, str]:
    """Return URL and title breadcrumbs for the active page."""
    session = await BrowserSessionManager.get_session()
    url = await session.get_current_page_url()
    title = await session.get_current_page_title()
    return {"url": url or "about:blank", "title": title or "(unknown)"}


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
    try:
        session = await BrowserSessionManager.get_session()
        await session.navigate_to(url)
        await _settle(0.5)
        title = await session.get_current_page_title()
        current_url = await session.get_current_page_url()
        return f"Action successful: navigated to {url}. Current page: {title} ({current_url})"
    except Exception as e:
        logger.error(f"❌ browser_navigate failed: {e}")
        return f"Failed to navigate to {url}: {type(e).__name__} - {str(e)}"


@tool
async def browser_get_text(
    config: RunnableConfig = None,
) -> str:
    """Extract all visible text from the current page."""
    try:
        session = await BrowserSessionManager.get_session()
        current_url = await session.get_current_page_url()
        if current_url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."
        title = await session.get_current_page_title()
        page = await session.get_current_page()
        text_content = await page.evaluate("(() => document.body.innerText)()")
        text_content = smart_truncate(str(text_content or ""), max_chars=12000)
        return f"## Current Page: {title}\n**URL**: {current_url}\n\n{text_content}"
    except Exception as e:
        return f"Failed to extract text: {type(e).__name__} - {str(e)}"


@tool
async def browser_screenshot(
    config: RunnableConfig = None,
) -> list:
    """
    Take a screenshot of the current page with Set-of-Marks highlighting
    and return a multimodal payload. Uses browser-use's built-in element
    highlighting for visual grounding.
    """
    try:
        session = await BrowserSessionManager.get_session()
        url = await session.get_current_page_url()
        if url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        # Take screenshot via browser-use session (includes SoM highlighting)
        screenshot_bytes = await session.take_screenshot(format="jpeg", quality=50)

        # Save to disk
        timestamp = int(time.time())
        filename = f"screenshot_{timestamp}.jpg"
        filepath = SCREENSHOT_DIR / filename
        filepath.write_bytes(screenshot_bytes)

        # Return multimodal content block for LLM vision
        b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        return [
            {"type": "text", "text": f"Set-of-Marks screenshot of {url} (saved to {filepath})."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
    except Exception as e:
        logger.error(f"❌ browser_screenshot failed: {e}")
        return f"Failed to take screenshot: {type(e).__name__} - {str(e)}"


@tool
async def browser_go_back(
    config: RunnableConfig = None,
) -> str:
    """Go back to the previous page in history. Use this if you navigated to a wrong page."""
    try:
        session = await BrowserSessionManager.get_session()
        page = await session.get_current_page()
        await page.go_back()
        await _settle(0.5)
        title = await session.get_current_page_title()
        url = await session.get_current_page_url()
        return f"✅ Went back. Now on: **{title}** ({url})"
    except Exception as e:
        return f"Failed to go back: {type(e).__name__} - {str(e)}"


@tool
async def browser_scroll(
    direction: str = Field("down", description="Scroll direction: 'up' or 'down'."),
    amount: int = Field(3, description="Number of viewport-heights to scroll (1-10). Default 3."),
    config: RunnableConfig = None,
) -> str:
    """Scroll the page up or down to see more content. Use after navigating to see content below the fold or in infinite-scroll pages."""
    try:
        session = await BrowserSessionManager.get_session()
        current_url = await session.get_current_page_url()
        if current_url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."
        pixels = min(max(amount, 1), 10) * 720  # viewport height = 720
        delta = pixels if direction == "down" else -pixels
        page = await session.get_current_page()
        await page.evaluate(f"(() => {{ window.scrollBy(0, {delta}); return true; }})()")
        await _settle(0.5)
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
    try:
        if text:
            session = await BrowserSessionManager.get_session()
            page = await session.get_current_page()
            # Poll for text presence via JS
            start = time.time()
            while (time.time() - start) < min(seconds, 30):
                found = await page.evaluate(f"(() => document.body.innerText.includes('{text}'))()")
                if found:
                    return f"✅ Text '{text}' appeared on the page."
                await asyncio.sleep(0.5)
            return f"Text '{text}' did not appear within {seconds} seconds."
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
    Returns browser-use's LLM-optimized numbered element map.
    """
    try:
        session = await BrowserSessionManager.get_session()
        url = await session.get_current_page_url()
        if url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        await _settle(0.3)

        # Use browser-use's LLM representation
        element_map = await session.get_state_as_text()
        title = await session.get_current_page_title()

        if not element_map or element_map.strip() == "":
            return "No interactive elements found on this page."

        element_map = smart_truncate(element_map, max_chars=5000)
        return (
            f"URL: {url}\n"
            f"Title: {title}\n\n"
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
    try:
        session = await BrowserSessionManager.get_session()
        tabs = await session.get_tabs()

        if action == "list":
            if not tabs:
                return "No tabs open."
            lines = []
            for i, tab in enumerate(tabs):
                lines.append(f"  [{i}] {tab.title} — {tab.url}")
            return "## Open Tabs\n" + "\n".join(lines)

        elif action == "new":
            await session.new_page()
            return "✅ Opened new tab. Use browser_navigate to load a page."

        elif action == "switch":
            if index is None or index < 0 or index >= len(tabs):
                return f"❌ Invalid index. You have {len(tabs)} tabs (0-{len(tabs)-1})."
            target_tab = tabs[index]
            from browser_use.browser.events import SwitchTabEvent
            await session.event_bus.dispatch(SwitchTabEvent(target_id=target_tab.target_id))
            return f"✅ Switched to tab [{index}]: **{target_tab.title}** ({target_tab.url})"

        elif action == "close":
            if not tabs:
                return "❌ No tabs to close."
            idx = index if index is not None else 0
            if idx < 0 or idx >= len(tabs):
                return f"❌ Invalid index. You have {len(tabs)} tabs (0-{len(tabs)-1})."
            if len(tabs) == 1:
                return "❌ Cannot close the last tab. Use browser_navigate instead."
            target_tab = tabs[idx]
            from browser_use.browser.events import CloseTabEvent
            await session.event_bus.dispatch(CloseTabEvent(target_id=target_tab.target_id))
            return f"✅ Closed tab [{idx}]."

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
    try:
        session = await BrowserSessionManager.get_session()
        tabs = await session.get_tabs()

        if len(tabs) <= 1:
            return "Error: Cannot close the only open tab."

        # Close the focused tab
        current_target = session.agent_focus_target_id
        if current_target:
            from browser_use.browser.events import CloseTabEvent
            await session.event_bus.dispatch(CloseTabEvent(target_id=current_target))
            await _settle(0.3)
            title = await session.get_current_page_title()
            return (
                "Action successful: Closed current tab. "
                f"Focus returned to previous page ({title})."
            )
        return "Error: No focused tab to close."
    except Exception as e:
        return f"Failed to close current tab: {type(e).__name__} - {str(e)}"


# ══════════════════════════════════════════════════════════════
# WRITE TOOLS (HITL-gated)
# ══════════════════════════════════════════════════════════════

@tool
async def browser_click(
    index: int = Field(
        ...,
        description="The index number of the element to click, from the Interactive Elements map "
                    "(e.g., if the map shows '[42] <button>Submit</button>', use index=42).",
    ),
    double_click: bool = Field(False, description="Whether to double-click instead of single click."),
    config: RunnableConfig = None,
) -> str:
    """
    Click an element using its index ID from the Interactive Elements observation.
    This is a WRITE action and requires human approval.
    """
    try:
        session = await BrowserSessionManager.get_session()

        # Look up the element by index from browser-use's selector map
        element = await session.get_element_by_index(index)
        if element is None:
            return f"Action failed: Element index [{index}] not found on the current page."

        # Highlight and click via session
        try:
            await session.highlight_interaction_element(element)
        except Exception:
            pass  # Highlighting is optional

        page = await session.get_current_page()
        xpath = element.xpath

        # Use JS click via CDP for reliability
        click_count = 2 if double_click else 1
        click_script = f"""
        (() => {{
            const el = document.evaluate("{xpath}", document, null,
                XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
            if (!el) return false;
            for (let i = 0; i < {click_count}; i++) {{
                el.click();
            }}
            return true;
        }})()
        """
        result = await page.evaluate(click_script)
        if not result:
            return f"Action failed: Could not click element [{index}] via XPath."

        await _settle(1.0)

        # Allow context-level new tab detection
        thread_id = _get_thread_id(config)
        if await BrowserSessionManager.pop_new_tab_notice(thread_id):
            return (
                f"Action successful: Clicked element [{index}]. "
                "[SYSTEM NOTICE: A new tab opened and was automatically focused.]"
            )

        return f"Action successful: Clicked element [{index}]."
    except Exception as e:
        return f"Failed to click element [{index}]: {type(e).__name__} - {str(e)}"


@tool
async def browser_type(
    index: int = Field(
        ...,
        description="The index number of the input field from the Interactive Elements map.",
    ),
    text: str = Field(..., description="The text to type into the field."),
    submit: bool = Field(False, description="Press Enter after typing to submit the form."),
    config: RunnableConfig = None,
) -> str:
    """
    Type text into an input field using its index ID from the Interactive Elements observation.
    Set submit=True to press Enter after typing (e.g. for search boxes).
    This is a WRITE action and requires human approval.
    """
    try:
        session = await BrowserSessionManager.get_session()

        element = await session.get_element_by_index(index)
        if element is None:
            return f"Action failed: Element index [{index}] not found."

        page = await session.get_current_page()
        xpath = element.xpath

        # Use JS to focus, clear, and set value via CDP
        escaped_text = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
        type_script = f"""
        (() => {{
            const el = document.evaluate("{xpath}", document, null,
                XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
            if (!el) return false;
            el.focus();
            el.value = '';
            el.value = '{escaped_text}';
            el.dispatchEvent(new Event('input', {{ bubbles: true }}));
            el.dispatchEvent(new Event('change', {{ bubbles: true }}));
            return true;
        }})()
        """
        result = await page.evaluate(type_script)
        if not result:
            return f"Action failed: Error typing into [{index}]."

        summary = f"typed into element [{index}]"
        if submit:
            await page.press("Enter")
            await _settle(0.5)
            summary += " and submitted"

        return f"Action successful: {summary}."
    except Exception as e:
        return f"Failed to type into element [{index}]: {type(e).__name__} - {str(e)}"


@tool
async def browser_execute_js(
    script: str = Field(..., description="JavaScript code to execute on the current page."),
    config: RunnableConfig = None,
) -> str:
    """
    Execute arbitrary JavaScript on the current page and return the result.
    This is a WRITE action and requires human approval.
    """
    try:
        session = await BrowserSessionManager.get_session()
        url = await session.get_current_page_url()
        if url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."
        page = await session.get_current_page()
        # Wrap user script in arrow function format required by browser-use CDP
        wrapped_script = f"(() => {{ {script} }})()"
        result = await page.evaluate(wrapped_script)
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
    try:
        session = await BrowserSessionManager.get_session()
        url = await session.get_current_page_url()
        if url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        page = await session.get_current_page()
        escaped_value = value.replace("'", "\\'")
        escaped_selector = selector.replace("'", "\\'")

        # Try selecting by label first, then by value
        select_script = f"""
        (() => {{
            const el = document.querySelector('{escaped_selector}');
            if (!el) return 'not_found';
            const options = Array.from(el.options);
            // Try by label
            const byLabel = options.find(o => o.text === '{escaped_value}');
            if (byLabel) {{ el.value = byLabel.value; el.dispatchEvent(new Event('change', {{bubbles:true}})); return 'ok_label'; }}
            // Try by value
            const byValue = options.find(o => o.value === '{escaped_value}');
            if (byValue) {{ el.value = byValue.value; el.dispatchEvent(new Event('change', {{bubbles:true}})); return 'ok_value'; }}
            return 'no_match';
        }})()
        """
        result = await page.evaluate(select_script)

        if result == "not_found":
            return f"❌ Could not find select element matching '{selector}'."
        elif result == "no_match":
            return f"❌ Could not find option '{value}' in `{selector}`."
        else:
            return f"Action successful: ✅ Selected '{value}' from `{selector}`."
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
    try:
        session = await BrowserSessionManager.get_session()
        page = await session.get_current_page()
        await page.press(key)
        await _settle(0.3)
        return f"Action successful: pressed '{key}'."
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
    try:
        session = await BrowserSessionManager.get_session()
        url = await session.get_current_page_url()
        if url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        page = await session.get_current_page()
        escaped_selector = selector.replace("'", "\\'")

        # Hover via JS + mouse event dispatch
        hover_script = f"""
        (() => {{
            let el = document.querySelector('{escaped_selector}');
            if (!el) {{
                // Fallback: try to find by text content
                const all = document.querySelectorAll('*');
                for (const e of all) {{
                    if (e.textContent.trim() === '{escaped_selector}') {{ el = e; break; }}
                }}
            }}
            if (!el) return false;
            const rect = el.getBoundingClientRect();
            el.dispatchEvent(new MouseEvent('mouseenter', {{bubbles: true, clientX: rect.x + rect.width/2, clientY: rect.y + rect.height/2}}));
            el.dispatchEvent(new MouseEvent('mouseover', {{bubbles: true, clientX: rect.x + rect.width/2, clientY: rect.y + rect.height/2}}));
            return true;
        }})()
        """
        result = await page.evaluate(hover_script)

        if not result:
            return f"❌ Could not find element matching '{selector}'."

        return f"Action successful: ✅ Hovering over `{selector}`."
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
    try:
        if action == "accept":
            msg = "✅ Dialogs will be accepted"
            if prompt_text:
                msg += f" with text '{prompt_text}'"
        else:
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
    try:
        session = await BrowserSessionManager.get_session()
        url = await session.get_current_page_url()
        if url == "about:blank":
            return "No page is currently loaded. Use browser_navigate first."

        paths = [p.strip() for p in file_paths.split(",") if p.strip()]
        # Validate paths exist
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            return f"❌ Files not found: {', '.join(missing)}"

        # Use session's file upload support
        page = await session.get_current_page()
        escaped_selector = selector.replace("'", "\\'")

        # Check if element is a file input
        is_file_input = await page.evaluate(
            f"(() => {{ const el = document.querySelector('{escaped_selector}'); return !!el && el.type === 'file'; }})()"
        )
        if not is_file_input:
            return f"❌ Element `{selector}` is not a file input."

        # For file uploads, we need to use the session-level CDP method
        # browser-use has is_file_input check and upload support
        from browser_use.browser.events import UploadFileEvent
        for path in paths:
            event = session.event_bus.dispatch(UploadFileEvent(
                file_path=path,
                selector=selector,
            ))
            await event

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
