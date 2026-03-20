"""Environment observers for forced action-observation loops."""

import base64
import logging
from typing import Any
from langchain_core.runnables import RunnableConfig

from core.text_utils import get_thread_id

logger = logging.getLogger(__name__)


def _is_multimodal_observation_enabled(config: RunnableConfig | None) -> bool:
    """Toggle multimodal observation payloads through config.configurable."""
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    raw_flag = configurable.get("enable_multimodal_observation", False)
    if isinstance(raw_flag, bool):
        return raw_flag
    if isinstance(raw_flag, str):
        return raw_flag.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw_flag)


async def get_browser_environment_state(config: RunnableConfig = None) -> str | list[dict[str, Any]]:
    """Return compressed browser environment state using browser-use's perception engine."""
    from core.browser_session import BrowserSessionManager

    try:
        session = await BrowserSessionManager.get_session()
        url = await session.get_current_page_url()
        title = await session.get_current_page_title()

        # Get the LLM-optimized DOM representation (numbered interactive elements)
        element_map = await BrowserSessionManager.get_state_text(
            settle_seconds=0.35,
            retries=2,
        )
        if not str(element_map or "").strip() or "empty dom tree" in str(element_map).lower():
            raise RuntimeError("Perception scan returned empty DOM tree")

        text_observation = (
            f"🌐 URL: {url}\n"
            f"🏷️ Title: {title}\n\n"
            f"🕸️ Interactive Elements:\n{element_map}"
        )

        # Handle multimodal vision (Set-of-Marks screenshot)
        multimodal_enabled = _is_multimodal_observation_enabled(config)
        if multimodal_enabled:
            try:
                screenshot_bytes = await session.take_screenshot(
                    format="jpeg", quality=50
                )
                b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
                return [
                    {"type": "text", "text": text_observation},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]
            except Exception as ss_error:
                logger.warning("Browser observer: screenshot failed: %s", ss_error)

        return text_observation
    except Exception as snapshot_error:
        logger.warning("Browser observer: snapshot failed: %s", snapshot_error)

    # Fallback: try raw text extraction
    try:
        from mcp_servers.browser_tools import browser_get_text
        fallback_text = await browser_get_text.ainvoke({}, config=config)
        return (
            "Warning: Interactive map failed. Raw text fallback follows.\n\n"
            f"{fallback_text}"
        )
    except Exception as text_error:
        logger.error("Browser observer: get_text fallback failed: %s", text_error)
        return "Browser environment state unavailable."


async def get_exec_environment_state(config: RunnableConfig = None) -> str:
    """Return the latest exec environment state for the active thread."""
    from mcp_servers.exec_tools import get_exec_environment_state_for_thread

    thread_id = get_thread_id(config, default="default")
    raw_state = get_exec_environment_state_for_thread(thread_id)
    return f"Terminal State:\n{raw_state}"
