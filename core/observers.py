"""Environment observers for forced action-observation loops."""

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


def _coerce_image_block(screenshot_payload: Any) -> dict[str, Any] | None:
    """Extract an image_url block from screenshot tool payloads."""
    if isinstance(screenshot_payload, list):
        for block in screenshot_payload:
            if isinstance(block, dict) and block.get("type") == "image_url":
                return block
    return None


async def get_browser_environment_state(config: RunnableConfig = None) -> str | list[dict[str, Any]]:
    """Return compressed browser environment state with breadcrumbs."""
    from mcp_servers.browser_tools import (
        browser_get_text,
        browser_screenshot,
        get_current_page_context,
        get_interactive_element_index,
    )

    try:
        context = await get_current_page_context(config)
        multimodal_enabled = _is_multimodal_observation_enabled(config)
        indexed_state = await get_interactive_element_index(
            config=config,
            max_elements=80,
            annotate_dom=multimodal_enabled,
        )
        element_lines: list[str] = []
        for item in indexed_state.get("elements", []) or []:
            idx = item.get("index", "?")
            tag = str(item.get("tag", "element"))
            attrs = item.get("attrs", []) or []
            attrs_str = f" {' '.join(attrs)}" if attrs else ""
            text = str(item.get("text", "")).strip()
            if tag == "input":
                element_lines.append(f"[{idx}] <{tag}{attrs_str}>")
            else:
                element_lines.append(
                    f"[{idx}] <{tag}{attrs_str}>{text}</{tag}>" if text else f"[{idx}] <{tag}{attrs_str}></{tag}>"
                )

        element_map = "\n".join(element_lines) if element_lines else "(no visible interactive elements found)"

        text_observation = (
            f"URL: {context.get('url', '(unknown)')}\n"
            f"Title: {context.get('title', '(unknown)')}\n\n"
            f"Interactive Elements:\n{element_map}"
        )

        if multimodal_enabled:
            screenshot_payload = await browser_screenshot.ainvoke({}, config=config)
            image_block = _coerce_image_block(screenshot_payload)
            if image_block:
                return [
                    {"type": "text", "text": text_observation},
                    image_block,
                ]

        return text_observation
    except Exception as snapshot_error:
        logger.warning("Browser observer: snapshot failed: %s", snapshot_error)

    try:
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
