"""
core/llm.py — Shared LLM initialization helper.

Used by agent.py and subagents.py to avoid duplicating the
model-agnostic init + fallback pattern.
"""

import logging
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "google_genai/gemini-3-flash-preview"


def init_agent_llm(active_model: str | None = None) -> BaseChatModel:
    """
    Initialize a chat model using the model-agnostic pattern.

    Args:
        active_model: Model string like "google_genai/gemini-3-flash-preview"
                      or "anthropic/claude-3-5-sonnet-20241022".
                      Falls back to the default if None or if loading fails.

    Returns:
        A BaseChatModel instance ready for use.
    """
    from langchain.chat_models import init_chat_model

    model_string = active_model or _DEFAULT_MODEL

    provider, actual_model = (
        model_string.split("/", 1) if "/" in model_string
        else ("google_genai", model_string)
    )

    try:
        llm = init_chat_model(actual_model, model_provider=provider)
        logger.info(f"  -> LLM loaded: {model_string}")
        return llm
    except Exception as e:
        logger.error(f"  -> Failed to load '{model_string}', falling back to default. Error: {e}")
        return init_chat_model("gemini-3-flash-preview", model_provider="google_genai")


def extract_response(state_values: dict, fallback: str = "Done!") -> str:
    """
    Extract the final text response from graph state values.

    Handles both simple string content and multimodal list content.
    Used by worker.py daemons and background_tools.py sub-agent runner.
    """
    messages = state_values.get("messages", [])
    if not messages:
        return fallback

    last_msg = messages[-1]
    if not hasattr(last_msg, "content"):
        return fallback

    content = last_msg.content
    if isinstance(content, str) and content:
        return content
    elif isinstance(content, list):
        texts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and "text" in item
        ]
        if texts:
            return "\n".join(texts)

    return fallback
