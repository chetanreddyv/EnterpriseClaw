"""
core/llm.py — Shared LLM initialization helper.
model-agnostic init + fallback pattern.
"""

import logging
from collections import deque
from langchain_core.language_models import BaseChatModel
from config.settings import settings

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "openai/gpt-5.4-mini"


def is_llm_connection_error(exc: Exception) -> bool:
    """Return True when an exception chain indicates transport-level model outage."""
    seen = set()
    queue = deque([exc])

    while queue:
        current = queue.popleft()
        if not current:
            continue
        if id(current) in seen:
            continue
        seen.add(id(current))

        if isinstance(
            current,
            (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
            ),
        ):
            return True

        name = type(current).__name__
        if name in {"APIConnectionError", "APITimeoutError"}:
            return True

        queue.append(getattr(current, "__cause__", None))
        queue.append(getattr(current, "__context__", None))

    return False

def init_agent_llm(model_string: str = None) -> BaseChatModel:
    """
    Initializes a LangChain chat model based on a provider/model string.
    Expected format: "provider/model_name"
    Example: "google_genai/gemini-3-flash-preview" or "lmstudio/qwen/qwen3.5-9b"
    """
    from langchain.chat_models import init_chat_model
    
    if not model_string:
        # Default back to google_genai if nothing is set (cost-effective)
        model_string = "openai/gpt-5.4-mini"

    # Robust splitting: provider/model
    if "/" in model_string:
        provider, actual_model = model_string.split("/", 1)
    else:
        # Intelligently guess provider if missing
        actual_model = model_string
        if model_string.startswith("gpt"):
            provider = "openai"
        elif model_string.startswith("claude"):
            provider = "anthropic"
        elif model_string.startswith("gemini"):
            provider = "google_genai"
        else:
            provider = "google_genai" # Default fallback

    # Initialize configuration kwargs that will be passed to LangChain
    init_kwargs = {}

    # Routing logic: Intercept custom providers and map to standard ones
    if provider in ("lmstudio", "lm_studio"):
        # LM Studio is a drop-in replacement for the OpenAI API format
        provider = "openai" 
        init_kwargs["api_key"] = settings.lm_studio_api_key
        init_kwargs["base_url"] = settings.lm_studio_base_url
        init_kwargs["timeout"] = 180.0
        
    elif provider == "openai":
        init_kwargs["api_key"] = settings.openai_api_key
        
    elif provider == "google_genai":
        init_kwargs["api_key"] = settings.google_api_key

    logger.info(f"LLM: Resolving '{model_string}' -> [provider={provider}, model={actual_model}]")
    if "base_url" in init_kwargs:
        logger.info(f"LLM: Custom Base URL: {init_kwargs['base_url']}")

    try:
        llm = init_chat_model(actual_model, model_provider=provider, **init_kwargs)
        return llm
    except Exception as e:
        logger.error(f"  -> Failed to load '{model_string}', falling back to default. Error: {e}")
        # Robust fallback
        return init_chat_model(
            "gemini-3-flash-preview", 
            model_provider="google_genai", 
            api_key=settings.google_api_key
        )


def extract_response(state_values: dict, fallback: str = "Done!") -> str:
    """
    Extract the final text response from graph state values.

    Handles both simple string content and multimodal list content.
    Used by worker.py daemons and background_tools.py sub-agent runner.
    """
    messages = state_values.get("messages", [])
    if not messages:
        return fallback

    # Check if we hit a hard failure in the graph
    if state_values.get("tool_failure_count", 0) >= 3 and not state_values.get("_retry", True):
         return "❌ I encountered an internal error while trying to process that request. Please try asking again."

    # Scan backwards to find the last true AI response
    from langchain_core.messages import AIMessage
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content:
                # Some local models pad the answer with newlines or "bot: "
                return content.strip()
            elif isinstance(content, list):
                texts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and "text" in item
                ]
                if texts:
                    return "\n".join(texts).strip()

    return fallback
