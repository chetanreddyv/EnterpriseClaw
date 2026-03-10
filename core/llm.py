"""
core/llm.py — Shared LLM initialization helper.
model-agnostic init + fallback pattern.
"""

import logging
from langchain_core.language_models import BaseChatModel
from config.settings import settings

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "lmstudio/qwen/qwen3.5-9b"

def init_agent_llm(model_string: str = None) -> BaseChatModel:
    """
    Initializes a LangChain chat model based on a provider/model string.
    Expected format: "provider/model_name"
    Example: "google_genai/gemini-3-flash-preview" or "lmstudio/qwen/qwen3.5-9b"
    """
    from langchain.chat_models import init_chat_model
    
    if not model_string:
        # User requested switch to Qwen as the default local testing model
        model_string = "lmstudio/qwen/qwen3.5-9b"

    # The maxsplit=1 ensures that lmstudio/qwen/qwen3.5-9b 
    # splits into "lmstudio" and "qwen/qwen3.5-9b"
    provider, actual_model = (
        model_string.split("/", 1) if "/" in model_string
        else ("google_genai", model_string)
    )

    # Initialize configuration kwargs that will be passed to LangChain
    init_kwargs = {}

    # Routing logic: Intercept custom providers and map to standard ones
    if provider in ("lmstudio", "lm_studio"):
        # LM Studio is a drop-in replacement for the OpenAI API format
        provider = "openai" 
        init_kwargs["api_key"] = settings.lm_studio_api_key
        init_kwargs["base_url"] = settings.lm_studio_base_url
        
    elif provider == "openai":
        init_kwargs["api_key"] = settings.openai_api_key
        
    elif provider == "google_genai":
        init_kwargs["api_key"] = settings.google_api_key

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
