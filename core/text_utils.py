"""Shared text utilities for truncation and chunking."""


def smart_truncate(
    text: str,
    max_chars: int,
    suffix: str = "\n\n... [Content truncated]",
) -> str:
    """Truncate text at a newline boundary when possible."""
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_newline = truncated.rfind("\n")
    if last_newline > int(max_chars * 0.8):
        truncated = truncated[:last_newline]

    return truncated + suffix


def chunk_text(text: str, chunk_size: int) -> list[str]:
    """Split a string into fixed-size chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def get_thread_id(config, default: str = "default") -> str:
    """Extract thread_id from RunnableConfig-like payloads."""
    if not config:
        return default
    configurable = config.get("configurable", {})
    return configurable.get("thread_id", default)
