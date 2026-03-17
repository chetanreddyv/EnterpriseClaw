"""
config/settings.py — Centralized configuration via Pydantic Settings.

All env vars are loaded from .env and validated at startup.
"""

import logging
from dotenv import load_dotenv

load_dotenv()

# Note: logging.basicConfig is called in app.py (the entrypoint)

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Telegram ──────────────────────────────────────────────
    telegram_bot_token: str = Field(default="", description="Telegram bot token from @BotFather")
    telegram_secret_token: str = Field(
        default="", description="Secret token for webhook verification"
    )
    allowed_chat_ids: str = Field(
        default="", description="Comma-separated list of allowed Telegram chat IDs"
    )

    # ── LLM Providers ─────────────────────────────────────────
    google_api_key: str = Field(default="", description="Google AI Studio API key")
    
    # NEW: LM Studio / OpenAI settings
    openai_api_key: str = Field(default="", description="OpenAI API Key (if using standard OpenAI)")
    claude_api_key: str = Field(default="", description="Anthropic Claude API key")
    lm_studio_base_url: str = Field(default="http://localhost:1234/v1", description="Local URL for LM Studio")
    lm_studio_api_key: str = Field(default="lm-studio", description="Mock API key for LM Studio")

    # ── Observability (optional) ──────────────────────────────
    langchain_tracing_v2: bool = Field(default=False)
    langsmith_api_key: Optional[str] = Field(default=None)


    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def allowed_chat_id_list(self) -> list[int]:
        """Parse comma-separated chat IDs into a list of ints."""
        if not self.allowed_chat_ids:
            return []
        return [int(cid.strip()) for cid in self.allowed_chat_ids.split(",") if cid.strip()]

    @property
    def needs_onboarding(self) -> bool:
        """True if critical keys are missing and onboarding should run."""
        return not self.google_api_key or not self.telegram_bot_token

    @property
    def is_configured(self) -> bool:
        """True if all critical keys are set."""
        return bool(self.google_api_key and self.telegram_bot_token)


# Singleton — import this across the app
settings = Settings()
