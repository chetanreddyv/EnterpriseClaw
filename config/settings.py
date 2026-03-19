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

    # ── WhatsApp (Bridge) ─────────────────────────────────────
    whatsapp_enabled: bool = Field(default=False, description="Enable WhatsApp channel")
    whatsapp_bridge_url: str = Field(
        default="ws://localhost:3001",
        description="WebSocket URL for the Node.js WhatsApp bridge",
    )
    whatsapp_bridge_token: str = Field(
        default="",
        description="Optional auth token for WhatsApp bridge",
    )
    whatsapp_allow_from: str = Field(
        default="",
        description="Comma-separated WhatsApp allowlist (phone/LID/JID)",
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

    # ── HITL (Human-In-The-Loop) ─────────────────────────────
    hitl_enabled: bool = Field(
        default=True,
        description="Global toggle for HITL approval gates. When False, all tools are auto-approved.",
    )

    # ── Browser Observations ──────────────────────────────────
    enable_multimodal_observation: bool = Field(
        default=True,
        description="Enable multimodal observations (browser screenshots) in Worker environment state.",
    )

    # ── Default Model ────────────────────────────────────────
    default_model: str = Field(
        default="openai/gpt-5.4-mini",
        description="Default LLM model string (provider/model). Used when no model is explicitly set.",
    )

    # ── Worker Limits ────────────────────────────────────────
    worker_max_steps: int = Field(
        default=15,
        description="Maximum action steps a Worker can take before giving up.",
    )
    worker_max_observation_chars: int = Field(
        default=50_000,
        description="Max characters for environment observation payloads (~12,500 tokens).",
    )
    worker_max_skill_prompt_chars: int = Field(
        default=10_000,
        description="Max characters for skill prompt context injected into Worker.",
    )

    # ── Supervisor Limits ────────────────────────────────────
    supervisor_token_budget: int = Field(
        default=6000,
        description="Token budget for supervisor message history trimming.",
    )
    supervisor_content_truncation: int = Field(
        default=3000,
        description="Max characters for AI/Tool message content before truncation.",
    )

    # ── Scheduler ────────────────────────────────────────────
    scheduler_max_concurrent_jobs: int = Field(
        default=4,
        description="Maximum number of cron jobs that can run concurrently.",
    )
    scheduler_heartbeat_seconds: int = Field(
        default=10,
        description="Seconds between scheduler heartbeat checks for due jobs.",
    )
    scheduler_run_history_max: int = Field(
        default=200,
        description="Maximum run history records to keep per job.",
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def allowed_chat_id_list(self) -> list[int]:
        """Parse comma-separated chat IDs into a list of ints."""
        if not self.allowed_chat_ids:
            return []
        return [int(cid.strip()) for cid in self.allowed_chat_ids.split(",") if cid.strip()]

    @property
    def whatsapp_allow_from_list(self) -> list[str]:
        """Parse comma-separated WhatsApp sender identifiers into a list."""
        if not self.whatsapp_allow_from:
            return []
        return [value.strip() for value in self.whatsapp_allow_from.split(",") if value.strip()]

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
