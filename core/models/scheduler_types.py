import uuid
from datetime import datetime, timezone
from typing import Optional

from config.settings import settings

SCHEDULER_SCHEMA_VERSION = 2


class SkillResolutionError(RuntimeError):
    """Raised when explicit skill binding cannot be resolved deterministically."""


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _to_int(value, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


class SchedulerJob:
    """In-memory representation of a persisted cron job."""

    def __init__(self, data: dict):
        self.schema_version = _to_int(data.get("schemaVersion"), 1)
        self.id = data.get("id", str(uuid.uuid4())[:8])
        self.name = data.get("name", "Unnamed Job")
        self.enabled = bool(data.get("enabled", True))

        schedule = data.get("schedule", {})
        self.schedule_kind = schedule.get("kind", "cron")
        self.schedule_value = schedule.get("schedule_value") or schedule.get("expr", "")
        self.schedule_tz = schedule.get("tz")
        self.schedule_every_ms = _to_int(schedule.get("everyMs"))
        self.schedule_at_ms = _to_int(schedule.get("atMs"))

        payload = data.get("payload", {})
        self.payload_kind = payload.get("kind", "system_event")
        self.payload_objective = payload.get("message", payload.get("objective", ""))
        self.deliver_mode = payload.get("deliver_mode", "silent")
        self.target = payload.get("target", "isolated")
        self.channel = payload.get("channel")
        self.platform = payload.get("platform", "system")
        self.skill_name = payload.get("skill_name")

        state = data.get("state", {})
        self.next_run_at_ms = _to_int(state.get("nextRunAtMs"))
        self.last_run_at_ms = _to_int(state.get("lastRunAtMs"))
        self.last_status = state.get("lastStatus", "pending")
        self.last_error = state.get("lastError")
        self.retry_count = _to_int(state.get("retryCount"), 0) or 0
        self.next_retry_at_ms = _to_int(state.get("nextRetryAtMs"))
        self.retry_reason = state.get("retryReason")

        self.retry_max_retries = _to_int(
            state.get("retryMaxRetries"), settings.scheduler_retry_max_retries
        ) or settings.scheduler_retry_max_retries
        self.retry_initial_backoff_seconds = _to_int(
            state.get("retryInitialBackoffSeconds"), settings.scheduler_retry_initial_backoff_seconds
        ) or settings.scheduler_retry_initial_backoff_seconds
        self.retry_max_backoff_seconds = _to_int(
            state.get("retryMaxBackoffSeconds"), settings.scheduler_retry_max_backoff_seconds
        ) or settings.scheduler_retry_max_backoff_seconds

        self.created_at_ms = _to_int(data.get("createdAtMs"), _now_ms()) or _now_ms()
        self.delete_after_run = bool(data.get("deleteAfterRun", False))

    def to_dict(self) -> dict:
        return {
            "schemaVersion": SCHEDULER_SCHEMA_VERSION,
            "id": self.id,
            "name": self.name,
            "enabled": self.enabled,
            "schedule": {
                "kind": self.schedule_kind,
                "expr": self.schedule_value if self.schedule_kind == "cron" else None,
                "everyMs": self.schedule_every_ms,
                "atMs": self.schedule_at_ms,
                "tz": self.schedule_tz,
            },
            "payload": {
                "kind": self.payload_kind,
                "message": self.payload_objective,
                "deliver_mode": self.deliver_mode,
                "target": self.target,
                "channel": self.channel,
                "platform": self.platform,
                "skill_name": self.skill_name,
            },
            "state": {
                "nextRunAtMs": self.next_run_at_ms,
                "lastRunAtMs": self.last_run_at_ms,
                "lastStatus": self.last_status,
                "lastError": self.last_error,
                "retryCount": self.retry_count,
                "nextRetryAtMs": self.next_retry_at_ms,
                "retryReason": self.retry_reason,
                "retryMaxRetries": self.retry_max_retries,
                "retryInitialBackoffSeconds": self.retry_initial_backoff_seconds,
                "retryMaxBackoffSeconds": self.retry_max_backoff_seconds,
            },
            "createdAtMs": self.created_at_ms,
            "deleteAfterRun": self.delete_after_run,
        }
