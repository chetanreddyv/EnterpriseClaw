"""
core/scheduler.py - System Scheduler Service

The scheduler is responsible for:
1. Loading persisted cron jobs from data/cron/jobs.json
2. Computing next run times using croniter
3. Dispatching isolated workers when schedules trigger
4. Persisting run history and state
5. Managing retries, delivery hooks, and concurrency limits
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from config.settings import settings
from core.execution import execute_job
from core.models import (
    SCHEDULER_SCHEMA_VERSION,
    SchedulerJob,
    SkillResolutionError,
    _now_ms,
    _to_int,
)

try:
    from croniter import croniter
except ImportError:
    croniter = None

logger = logging.getLogger(__name__)

JOBS_FILE = Path("data/cron/jobs.json")
JOBS_RUNS_DIR = Path("data/cron/runs")
RUN_HISTORY_MAX_RECORDS = settings.scheduler_run_history_max
MAX_RESULT_SUMMARY_CHARS = 1000


class SystemScheduler:
    """
    Lightweight async scheduler that:
    - Loads jobs from data/cron/jobs.json
    - Computes next run times
    - Fires workers in isolated, stateful sessions
    - Persists state
    """

    def __init__(self):
        self.jobs: Dict[str, SchedulerJob] = {}
        self.running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self.max_concurrent_jobs = settings.scheduler_max_concurrent_jobs
        self.active_jobs = 0
        self._in_flight_job_ids: set[str] = set()
        self._execution_tasks: set[asyncio.Task] = set()
        self._lifecycle_lock = asyncio.Lock()
        # Persistent SQLite checkpointer - shared across all cron sessions.
        self._checkpointer = None
        self._cp_ctx = None
        self._owns_checkpointer = False

    def set_checkpointer(self, checkpointer, owned: bool = False) -> None:
        """Inject an external checkpointer (typically app lifespan-owned)."""
        self._checkpointer = checkpointer
        self._owns_checkpointer = owned

    def load_jobs(self) -> None:
        """Load all jobs from data/cron/jobs.json at startup."""
        self.jobs = {}
        if not JOBS_FILE.exists():
            logger.info("SchedulerService: No jobs.json found, starting fresh")
            return

        try:
            with open(JOBS_FILE, "r") as f:
                data = json.load(f)

            jobs_list = data.get("jobs", []) if isinstance(data, dict) else []
            for job_data in jobs_list:
                job = SchedulerJob(job_data)
                previous_next_run = job.next_run_at_ms
                self._calculate_next_run(job, reason="startup_load")
                if previous_next_run and previous_next_run <= _now_ms() and job.next_run_at_ms:
                    logger.info(
                        "SchedulerService: Skipping missed runs for %s; previousNext=%s next=%s",
                        job.id,
                        previous_next_run,
                        job.next_run_at_ms,
                    )
                self.jobs[job.id] = job
                logger.info("SchedulerService: Loaded job '%s' (id=%s)", job.name, job.id)

            logger.info("SchedulerService: Loaded %d jobs from %s", len(self.jobs), JOBS_FILE)
        except Exception as e:
            logger.error("SchedulerService: Failed to load jobs.json: %s", e)

    def _calculate_next_run(
        self,
        job: SchedulerJob,
        *,
        anchor_ms: Optional[int] = None,
        reason: str = "regular",
    ) -> None:
        """Calculate next fire time while preserving cadence and skip-missed policy."""
        now_ms = _now_ms()

        if job.schedule_kind == "at":
            if job.last_run_at_ms:
                job.next_run_at_ms = None
            else:
                job.next_run_at_ms = job.schedule_at_ms
            return

        if job.schedule_kind == "every":
            every_ms = _to_int(job.schedule_every_ms)
            if not every_ms or every_ms <= 0:
                logger.error("SchedulerService: Invalid every schedule for job %s", job.id)
                job.next_run_at_ms = None
                return

            if anchor_ms is not None:
                candidate = anchor_ms + every_ms
            elif job.last_run_at_ms:
                candidate = job.last_run_at_ms + every_ms
            elif job.next_run_at_ms:
                candidate = job.next_run_at_ms
            else:
                candidate = now_ms + every_ms

            if candidate <= now_ms:
                jumps = ((now_ms - candidate) // every_ms) + 1
                candidate += jumps * every_ms

            job.next_run_at_ms = candidate
            return

        if job.schedule_kind == "cron":
            if not croniter:
                logger.error("SchedulerService: croniter not installed; cannot parse cron expressions")
                job.next_run_at_ms = None
                return

            try:
                base_dt = datetime.fromtimestamp((anchor_ms or now_ms) / 1000, tz=timezone.utc)
                if job.schedule_tz and job.schedule_tz not in ("UTC", "utc"):
                    try:
                        from zoneinfo import ZoneInfo

                        base_dt = base_dt.astimezone(ZoneInfo(job.schedule_tz))
                    except Exception as e:
                        logger.warning(
                            "SchedulerService: Invalid timezone '%s' for job %s; falling back to UTC (%s)",
                            job.schedule_tz,
                            job.id,
                            e,
                        )

                cron = croniter(job.schedule_value, base_dt)
                next_run = cron.get_next(datetime)
                if next_run.tzinfo is None:
                    next_run = next_run.replace(tzinfo=timezone.utc)

                guard = 0
                while int(next_run.timestamp() * 1000) <= now_ms and guard < 200:
                    next_run = cron.get_next(datetime)
                    if next_run.tzinfo is None:
                        next_run = next_run.replace(tzinfo=timezone.utc)
                    guard += 1

                job.next_run_at_ms = int(next_run.timestamp() * 1000)
                return
            except Exception as e:
                logger.error("SchedulerService: Failed to parse cron '%s': %s", job.schedule_value, e)
                job.next_run_at_ms = None
                return

        logger.error("SchedulerService: Unknown schedule kind '%s' for job %s", job.schedule_kind, job.id)
        job.next_run_at_ms = None

    def _compute_retry_delay_seconds(self, job: SchedulerJob, attempt: int) -> int:
        raw_delay = job.retry_initial_backoff_seconds * (2 ** max(0, attempt - 1))
        capped = min(raw_delay, max(1, job.retry_max_backoff_seconds))
        jitter_ratio = max(0.0, settings.scheduler_retry_jitter_ratio)
        if jitter_ratio > 0:
            jitter_low = max(0.1, 1.0 - jitter_ratio)
            jitter_high = 1.0 + jitter_ratio
            capped = int(capped * random.uniform(jitter_low, jitter_high))
        return max(1, capped)

    def _is_transient_failure(self, status: str, error: Optional[str]) -> bool:
        if status == "interrupted":
            return True
        if status not in {"error", "failed"}:
            return False
        if not error:
            return True

        text = str(error).lower()
        transient_markers = (
            "timeout",
            "timed out",
            "rate limit",
            "429",
            "connection reset",
            "connection aborted",
            "temporar",
            "service unavailable",
            "gateway timeout",
            "503",
            "502",
            "network",
        )
        deterministic_markers = (
            "skill_resolution",
            "validation",
            "invalid",
            "missing required",
            "not found",
            "permission denied",
            "unauthorized",
        )
        if any(marker in text for marker in deterministic_markers):
            return False
        return any(marker in text for marker in transient_markers)

    def _schedule_retry(self, job: SchedulerJob, status: str, error: Optional[str], now_ms: int) -> bool:
        if not settings.scheduler_retry_enabled:
            return False
        if not self._is_transient_failure(status, error):
            return False
        if job.retry_count >= job.retry_max_retries:
            return False

        attempt = job.retry_count + 1
        delay_seconds = self._compute_retry_delay_seconds(job, attempt)
        job.retry_count = attempt
        job.next_retry_at_ms = now_ms + (delay_seconds * 1000)
        job.retry_reason = f"{status}: {error or 'unspecified'}"
        job.next_run_at_ms = job.next_retry_at_ms
        logger.warning(
            "SchedulerService: Scheduling retry for job %s attempt=%d delay=%ds reason=%s",
            job.id,
            attempt,
            delay_seconds,
            job.retry_reason,
        )
        return True

    def add_job(
        self,
        name: str,
        objective: str,
        schedule_type: str,
        schedule_value: str,
        deliver_mode: str = "silent",
        target: str = "isolated",
        tz: Optional[str] = None,
        skill_name: Optional[str] = None,
        channel: Optional[str] = None,
        platform: Optional[str] = None,
        retry_max_retries: Optional[int] = None,
        retry_initial_backoff_seconds: Optional[int] = None,
        retry_max_backoff_seconds: Optional[int] = None,
    ) -> str:
        """Add a scheduled job. Returns job ID."""
        if schedule_type not in ("at", "every", "cron"):
            raise ValueError(f"Invalid schedule_type: {schedule_type}")

        import uuid

        job_id = str(uuid.uuid4())[:8]
        job = SchedulerJob(
            {
                "schemaVersion": SCHEDULER_SCHEMA_VERSION,
                "id": job_id,
                "name": name,
                "enabled": True,
                "schedule": {
                    "kind": schedule_type,
                    "expr": schedule_value if schedule_type == "cron" else None,
                    "everyMs": int(schedule_value) if schedule_type == "every" else None,
                    "atMs": int(schedule_value) if schedule_type == "at" else None,
                    "tz": tz,
                },
                "payload": {
                    "kind": "system_event",
                    "message": objective,
                    "deliver_mode": deliver_mode,
                    "target": target,
                    "channel": channel,
                    "platform": platform or "system",
                    "skill_name": skill_name,
                },
                "state": {
                    "retryMaxRetries": _to_int(retry_max_retries, settings.scheduler_retry_max_retries),
                    "retryInitialBackoffSeconds": _to_int(
                        retry_initial_backoff_seconds,
                        settings.scheduler_retry_initial_backoff_seconds,
                    ),
                    "retryMaxBackoffSeconds": _to_int(
                        retry_max_backoff_seconds,
                        settings.scheduler_retry_max_backoff_seconds,
                    ),
                },
            }
        )

        self._calculate_next_run(job, reason="new_job")
        self.jobs[job_id] = job
        self._persist_jobs()
        logger.info("SchedulerService: Added job '%s' (id=%s)", name, job_id)
        return job_id

    def cancel_job(self, job_id: str) -> bool:
        """Remove a job."""
        if job_id in self.jobs:
            del self.jobs[job_id]
            self._persist_jobs()
            logger.info("SchedulerService: Cancelled job %s", job_id)
            return True
        return False

    def list_jobs(self) -> list[dict]:
        """Return list of all jobs with current state."""
        return [job.to_dict() for job in self.jobs.values()]

    def get_health_snapshot(self) -> dict:
        """Return scheduler health data for diagnostics."""
        now_ms = _now_ms()
        due_jobs = [
            job
            for job in self.jobs.values()
            if job.enabled and job.next_run_at_ms is not None and job.next_run_at_ms <= now_ms
        ]
        oldest_due_lag_ms = 0
        if due_jobs:
            oldest_due_lag_ms = max(now_ms - min(job.next_run_at_ms for job in due_jobs), 0)

        retry_queued = [
            job.id
            for job in self.jobs.values()
            if job.next_retry_at_ms is not None and job.next_retry_at_ms >= now_ms
        ]

        return {
            "running": self.running,
            "jobCount": len(self.jobs),
            "activeJobs": self.active_jobs,
            "inFlightJobIds": sorted(self._in_flight_job_ids),
            "executionTaskCount": len(self._execution_tasks),
            "dueJobCount": len(due_jobs),
            "oldestDueLagMs": oldest_due_lag_ms,
            "retryQueuedJobIds": retry_queued,
        }

    def _persist_json_atomic(self, path: Path, payload: dict) -> None:
        """Persist JSON atomically to avoid partial writes on crash."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
        tmp_path.replace(path)

    def _trim_run_history(self, run_file: Path) -> None:
        """Keep only the most recent bounded run records per job."""
        try:
            if not run_file.exists():
                return
            with open(run_file, "r") as f:
                lines = f.readlines()
            if len(lines) <= RUN_HISTORY_MAX_RECORDS:
                return
            kept_lines = lines[-RUN_HISTORY_MAX_RECORDS:]
            with open(run_file, "w") as f:
                f.writelines(kept_lines)
        except Exception as e:
            logger.warning("SchedulerService: Failed trimming run history for %s: %s", run_file, e)

    def _persist_jobs(self) -> None:
        """Write current jobs to data/cron/jobs.json."""
        data = {
            "version": SCHEDULER_SCHEMA_VERSION,
            "jobs": [job.to_dict() for job in self.jobs.values()],
        }
        try:
            self._persist_json_atomic(JOBS_FILE, data)
        except Exception as e:
            logger.error("SchedulerService: Failed to persist jobs: %s", e)

    async def _write_run_record(
        self,
        job: SchedulerJob,
        status: str,
        error: Optional[str] = None,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        started_at_ms: Optional[int] = None,
        finished_at_ms: Optional[int] = None,
        duration_ms: Optional[int] = None,
        execution_mode: str = "cron",
        result_summary: Optional[str] = None,
        reason_tag: Optional[str] = None,
    ) -> None:
        """Append a structured run record to per-job JSONL history."""
        JOBS_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        run_file = JOBS_RUNS_DIR / f"{job.id}.jsonl"
        now_iso = datetime.now(timezone.utc).isoformat()

        summary = None
        if result_summary is not None:
            summary = str(result_summary).strip()
            if len(summary) > MAX_RESULT_SUMMARY_CHARS:
                summary = summary[:MAX_RESULT_SUMMARY_CHARS] + "...[truncated]"

        record = {
            "timestamp": now_iso,
            "runId": run_id,
            "jobId": job.id,
            "sessionId": session_id,
            "status": status,
            "error": error,
            "reasonTag": reason_tag,
            "executionMode": execution_mode,
            "skillName": job.skill_name,
            "startedAtMs": started_at_ms,
            "finishedAtMs": finished_at_ms,
            "durationMs": duration_ms,
            "resultSummary": summary,
            "retryCount": job.retry_count,
            "nextRetryAtMs": job.next_retry_at_ms,
        }

        try:
            with open(run_file, "a") as f:
                f.write(json.dumps(record) + "\n")
            self._trim_run_history(run_file)
        except Exception as e:
            logger.error("SchedulerService: Failed to write run record for %s: %s", job.id, e)

    async def _execute_job(
        self,
        job: SchedulerJob,
        *,
        claimed: bool = False,
        planned_fire_at_ms: Optional[int] = None,
    ) -> None:
        """Compatibility wrapper used by tests and direct callers."""
        await execute_job(self, job, claimed=claimed, planned_fire_at_ms=planned_fire_at_ms)

    async def start(self) -> None:
        """Start the background scheduler loop."""
        async with self._lifecycle_lock:
            if self.running:
                return

            if self._checkpointer is None:
                import os
                from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

                db_path = os.getenv("DB_PATH", "./data/checkpoints_v2.db")
                try:
                    self._cp_ctx = AsyncSqliteSaver.from_conn_string(db_path)
                    self._checkpointer = await self._cp_ctx.__aenter__()
                    self._owns_checkpointer = True
                    logger.info("SchedulerService: SQLite checkpointer opened for cron sessions")
                except Exception as e:
                    logger.warning(
                        "SchedulerService: Could not open checkpointer (%s); cron sessions will be stateless",
                        e,
                    )
                    self._checkpointer = None
            else:
                logger.info("SchedulerService: Using injected checkpointer for cron sessions")

            self.running = True
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            logger.info("SchedulerService: Scheduler started")

    def _on_execution_task_done(self, job_id: str, task: asyncio.Task) -> None:
        self._execution_tasks.discard(task)
        self._in_flight_job_ids.discard(job_id)
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error("SchedulerService: Background task crashed for job %s: %s", job_id, exc)

    async def stop(self) -> None:
        """Stop the background scheduler loop."""
        async with self._lifecycle_lock:
            self.running = False
            scheduler_task = self._scheduler_task
            self._scheduler_task = None

            if scheduler_task:
                scheduler_task.cancel()
                try:
                    await scheduler_task
                except asyncio.CancelledError:
                    pass

            pending = [task for task in self._execution_tasks if not task.done()]
            grace_seconds = max(0, settings.scheduler_shutdown_grace_seconds)
            if pending and grace_seconds > 0:
                done, not_done = await asyncio.wait(pending, timeout=grace_seconds)
                if not_done:
                    logger.warning(
                        "SchedulerService: %d cron tasks still running after %ds; cancelling",
                        len(not_done),
                        grace_seconds,
                    )
                    for task in not_done:
                        task.cancel()
                    await asyncio.gather(*not_done, return_exceptions=True)
                if done:
                    await asyncio.gather(*done, return_exceptions=True)

            self._execution_tasks.clear()
            self._in_flight_job_ids.clear()
            logger.info("SchedulerService: Scheduler stopped")

            if self._owns_checkpointer and self._cp_ctx is not None:
                try:
                    await self._cp_ctx.__aexit__(None, None, None)
                except Exception:
                    pass
                self._cp_ctx = None
                self._checkpointer = None
                self._owns_checkpointer = False

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop: check jobs every heartbeat and execute due jobs."""
        while self.running:
            try:
                now_ms = _now_ms()
                due_lags = []

                for job in list(self.jobs.values()):
                    if not job.enabled or job.next_run_at_ms is None:
                        continue
                    if job.id in self._in_flight_job_ids:
                        continue
                    if job.next_run_at_ms > now_ms:
                        continue

                    due_lags.append(max(0, now_ms - job.next_run_at_ms))
                    planned_fire_at_ms = job.next_run_at_ms
                    self._in_flight_job_ids.add(job.id)

                    try:
                        task = asyncio.create_task(
                            execute_job(
                                self,
                                job,
                                claimed=True,
                                planned_fire_at_ms=planned_fire_at_ms,
                            )
                        )
                    except Exception as e:
                        self._in_flight_job_ids.discard(job.id)
                        logger.error(
                            "SchedulerService: Failed creating execution task for job %s: %s",
                            job.id,
                            e,
                            exc_info=True,
                        )
                        continue

                    self._execution_tasks.add(task)
                    task.add_done_callback(lambda t, jid=job.id: self._on_execution_task_done(jid, t))

                if due_lags:
                    logger.info(
                        "SchedulerService: dueJobs=%d maxDueLagMs=%d inFlight=%d",
                        len(due_lags),
                        max(due_lags),
                        len(self._in_flight_job_ids),
                    )

                await asyncio.sleep(settings.scheduler_heartbeat_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("SchedulerService: Loop error: %s", e, exc_info=True)
                await asyncio.sleep(settings.scheduler_heartbeat_seconds)


_scheduler = SystemScheduler()


async def get_scheduler() -> SystemScheduler:
    """Get the global scheduler instance."""
    return _scheduler


async def initialize_scheduler(checkpointer=None) -> None:
    """Initialize the scheduler at app startup."""
    if _scheduler.running:
        return

    from memory.retrieval import memory_retrieval

    await memory_retrieval.initialize()
    if not memory_retrieval.is_ready():
        raise RuntimeError("Skill index not ready; cannot start scheduler")
    logger.info("SchedulerService: Memory/skill index initialized")

    if checkpointer is not None:
        _scheduler.set_checkpointer(checkpointer, owned=False)

    _scheduler.load_jobs()
    await _scheduler.start()


async def shutdown_scheduler() -> None:
    """Shutdown the scheduler at app teardown."""
    await _scheduler.stop()


__all__ = [
    "SystemScheduler",
    "SchedulerJob",
    "SkillResolutionError",
    "get_scheduler",
    "initialize_scheduler",
    "shutdown_scheduler",
    "_now_ms",
]
