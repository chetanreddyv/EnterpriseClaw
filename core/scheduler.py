"""
core/scheduler.py — System Scheduler Service (Nanobot/OpenClaw approach)

The scheduler is responsible for:
1. Loading persisted cron jobs from data/cron/jobs.json
2. Computing next run times using croniter
3. Executing isolated workers when schedules trigger
4. Persisting run history and state
5. Handling retries and concurrency limits

The LLM does NOT need to understand scheduling. It just calls
schedule_background_task() to register a job. The backend handles the rest.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict

try:
    from croniter import croniter
except ImportError:
    croniter = None

logger = logging.getLogger(__name__)

JOBS_FILE = Path("data/cron/jobs.json")
JOBS_RUNS_DIR = Path("data/cron/runs")
RUN_HISTORY_MAX_RECORDS = 200
MAX_RESULT_SUMMARY_CHARS = 1000


class SchedulerJob:
    """In-memory representation of a persisted cron job."""
    
    def __init__(self, data: dict):
        self.id = data.get("id", str(uuid.uuid4())[:8])
        self.name = data.get("name", "Unnamed Job")
        self.enabled = data.get("enabled", True)
        
        # Schedule config
        schedule = data.get("schedule", {})
        self.schedule_kind = schedule.get("kind", "cron")  # at, every, cron
        self.schedule_value = schedule.get("schedule_value") or schedule.get("expr", "")
        self.schedule_tz = schedule.get("tz")
        self.schedule_every_ms = schedule.get("everyMs")
        self.schedule_at_ms = schedule.get("atMs")
        
        # Payload (what to execute)
        payload = data.get("payload", {})
        self.payload_kind = payload.get("kind", "system_event")  # system_event, agent_turn
        self.payload_objective = payload.get("message", payload.get("objective", ""))
        self.deliver_mode = payload.get("deliver_mode", "silent")  # announce, silent
        self.target = payload.get("target", "isolated")  # main, isolated, session:<id>
        self.channel = payload.get("channel")
        # Explicit skill binding — bypasses semantic search; e.g. "job-finder"
        self.skill_name = payload.get("skill_name")
        
        # State tracking
        state = data.get("state", {})
        self.next_run_at_ms = state.get("nextRunAtMs")
        self.last_run_at_ms = state.get("lastRunAtMs")
        self.last_status = state.get("lastStatus", "pending")
        self.last_error = state.get("lastError")
        self.retry_count = state.get("retryCount", 0)
        
        # Lifecycle
        self.created_at_ms = data.get("createdAtMs", int(datetime.now(timezone.utc).timestamp() * 1000))
        self.delete_after_run = data.get("deleteAfterRun", False)
    
    def to_dict(self) -> dict:
        """Serialize job back to dict for persistence."""
        return {
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
                "skill_name": self.skill_name,
            },
            "state": {
                "nextRunAtMs": self.next_run_at_ms,
                "lastRunAtMs": self.last_run_at_ms,
                "lastStatus": self.last_status,
                "lastError": self.last_error,
                "retryCount": self.retry_count,
            },
            "createdAtMs": self.created_at_ms,
            "deleteAfterRun": self.delete_after_run,
        }


class SystemScheduler:
    """
    Lightweight async scheduler that:
    - Loads jobs from data/cron/jobs.json
    - Computes next run times
    - Fires workers in isolated, stateful sessions (Option B)
    - Persists state
    """
    
    def __init__(self):
        self.jobs: Dict[str, SchedulerJob] = {}
        self.running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self.max_concurrent_jobs = 4
        self.active_jobs = 0
        self._in_flight_job_ids: set[str] = set()
        self._lifecycle_lock = asyncio.Lock()
        # Persistent SQLite checkpointer — shared across all cron sessions
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
            
            # Expect dict format with "jobs" key
            jobs_list = data.get("jobs", []) if isinstance(data, dict) else []

            for job_data in jobs_list:
                job = SchedulerJob(job_data)
                self.jobs[job.id] = job
                self._calculate_next_run(job)
                logger.info(f"SchedulerService: Loaded job '{job.name}' (id={job.id})")
            
            logger.info(f"SchedulerService: Loaded {len(self.jobs)} jobs from {JOBS_FILE}")
        except Exception as e:
            logger.error(f"SchedulerService: Failed to load jobs.json: {e}")
    
    def _calculate_next_run(self, job: SchedulerJob) -> None:
        """Calculate the next run time for a job based on its schedule."""
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        if job.schedule_kind == "at":
            # One-shot at specific time
            job.next_run_at_ms = job.schedule_at_ms
        
        elif job.schedule_kind == "every":
            # Recurring interval in milliseconds
            if job.last_run_at_ms:
                job.next_run_at_ms = job.last_run_at_ms + job.schedule_every_ms
            else:
                job.next_run_at_ms = now_ms + job.schedule_every_ms
        
        elif job.schedule_kind == "cron":
            # Standard cron expression
            if not croniter:
                logger.error("croniter not installed; cannot parse cron expressions")
                job.next_run_at_ms = None
                return
            
            try:
                # Convert current time to job's timezone before passing to croniter
                if job.schedule_tz and job.schedule_tz not in ("UTC", "utc"):
                    try:
                        from zoneinfo import ZoneInfo
                        now_local = datetime.now(ZoneInfo(job.schedule_tz))
                    except Exception as e:
                        logger.warning(
                            "SchedulerService: Invalid timezone '%s' for job %s; falling back to UTC (%s)",
                            job.schedule_tz,
                            job.id,
                            e,
                        )
                        now_local = datetime.now(timezone.utc)
                else:
                    now_local = datetime.now(timezone.utc)
                
                cron = croniter(job.schedule_value, now_local)
                next_run = cron.get_next(datetime)
                
                # Ensure timezone info is attached
                if next_run.tzinfo is None:
                    next_run = next_run.replace(tzinfo=timezone.utc)
                
                job.next_run_at_ms = int(next_run.timestamp() * 1000)
            except Exception as e:
                logger.error(f"SchedulerService: Failed to parse cron '{job.schedule_value}': {e}")
                job.next_run_at_ms = None
    
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
    ) -> str:
        """Add or update a scheduled job. Returns job ID."""
        # Normalize schedule_type
        if schedule_type not in ("at", "every", "cron"):
            raise ValueError(f"Invalid schedule_type: {schedule_type}")
        
        job_id = str(uuid.uuid4())[:8]
        job = SchedulerJob({
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
                "skill_name": skill_name,
            },
        })
        
        self._calculate_next_run(job)
        self.jobs[job_id] = job
        self._persist_jobs()
        logger.info(f"SchedulerService: Added job '{name}' (id={job_id})")
        return job_id
    
    def cancel_job(self, job_id: str) -> bool:
        """Remove a job."""
        if job_id in self.jobs:
            del self.jobs[job_id]
            self._persist_jobs()
            logger.info(f"SchedulerService: Cancelled job {job_id}")
            return True
        return False
    
    def list_jobs(self) -> list[dict]:
        """Return list of all jobs with current state."""
        return [job.to_dict() for job in self.jobs.values()]

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
            "version": 1,
            "jobs": [job.to_dict() for job in self.jobs.values()],
        }
        try:
            self._persist_json_atomic(JOBS_FILE, data)
        except Exception as e:
            logger.error(f"SchedulerService: Failed to persist jobs: {e}")
    
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
            "executionMode": execution_mode,
            "skillName": job.skill_name,
            "startedAtMs": started_at_ms,
            "finishedAtMs": finished_at_ms,
            "durationMs": duration_ms,
            "resultSummary": summary,
        }
        
        try:
            with open(run_file, "a") as f:
                f.write(json.dumps(record) + "\n")
            self._trim_run_history(run_file)
        except Exception as e:
            logger.error(f"SchedulerService: Failed to write run record for {job.id}: {e}")
    
    async def _execute_job(self, job: SchedulerJob, claimed: bool = False) -> None:
        """Execute a single job by invoking a Worker graph in isolated context."""
        if not job.enabled:
            return

        if not claimed:
            if job.id in self._in_flight_job_ids:
                logger.info("SchedulerService: Job %s is already running; skipping duplicate trigger", job.id)
                return
            self._in_flight_job_ids.add(job.id)
        
        # Prevent concurrent execution of the same job
        if self.active_jobs >= self.max_concurrent_jobs:
            logger.warning(f"SchedulerService: Concurrency limit reached; deferring {job.id}")
            self._in_flight_job_ids.discard(job.id)
            return
        
        self.active_jobs += 1
        run_id = uuid.uuid4().hex[:12]
        started_at_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        session_id = f"cron_{job.id}"
        
        try:
            # Compute isolated session ID
            if job.target.startswith("session:"):
                session_id = job.target[len("session:"):]
            elif job.target == "main":
                # For main target, use a system thread (could be user's thread if known)
                session_id = "system_cron_thread"
            
            logger.info(f"SchedulerService: Executing job '{job.name}' (id={job.id}) in session {session_id}")
            await self._write_run_record(
                job,
                status="started",
                run_id=run_id,
                session_id=session_id,
                started_at_ms=started_at_ms,
                execution_mode="cron",
            )
            
            # Import here to avoid circular dependency
            from core.graphs.worker import build_worker_graph
            
            # If an explicit skill is declared, pre-resolve its tools directly
            # This is the reliable path that bypasses semantic search
            pre_skill_prompts = ""
            pre_skill_tools: list[str] = []
            prebound_skills: list[str] = []
            if job.skill_name:
                try:
                    from memory.retrieval import memory_retrieval
                    # Search by skill name directly (short, exact query)
                    prompts, matched, tool_map = await memory_retrieval.get_relevant_skills_with_metadata(job.skill_name)
                    if matched:
                        pre_skill_prompts = prompts
                        prebound_skills = [s for s in matched if isinstance(s, str) and s.strip()]
                        for sname in matched:
                            pre_skill_tools.extend(tool_map.get(sname, []))
                        # Deduplicate while preserving order
                        pre_skill_tools = list(dict.fromkeys(pre_skill_tools))
                        logger.info(f"SchedulerService: Explicit skill '{job.skill_name}' resolved tools: {pre_skill_tools}")
                    else:
                        logger.warning(f"SchedulerService: Explicit skill '{job.skill_name}' not found in index; falling back to semantic match")
                except Exception as e:
                    logger.warning(f"SchedulerService: Skill pre-resolution failed: {e}")
            
            # Build worker graph with shared checkpointer (enables stateful runs)
            worker_graph = build_worker_graph(checkpointer=self._checkpointer)
            
            # Config with thread_id gives this job its own persistent history
            config = {
                "configurable": {
                    "thread_id": session_id,
                    "platform": "system",
                    "execution_mode": "cron",
                },
                "recursion_limit": 50,
            }
            
            result = await worker_graph.ainvoke(
                {
                    "objective": job.payload_objective,
                    "skill_prompts": pre_skill_prompts,
                    "active_skills": prebound_skills,
                    "active_skill_tools": pre_skill_tools,
                    "max_steps": 15,
                    "step_count": 0,
                    "status": "running",
                    "messages": [],
                    "observation": "Starting isolated background task.",
                    "result_summary": "",
                    "tool_failure_count": 0,
                    "_retry": False,
                    "_formatted_prompt": [],
                    "active_model": "",
                    # Cron policy: auto-approve only skill-declared tools.
                    "approved_tools": pre_skill_tools,
                    "execution_mode": "cron",
                    "execution_source": f"scheduler:{job.id}",
                },
                config=config,
            )
            
            # Record run outcome with normalized lifecycle state.
            finished_at_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            duration_ms = max(0, finished_at_ms - started_at_ms)
            raw_status = str(result.get("status", "completed")).strip().lower()
            normalized_status = raw_status or "completed"
            normalized_error = None
            if normalized_status == "running":
                normalized_status = "interrupted"
                normalized_error = "worker returned status=running without completion"

            job.last_run_at_ms = finished_at_ms
            job.last_status = normalized_status
            job.last_error = normalized_error

            if normalized_status in {"completed", "escalated"}:
                job.retry_count = 0
            else:
                job.retry_count = job.retry_count + 1

            await self._write_run_record(
                job,
                status=normalized_status,
                error=normalized_error,
                run_id=run_id,
                session_id=session_id,
                started_at_ms=started_at_ms,
                finished_at_ms=finished_at_ms,
                duration_ms=duration_ms,
                execution_mode="cron",
                result_summary=result.get("result_summary"),
            )
            
            # If job should be deleted after run, do so
            if job.delete_after_run:
                self.cancel_job(job.id)
            else:
                # Calculate next run and persist
                self._calculate_next_run(job)
                self._persist_jobs()
            
            logger.info(
                "SchedulerService: Job '%s' completed with status=%s durationMs=%d",
                job.name,
                job.last_status,
                duration_ms,
            )
        
        except Exception as e:
            logger.error(f"SchedulerService: Job execution failed: {e}", exc_info=True)
            finished_at_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            duration_ms = max(0, finished_at_ms - started_at_ms)
            job.last_status = "error"
            job.last_error = str(e)
            job.last_run_at_ms = finished_at_ms
            job.retry_count = job.retry_count + 1
            await self._write_run_record(
                job,
                status="error",
                error=str(e),
                run_id=run_id,
                session_id=session_id,
                started_at_ms=started_at_ms,
                finished_at_ms=finished_at_ms,
                duration_ms=duration_ms,
                execution_mode="cron",
            )
            
            # Persist updated state
            self._persist_jobs()
        
        finally:
            self.active_jobs -= 1
            self._in_flight_job_ids.discard(job.id)
    
    async def start(self) -> None:
        """Start the background scheduler loop."""
        async with self._lifecycle_lock:
            if self.running:
                return

            # Open a persistent SQLite checkpointer only if none was injected.
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
                    logger.warning(f"SchedulerService: Could not open checkpointer ({e}); cron sessions will be stateless")
                    self._checkpointer = None
            else:
                logger.info("SchedulerService: Using injected checkpointer for cron sessions")

            self.running = True
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            logger.info("SchedulerService: Scheduler started")
    
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

            self._in_flight_job_ids.clear()
            logger.info("SchedulerService: Scheduler stopped")

            # Close checkpointer only when scheduler owns it.
            if self._owns_checkpointer and self._cp_ctx is not None:
                try:
                    await self._cp_ctx.__aexit__(None, None, None)
                except Exception:
                    pass
                self._cp_ctx = None
                self._checkpointer = None
                self._owns_checkpointer = False
    
    async def _scheduler_loop(self) -> None:
        """Main scheduler loop: check jobs every 10 seconds, execute if due."""
        while self.running:
            try:
                now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                
                # Find jobs that are due to run
                for job in list(self.jobs.values()):
                    if not job.enabled or job.next_run_at_ms is None:
                        continue

                    if job.id in self._in_flight_job_ids:
                        continue
                    
                    if job.next_run_at_ms <= now_ms:
                        # Fire the job (non-blocking)
                        self._in_flight_job_ids.add(job.id)
                        asyncio.create_task(self._execute_job(job, claimed=True))
                
                # Sleep before next check
                await asyncio.sleep(10)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"SchedulerService: Loop error: {e}", exc_info=True)
                await asyncio.sleep(10)


# Singleton instance
_scheduler = SystemScheduler()


async def get_scheduler() -> SystemScheduler:
    """Get the global scheduler instance."""
    return _scheduler


async def initialize_scheduler(checkpointer=None) -> None:
    """Initialize the scheduler at app startup."""
    if _scheduler.running:
        return

    # Ensure skill index is loaded so cron jobs can match skills without the app context
    try:
        from memory.retrieval import memory_retrieval
        await memory_retrieval.initialize()
        logger.info("SchedulerService: Memory/skill index initialized")
    except Exception as e:
        logger.warning(f"SchedulerService: Memory initialization failed ({e}); skill matching may be unavailable")
    
    if checkpointer is not None:
        _scheduler.set_checkpointer(checkpointer, owned=False)

    _scheduler.load_jobs()
    await _scheduler.start()


async def shutdown_scheduler() -> None:
    """Shutdown the scheduler at app teardown."""
    await _scheduler.stop()
