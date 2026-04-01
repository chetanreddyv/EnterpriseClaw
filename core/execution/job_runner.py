import logging
import uuid
from typing import Optional, TYPE_CHECKING

from config.settings import settings
from core.models import SchedulerJob, SkillResolutionError, _now_ms

if TYPE_CHECKING:
    from core.scheduler import SystemScheduler

logger = logging.getLogger(__name__)


async def _send_completion_notification(
    job: SchedulerJob,
    status: str,
    summary: Optional[str],
    error: Optional[str],
) -> None:
    if not settings.scheduler_delivery_enabled:
        return
    if job.deliver_mode not in {"reply", "announce"}:
        return
    if not job.channel:
        return

    from core.channel_manager import channel_manager

    status_label = status.upper()
    message_lines = [f"📬 Cron job '{job.name}' finished with status={status_label}."]
    if summary:
        message_lines.append(f"Summary: {str(summary).strip()}")
    if error:
        message_lines.append(f"Error: {error}")

    try:
        await channel_manager.send_message(
            platform=job.platform or "system",
            thread_id=str(job.channel),
            content="\n".join(message_lines),
        )
    except Exception as e:
        logger.error(
            "SchedulerService: Failed delivery for job %s platform=%s channel=%s: %s",
            job.id,
            job.platform,
            job.channel,
            e,
        )


async def execute_job(
    scheduler: "SystemScheduler",
    job: SchedulerJob,
    *,
    claimed: bool = False,
    planned_fire_at_ms: Optional[int] = None,
) -> None:
    """Execute a single job by invoking a Worker graph in isolated context."""
    if not job.enabled:
        return

    if not claimed:
        if job.id in scheduler._in_flight_job_ids:
            logger.info("SchedulerService: Job %s is already running; skipping duplicate trigger", job.id)
            return
        scheduler._in_flight_job_ids.add(job.id)

    if scheduler.active_jobs >= scheduler.max_concurrent_jobs:
        logger.warning("SchedulerService: Concurrency limit reached; deferring %s", job.id)
        scheduler._in_flight_job_ids.discard(job.id)
        return

    scheduler.active_jobs += 1
    run_id = uuid.uuid4().hex[:12]
    started_at_ms = _now_ms()
    session_id = f"cron_{job.id}"

    try:
        if job.target.startswith("session:"):
            session_id = job.target[len("session:") :]
        elif job.target == "main":
            session_id = "system_cron_thread"

        logger.info(
            "SchedulerService: Executing job '%s' (id=%s) session=%s plannedFireAtMs=%s",
            job.name,
            job.id,
            session_id,
            planned_fire_at_ms,
        )
        await scheduler._write_run_record(
            job,
            status="started",
            run_id=run_id,
            session_id=session_id,
            started_at_ms=started_at_ms,
            execution_mode="cron",
        )

        from core.graphs.worker import build_worker_graph

        pre_skill_prompts = ""
        pre_skill_tools: list[str] = []
        prebound_skills: list[str] = []
        if job.skill_name:
            from memory.retrieval import memory_retrieval

            if not memory_retrieval.is_ready():
                raise SkillResolutionError("skill_resolution_not_ready")

            try:
                prompts, matched, tool_map = await memory_retrieval.get_relevant_skills_with_metadata(
                    job.skill_name
                )
                if matched:
                    pre_skill_prompts = prompts
                    prebound_skills = [s for s in matched if isinstance(s, str) and s.strip()]
                    for sname in matched:
                        pre_skill_tools.extend(tool_map.get(sname, []))
                    pre_skill_tools = list(dict.fromkeys(pre_skill_tools))
                    logger.info(
                        "SchedulerService: Explicit skill '%s' resolved tools: %s",
                        job.skill_name,
                        pre_skill_tools,
                    )
                elif settings.allow_cron_skill_fallback:
                    logger.warning(
                        "SchedulerService: Explicit skill '%s' not found; fallback enabled",
                        job.skill_name,
                    )
                else:
                    raise SkillResolutionError(f"skill_resolution_miss:{job.skill_name}")
            except SkillResolutionError:
                raise
            except Exception as e:
                if settings.allow_cron_skill_fallback:
                    logger.warning(
                        "SchedulerService: Skill pre-resolution failed with fallback enabled: %s",
                        e,
                    )
                else:
                    raise SkillResolutionError(f"skill_resolution_error:{e}")

        worker_graph = build_worker_graph(checkpointer=scheduler._checkpointer)
        config = {
            "configurable": {
                "thread_id": session_id,
                "platform": "system",
                "execution_mode": "cron",
                "enable_multimodal_observation": settings.enable_multimodal_observation,
            },
            "recursion_limit": 50,
        }

        result = await worker_graph.ainvoke(
            {
                "objective": job.payload_objective,
                "skill_prompts": pre_skill_prompts,
                "active_skills": prebound_skills,
                "active_skill_tools": pre_skill_tools,
                "max_steps": settings.worker_max_steps,
                "step_count": 0,
                "status": "running",
                "messages": [],
                "environment_snapshot": "Starting isolated background task.",
                "result_summary": "",
                "tool_failure_count": 0,
                "_retry": False,
                "_formatted_prompt": [],
                "active_model": "",
                "approved_tools": pre_skill_tools,
                "execution_mode": "cron",
                "execution_source": f"scheduler:{job.id}",
            },
            config=config,
        )

        finished_at_ms = _now_ms()
        duration_ms = max(0, finished_at_ms - started_at_ms)
        raw_status = str(result.get("status", "completed")).strip().lower()
        normalized_status = raw_status or "completed"
        normalized_error = None
        reason_tag = None
        if normalized_status == "running":
            normalized_status = "interrupted"
            normalized_error = "worker returned status=running without completion"
            reason_tag = "worker_incomplete"

        summary = result.get("result_summary")
        job.last_run_at_ms = finished_at_ms
        job.last_status = normalized_status
        job.last_error = normalized_error

        retry_scheduled = False
        if normalized_status in {"completed", "escalated"}:
            job.retry_count = 0
            job.next_retry_at_ms = None
            job.retry_reason = None
            scheduler._calculate_next_run(
                job,
                anchor_ms=planned_fire_at_ms or job.next_run_at_ms,
                reason="post_success",
            )
        else:
            retry_scheduled = scheduler._schedule_retry(job, normalized_status, normalized_error, finished_at_ms)
            if not retry_scheduled:
                if settings.scheduler_retry_enabled and scheduler._is_transient_failure(normalized_status, normalized_error):
                    job.retry_count = min(job.retry_count + 1, job.retry_max_retries)
                job.next_retry_at_ms = None
                job.retry_reason = None
                scheduler._calculate_next_run(
                    job,
                    anchor_ms=planned_fire_at_ms or job.next_run_at_ms,
                    reason="post_failure_skip",
                )

        await scheduler._write_run_record(
            job,
            status=normalized_status,
            error=normalized_error,
            run_id=run_id,
            session_id=session_id,
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            duration_ms=duration_ms,
            execution_mode="cron",
            result_summary=summary,
            reason_tag=reason_tag,
        )

        await _send_completion_notification(
            job=job,
            status=normalized_status,
            summary=summary,
            error=normalized_error,
        )

        if job.delete_after_run and normalized_status in {"completed", "escalated"} and not retry_scheduled:
            scheduler.cancel_job(job.id)
        else:
            scheduler._persist_jobs()

        logger.info(
            "SchedulerService: Job '%s' completed status=%s durationMs=%d retryScheduled=%s",
            job.name,
            job.last_status,
            duration_ms,
            retry_scheduled,
        )

    except SkillResolutionError as e:
        finished_at_ms = _now_ms()
        duration_ms = max(0, finished_at_ms - started_at_ms)
        error_text = str(e)
        job.last_status = "failed"
        job.last_error = error_text
        job.last_run_at_ms = finished_at_ms

        retry_scheduled = scheduler._schedule_retry(job, "failed", error_text, finished_at_ms)
        if not retry_scheduled:
            scheduler._calculate_next_run(
                job,
                anchor_ms=planned_fire_at_ms or job.next_run_at_ms,
                reason="skill_resolution_failed",
            )

        await scheduler._write_run_record(
            job,
            status="failed",
            error=error_text,
            run_id=run_id,
            session_id=session_id,
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            duration_ms=duration_ms,
            execution_mode="cron",
            reason_tag="skill_resolution",
        )
        scheduler._persist_jobs()
        logger.error("SchedulerService: Deterministic skill resolution failure for job %s: %s", job.id, error_text)

    except Exception as e:
        logger.error("SchedulerService: Job execution failed: %s", e, exc_info=True)
        finished_at_ms = _now_ms()
        duration_ms = max(0, finished_at_ms - started_at_ms)
        error_text = str(e)
        job.last_status = "error"
        job.last_error = error_text
        job.last_run_at_ms = finished_at_ms

        retry_scheduled = scheduler._schedule_retry(job, "error", error_text, finished_at_ms)
        if not retry_scheduled:
            if settings.scheduler_retry_enabled and scheduler._is_transient_failure("error", error_text):
                job.retry_count = min(job.retry_count + 1, job.retry_max_retries)
            job.next_retry_at_ms = None
            job.retry_reason = None
            scheduler._calculate_next_run(
                job,
                anchor_ms=planned_fire_at_ms or job.next_run_at_ms,
                reason="exception_path",
            )

        await scheduler._write_run_record(
            job,
            status="error",
            error=error_text,
            run_id=run_id,
            session_id=session_id,
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            duration_ms=duration_ms,
            execution_mode="cron",
            reason_tag="exception",
        )

        await _send_completion_notification(
            job=job,
            status="error",
            summary=None,
            error=error_text,
        )
        scheduler._persist_jobs()

    finally:
        scheduler.active_jobs = max(0, scheduler.active_jobs - 1)
        scheduler._in_flight_job_ids.discard(job.id)
