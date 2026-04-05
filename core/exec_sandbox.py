"""
core/exec_sandbox.py — Hardened command execution with Docker sandboxing,
environment filtering, command blocklist, and persistent CWD management.

This is a CORE tool — always available to Workers without skill matching.
Security is enforced via:
  1. HITL approval gates (core/hitl.py — exec_command is NOT_ALLOWED tier)
  2. Environment variable filtering (API keys, secrets, tokens stripped)
  3. Command blocklist (regex patterns block destructive operations)
  4. Optional Docker sandbox (ephemeral containers with resource limits)
  5. CWD persistence per thread (no more losing directory context)
"""

import asyncio
import logging
import os
import re
import shutil
import signal
from datetime import datetime, timezone
from typing import Dict, Optional

from pydantic import Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import httpx

from core.text_utils import get_thread_id

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Per-Thread Exec State (CWD persistence + observer data)
# ═══════════════════════════════════════════════════════════════

_EXEC_STATE_BY_THREAD: Dict[str, dict] = {}


def _truncate_tail(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return "...[TRUNCATED]...\n" + text[-max_chars:]


def _get_persistent_cwd(thread_id: str) -> str:
    """Return the last known CWD for a thread, or os.getcwd() if none."""
    entry = _EXEC_STATE_BY_THREAD.get(thread_id)
    if entry and entry.get("cwd"):
        cwd = entry["cwd"]
        if os.path.isdir(cwd):
            return cwd
    return os.getcwd()


def _update_exec_state(
    thread_id: str, command: str, cwd: str, status: str, output: str = "",
    sandbox_mode: str = "host",
) -> None:
    """Update thread-scoped exec state, preserving CWD for next call."""
    existing = _EXEC_STATE_BY_THREAD.get(thread_id, {})
    _EXEC_STATE_BY_THREAD[thread_id] = {
        "cwd": cwd,
        "last_command": command,
        "status": status,
        "last_output": output,
        "sandbox_mode": sandbox_mode,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        # Preserve history of CWD changes for debugging
        "cwd_history": (existing.get("cwd_history") or [])[-9:] + [cwd],
    }


def get_exec_environment_state_for_thread(thread_id: str, tail_chars: int = 8000) -> str:
    """Return a compact terminal state snapshot for a specific thread.

    Used by the Worker observer loop (core/observers.py) to inject terminal
    context into the environment_snapshot each turn.
    """
    entry = _EXEC_STATE_BY_THREAD.get(thread_id)
    if not entry:
        return (
            "## Exec Environment State\n"
            f"- Working Directory: {os.getcwd()}\n"
            "- Last Command: (none)\n"
            "- Status: idle\n"
            "- Sandbox: not initialized\n\n"
            "No terminal output captured yet for this thread."
        )

    output = _truncate_tail(str(entry.get("last_output", "")), max_chars=tail_chars)
    output_block = output if output.strip() else "(no output captured)"
    return (
        "## Exec Environment State\n"
        f"- Working Directory: {entry.get('cwd', os.getcwd())}\n"
        f"- Last Command: {entry.get('last_command', '(none)')}\n"
        f"- Status: {entry.get('status', 'unknown')}\n"
        f"- Sandbox: {entry.get('sandbox_mode', 'host')}\n"
        f"- Updated At (UTC): {entry.get('updated_at', '(unknown)')}\n\n"
        f"### Last Output (tail)\n{output_block}"
    )


# ═══════════════════════════════════════════════════════════════
# Security: Environment Filtering
# ═══════════════════════════════════════════════════════════════

# Patterns that match sensitive environment variable names.
# Applied in BOTH host and docker modes.
_BLOCKED_ENV_PATTERNS = [
    re.compile(r".*API_KEY.*", re.IGNORECASE),
    re.compile(r".*SECRET.*", re.IGNORECASE),
    re.compile(r".*TOKEN.*", re.IGNORECASE),
    re.compile(r".*PASSWORD.*", re.IGNORECASE),
    re.compile(r"^DATABASE_URL$", re.IGNORECASE),
    re.compile(r"^DB_.*", re.IGNORECASE),
    re.compile(r"^AWS_.*", re.IGNORECASE),
    re.compile(r"^AZURE_.*", re.IGNORECASE),
    re.compile(r"^GCP_.*", re.IGNORECASE),
    re.compile(r"^LANGSMITH_.*", re.IGNORECASE),
    re.compile(r"^LOGFIRE_.*", re.IGNORECASE),
]


def _build_safe_env(extra_blocked: list[str] | None = None) -> dict[str, str]:
    """Build an environment dict with sensitive variables stripped."""
    patterns = list(_BLOCKED_ENV_PATTERNS)
    for raw in extra_blocked or []:
        raw = raw.strip()
        if raw:
            try:
                patterns.append(re.compile(raw, re.IGNORECASE))
            except re.error:
                logger.warning("ExecSandbox: Ignoring invalid blocked env pattern: %s", raw)

    safe: dict[str, str] = {}
    blocked_count = 0
    for key, value in os.environ.items():
        if any(p.match(key) for p in patterns):
            blocked_count += 1
            continue
        safe[key] = value

    if blocked_count:
        logger.debug("ExecSandbox: Filtered %d sensitive env vars from subprocess.", blocked_count)
    return safe


# ═══════════════════════════════════════════════════════════════
# Security: Command Blocklist
# ═══════════════════════════════════════════════════════════════

_BLOCKED_COMMAND_PATTERNS = [
    # Destructive filesystem operations
    (re.compile(r"rm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?/\s*$"), "Blocked: recursive delete of root filesystem"),
    (re.compile(r"rm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?/\s+"), "Blocked: recursive delete of root filesystem"),
    (re.compile(r"rm\s+-rf\s+/(?!\w)"), "Blocked: recursive delete of root filesystem"),
    (re.compile(r"rm\s+-rf\s+~\s*$"), "Blocked: recursive delete of home directory"),
    # Disk/partition destruction
    (re.compile(r"mkfs\b"), "Blocked: filesystem formatting command"),
    (re.compile(r"dd\s+if=/dev/"), "Blocked: raw disk write operation"),
    (re.compile(r">\s*/dev/sd[a-z]"), "Blocked: direct device write"),
    # Fork bombs
    (re.compile(r":\(\)\s*\{.*\}"), "Blocked: fork bomb pattern"),
    (re.compile(r"\.\s*\(\)\s*\{.*\|\s*\.\s*&"), "Blocked: fork bomb pattern"),
    # System control (unintentional)
    (re.compile(r"\bshutdown\b"), "Blocked: system shutdown command"),
    (re.compile(r"\breboot\b"), "Blocked: system reboot command"),
    (re.compile(r"\binit\s+0\b"), "Blocked: system halt command"),
    (re.compile(r"\bhalt\b"), "Blocked: system halt command"),
    # Privilege escalation
    (re.compile(r"\bchmod\s+[0-7]*777\s+/\s*$"), "Blocked: chmod 777 on root"),
    (re.compile(r"\bchown\s+-R\s+.*\s+/\s*$"), "Blocked: recursive chown on root"),
]


def _check_command_safety(command: str) -> tuple[bool, str]:
    """Validate command against blocklist. Returns (is_safe, reason)."""
    for pattern, reason in _BLOCKED_COMMAND_PATTERNS:
        if pattern.search(command):
            logger.warning("ExecSandbox: BLOCKED command '%s' — %s", command[:80], reason)
            return False, reason
    return True, ""


# ═══════════════════════════════════════════════════════════════
# CWD Tracking: Detect cd commands and update persistent CWD
# ═══════════════════════════════════════════════════════════════

_CD_PATTERN = re.compile(
    r"""
    (?:^|&&\s*|;\s*)       # start of command or chained
    cd\s+                  # cd followed by space
    ("[^"]+"|'[^']+'|\S+)  # quoted or unquoted path
    \s*(?:$|&&|;)          # end or chain
    """,
    re.VERBOSE,
)


def _detect_final_cwd(command: str, current_cwd: str) -> str:
    """Parse cd commands in pipeline to predict final CWD after execution.

    This is used to update persistent CWD state. The actual subprocess CWD
    is set via proc_kwargs["cwd"], so this is a best-effort prediction for
    the *next* call.
    """
    cwd = current_cwd
    for match in _CD_PATTERN.finditer(command):
        raw_dir = match.group(1).strip("\"'")
        if raw_dir.startswith("/"):
            candidate = raw_dir
        elif raw_dir == "~" or raw_dir.startswith("~/"):
            candidate = os.path.expanduser(raw_dir)
        elif raw_dir == "-":
            continue  # cd - requires shell state we can't track
        else:
            candidate = os.path.join(cwd, raw_dir)
        candidate = os.path.normpath(candidate)
        if os.path.isdir(candidate):
            cwd = candidate
    return cwd


# ═══════════════════════════════════════════════════════════════
# Execution Backends
# ═══════════════════════════════════════════════════════════════

async def _execute_host(
    command: str, cwd: str, timeout_seconds: int, safe_env: dict[str, str],
) -> tuple[int, str, str]:
    """Execute command directly on host via asyncio subprocess."""
    proc_kwargs = {
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "cwd": cwd,
        "env": safe_env,
    }
    if os.name == "posix":
        proc_kwargs["preexec_fn"] = os.setsid

    process = await asyncio.create_subprocess_shell(command, **proc_kwargs)

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        # Aggressively kill the entire process group to free resources
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        raise

    output = stdout.decode("utf-8", errors="replace").strip()
    error = stderr.decode("utf-8", errors="replace").strip()
    return process.returncode, output, error


async def _execute_docker(
    command: str, cwd: str, timeout_seconds: int,
    image: str, memory_limit: str, cpu_limit: float,
    network: str, extra_volumes: list[str],
) -> tuple[int, str, str]:
    """Execute command inside an ephemeral Docker container."""
    docker_bin = shutil.which("docker")
    if not docker_bin:
        raise RuntimeError(
            "Docker is not installed or not in PATH. "
            "Set EXEC_SANDBOX_MODE=host to use direct execution."
        )

    docker_cmd = [
        docker_bin, "run", "--rm",
        "--memory", memory_limit,
        f"--cpus={cpu_limit}",
        "--network", network,
        "--pids-limit", "256",
        "-v", f"{cwd}:/work:rw",
        "-w", "/work",
    ]

    # Mount extra volumes if configured
    for vol in extra_volumes:
        vol = vol.strip()
        if vol and ":" in vol:
            docker_cmd.extend(["-v", vol])

    docker_cmd.extend([image, "sh", "-c", command])

    proc_kwargs = {
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
    }
    if os.name == "posix":
        proc_kwargs["preexec_fn"] = os.setsid

    process = await asyncio.create_subprocess_exec(*docker_cmd, **proc_kwargs)

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds + 10,  # extra grace for container startup
        )
    except asyncio.TimeoutError:
        # Kill the docker container
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        raise

    output = stdout.decode("utf-8", errors="replace").strip()
    error = stderr.decode("utf-8", errors="replace").strip()
    return process.returncode, output, error


# ═══════════════════════════════════════════════════════════════
# Background Process Monitor
# ═══════════════════════════════════════════════════════════════

async def _monitor_background_process(
    process, command: str, thread_id: str, platform: str, cwd: str,
    sandbox_mode: str,
):
    """Waits for a background process to finish and notifies the main agent."""
    stdout, stderr = await process.communicate()
    output = stdout.decode("utf-8", errors="replace").strip()
    error = stderr.decode("utf-8", errors="replace").strip()
    combined_output = output
    if error:
        combined_output = f"{combined_output}\n\nSTDERR:\n{error}" if combined_output else f"STDERR:\n{error}"

    final_status = "completed" if process.returncode == 0 else f"failed ({process.returncode})"
    _update_exec_state(thread_id, command, cwd, final_status, combined_output, sandbox_mode)

    # Format the completion notification
    msg = f"🔔 **[Background Process Complete]**\nCommand: `{command}`\nExit Code: {process.returncode}"
    if output:
        msg += f"\n\nSTDOUT:\n```\n{output[:1000]}\n```"
    if error:
        msg += f"\n\nSTDERR:\n```\n{error[:1000]}\n```"

    # Inject via Universal Gateway
    if thread_id:
        base_url = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{base_url}/api/v1/system/{thread_id}/notify",
                    json={"message": msg, "platform": platform},
                    timeout=10.0,
                )
        except Exception as e:
            logger.error("Failed to push background process result for %s: %s", thread_id, e)


# ═══════════════════════════════════════════════════════════════
# Core Tool: exec_command
# ═══════════════════════════════════════════════════════════════

@tool
async def exec_command(
    command: str = Field(
        ...,
        description=(
            "The exact shell command to run. DO NOT use interactive commands "
            "(like vim, nano, or top). The working directory persists between "
            "calls — use 'cd' naturally to navigate."
        ),
    ),
    timeout_seconds: int = Field(
        ...,
        description="Timeout in seconds. You MUST provide 60 if unsure.",
    ),
    background: bool = Field(
        ...,
        description="Set to True to run asynchronously in the background, otherwise False.",
    ),
    work_dir: str = Field(
        None,
        description=(
            "Optional absolute or relative path to execute the command in. "
            "If omitted, uses the persistent working directory from your last command."
        ),
    ),
    config: RunnableConfig = None,
) -> str:
    """
    Executes a shell command in a sandboxed environment.
    Returns the standard output and standard error.

    Security: Commands are validated against a blocklist, environment
    variables containing secrets are stripped, and execution can optionally
    run inside an ephemeral Docker container.
    """
    from config.settings import settings

    thread_id = get_thread_id(config, default="default")
    platform = config.get("configurable", {}).get("platform", "telegram") if config else "telegram"
    sandbox_mode = settings.exec_sandbox_mode

    # ── CWD Resolution (persistent per thread) ────────────────
    if work_dir:
        cwd = os.path.abspath(work_dir)
    else:
        cwd = _get_persistent_cwd(thread_id)

    if not os.path.exists(cwd):
        return f"Action failed: The specified working directory '{cwd}' does not exist."

    # ── Security: Command Blocklist ───────────────────────────
    is_safe, reason = _check_command_safety(command)
    if not is_safe:
        _update_exec_state(thread_id, command, cwd, "blocked", reason, sandbox_mode)
        return f"Action blocked: {reason}"

    # ── Security: Environment Filtering ───────────────────────
    extra_blocked_raw = settings.exec_blocked_env_patterns
    extra_blocked = [p.strip() for p in extra_blocked_raw.split(",") if p.strip()] if extra_blocked_raw else []
    safe_env = _build_safe_env(extra_blocked)

    # ── Detect CWD changes for persistence ────────────────────
    final_cwd = _detect_final_cwd(command, cwd)

    # ── BACKGROUND MODE ───────────────────────────────────────
    if background:
        if sandbox_mode == "docker":
            return (
                "Action failed: Background mode is not supported in Docker sandbox. "
                "Use `background=False` with a reasonable timeout, or set EXEC_SANDBOX_MODE=host."
            )
        try:
            _update_exec_state(thread_id, command, final_cwd, "running", "Background process started.", sandbox_mode)
            proc_kwargs = {
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
                "cwd": cwd,
                "env": safe_env,
            }
            if os.name == "posix":
                proc_kwargs["preexec_fn"] = os.setsid
            process = await asyncio.create_subprocess_shell(command, **proc_kwargs)
            asyncio.create_task(
                _monitor_background_process(process, command, thread_id, platform, final_cwd, sandbox_mode)
            )
            return f"Action successful: started background command (pid={process.pid})."
        except Exception as e:
            _update_exec_state(thread_id, command, cwd, "failed", f"Failed to start background process: {e}", sandbox_mode)
            return f"Action failed: could not start background command. {str(e)}"

    # ── BLOCKING MODE ─────────────────────────────────────────
    try:
        _update_exec_state(thread_id, command, cwd, "running", "", sandbox_mode)

        if sandbox_mode == "docker":
            extra_volumes_raw = settings.exec_docker_volumes
            extra_volumes = [v.strip() for v in extra_volumes_raw.split(",") if v.strip()] if extra_volumes_raw else []
            returncode, output, error = await _execute_docker(
                command=command,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                image=settings.exec_sandbox_image,
                memory_limit=settings.exec_docker_memory_limit,
                cpu_limit=settings.exec_docker_cpu_limit,
                network=settings.exec_docker_network,
                extra_volumes=extra_volumes,
            )
        else:
            returncode, output, error = await _execute_host(
                command=command,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                safe_env=safe_env,
            )

        combined_output = output
        if error:
            combined_output = f"{combined_output}\n\nSTDERR:\n{error}" if combined_output else f"STDERR:\n{error}"

        if returncode == 0:
            _update_exec_state(thread_id, command, final_cwd, "completed", combined_output, sandbox_mode)
            return f"Action successful: command completed (exit code {returncode})."

        _update_exec_state(thread_id, command, cwd, f"failed ({returncode})", combined_output, sandbox_mode)
        return f"Action failed: command exited with code {returncode}."

    except asyncio.TimeoutError:
        timeout_msg = (
            f"Command timed out after {timeout_seconds} seconds. "
            "The process was killed."
        )
        _update_exec_state(thread_id, command, cwd, "timeout", timeout_msg, sandbox_mode)
        return f"Action failed: {timeout_msg}"

    except RuntimeError as e:
        # Raised when Docker is not available
        _update_exec_state(thread_id, command, cwd, "failed", str(e), sandbox_mode)
        return f"Action failed: {str(e)}"

    except Exception as e:
        _update_exec_state(thread_id, command, cwd, "failed", f"Execution failed: {e}", sandbox_mode)
        return f"Action failed: execution error. {str(e)}"
