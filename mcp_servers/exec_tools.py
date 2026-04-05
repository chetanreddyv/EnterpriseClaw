import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from typing import Dict
from pydantic import Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import httpx

from core.text_utils import get_thread_id

logger = logging.getLogger(__name__)

_EXEC_STATE_BY_THREAD: Dict[str, dict] = {}


def _truncate_tail(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return "...[TRUNCATED]...\n" + text[-max_chars:]


def _update_exec_state(thread_id: str, command: str, cwd: str, status: str, output: str = "") -> None:
    _EXEC_STATE_BY_THREAD[thread_id] = {
        "cwd": cwd,
        "last_command": command,
        "status": status,
        "last_output": output,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def get_exec_environment_state_for_thread(thread_id: str, tail_chars: int = 8000) -> str:
    """Return a compact terminal state snapshot for a specific thread."""
    entry = _EXEC_STATE_BY_THREAD.get(thread_id)
    if not entry:
        return (
            "## Exec Environment State\n"
            f"- Working Directory: {os.getcwd()}\n"
            "- Last Command: (none)\n"
            "- Status: idle\n\n"
            "No terminal output captured yet for this thread."
        )

    output = _truncate_tail(str(entry.get("last_output", "")), max_chars=tail_chars)
    output_block = output if output.strip() else "(no output captured)"
    return (
        "## Exec Environment State\n"
        f"- Working Directory: {entry.get('cwd', os.getcwd())}\n"
        f"- Last Command: {entry.get('last_command', '(none)')}\n"
        f"- Status: {entry.get('status', 'unknown')}\n"
        f"- Updated At (UTC): {entry.get('updated_at', '(unknown)')}\n\n"
        f"### Last Output (tail)\n{output_block}"
    )

async def _monitor_background_process(process, command: str, thread_id: str, platform: str, cwd: str):
    """Waits for a background process to finish and notifies the main agent."""
    stdout, stderr = await process.communicate()
    output = stdout.decode('utf-8', errors='replace').strip()
    error = stderr.decode('utf-8', errors='replace').strip()
    combined_output = output
    if error:
        combined_output = f"{combined_output}\n\nSTDERR:\n{error}" if combined_output else f"STDERR:\n{error}"

    final_status = "completed" if process.returncode == 0 else f"failed ({process.returncode})"
    _update_exec_state(thread_id, command, cwd, final_status, combined_output)
    
    # Format the completion notification
    msg = f"🔔 **[Background Process Complete]**\nCommand: `{command}`\nExit Code: {process.returncode}"
    if output: msg += f"\n\nSTDOUT:\n```\n{output[:1000]}\n```"
    if error: msg += f"\n\nSTDERR:\n```\n{error[:1000]}\n```"

    # Inject via Universal Gateway
    if thread_id:
        base_url = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{base_url}/api/v1/system/{thread_id}/notify",
                    json={
                        "message": msg,
                        "platform": platform
                    },
                    timeout=10.0
                )
        except Exception as e:
            logger.error(f"Failed to push background process result for {thread_id}: {e}")

@tool
async def exec_command(
    command: str = Field(
        ..., 
        description="The exact shell command to run. DO NOT use interactive commands (like vim, nano, or top). "
                    "Note: 'cd' commands do not persist between steps. Use the 'work_dir' parameter instead or chain commands (e.g., 'cd foo && ls')."
    ),
    timeout_seconds: int = Field(
        ..., 
        description="Timeout in seconds. You MUST provide 60 if unsure."
    ),
    background: bool = Field(
        ..., 
        description="Set to True to run asynchronously in the background, otherwise False."
    ),
    work_dir: str = Field(
        None,
        description="Optional absolute or relative path to execute the command in. Defaults to the current active directory."
    ),
    config: RunnableConfig = None
) -> str:
    """
    Executes a shell command on the host system. 
    Returns the standard output and standard error.
    """
    
    thread_id = get_thread_id(config, default="default")
    platform = config.get("configurable", {}).get("platform", "telegram") if config else "telegram"
    cwd = os.path.abspath(work_dir) if work_dir else os.getcwd()
    
    if not os.path.exists(cwd):
        return f"Action failed: The specified working directory '{cwd}' does not exist."

    # Prepare kwargs for process creation, including process group tracking for POSIX
    proc_kwargs = {
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "cwd": cwd
    }
    if os.name == "posix":
        proc_kwargs["preexec_fn"] = os.setsid
    
    # --- FIRE AND FORGET MODE (NOW WITH MONITORING) ---
    if background:
        try:
            _update_exec_state(thread_id, command, cwd, "running", "Background process started.")
            process = await asyncio.create_subprocess_shell(command, **proc_kwargs)
            asyncio.create_task(_monitor_background_process(process, command, thread_id, platform, cwd))
            return f"Action successful: started background command (pid={process.pid})."
        except Exception as e:
            _update_exec_state(thread_id, command, cwd, "failed", f"Failed to start background process: {e}")
            return f"Action failed: could not start background command. {str(e)}"

    # --- BLOCKING MODE (WITH STRICT TIMEOUTS) ---
    try:
        _update_exec_state(thread_id, command, cwd, "running", "")
        process = await asyncio.create_subprocess_shell(command, **proc_kwargs)
        
        # We wrap the communication in wait_for to prevent infinite hangs
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), 
            timeout=timeout_seconds
        )
        
        output = stdout.decode('utf-8', errors='replace').strip()
        error = stderr.decode('utf-8', errors='replace').strip()

        combined_output = output
        if error:
            combined_output = f"{combined_output}\n\nSTDERR:\n{error}" if combined_output else f"STDERR:\n{error}"

        if process.returncode == 0:
            _update_exec_state(thread_id, command, cwd, "completed", combined_output)
            return f"Action successful: command completed (exit code {process.returncode})."

        _update_exec_state(thread_id, command, cwd, f"failed ({process.returncode})", combined_output)
        return f"Action failed: command exited with code {process.returncode}."
        
    except asyncio.TimeoutError:
        # If the command hangs, aggressively kill the entire process group to free resources
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        timeout_msg = (
            f"Command timed out after {timeout_seconds} seconds. "
            "The process was killed."
        )
        _update_exec_state(thread_id, command, cwd, "timeout", timeout_msg)
        return f"Action failed: {timeout_msg}"
        
    except Exception as e:
        _update_exec_state(thread_id, command, cwd, "failed", f"Execution failed: {e}")
        return f"Action failed: execution error. {str(e)}"

TOOL_REGISTRY = {
    "exec_command": exec_command
}
