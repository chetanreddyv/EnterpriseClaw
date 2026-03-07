import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

REMINDERS_FILE = Path("data/reminders.json")

@tool
def check_current_time() -> str:
    """
    Returns the current date and time in ISO format (UTC) and the local system timezone.
    Use this to orient yourself chronologically before making time-sensitive decisions or creating reminders.
    """
    now = datetime.now(timezone.utc)
    local = datetime.now().astimezone()
    return f"Current UTC Time: {now.isoformat()}\nLocal System Time: {local.isoformat()}\nTimezone: {local.tzname()}"

@tool
def schedule_reminder(message: str, time_str: str) -> str:
    """
    Schedule a reminder or future task. The heartbeat system will check this schedule
    every 15 minutes and inject the message back into the conversation context.
    
    Args:
        message (str): The reminder text (e.g. "Check the stock price of AAPL", "Reply to Bob's email")
        time_str (str): When you want to be reminded. You can use ISO format (2025-10-15T14:30:00Z) or plain English (e.g., "in 30 mins", "tomorrow at 9am").
    """
    try:
        REMINDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        reminders = []
        if REMINDERS_FILE.exists():
            with open(REMINDERS_FILE, "r") as f:
                reminders = json.load(f)
                
        reminders.append({
            "message": message,
            "target_time": time_str,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending"
        })
        
        with open(REMINDERS_FILE, "w") as f:
            json.dump(reminders, f, indent=2)
            
        logger.info(f"📅 Reminder scheduled: '{message}' for {time_str}")
        return f"Successfully scheduled reminder '{message}' for '{time_str}'."
    except Exception as e:
        logger.error(f"Failed to schedule reminder: {e}")
        return f"Failed to schedule reminder: {e}"

@tool
async def save_to_long_term_memory(fact: str) -> str:
    """
    Save an important fact, preference, or rule about the user into your long-term 
    semantic memory vector store. Use this whenever the user tells you something 
    you need to remember for future conversations. DO NOT save temporary/ephemeral context.
    
    Args:
        fact (str): The specific, detailed fact to remember (e.g., "User's favorite programming language is Python.", "User prefers concise answers without apologies.", "User lives in New York.")
    """
    try:
        from memory.retrieval import memory_retrieval
        from pydantic import BaseModel
        
        # We simulate the ExtractedMemory structure that vectorstore expects
        class ExtractedItem(BaseModel):
            text: str
            importance: str
            confidence: float = 1.0
            
        class TempMemoryObj:
            facts = [ExtractedItem(text=fact, importance="high")]
            preferences = []
            rules = []
            updates = []
            obsolete_items = []
            important = True
            
        mem_obj = TempMemoryObj()
        # Default to a generic source thread since active control is manual
        await memory_retrieval.store.apply_updates(mem_obj, source_thread_id="agent_manual_memory")
        logger.info(f"🧠 Memory saved: '{fact}'")
        return f"Successfully saved to long-term memory: '{fact}'."
    except Exception as e:
        logger.error(f"Failed to save memory: {e}")
        return f"Error saving memory: {str(e)}"

# Register tools so they are dynamically loaded by the GLOBAL_TOOL_REGISTRY in __init__.py
TOOL_REGISTRY = {
    "check_current_time": check_current_time,
    "schedule_reminder": schedule_reminder,
    "save_to_long_term_memory": save_to_long_term_memory
}
