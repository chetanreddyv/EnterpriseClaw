import logging
from typing import Dict, Any

from interfaces.base import ClientInterface

logger = logging.getLogger(__name__)

class CLIClient(ClientInterface):
    """
    Command Line Interface Client for EnterpriseClaw.
    Allows interaction directly from the terminal without webhooks.
    """
    
    async def send_message(self, thread_id: str, content: str) -> None:
        """Prints a standard text message to the terminal."""
        print(f"\n🤖 [Agent]:\n{content}\n")
        
    async def request_approval(self, thread_id: str, tool_name: str, args: Dict[str, Any]) -> None:
        """Prompts the user for approval directly in the terminal."""
        args_text = "\n".join(f"  • {k}: {v}" for k, v in args.items()) if args else "  (No arguments)"
        print(f"\n⚠️  [HITL]: Action Requires Approval!")
        print(f"Tool: {tool_name}")
        print(f"Arguments:\n{args_text}\n")
        
        # We need a way to resume the graph.
        # The easiest way to handle async input without blocking the main event loop
        # is to prompt via a thread executor or simple input() if we strictly control the loop.
        # For our simple CLI runner, we'll assume the main event loop will handle resumption.
        
        print("Type 'y' to Approve, 'n' to Reject, or 'e' to Edit in the input prompt.")
