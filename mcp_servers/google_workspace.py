"""
mcp_servers/google_workspace.py — Gmail + Calendar tools.

All tools are plain Python functions that can be called programmatically.
The FastMCP decorator is kept for optional MCP-protocol serving, but
the primary usage is direct import + call from the approval node.

Tool registry
─────────────
    TOOL_REGISTRY: dict[str, callable]

    Maps tool names (e.g. "send_email") to their implementing functions.
    Used by nodes/approval.py to dispatch approved write actions.
"""

import sys
import uuid
import base64
import logging
import datetime
from email.mime.text import MIMEText
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from scripts.google_auth_helper import get_google_creds
from googleapiclient.discovery import build

load_dotenv()

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("mcp.google_workspace")




# ══════════════════════════════════════════════════════════════
# Service builders
# ══════════════════════════════════════════════════════════════

def _get_gmail_service():
    """Build an authenticated Gmail API client."""
    creds = get_google_creds()
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _get_calendar_service():
    """Build an authenticated Google Calendar API client."""
    creds = get_google_creds()
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


# ══════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════

def _b64url_decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data.encode("utf-8"))


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8")


def _find_header(headers: List[Dict[str, str]], name: str) -> str:
    name_lower = name.lower()
    for h in headers or []:
        if h.get("name", "").lower() == name_lower:
            return h.get("value", "")
    return ""


def _extract_text_from_payload(payload: Dict[str, Any]) -> Tuple[str, str]:
    """Walk MIME parts and return (text_plain, text_html)."""
    text_plain = ""
    text_html = ""

    def walk(part: Dict[str, Any]):
        nonlocal text_plain, text_html
        mime_type = part.get("mimeType", "")
        body = part.get("body", {}) or {}
        data = body.get("data")
        if data:
            try:
                decoded = _b64url_decode(data).decode("utf-8", errors="replace")
            except Exception:
                decoded = ""
            if mime_type == "text/plain" and not text_plain:
                text_plain = decoded
            elif mime_type == "text/html" and not text_html:
                text_html = decoded
        for child in part.get("parts", []) or []:
            walk(child)

    walk(payload or {})
    return text_plain, text_html


def _resolve_label_id(service, label_id_or_name: str) -> str:
    """Accept a label ID or display name; return the canonical ID."""
    system = {"INBOX", "UNREAD", "STARRED", "IMPORTANT", "SENT", "TRASH", "SPAM", "DRAFT"}
    if label_id_or_name in system:
        return label_id_or_name

    resp = service.users().labels().list(userId="me").execute()
    for lbl in resp.get("labels", []) or []:
        if lbl.get("id") == label_id_or_name or lbl.get("name") == label_id_or_name:
            return lbl["id"]

    raise ValueError(f"Label not found: {label_id_or_name}")


# ══════════════════════════════════════════════════════════════
# Gmail Tools
# ══════════════════════════════════════════════════════════════

def list_messages(max_results: int, query: str, include_spam_trash: bool) -> str:
    """List Gmail messages.
    
    Args:
        max_results: Maximum number of messages to return. You MUST provide a reasonable integer, e.g., 10.
        query: Standard Gmail search syntax (e.g. 'from:someone@example.com is:unread'). You MUST pass an empty string '' if no filter is needed.
        include_spam_trash: Whether to include spam and trash. You MUST provide True or False explicitly.
    """
    try:
        service = _get_gmail_service()
        resp = service.users().messages().list(
            userId="me",
            maxResults=max_results,
            q=query if query else None,
            includeSpamTrash=include_spam_trash,
        ).execute()

        msgs = resp.get("messages", []) or []
        if not msgs:
            return "No messages found."

        lines = [f"Messages (showing up to {max_results}):"]
        for m in msgs:
            msg = service.users().messages().get(
                userId="me", id=m["id"], format="metadata",
                metadataHeaders=["From", "To", "Subject", "Date"],
            ).execute()
            headers = (msg.get("payload", {}) or {}).get("headers", []) or []
            subject = _find_header(headers, "Subject")
            sender = _find_header(headers, "From")
            date = _find_header(headers, "Date")
            snippet = msg.get("snippet", "")
            lines.append(f"- ID: {msg['id']} | {date} | {sender} | {subject}")
            if snippet:
                lines.append(f"  Snippet: {snippet}")

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"❌ list_messages error: {e}")
        return f"Error listing messages: {e}"


def get_message(message_id: str) -> str:
    """Fetch a single email by message ID and return headers + body.
    
    Args:
        message_id: The exact ID of the message to retrieve.
    """
    try:
        service = _get_gmail_service()
        msg = service.users().messages().get(userId="me", id=message_id, format="full").execute()
        payload = msg.get("payload", {}) or {}
        headers = payload.get("headers", []) or []

        subject = _find_header(headers, "Subject")
        sender = _find_header(headers, "From")
        to = _find_header(headers, "To")
        date = _find_header(headers, "Date")
        text_plain, text_html = _extract_text_from_payload(payload)

        out = [
            f"Message ID: {msg['id']}",
            f"Thread ID: {msg.get('threadId')}",
            f"Date: {date}",
            f"From: {sender}",
            f"To: {to}",
            f"Subject: {subject}",
            "\n--- BODY ---",
            text_plain.strip() if text_plain else (text_html.strip() if text_html else "[No body]"),
        ]
        return "\n".join(out)
    except Exception as e:
        logger.error(f"❌ get_message error: {e}")
        return f"Error getting message: {e}"


def send_email(to: str, subject: str, body: str, cc: str, bcc: str) -> str:
    """Send an email via Gmail.

    Args:
        to: Primary recipient email address.
        subject: Email subject line.
        body: Email body text.
        cc: Optional CC recipients (comma-separated). You MUST pass an empty string '' if there are no CCs.
        bcc: Optional BCC recipients (comma-separated). You MUST pass an empty string '' if there are no BCCs.
    """
    try:
        service = _get_gmail_service()
        msg = MIMEText(body, _subtype="plain", _charset="utf-8")
        msg["to"] = to
        msg["subject"] = subject
        if cc.strip():
            msg["cc"] = cc
        if bcc.strip():
            msg["bcc"] = bcc

        raw = _b64url_encode(msg.as_bytes())
        sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()

        return f"Email sent successfully. Message ID: {sent.get('id')}"
    except Exception as e:
        logger.error(f"❌ send_email error: {e}")
        return f"Error sending email: {e}"


def mark_read(message_id: str) -> str:
    """Mark a Gmail message as read.
    
    Args:
        message_id: The exact ID of the message to mark read.
    """
    try:
        service = _get_gmail_service()
        service.users().messages().modify(
            userId="me", id=message_id,
            body={"removeLabelIds": ["UNREAD"]},
        ).execute()
        return f"Marked as read: {message_id}"
    except Exception as e:
        return f"Error marking read: {e}"


def mark_unread(message_id: str) -> str:
    """Mark a Gmail message as unread.
    
    Args:
        message_id: The exact ID of the message to mark unread.
    """
    try:
        service = _get_gmail_service()
        service.users().messages().modify(
            userId="me", id=message_id,
            body={"addLabelIds": ["UNREAD"]},
        ).execute()
        return f"Marked as unread: {message_id}"
    except Exception as e:
        return f"Error marking unread: {e}"


def list_labels() -> str:
    """List all available Gmail labels."""
    try:
        service = _get_gmail_service()
        resp = service.users().labels().list(userId="me").execute()
        labels = resp.get("labels", []) or []
        if not labels:
            return "No labels found."
        lines = ["Labels:"]
        for lbl in labels:
            lines.append(f"- {lbl.get('name')} (ID: {lbl.get('id')})")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing labels: {e}"


def add_label(message_id: str, label: str) -> str:
    """Add a label to a Gmail message (by label name or ID).
    
    Args:
        message_id: The ID of the message.
        label: The label ID or name.
    """
    try:
        service = _get_gmail_service()
        label_id = _resolve_label_id(service, label)
        service.users().messages().modify(
            userId="me", id=message_id,
            body={"addLabelIds": [label_id]},
        ).execute()
        return f"Added label '{label}' to message {message_id}"
    except Exception as e:
        return f"Error adding label: {e}"


def remove_label(message_id: str, label: str) -> str:
    """Remove a label from a Gmail message (by label name or ID).
    
    Args:
        message_id: The ID of the message.
        label: The label ID or name.
    """
    try:
        service = _get_gmail_service()
        label_id = _resolve_label_id(service, label)
        service.users().messages().modify(
            userId="me", id=message_id,
            body={"removeLabelIds": [label_id]},
        ).execute()
        return f"Removed label '{label}' from message {message_id}"
    except Exception as e:
        return f"Error removing label: {e}"


# ══════════════════════════════════════════════════════════════
# Calendar Tools
# ══════════════════════════════════════════════════════════════

def list_events(max_results: int) -> str:
    """List upcoming Google Calendar events.

    Args:
        max_results: Maximum number of events to return. You MUST explicitly provide a reasonable integer like 10.
    """
    try:
        service = _get_calendar_service()
        now = datetime.datetime.utcnow().isoformat() + "Z"

        events_result = service.events().list(
            calendarId="primary", timeMin=now,
            maxResults=max_results, singleEvents=True,
            orderBy="startTime",
        ).execute()
        events = events_result.get("items", [])

        if not events:
            return "No upcoming events found."

        lines = ["Upcoming events:"]
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            summary = event.get("summary", "(No title)")
            lines.append(f"- {start}: {summary}")

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"❌ list_events error: {e}")
        return f"Error listing events: {e}"


def create_event(summary: str, start_time: str, end_time: str, description: str) -> str:
    """Create a new Google Calendar event.

    Args:
        summary: Title of the event.
        start_time: Start time in ISO 8601 format (e.g. '2024-03-15T10:00:00').
        end_time: End time in ISO 8601 format.
        description: Description of the event. You MUST pass an empty string '' if no description is needed.
    """
    try:
        service = _get_calendar_service()
        event = {
            "summary": summary,
            "description": description if description.strip() else "",
            "start": {"dateTime": start_time, "timeZone": "UTC"},
            "end": {"dateTime": end_time, "timeZone": "UTC"},
        }
        created = service.events().insert(calendarId="primary", body=event).execute()
        return f"Event created: {created.get('htmlLink')}"
    except Exception as e:
        logger.error(f"❌ create_event error: {e}")
        return f"Error creating event: {e}"


def create_meeting(summary: str, start_time: str, end_time: str, attendees_csv: str, description: str) -> str:
    """Create a Google Calendar event with a Google Meet link.

    Args:
        summary: Title of the meeting.
        start_time: Start time in ISO 8601 format.
        end_time: End time in ISO 8601 format.
        attendees_csv: Comma-separated list of email addresses to invite. You MUST pass an empty string '' if no attendees.
        description: Description of the meeting. You MUST pass an empty string '' if no description is needed.
    """
    try:
        service = _get_calendar_service()
        
        attendee_list = []
        if attendees_csv.strip():
            attendee_list = [{"email": email.strip()} for email in attendees_csv.split(",") if email.strip()]

        event = {
            "summary": summary,
            "description": description if description.strip() else "",
            "start": {"dateTime": start_time, "timeZone": "UTC"},
            "end": {"dateTime": end_time, "timeZone": "UTC"},
            "conferenceData": {
                "createRequest": {
                    "requestId": str(uuid.uuid4()),
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            },
            "attendees": attendee_list,
        }
        created = service.events().insert(
            calendarId="primary", body=event, conferenceDataVersion=1,
        ).execute()

        meet_link = created.get("hangoutLink", "No link generated")
        return f"Meeting created: {created.get('htmlLink')}\nGoogle Meet Link: {meet_link}"
    except Exception as e:
        logger.error(f"❌ create_meeting error: {e}")
        return f"Error creating meeting: {e}"


# ══════════════════════════════════════════════════════════════
# Tool Registry — programmatic access for approval node
# ══════════════════════════════════════════════════════════════

TOOL_REGISTRY: Dict[str, Any] = {
    "list_messages": list_messages,
    "get_message": get_message,
    "send_email": send_email,
    "mark_read": mark_read,
    "mark_unread": mark_unread,
    "list_labels": list_labels,
    "add_label": add_label,
    "remove_label": remove_label,
    "list_events": list_events,
    "create_event": create_event,
    "create_meeting": create_meeting,
}



