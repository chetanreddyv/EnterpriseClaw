"""
onboarding.py — Interactive setup wizard for EnterpriseClaw.

Launches a beautiful localhost web UI that walks non-technical users
through every configuration step.  Validates API keys in real-time
against the actual APIs and writes a .env file on completion.

Usage:
    uv run python onboarding.py           # opens browser to localhost:8000
    uv run python onboarding.py --cli     # CLI-only fallback (headless)
"""

import sys
import asyncio
import webbrowser
import logging
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
ENV_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"

# ══════════════════════════════════════════════════════════════
# 1. Validation helpers
# ══════════════════════════════════════════════════════════════

async def validate_gemini_key(api_key: str) -> dict:
    """Test a Gemini API key with a real lightweight call."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return {"valid": True, "message": "Gemini API key is valid ✅"}
            elif resp.status_code == 400:
                return {"valid": False, "message": "Invalid API key format. Check for extra spaces or characters."}
            elif resp.status_code == 403:
                return {"valid": False, "message": "API key is forbidden — it may be restricted or disabled."}
            else:
                return {"valid": False, "message": f"Unexpected response ({resp.status_code}). Double-check the key."}
    except Exception as e:
        return {"valid": False, "message": f"Connection error: {str(e)}"}


async def validate_telegram_token(token: str) -> dict:
    """Test a Telegram bot token via getMe."""
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            data = resp.json()
            if data.get("ok"):
                bot_name = data["result"].get("first_name", "Unknown")
                username = data["result"].get("username", "")
                return {
                    "valid": True,
                    "message": f"Connected to bot: {bot_name} (@{username}) ✅",
                    "bot_name": bot_name,
                    "username": username,
                }
            else:
                return {"valid": False, "message": "Invalid token. Make sure you copied the full token from @BotFather."}
    except Exception as e:
        return {"valid": False, "message": f"Connection error: {str(e)}"}


def validate_chat_ids(chat_ids: str) -> dict:
    """Validate comma-separated chat IDs."""
    if not chat_ids.strip():
        return {"valid": False, "message": "At least one Chat ID is required."}
    try:
        ids = [int(cid.strip()) for cid in chat_ids.split(",") if cid.strip()]
        if not ids:
            return {"valid": False, "message": "No valid IDs found. Enter numeric Telegram chat IDs."}
        return {"valid": True, "message": f"{len(ids)} Chat ID(s) configured ✅"}
    except ValueError:
        return {"valid": False, "message": "Chat IDs must be numbers, separated by commas."}


def write_env_file(config: dict) -> str:
    """Write the .env file from the collected configuration."""
    lines = [
        "# ── Telegram ─────────────────────────────────────────────────",
        f"TELEGRAM_BOT_TOKEN={config.get('telegram_bot_token', '')}",
        f"TELEGRAM_SECRET_TOKEN={config.get('telegram_secret_token', '')}",
        f"ALLOWED_CHAT_IDS={config.get('allowed_chat_ids', '')}",
        "",
        "# ── LLM (Gemini) ────────────────────────────────────────────",
        f"GOOGLE_API_KEY={config.get('google_api_key', '')}",
        "",
        "# ── Google Workspace ─────────────────────────────────────────",
        f"GOOGLE_TOKEN_JSON={config.get('google_token_json', '')}",
        "",
        "# ── Observability (optional) ─────────────────────────────────",
        f"LANGCHAIN_TRACING_V2={config.get('langchain_tracing', 'false')}",
        f"LANGSMITH_API_KEY={config.get('langsmith_key', '')}",

    ]
    content = "\n".join(lines) + "\n"
    ENV_PATH.write_text(content)
    return str(ENV_PATH)


# ══════════════════════════════════════════════════════════════
# 2. FastAPI mini-app
# ══════════════════════════════════════════════════════════════

app = FastAPI(title="EnterpriseClaw Setup Wizard")
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


class ValidateRequest(BaseModel):
    key: str

class CompleteRequest(BaseModel):
    google_api_key: str
    telegram_bot_token: str
    allowed_chat_ids: str
    telegram_secret_token: str = ""
    google_token_json: str = ""


@app.get("/", response_class=HTMLResponse)
async def serve_wizard(request: Request):
    """Serve the single-page onboarding wizard."""
    return templates.TemplateResponse("onboarding.html", {"request": request})


@app.post("/api/validate/gemini")
async def api_validate_gemini(req: ValidateRequest):
    result = await validate_gemini_key(req.key.strip())
    return JSONResponse(content=result)


@app.post("/api/validate/telegram")
async def api_validate_telegram(req: ValidateRequest):
    result = await validate_telegram_token(req.key.strip())
    return JSONResponse(content=result)


@app.post("/api/validate/chat-id")
async def api_validate_chat_id(req: ValidateRequest):
    result = validate_chat_ids(req.key.strip())
    return JSONResponse(content=result)


@app.post("/api/complete")
async def api_complete(req: CompleteRequest):
    """Write .env and return success."""
    config = req.model_dump()
    env_path = write_env_file(config)
    return JSONResponse(content={
        "success": True,
        "env_path": env_path,
        "message": "Configuration saved! You can now start EnterpriseClaw.",
    })


# ══════════════════════════════════════════════════════════════
# 3. HTML/CSS/JS — Indian Mythology-Inspired UI
# ══════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════
# 4. CLI fallback
# ══════════════════════════════════════════════════════════════

async def run_cli():
    """Interactive CLI-only onboarding for headless environments."""
    print("\n" + "═" * 56)
    print("  ⚡  EnterpriseClaw — Setup Wizard (CLI Mode)")
    print("═" * 56)
    print("\nThis wizard will configure your .env file step by step.\n")

    config = {}

    # Step 1: Gemini API Key
    print("── Step 1/4: Google API Key ──────────────────────────")
    print("  Get one at: https://aistudio.google.com/apikey\n")
    while True:
        key = input("  Paste your Google API Key: ").strip()
        if not key:
            print("  ⚠️  Key cannot be empty.\n")
            continue
        result = await validate_gemini_key(key)
        if result["valid"]:
            print(f"  {result['message']}\n")
            config["google_api_key"] = key
            break
        else:
            print(f"  ❌ {result['message']} Try again.\n")

    # Step 2: Telegram Bot Token
    print("── Step 2/4: Telegram Bot Token ─────────────────────")
    print("  Create a bot: https://t.me/BotFather → /newbot\n")
    while True:
        token = input("  Paste your Bot Token: ").strip()
        if not token:
            print("  ⚠️  Token cannot be empty.\n")
            continue
        result = await validate_telegram_token(token)
        if result["valid"]:
            print(f"  {result['message']}\n")
            config["telegram_bot_token"] = token
            break
        else:
            print(f"  ❌ {result['message']} Try again.\n")

    # Step 3: Chat IDs
    print("── Step 3/4: Telegram Chat ID ───────────────────────")
    print("  Find yours: message @userinfobot on Telegram\n")
    while True:
        ids = input("  Enter your Chat ID(s): ").strip()
        result = validate_chat_ids(ids)
        if result["valid"]:
            print(f"  {result['message']}\n")
            config["allowed_chat_ids"] = ids
            break
        else:
            print(f"  ❌ {result['message']} Try again.\n")

    # Step 4: Google Workspace
    print("── Step 4/4: Google Workspace (Optional) ────────────")
    ws = input("  Set up Gmail/Calendar? (y/N): ").strip().lower()
    if ws == "y":
        print("  → Place credentials.json in the project root")
        print("  → Then run: uv run python google_auth_helper.py\n")
    else:
        print("  → Skipped. You can set this up later.\n")

    # Write .env
    env_path = write_env_file(config)
    print("═" * 56)
    print(f"  ✅ Configuration saved to {env_path}")
    print(f"\n  Start EnterpriseClaw with:  uv run python app.py")
    print("═" * 56 + "\n")


# ══════════════════════════════════════════════════════════════
# 5. Entrypoint
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(run_cli())
    else:
        import uvicorn

        print("\n  ⚡ EnterpriseClaw Setup Wizard")
        print("  Opening http://localhost:8888 in your browser...\n")

        # Open browser after a short delay (server needs to start)
        import threading
        threading.Timer(1.5, lambda: webbrowser.open("http://localhost:8888")).start()

        uvicorn.run(app, host="0.0.0.0", port=8888, log_level="warning")
