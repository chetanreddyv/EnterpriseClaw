import os
import sys
import shutil
import subprocess
from pathlib import Path
import httpx
import asyncio

PROJECT_ROOT = Path(__file__).parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"

async def validate_gemini_key(api_key: str) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                return {"valid": True, "message": "Gemini API key is valid ✅"}
            return {"valid": False, "message": "Invalid API key format or restricted."}
    except Exception as e:
        return {"valid": False, "message": f"Connection error: {e}"}

async def validate_telegram_token(token: str) -> dict:
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            data = resp.json()
            if data.get("ok"):
                bot_name = data["result"].get("first_name", "Unknown")
                username = data["result"].get("username", "")
                return {"valid": True, "message": f"Connected to bot: {bot_name} (@{username}) ✅"}
            return {"valid": False, "message": "Invalid token. Double check."}
    except Exception as e:
        return {"valid": False, "message": f"Connection error: {e}"}

def run_playwright_install():
    print("\n📦 Installing Playwright browsers...")
    try:
        subprocess.run(["uv", "run", "playwright", "install", "chromium"], check=True)
        print("✅ Playwright browsers installed.")
    except Exception as e:
        print(f"⚠️ Failed to install Playwright browsers: {e}")

async def async_main():
    print("\n" + "═" * 56)
    print("  🚀 EnterpriseClaw — Setup Wizard")
    print("═" * 56)
    print("\nLet's get your environment configured in minutes.\n")

    config = {}

    # Copy .env.example if .env doesn't exist
    if not ENV_PATH.exists() and ENV_EXAMPLE_PATH.exists():
        shutil.copy(ENV_EXAMPLE_PATH, ENV_PATH)
        print("✅ Created .env from .env.example")
    elif not ENV_PATH.exists():
        ENV_PATH.touch()

    # Read existing env to see what's missing
    from dotenv import dotenv_values
    existing_env = dotenv_values(ENV_PATH)

    google_key = existing_env.get("GOOGLE_API_KEY", "")
    if not google_key:
        print("── Step 1/3: Google API Key ─────────────────────────")
        print("Get your API Key at: https://aistudio.google.com/apikey\n")
        while True:
            key = input("Paste your Google API Key (or press enter to skip): ").strip()
            if not key:
                print("Skipping Google API key.")
                break
            result = await validate_gemini_key(key)
            if result["valid"]:
                print(f"  {result['message']}\n")
                config["GOOGLE_API_KEY"] = key
                break
            else:
                print(f"  ❌ {result['message']} Try again.\n")
    else:
        config["GOOGLE_API_KEY"] = google_key

    telegram_token = existing_env.get("TELEGRAM_BOT_TOKEN", "")
    if not telegram_token:
        print("── Step 2/3: Telegram Bot Token ─────────────────────")
        print("Create a bot via https://t.me/BotFather -> /newbot\n")
        while True:
            token = input("Paste your Bot Token (or press enter to skip): ").strip()
            if not token:
                print("Skipping Telegram Bot Token.")
                break
            result = await validate_telegram_token(token)
            if result["valid"]:
                print(f"  {result['message']}\n")
                config["TELEGRAM_BOT_TOKEN"] = token
                break
            else:
                print(f"  ❌ {result['message']} Try again.\n")
    else:
        config["TELEGRAM_BOT_TOKEN"] = telegram_token

    allowed_chats = existing_env.get("ALLOWED_CHAT_IDS", "")
    if not allowed_chats:
        print("── Step 3/3: Telegram Chat IDs ───────────────────────")
        print("Message @userinfobot on Telegram to get your numeric Chat ID.\n")
        ids = input("Enter your Chat ID(s) comma-separated (or press enter to skip): ").strip()
        if ids:
            config["ALLOWED_CHAT_IDS"] = ids
    else:
        config["ALLOWED_CHAT_IDS"] = allowed_chats

    # Write configs back to .env
    lines = []
    if ENV_PATH.exists():
        with open(ENV_PATH, "r") as f:
            lines = f.readlines()

    # Simple inplace replacement for the known keys
    def replace_or_append(lines, key, value):
        found = False
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}\n"
                found = True
                break
        if not found and value:
            lines.append(f"{key}={value}\n")

    for k, v in config.items():
        replace_or_append(lines, k, v)

    with open(ENV_PATH, "w") as f:
        f.writelines(lines)

    print("\n✅ Configuration saved to .env")

    # Install Playwright
    run_playwright_install()

    print("\n═" * 56)
    print("🎉 All set! You're ready to go.")
    print("Start EnterpriseClaw with:  make run")
    print("═" * 56 + "\n")

def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted. You can run 'make setup' again later.")
        sys.exit(1)

if __name__ == "__main__":
    main()
