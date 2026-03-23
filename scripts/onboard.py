import os
import sys
import shutil
import subprocess
from pathlib import Path
import httpx
import asyncio

USE_RICH_PROMPTS = sys.stdin.isatty()

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    console = Console()
except ImportError:
    print("Please install requirements first!")
    print("Run: uv run --extra setup python scripts/onboard.py")
    sys.exit(1)

if USE_RICH_PROMPTS:
    try:
        import questionary
    except ImportError:
        console.print("[red]Missing questionary. Please install requirements![/red]")
        console.print("Run: uv run --extra setup python scripts/onboard.py")
        sys.exit(1)

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
    try:
        subprocess.run(["uv", "run", "playwright", "install", "chromium"], check=True, capture_output=True)
        console.print("[bold green]✅ Playwright browsers installed.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]⚠️ Failed to install Playwright browsers: {e}[/bold red]")


async def prompt_for_input(prompt_msg: str, is_password: bool = False) -> str:
    """Prompt user for input, using questionary if TTY is available, otherwise falling back to input()."""
    if USE_RICH_PROMPTS:
        if is_password:
            val = await questionary.password(prompt_msg).ask_async()
        else:
            val = await questionary.text(prompt_msg).ask_async()
        if val is None:
            # User triggered KeyboardInterrupt/Cancellation
            raise KeyboardInterrupt
        return val.strip()
    else:
        # Graceful fallback for non-interactive environments
        try:
            val = input(f"{prompt_msg} ")
            return val.strip()
        except EOFError:
            return ""


def validate_chat_ids_format(ids_str: str) -> bool:
    if not ids_str:
        return True
    try:
        [int(cid.strip()) for cid in ids_str.split(",") if cid.strip()]
        return True
    except ValueError:
        return False


async def async_main():
    console.print()
    console.print(Panel.fit(
        "[bold cyan] EnterpriseClaw Setup Wizard [/bold cyan]\n\n"
        "[white]Let's get your environment configured in minutes.[/white]",
        border_style="cyan"
    ))
    console.print()

    config = {}
    summary_status = {
        "GOOGLE_API_KEY": "⚠️ not set",
        "TELEGRAM_BOT_TOKEN": "⚠️ not set",
        "ALLOWED_CHAT_IDS": "⚠️ not set"
    }

    # Copy .env.example if .env doesn't exist
    if not ENV_PATH.exists() and ENV_EXAMPLE_PATH.exists():
        shutil.copy(ENV_EXAMPLE_PATH, ENV_PATH)
        console.print("[dim]✅ Created .env from .env.example[/dim]")
    elif not ENV_PATH.exists():
        ENV_PATH.touch()

    # Read existing env to see what's missing
    from dotenv import dotenv_values
    existing_env = dotenv_values(ENV_PATH)

    google_key = existing_env.get("GOOGLE_API_KEY", "")
    if not google_key:
        console.print("[bold]── Step 1/3: Google API Key ─────────────────────────[/bold]")
        console.print("[dim]Get your API Key at: https://aistudio.google.com/apikey[/dim]\n")
        while True:
            key = await prompt_for_input("Paste your Google API Key (or press enter to skip):", is_password=True)
            if not key:
                console.print("[yellow]Skipping Google API key.[/yellow]\n")
                break
            
            with console.status("[bold green]Validating Gemini API Key..."):
                result = await validate_gemini_key(key)
                
            if result["valid"]:
                console.print(f"  [green]{result['message']}[/green]\n")
                config["GOOGLE_API_KEY"] = key
                summary_status["GOOGLE_API_KEY"] = "✅ set (new)"
                break
            else:
                console.print(f"  [red]❌ {result['message']} Try again.[/red]\n")
    else:
        config["GOOGLE_API_KEY"] = google_key
        summary_status["GOOGLE_API_KEY"] = "✅ set (existing)"

    telegram_token = existing_env.get("TELEGRAM_BOT_TOKEN", "")
    if not telegram_token:
        console.print("[bold]── Step 2/3: Telegram Bot Token ─────────────────────[/bold]")
        console.print("[dim]Create a bot via https://t.me/BotFather -> /newbot[/dim]\n")
        while True:
            token = await prompt_for_input("Paste your Bot Token (or press enter to skip):", is_password=True)
            if not token:
                console.print("[yellow]Skipping Telegram Bot Token.[/yellow]\n")
                break
                
            with console.status("[bold green]Validating Bot Token..."):
                result = await validate_telegram_token(token)
                
            if result["valid"]:
                console.print(f"  [green]{result['message']}[/green]\n")
                config["TELEGRAM_BOT_TOKEN"] = token
                summary_status["TELEGRAM_BOT_TOKEN"] = "✅ set (new)"
                break
            else:
                console.print(f"  [red]❌ {result['message']} Try again.[/red]\n")
    else:
        config["TELEGRAM_BOT_TOKEN"] = telegram_token
        summary_status["TELEGRAM_BOT_TOKEN"] = "✅ set (existing)"

    allowed_chats = existing_env.get("ALLOWED_CHAT_IDS", "")
    if not allowed_chats:
        console.print("[bold]── Step 3/3: Telegram Chat IDs ───────────────────────[/bold]")
        console.print("[dim]Message @userinfobot on Telegram to get your numeric Chat ID.[/dim]\n")
        while True:
            ids = await prompt_for_input("Enter your Chat ID(s) comma-separated (or press enter to skip):")
            if not ids:
                console.print("[yellow]Skipping Telegram Chat IDs.[/yellow]\n")
                break
            
            if validate_chat_ids_format(ids):
                config["ALLOWED_CHAT_IDS"] = ids
                summary_status["ALLOWED_CHAT_IDS"] = "✅ set (new)"
                console.print(f"  [green]Chat IDs formatted correctly ✅[/green]\n")
                break
            else:
                console.print("  [red]❌ Invalid format. Please enter comma-separated integers (e.g., 12345,67890).[/red]\n")
    else:
        config["ALLOWED_CHAT_IDS"] = allowed_chats
        summary_status["ALLOWED_CHAT_IDS"] = "✅ set (existing)"

    # Write configs back to .env
    lines = []
    if ENV_PATH.exists():
        with open(ENV_PATH, "r") as f:
            lines = f.readlines()

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
        if v: # Only write non-empty items
            replace_or_append(lines, k, v)

    with open(ENV_PATH, "w") as f:
        f.writelines(lines)

    console.print("[bold green]✅ Configuration saved to .env[/bold green]\n")

    # Install Playwright
    with console.status("[bold cyan]📦 Installing Playwright browsers..."):
        run_playwright_install()

    # Dry-run check for application config
    console.print("\n[bold]── Dry Run Validation ────────────────────────────────[/bold]")
    with console.status("[bold blue]Testing configuration validity..."):
        try:
            from dotenv import load_dotenv
            load_dotenv(ENV_PATH, override=True)
            
            # Setting environment variables from current .env so pydantic can access them
            for k, v in dotenv_values(ENV_PATH).items():
                if v is not None:
                    os.environ[k] = v

            sys.path.insert(0, str(PROJECT_ROOT))
            from config.settings import Settings
            
            _ = Settings()
            config_valid = True
            config_msg = "[bold green]✅ Application configuration is valid![/bold green]"
        except ImportError:
            config_valid = False
            config_msg = "[yellow]⚠️ Could not check config validity (app modules not found).[/yellow]"
        except Exception as e:
            config_valid = False
            # Pydantic validation errors could be large; let's format them
            import traceback
            err_str = str(e)
            config_msg = f"[bold red]❌ Configuration Error Detected:[/bold red]\n{err_str}"

    console.print(config_msg)

    # Summary Panel
    table = Table(title="Setup Summary", show_header=True, header_style="bold magenta")
    table.add_column("Environment Variable", style="cyan")
    table.add_column("Status", justify="left")
    
    for key, status in summary_status.items():
        if "⚠️" in status:
            colored_status = f"[yellow]{status}[/yellow]"
        elif "(new)" in status:
            colored_status = f"[green]{status}[/green]"
        else:
            colored_status = f"[dim green]{status}[/dim green]"
        table.add_row(key, colored_status)
    
    console.print("\n")
    console.print(table)

    if "⚠️" in str(summary_status.values()) or not config_valid:
        console.print(Panel(
            "[yellow]Some settings are still missing or invalid.[/yellow] "
            "EnterpriseClaw might not run properly until they are resolved.\n"
            "You can run [bold]make setup[/bold] again later or edit [bold].env[/bold] directly.",
            border_style="yellow",
            title="Warning"
        ))
    else:
        console.print(Panel.fit(
            "[bold green]🎉 All set! You're ready to go.[/bold green]\n\n"
            "Start EnterpriseClaw with:  [bold cyan]make run[/bold cyan]",
            border_style="green",
        ))

def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted. You can run 'make setup' again later.")
        sys.exit(1)

if __name__ == "__main__":
    main()
