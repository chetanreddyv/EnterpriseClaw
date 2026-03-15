# 🦅 EnterpriseClaw

**A deterministic, highly-scalable agentic framework built for production.**

EnterpriseClaw is a stateful AI orchestration engine powered by **LangGraph**. It abandons "black box" agent loops in favor of explicit state machines, strict schema enforcement, and a decoupled Gateway architecture. It is designed to run complex, multi-step agentic workflows with bulletproof Human-In-The-Loop (HITL) safeguards.

## ⚡ Why EnterpriseClaw?

Most agent frameworks (like LangChain agents or Pydantic AI) bury the execution loop inside a single runtime method. When tools fail or require human approval, the orchestration breaks, threads block, and context is lost.

EnterpriseClaw solves the "Orchestrator Clash" by splitting the brain from the brawn:

1. **The Brain (LLM + Flat Schemas):** We use native model tool-calling with strictly flattened Pydantic schemas (zero implicit defaults) to eliminate LLM hallucinations and API warnings.
2. **The Brawn (LangGraph):** The entire execution loop, memory check-pointing, and HITL pausing are handled natively by LangGraph's state machine.
3. **The Direct Gateway:** Webhooks and API endpoints stream-invoke the LangGraph state machine directly via `astream()`, drastically simplifying background execution without bloated queues.

---

## ✨ Core Features

* **🛡️ True Human-In-The-Loop (HITL):** Dangerous "Write" operations (like `exec_command` or `send_email`) instantly pause the LangGraph state. The LLM's intent is routed to the user (via Telegram or Web) for approval, rejection, or feedback. Rejections are fed *back* to the LLM so it can course-correct.
* **🧠 Enterprise-Grade Memory (MemoryGate):** Unlike standard frameworks that dump agent memory into a single, fragile markdown file, EnterpriseClaw uses a dual-layer memory architecture powered by SQLite and Zvec vector indexing for lightning-fast, semantic context retrieval.
* **🔌 JIT Skill Retrieval + Dynamic Tool Scoping:** The Supervisor delegates an objective with a single `domain` (`browser`, `exec`, or `all`), and the Worker performs just-in-time skill retrieval from the vector index using that objective. The Worker binds tools inside that domain and applies matched skill metadata to keep context and blast radius tight.
* **🚦 The Gateway Pattern:** The core execution engine has zero knowledge of the delivery channel. A central `ChannelManager` translates abstract agent actions into UI-specific formats (Telegram Inline Keyboards, Web UI buttons, etc.).
* **💓 System Heartbeat:** A lightweight asynchronous loop wakes the agent every 15 minutes to perform time-based tasks natively using the injected system clock context and `schedule_reminder`.
* **👀 Background Process Monitoring:** When the agent executes a background shell command, it doesn't just blind-fire it. EnterpriseClaw attaches an async monitor to capture `stdout`/`stderr` and automatically notifies the primary agent thread the moment the process concludes.
* **🕒 Strict Temporal Awareness:** Supervisor and Worker prompts inject current UTC and local time in a dedicated "System Clock" section, improving time-sensitive reasoning without requiring explicit time-tool calls for basic date awareness.

* **🔀 Model Agnosticism:** Swap the underlying LLM at runtime without touching code. Backed by LangChain's `init_chat_model` factory, each conversation thread stores its preferred model in the SQLite checkpointer — meaning different threads can run different providers simultaneously, and preferences survive restarts.

* **🌐 Browser Use:** Control a real Chromium browser via Playwright. Navigate websites, click buttons, fill forms, take screenshots, and extract content — all with HITL safeguards on interaction tools. Each thread gets an isolated browser session with separate cookies and storage.

---

## 🌐 Browser Use

EnterpriseClaw can control a **real headless Chromium browser** for tasks that require interaction beyond simple HTTP fetching — logging into dashboards, filling forms, clicking through multi-step flows, or scraping JavaScript-rendered content.

### Setup

After installing dependencies, download the Chromium binary (one-time):

```bash
playwright install chromium
```

### Tools

| Tool | Safety | Description |
|---|---|---|
| `browser_navigate(url)` | ✅ Read | Navigate to a URL, returns page title and text |
| `browser_get_text()` | ✅ Read | Extract visible text from the current page |
| `browser_screenshot()` | ✅ Read | Save screenshot to `./data/screenshots/` |
| `browser_click(selector)` | 🔐 HITL | Click an element by CSS selector or visible text |
| `browser_type(selector, text)` | 🔐 HITL | Type text into an input field |
| `browser_execute_js(script)` | 🔐 HITL | Run JavaScript on the page |

> **Tiered Autonomy:** Read-only tools (`navigate`, `get_text`, `screenshot`) run freely. Dangerous interaction tools (`click`, `type`, `execute_js`) pause for human approval before executing, exactly like `exec_command`. Browser contexts are isolated per conversation thread, and idle sessions are automatically garbage-collected after 30 minutes.



## 🔀 Model Agnosticism

Send `/model <provider/model>` from **any channel** (Telegram or Web) to hot-swap the LLM for that specific conversation thread. The preference is persisted in SQLite — it survives restarts and is isolated per-thread.

```
/model google_genai/gemini-2.5-flash      # default
/model openai/gpt-4o
/model anthropic/claude-3-5-sonnet-20241022
/model ollama/llama3                       # local, no API key needed
```

| Provider | Format | Required `.env` key |
|---|---|---|
| Google Gemini | `google_genai/<model>` | `GEMINI_API_KEY` |
| OpenAI | `openai/<model>` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/<model>` | `ANTHROPIC_API_KEY` |
| Ollama (local) | `ollama/<model>` | *(none — server must be running)* |
| Any OpenAI-compat | `openai/<model>` + `OPENAI_BASE_URL` | `OPENAI_API_KEY` |

> The agent automatically falls back to Gemini if the requested model fails to initialise.

## 🛠️ Interactive Commands

EnterpriseClaw provides several slash commands to manipulate agent state mid-conversation without needing to restart the server or edit config files:

*   `/model <provider/model>` — Hot-swap the underlying LLM for the current thread (e.g. `/model openai/gpt-4o`).
*   `/permit <tool_name>` — Permit a tool for the current thread. For high-risk tools (for example `exec_command`), this bypasses repeated HITL prompts until revoked.
*   `/deny <tool_name>` — Force confirmation for an auto-allowed tool (for example `browser_navigate` or `delegate_task`) in the current thread.
*   `/tools` — Print all tools and their thread-specific policy state (`autonomous`, `auto-allowed`, `permitted`, `denied/requires approval`).

## 🔁 Delegation Contract (Current)

Complex multi-step execution should be delegated through:

```python
delegate_task(
	objective="Go to example.com, extract headline text, and summarize it.",
	domain="browser",
	max_steps=15,
)
```

- `objective`: specific task goal for the Worker.
- `domain`: strict worker scope for this delegation (`browser`, `exec`, or `all`).
- `max_steps`: worker loop cap.

Notes:
- Domain categories are resolved dynamically from modules in `mcp_servers/`.
- Worker always retains `escalate_to_supervisor` as a safety exit.
- Worker execution mode is metadata-aware: stateful tools run sequentially, read-only sets may run concurrently.

### 1. Installation

Clone the repository and install dependencies using `uv` (recommended for strict lockfile management):

```bash
git clone https://github.com/chetanreddyv/EnterpriseClaw.git
cd EnterpriseClaw
uv sync

```

### 2. Configuration

Copy the example environment file and add your API keys:

```bash
cp .env.example .env

```

Ensure you set at least one LLM API key and your `TELEGRAM_BOT_TOKEN` (if using the Telegram channel). Add keys for any providers you want to hot-swap into at runtime:

```bash
# Required — at least one of these:
GEMINI_API_KEY=...        # Google Gemini (default)
OPENAI_API_KEY=...        # OpenAI (optional)
ANTHROPIC_API_KEY=...     # Anthropic Claude (optional)
# Ollama: no key needed, just run the server locally
```

### 3. Start the Engine

Run the setup wizard to initialize the SQLite checkpointer and vector stores, then start the server:

```bash
uv run python app.py

```

*Note: The FastAPI server returns quickly for webhook requests while orchestration continues asynchronously in the LangGraph runtime.*

---

## 🧠 Rethinking Agentic Memory: The MemoryGate Engine

Other typical local agent frameworks rely on reading and writing to plain text `.md` files to remember user preferences. This approach is slow, consumes massive amounts of the LLM's token context limit, and is prone to corruption during parallel execution.

EnterpriseClaw introduces **MemoryGate**, a highly-scalable, dual-layer memory architecture:

1. **Short-Term State (Thread Memory):** LangGraph's SQLite checkpointer natively tracks the sliding window of conversational context and pending tool executions. If the server crashes while waiting for human approval, the exact state is flawlessly preserved and instantly thawed upon reboot.
2. **Long-Term Memory (Agent-Driven Semantic Indexing):** Instead of wasting tokens passively parsing every message, the agent is empowered to actively use a `@tool` (`save_to_long_term_memory`) to save preferences and facts to a Zvec vector index. When a user asks a question, the agent performs a semantic search to inject *only* the highly relevant context into the prompt, saving tokens and vastly improving reasoning accuracy.

---

## 🛠️ Creating Skills (JIT Retrieval)

Skill files are used as prompt-time behavior modules. The Worker retrieves relevant skills at runtime based on the delegated objective.

Create a Markdown file in `skills/my_new_skill/skill.md`:

```markdown
---
name: github_manager
description: Manage GitHub repositories, check PRs, and review issues.
---

### TRIGGER_EXAMPLES
- "check open pull requests in my repo"
- "review latest github issues"
### END_TRIGGER_EXAMPLES

# GitHub Manager
You are a senior developer managing a GitHub repository. 
When asked to check PRs, gather facts first and then propose safe next actions.
Always verify the current directory before running commands.

```

**What happens automatically:**

1. Skills are embedded into the skill vector index at startup.
2. On delegation, Worker queries skills using the task `objective`.
3. Only matched skill content is injected into the Worker prompt (JIT).
4. Tool binding is controlled by delegated `domain` plus matched skill metadata.

To add new executable tools, register them in `mcp_servers/<module>.py` via `TOOL_REGISTRY`. The module name determines the category automatically (`web_tools.py` -> `web`, `exec_tools.py` -> `exec`).

---

## 🏗️ Architecture Deep Dive

EnterpriseClaw is built on three distinct layers, providing an enterprise-grade execution environment superior to typical monolithic agent loops:

1. **The Gateway API (`app.py`):** Fast, stateless endpoints that stream asynchronous LangGraph invocations directly from user inputs and the system heartbeat.
2. **The Control Plane (LangGraph):** A Supervisor graph delegates to Worker subgraphs. Workers start with a SkillContext step (JIT retrieval), then execute in an action-observation loop with category-scoped tools and HITL enforcement.
3. **The Channel Manager (`core/channel_manager.py`):** Intercepts outputs from the Control Plane and formats them for the specific user interface (e.g., rendering an "Approve/Reject" button in Telegram or Web Chat).

## 🤝 Contributing

We welcome contributions to make EnterpriseClaw even more robust. Please ensure that all new tools use **Flat Pydantic Schemas** (no `Optional` or default values) to maintain strict LLM determinism.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
