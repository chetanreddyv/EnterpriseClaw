# 🦅 EnterpriseClaw

**A deterministic, highly-scalable agentic framework built for production.**

EnterpriseClaw is a stateful AI orchestration engine powered by **LangGraph**. It abandons "black box" agent loops in favor of explicit state machines, strict schema enforcement, and a decoupled Gateway architecture. It is designed to run complex, multi-step agentic workflows with bulletproof Human-In-The-Loop (HITL) safeguards.

## ⚡ Why EnterpriseClaw?

Most agent frameworks (like LangChain agents or Pydantic AI) bury the execution loop inside a single runtime method. When tools fail or require human approval, the orchestration breaks, threads block, and context is lost.

EnterpriseClaw solves the "Orchestrator Clash" by splitting the brain from the brawn:

1. **The Brain (LLM + Flat Schemas):** We use native model tool-calling with strictly flattened Pydantic schemas (zero implicit defaults) to eliminate LLM hallucinations and API warnings.
2. **The Brawn (LangGraph):** The entire execution loop, memory check-pointing, and HITL pausing are handled natively by LangGraph's state machine.
3. **The Gateway (Decoupled I/O):** Webhooks and API endpoints never wait for the LLM. Every request is pushed to an async queue (`LaneManager`) and processed sequentially per-thread.

---

## ✨ Core Features

* **🛡️ True Human-In-The-Loop (HITL):** Dangerous "Write" operations (like `exec_command` or `send_email`) instantly pause the LangGraph state. The LLM's intent is routed to the user (via Telegram or Web) for approval, rejection, or feedback. Rejections are fed *back* to the LLM so it can course-correct.
* **🧠 Enterprise-Grade Memory (MemoryGate):** Unlike standard frameworks that dump agent memory into a single, fragile markdown file, EnterpriseClaw uses a dual-layer memory architecture powered by SQLite and Zvec vector indexing for lightning-fast, semantic context retrieval.
* **🔌 The Standard Library & Skill Auto-Loader:** EnterpriseClaw comes pre-equipped with a "Standard Library" of core capabilities (`exec_command`, `web_search`). You don't need to specify these in configuration files—the agent intrinsically knows how to use them, and they are permanently protected by HITL. For custom APIs, simply drop a `SKILL.md` file into the `/skills` directory and the framework binds Python tools on the fly.
* **🚦 The Gateway Pattern:** The core execution engine has zero knowledge of the delivery channel. A central `ChannelManager` translates abstract agent actions into UI-specific formats (Telegram Inline Keyboards, Web UI buttons, etc.).
* **⏳ True LangGraph Subagents:** Advanced background operations. Instead of simple fire-and-forget scripts, EnterpriseClaw spawns ephemeral LangGraph execute loops for complex background research. When a subagent finishes, it securely injects its report directly into the primary thread's checkpointer state.
* **👀 Background Process Monitoring:** When the agent executes a background shell command, it doesn't just blind-fire it. EnterpriseClaw attaches an async monitor to capture `stdout`/`stderr` and automatically notifies the primary agent thread the moment the process concludes.

* **🔀 Model Agnosticism:** Swap the underlying LLM at runtime without touching code. Backed by LangChain's `init_chat_model` factory, each conversation thread stores its preferred model in the SQLite checkpointer — meaning different threads can run different providers simultaneously, and preferences survive restarts.

---

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

*Note: The FastAPI server will instantly return HTTP 200s for webhooks, pushing the heavy lifting to the background `LaneManager` queues.*

---

## 🧠 Rethinking Agentic Memory: The MemoryGate Engine

Other typical local agent frameworks rely on reading and writing to plain text `.md` files to remember user preferences. This approach is slow, consumes massive amounts of the LLM's token context limit, and is prone to corruption during parallel execution.

EnterpriseClaw introduces **MemoryGate**, a highly-scalable, dual-layer memory architecture:

1. **Short-Term State (Thread Memory):** LangGraph's SQLite checkpointer natively tracks the sliding window of conversational context and pending tool executions. If the server crashes while waiting for human approval, the exact state is flawlessly preserved and instantly thawed upon reboot.
2. **Long-Term Memory (Zvec Semantic Indexing):** Background tasks asynchronously extract key facts and user preferences from completed conversational loops, storing them in a Zvec vector index. When a user asks a question, the agent performs a semantic search to inject *only* the highly relevant context into the prompt, saving tokens and vastly improving reasoning accuracy.

---

## 🛠️ Creating Skills (The Magic)

EnterpriseClaw uses a **Universal Skill Auto-Loader**. You don't need to write Python boilerplate to teach the agent a new workflow.

Just create a Markdown file in `skills/my_new_skill/skill.md`:

```markdown
---
name: github_manager
description: Manage GitHub repositories, check PRs, and review issues.
tools: exec_command, list_files
---

# GitHub Manager
You are a senior developer managing a GitHub repository. 
When asked to check PRs, use the `gh` CLI via the `exec_command` tool.
Always verify the current directory before running commands.

```

**What happens automatically:**

1. EnterpriseClaw reads the `tools` frontmatter.
2. It fetches `exec_command` and `list_files` from the Global Tool Registry.
3. It binds those tools to the LLM.
4. It flags `exec_command` as a dangerous "write" action, ensuring the agent pauses for your permission before executing any bash scripts.

---

## 🏗️ Architecture Deep Dive

EnterpriseClaw is built on three distinct layers, providing an enterprise-grade execution environment superior to typical monolithic agent loops:

1. **The Gateway API (`app.py`):** Fast, stateless endpoints that receive inputs and push `IncomingMessageEvent` payloads to the bus.
2. **The Control Plane (`core/worker.py` & LangGraph):** A continuous background loop that consumes events, steps through the LangGraph state machine, executes tools, and suspends state to SQLite when HITL is required. 
3. **The Subagent Plane (`nodes/subagents.py`):** Dedicated, ephemeral LangGraph loops for background tasks. Unlike standard `while` loop subagents that vanish if a server restarts, EnterpriseClaw's background tasks run within the LangGraph ecosystem and cleanly inject their outcomes into the primary conversation thread.
4. **The Channel Manager (`core/channel_manager.py`):** Intercepts outputs from the Control Plane and formats them for the specific user interface (e.g., rendering an "Approve/Reject" button in Telegram).

## 🤝 Contributing

We welcome contributions to make EnterpriseClaw even more robust. Please ensure that all new tools use **Flat Pydantic Schemas** (no `Optional` or default values) to maintain strict LLM determinism.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
