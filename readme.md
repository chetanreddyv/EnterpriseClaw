# EnterpriseClaw

EnterpriseClaw is a production-focused agent orchestration system built on LangGraph.

It is designed around one core idea:

- Keep orchestration deterministic.
- Keep model reasoning flexible.

In practice, this means:

- The Supervisor handles conversation, memory context, and delegation.
- The Worker executes objectives in a strict action-observation loop.
- Heavy environment state is replaced each turn (not accumulated forever).
- Tool safety, routing, and failure behavior are explicit and testable.

## Why This Project Exists

Most agent stacks fail in one of these ways:

- Context windows bloat over long tasks.
- Browser agents drift on dynamic UIs.
- Tool usage is too permissive or too rigid.
- Infrastructure failures burn tokens and loop forever.

EnterpriseClaw addresses these directly with deterministic graph routing, replace-only environment anchors, strict tool policy, and typed failure handling.

## Core Architecture

### 1. Supervisor (persistent orchestrator)

The Supervisor graph is user-facing and stateful.

Responsibilities:

- Ingest user input.
- Retrieve long-term context.
- Build the high-level prompt.
- Use only a fixed, narrow tool set.
- Delegate complex execution via `delegate_task(objective=...)`.

Primary flow:

- Intent -> Prompt Builder -> Executor -> Tools/Compact -> END

Code:

- `core/graphs/supervisor.py`
- `core/nodes/supervisor_nodes.py`

### 2. Worker (ephemeral executor)

The Worker graph is delegated, objective-bound, and execution-focused.

Responsibilities:

- Retrieve matching skills once at start.
- Build action-observation prompts.
- Execute tool calls under strict policy.
- Refresh current environment snapshot.
- Summarize result and return to Supervisor.

Primary flow:

- Skill Context -> Prompt Builder -> Executor -> Tools -> Refresh Environment -> Prompt Builder (loop) -> Summarize -> END

Code:

- `core/graphs/worker.py`
- `core/nodes/worker_nodes.py`

## Deterministic State Management (Efficiency + Reliability)

EnterpriseClaw uses a two-zone state model in Worker sessions.

### Zone A: Action Ledger (`messages`)

This is lightweight and append-only.

- Stores short action summaries and tool traces.
- Avoids dumping heavy raw state into history.
- Keeps token usage stable across long loops.

### Zone B: Ephemeral Anchor (`environment_snapshot`)

This is heavy, current reality, and replace-only.

- Contains current browser/terminal state.
- Replaced every refresh cycle.
- Never treated as permanent chat history.

Result:

- The model sees current ground truth without stale state pollution.
- Prompt growth is controlled.
- Attention is focused on now, not old screenshots/text maps.

State schema:

- `core/graphs/states.py`

## Context Management Strategy

### Prompt assembly is deliberate

Worker prompt builder composes context in this order:

- System rules and execution constraints.
- Objective anchor.
- Valid history tail (pair-safe for tool calls/results).
- Current environment snapshot at the end.

This ordering preserves recency and prevents tool-message orphaning.

Key logic:

- `_coerce_valid_worker_history(...)`
- `_build_worker_history_tail(...)`
- `_observation_to_prompt_text(...)`

Code:

- `core/nodes/worker_nodes.py`

## Memory System (Hybrid, Durable, and Fast)

EnterpriseClaw memory has two layers:

### 1. Durable source of truth: SQLite

`memory/db.py` stores:

- Thread history (`thread_history`).
- Long-term memory items (`memory_items`) with lifecycle state.
- FTS5 keyword index (`memory_fts`) for BM25 retrieval.

Important properties:

- WAL mode and tuned pragmas for concurrency.
- FTS integrity checks and rebuild support.
- Active/tombstoned item lifecycle.

### 2. Semantic retrieval layer: Zvec + FastEmbed

`memory/vectorstore.py` provides:

- Semantic retrieval over memory items.
- Hybrid ranking using semantic + FTS via Reciprocal Rank Fusion (RRF).
- Time-decay weighting for freshness.
- Async initialization and corruption recovery.
- Bounded embedding concurrency with semaphore.

### 3. Retrieval gateway

`memory/retrieval.py` merges:

- Recent conversation history.
- Relevant semantic memory context.
- Skill retrieval with metadata.

This gives strong recall without flooding prompt context.

## Skill System (JIT and Tool-Scoped)

Skills live in `skills/*/skill.md` and include frontmatter metadata:

- `name`
- `description`
- `tools`
- trigger examples

Worker start behavior:

- Retrieve top-matching skills for objective.
- Parse declared tools from skill frontmatter.
- Bind only resolved tools (+ `escalate_to_supervisor`, `complete_task`).
- If nothing executable is resolved, escalate instead of guessing.

This prevents broad tool exposure and reduces hallucinated tool calls.

Code:

- `memory/vectorstore.py`
- `core/nodes/worker_nodes.py`

## Browser Perception and Action Grounding

Browser automation is built around browser-use + CDP-backed state extraction.

### Session profile defaults

`core/browser_session.py` configures:

- 1920x1080 viewport.
- `device_scale_factor=1.0`.
- highlighted elements enabled for Set-of-Marks grounding.
- per-thread session tracking.
- idle GC for stale thread contexts.

### Observation model

`core/observers.py` returns browser state as:

- text map (AXTree-derived via browser-use state pipeline), and
- screenshot block (JPEG, quality 60),

with safe fallback to text-only if screenshot fails.

### Dynamic UI drift protection

`mcp_servers/browser_tools.py` supports `expected_text` checks in write actions.

- If element index content does not match expectation, action aborts with explicit cognitive failure.
- This protects against index drift on dynamic SPAs.

### Scroll correctness

Scrolling uses runtime viewport height (not hardcoded pixel jumps), improving consistency across layouts.

## Safety and Deterministic Execution Guards

### HITL policy tiers

`core/hitl.py` applies tiered approval:

- Autonomous: always allowed.
- Allowed: allowed by default, can be denied.
- Not allowed: requires explicit permit/approval.

Unknown tools default to strict handling.

### Cardinality guard

Worker strict action loop enforces one stateful action per turn.

- Extra stateful calls are dropped.
- System error feedback is injected so the model can reconcile.

### Failure split: cognitive vs infrastructure

- Cognitive/tool errors are returned to model context for correction.
- Infrastructure failures (dead browser/session transport issues) trip a circuit breaker and stop execution to prevent token burn.

Code:

- `core/errors.py`
- `core/browser_session.py`
- `core/nodes/worker_nodes.py`

## Scheduling and Background Jobs

`core/scheduler.py` implements persistent async scheduling with:

- at/every/cron schedule modes.
- atomic JSON persistence (`data/cron/jobs.json`).
- bounded run history (`data/cron/runs`).
- isolated worker execution per job.
- optional prebound skill execution path.

The scheduler is backend-owned; the model uses scheduling tools rather than implementing scheduling logic itself.

## Tool Plugin Runtime

Tools are loaded dynamically from `mcp_servers/`.

`mcp_servers/__init__.py`:

- discovers plugin modules.
- imports each module `TOOL_REGISTRY`.
- builds global registry + metadata.
- assigns execution characteristics (category/stateful mode/observation mode).

This keeps tool surface extensible while preserving deterministic runtime categorization.

## Interfaces

EnterpriseClaw supports multiple front doors:

- Telegram (`interfaces/telegram.py`)
- WhatsApp bridge (`interfaces/whatsapp.py`)
- Web chat (`interfaces/web_chat.py`, `templates/`, `static/`)
- CLI (`interfaces/cli.py`)

The FastAPI app and graph lifecycle are managed in `app.py`.

## Project Layout

- `app.py`: FastAPI entrypoint + runtime wiring.
- `config/settings.py`: environment configuration.
- `core/graphs/`: Supervisor/Worker graph definitions.
- `core/nodes/`: node implementations and routing behaviors.
- `core/browser_session.py`: browser-use session manager.
- `core/observers.py`: environment observation adapters.
- `core/hitl.py`: approval policy engine.
- `core/scheduler.py`: background scheduler service.
- `mcp_servers/`: tool plugins.
- `memory/`: DB + vector index + retrieval.
- `skills/`: skill packs used for JIT binding.
- `tests/`: architecture, tools, memory, scheduler, and integration tests.

## Setup

### 1. Prerequisites

- Python 3.12+
- `uv`
- Chromium (for browser tools)

### 2. Install

```bash
git clone https://github.com/chetanreddyv/EnterpriseClaw.git
cd EnterpriseClaw
uv sync
uv run playwright install chromium
```

### 3. Configure environment

Guided wizard (sets up API keys and browsers automatically):

```bash
make setup
```

Or without Make:

```bash
uv run onboard
```

Manual baseline:

```bash
cp .env.example .env
```

Common required keys:

- `GOOGLE_API_KEY`
- `TELEGRAM_BOT_TOKEN`

Common optional keys:

- `OPENAI_API_KEY`
- `LM_STUDIO_BASE_URL`
- `LM_STUDIO_API_KEY`
- `GOOGLE_TOKEN_JSON`
- `WHATSAPP_ENABLED`
- `WHATSAPP_BRIDGE_URL`
- `WHATSAPP_BRIDGE_TOKEN`
- `ALLOWED_CHAT_IDS`
- `TELEGRAM_SECRET_TOKEN`

### 4. Optional Google Workspace auth

```bash
uv run python scripts/google_auth_helper.py
```

### 5. Run

API:

```bash
uv run python app.py
```

CLI:

```bash
uv run python app.py --cli
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Chat Commands

Thread-scoped runtime commands:

- `/models`: list configured providers.
- `/model <provider/model>`: switch active model for current thread.
- `/tools`: inspect tool policy.
- `/permit <tool_name>`: allow tool.
- `/deny <tool_name>`: deny tool.
- `/cron`, `/cron list`, `/cron cancel <job_id>`, `/cron cancel all`: manage jobs.

## API Surface

Main endpoints in `app.py`:

- `GET /health`
- `POST /webhook` (Telegram)
- `POST /api/v1/chat/{thread_id}`
- `POST /api/v1/chat/{thread_id}/resume`
- `POST /api/v1/system/{thread_id}/notify`
- `GET /chat`

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat/demo-thread \
  -H "Content-Type: application/json" \
  -d '{"user_input":"Summarize the latest AI headlines"}'
```

## Testing

Run tests:

```bash
uv run pytest -q
```

Focused suites:

```bash
uv run pytest tests/test_dynamic_worker_architecture.py -q
uv run pytest tests/test_observers.py -q
uv run pytest tests/test_browser_tools.py -q
uv run pytest tests/test_scheduler_runtime.py -q
```

## What Makes EnterpriseClaw SOTA in Practice

- Deterministic graph routing with explicit state contracts.
- Replace-only environment anchors for stable long-horizon efficiency.
- Hybrid durable + semantic memory retrieval.
- JIT skill retrieval with strict tool scoping.
- Multimodal browser grounding with AXTree-style text map + visual context.
- Typed infrastructure failure handling and circuit-breaking.
- Tiered HITL policy that balances safety and usability.

This repository is engineered for real, long-running agent workloads where reliability, controllability, and context efficiency matter more than demo-only benchmarks.
