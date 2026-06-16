# ToolForge

ToolForge is an open-source research service that measures **how well language models can build agent pipelines and then self-evaluate those pipelines in production**.

It does this with two cooperating roles and an evaluation layer:

- A **Creator agent** ingests a natural-language description of an automation use case and designs a set of tools (MCP-exposed) that solve it — proposing, validating, and promoting each tool iteratively.
- A **Consumer agent** then executes the use case using only those tools, exactly as any downstream client would.
- A **Judge** layer scores the resulting runs — reliability, tool selection, structural stability, and more — so a model's ability to *create* a working pipeline and to *critique and improve it* can be benchmarked over time.

Every interaction is recorded, which is what makes the whole loop measurable and reproducible.

The project is framework-agnostic on the consumption side: tools are exposed via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io), so any MCP-capable client — Claude Desktop, Cursor, or the bundled Consumer agent — can consume them without modification. The Creator's own working surface is also MCP-based, enabling both fully autonomous runs and human-in-the-loop collaboration through any MCP client.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  toolforge CLI                                              │
│                                                             │
│  creator run ──► toolforge-mcp-creator (subprocess, stdio) │
│  │               ├─ propose / validate / promote / deprecate│
│  │               ├─ toolforge-registry  (SQLite + files)   │
│  │               └─ toolforge-sandbox   (uv / Docker)      │
│  │                                                         │
│  │  LLM ◄── toolforge-core (Anthropic / OpenAI-compat)    │
│  └─ toolforge-tui  (Textual interactive session)           │
│                                                             │
│  consumer run ─► toolforge-mcp-usecase (subprocess, stdio) │
│                  ├─ call active tools                       │
│                  ├─ toolforge-registry                      │
│                  └─ toolforge-sandbox                       │
│                                                             │
│  Runs are recorded by toolforge-telemetry, then scored by  │
│  toolforge-judge (reliability / selection / stability)     │
└─────────────────────────────────────────────────────────────┘
```

| Package | Role |
|---|---|
| `toolforge-core` | LLM abstraction (Anthropic + OpenAI-compat), agent loop, MCP client, `CreatorAgent`, `ConsumerAgent`, config |
| `toolforge-registry` | Filesystem + per-run SQLite store for use cases, runs, and tool versions |
| `toolforge-sandbox` | Isolated Python execution for handler code (`uv` or Docker mode) |
| `toolforge-mcp-creator` | MCP server exposing meta-tools to the Creator LLM |
| `toolforge-mcp-usecase` | MCP server exposing active run tools to the Consumer LLM |
| `toolforge-judge` | Metrics + Judges that score recorded runs and feed improvements back to the Creator |
| `toolforge-tui` | Textual TUI for interactive Creator sessions |
| `toolforge-telemetry` | Records creation and execution events for later evaluation |
| `toolforge-cli` | `toolforge` CLI entry point, wires everything together |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.11+** | 3.14 recommended; uses `tomllib` (stdlib) |
| **[uv](https://docs.astral.sh/uv/)** | Workspace dependency management |
| **Docker** | Required only when `mode = "docker"` in `[sandbox]`. Optional — the default `uv` mode runs handlers locally without Docker. |
| **LLM API key** | Anthropic, OpenAI, or any OpenAI-compatible provider (Z.ai, OpenRouter, vLLM) |

---

## Installation

```bash
git clone https://github.com/your-org/toolforge
cd toolforge
uv sync --all-extras
```

This installs all workspace packages and their dependencies into a single shared `.venv`. The `toolforge` CLI is immediately available:

```bash
uv run toolforge --help
```

---

## Quick Start

### 1. Configure

Copy the example config and edit it for your provider:

```bash
cp toolforge.example.toml toolforge.toml
```

The Creator and Consumer system prompts **ship with the repository** in `prompts/` (`creator_system.md`, `consumer_system.md`, and the judge prompts). You don't need to create anything — the default config already points at them. Edit those files if you want to change how the agents behave; nothing else is required to get started.

### 2. Choose a sandbox mode

The sandbox has two execution modes, set via `mode` in `[sandbox]` of `toolforge.toml`:

**`uv` mode (default)** — handler code runs in a local subprocess via `uv run`. No Docker required. Dependencies are installed into an ephemeral uv cache. Suitable for development and environments without Docker.

```toml
[sandbox]
timeout_seconds = 30
mode            = "uv"
```

**`docker` mode** — handler code runs inside an isolated Docker container. Only `/inputs` (read-only) and `/outputs` (read-write) are mounted. The host filesystem is not accessible.

```toml
[sandbox]
timeout_seconds = 30
mode            = "docker"
image           = "toolforge-sandbox:latest"
```

To use docker mode, build the image first:

```bash
uv run toolforge sandbox build
```

This builds `toolforge-sandbox:latest` from `Dockerfile.sandbox` (Python 3.12-slim, `uv` pre-installed, non-root `sandbox` user).

### 3. Launch the TUI (optional shortcut)

Instead of driving everything from the CLI, you can open the full interactive TUI directly:

```bash
uv run toolforge tui
```

This opens the **Selector screen** — a tree view of all use cases and runs. From there you can:
- Press `n` to create a new use case
- Select a draft run and press `Ctrl+P` → **Open Creator** to enter the Creator screen
- Select a validated run and press `Ctrl+P` → **Open Consumer** to test it interactively
- Press `v` to validate a run, `f` to fork it, or `e` to unlock and re-edit it

The steps below describe the same workflow from the CLI for scripting purposes.

### 4. Create a use case

This example is LLM-only — the tool reasons over text and needs no external service:

```bash
uv run toolforge usecase create \
  --id text_summarizer \
  --prompt "Summarize a block of text into a few concise bullet points."
```

Or read the prompt from a file:

```bash
uv run toolforge usecase create --id text_summarizer --prompt-file use_cases/summary.md
```

### 5. Create a run and start the Creator session

```bash
run_id=$(uv run toolforge run create --usecase text_summarizer)
echo "Run: $run_id"

uv run toolforge creator run \
  --usecase text_summarizer \
  --run "$run_id"
```

The Textual TUI opens. Type natural-language instructions to the Creator agent — it will propose tools, run them in the sandbox, iterate, and ask you to validate the result when done.

Example Creator session (TUI):

```
You: Design a tool that summarizes arbitrary text into 3-5 bullet points using an
     LLM. Test it on a short paragraph before promoting.

  ⚙ propose_tool
    (name='summarize_text', description='Summarize text into concise bullets...')
  ⚙ validate_in_sandbox
    (name='summarize_text', version=1, test_args='{"text": "ToolForge is..."}')
  ⚙ promote_tool
    (name='summarize_text', version=1)
  ⚙ request_human_validation
```

Press **Ctrl+C** to exit the TUI when the agent has finished.

### 6. Validate the run (human gate)

Inspect the tools that were created:

```bash
uv run toolforge run tools --usecase text_summarizer --run "$run_id"
```

When satisfied, lock the run as immutable:

```bash
uv run toolforge run validate --usecase text_summarizer --run "$run_id"
```

After validation the run is frozen — no further tool changes are possible. To iterate, fork it into a new draft run (see [Iterating](#iterating)).

### 7. Run the Consumer

```bash
uv run toolforge consumer run \
  --usecase text_summarizer \
  --run "$run_id" \
  --task "Summarize: ToolForge studies how well models can build and self-evaluate agent pipelines. A Creator designs tools, a Consumer runs them, and a Judge scores the result."
```

The Consumer agent connects to the Usecase MCP server, calls the active tools, and streams its response to the terminal. Tool calls appear on stderr so you can pipe just the answer:

```bash
uv run toolforge consumer run ... --task "..." 2>/dev/null
```

### Iterating

Once a run is validated you cannot modify its tools. Fork it to start a new draft that inherits the existing tools:

```bash
new_run=$(uv run toolforge run fork --usecase text_summarizer --from "$run_id")
uv run toolforge creator run --usecase text_summarizer --run "$new_run"
```

---

## Configuration reference

`toolforge.toml` is loaded from the current directory by default. Pass `--config <path>` to any command to override. System prompt paths are relative to the config file.

### Anthropic

```toml
[llm.providers.anthropic]
api_key_env = "ANTHROPIC_API_KEY"   # name of the env var holding the key

[llm.creator]
provider           = "anthropic"
model              = "claude-opus-4-7"
temperature        = 0.7
max_tokens         = 8192
system_prompt_file = "prompts/creator_system.md"   # ships with the repo

[llm.consumer]
provider           = "anthropic"
model              = "claude-sonnet-4-6"
temperature        = 0.3
max_tokens         = 4096
system_prompt_file = "prompts/consumer_system.md"

[sandbox]
timeout_seconds = 30
mode            = "uv"                        # "uv" (default) | "docker"
image           = "toolforge-sandbox:latest"  # used when mode = "docker"

[tui]
theme     = "dark"
log_level = "INFO"
```

### OpenAI-compatible (Z.ai / OpenRouter / vLLM)

```toml
[llm.providers.zai]
api_key_env = "ZAI_API_KEY"
base_url    = "https://api.z.ai/v1"

[llm.creator]
provider           = "zai"
model              = "claude-opus-4-7"
temperature        = 0.7
max_tokens         = 8192
system_prompt_file = "prompts/creator_system.md"

[llm.consumer]
provider           = "zai"
model              = "claude-sonnet-4-6"
temperature        = 0.3
max_tokens         = 4096
system_prompt_file = "prompts/consumer_system.md"
```

### Multiple providers

You can define several providers and mix them between Creator and Consumer:

```toml
[llm.providers.anthropic]
api_key_env = "ANTHROPIC_API_KEY"

[llm.providers.openrouter]
api_key_env = "OPENROUTER_API_KEY"
base_url    = "https://openrouter.ai/api/v1"

[llm.creator]
provider = "anthropic"
model    = "claude-opus-4-7"
...

[llm.consumer]
provider = "openrouter"
model    = "google/gemini-2.5-pro"
...
```

---

## CLI reference

All commands accept `--data-root <path>` (default: `data`) to set the registry location.

```
toolforge
├── tui       [--config <path>]   # full interactive launcher (selector → creator / consumer)
│
├── usecase
│   ├── create  --id <id>  (--prompt <text> | --prompt-file <path>)
│   └── list
│
├── run
│   ├── create    --usecase <id>
│   ├── list      --usecase <id>
│   ├── validate  --usecase <id>  --run <id>
│   ├── fork      --usecase <id>  --from <id>
│   └── tools     --usecase <id>  --run <id>
│
├── sandbox
│   └── build  [--tag <tag>]  [--no-cache]
│
├── creator
│   └── run  --usecase <id>  --run <id>
│            [--config <path>]
│
└── consumer
    └── run  --usecase <id>  --run <id>  --task <text>
             [--input-file <path>]...  [--config <path>]
```

`run create` prints only the run ID to stdout, making it suitable for shell scripting:

```bash
run_id=$(uv run toolforge run create --usecase my_uc)
```

---

## Data layout

The registry lives under `--data-root` (default `./data`):

```
data/
└── usecases/
    └── text_summarizer/
        ├── usecase.json
        ├── prompt.md
        └── runs/
            └── r_20260524_a1b2c3/
                ├── run.json
                ├── registry.db          # SQLite (WAL mode)
                └── tools/
                    └── summarize_text/
                        ├── v1.py        # handler source
                        └── v1.json      # JSON Schema
```

Runs are never modified in place after `validate`. Fork to iterate.

---

## Evaluation

Every creator action and tool call is recorded so runs can be replayed and scored. The `toolforge-judge` package turns those recordings into metrics across several families — reliability (error/retry/latency), tool contribution, structural stability, and LLM-scored selection/parameter quality — and a set of Judges that produce per-tool feedback for the Creator to act on.

This is what lets ToolForge answer its central question: not just *did the pipeline run*, but *how good was the model at building it, and at improving it once it saw production behavior*.

> Telemetry plumbing and the Judge scoring model are still evolving; see `architecture.md` for the current design.

---

## Development

Run the tests for any package with:

```bash
uv run python -m pytest packages/toolforge-core/tests/ -v
```

Run all suites:

```bash
for pkg in toolforge-core toolforge-registry toolforge-sandbox toolforge-telemetry \
           toolforge-mcp-creator toolforge-mcp-usecase toolforge-judge \
           toolforge-tui toolforge-cli; do
  uv run python -m pytest packages/$pkg/tests/ -q
done
```

### Package dependency graph

```
toolforge-cli
├── toolforge-core          (LLM, agents, config)
│   └── mcp
├── toolforge-registry      (filesystem + SQLite)
├── toolforge-sandbox       (uv / Docker)
├── toolforge-mcp-creator
│   ├── toolforge-registry
│   ├── toolforge-sandbox
│   └── toolforge-telemetry
├── toolforge-mcp-usecase
│   ├── toolforge-registry
│   ├── toolforge-sandbox
│   └── toolforge-telemetry
├── toolforge-judge
│   ├── toolforge-registry
│   └── toolforge-telemetry
├── toolforge-tui
│   └── toolforge-core
└── toolforge-telemetry
```

All inter-package dependencies are resolved via the `uv` workspace — no publishing required.
</content>
