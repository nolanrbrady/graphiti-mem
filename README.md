# Graphiti

Graphiti is a project-local memory engine for coding agents.

The intended product surface is now MCP-first for Codex or Claude Code. The CLI remains in the repo as a development and fallback interface.

The supported runtime surface in this repository is intentionally narrow:

- `graphiti mcp --transport stdio`
- MCP tools: `init_project`, `discover_history`, `list_history_sessions`, `read_history_session`, `import_history_sessions`, `apply_agents_instructions`, `store_memory`, `recall_memory`, `index_project`, `doctor`
- CLI fallback and dev tools: `graphiti init`, `graphiti recall`, `graphiti remember`, `graphiti index`, `graphiti doctor`
- Benchmark tooling: `python -m graphiti_core.memory.benchmark run`, `make benchmark-memory`, `make benchmark-memory-dogfood`

It stores repo-scoped memory under `.graphiti/` and uses Kuzu as the default local backend. Neo4j is retained as a fallback backend for the underlying graph engine, but the turnkey workflow is local-first and MCP-first.

## What It Stores

Graphiti is optimized for durable agent memory over time:

- decisions
- constraints
- patterns
- implementation notes
- workflows
- pitfalls
- indexed project artifacts

It preserves the parts of the original graph model that matter for memory:

- episodes as provenance
- entities and facts
- deduplication
- contradiction handling via invalidation
- fast local recall

## Quickstart

Requirements:

- Python 3.10+
- no separate model endpoint is required for MCP-first onboarding and history import
- an OpenAI-compatible endpoint is only optional if you want the engine itself to perform structured graph extraction

Install:

```bash
pip install graphiti-core
```

Or with Neo4j fallback support:

```bash
pip install "graphiti-core[neo4j]"
```

Initialize memory in a repo:

```bash
graphiti init
```

`graphiti init` now acts as a local development onboarding flow. It creates `.graphiti/`, detects project-local Codex or Claude transcript history from disk, bootstraps those sessions into source evidence plus durable memory by default, and can update `AGENTS.md` with a managed Graphiti block.
If you want Codex to know about Graphiti globally, `graphiti init --install-mcp` installs a `graphiti` MCP server entry into `~/.codex/config.toml` using the current Python executable.

Store a durable memory:

```bash
graphiti remember \
  --kind decision \
  --summary "Prefer pattern Y over pattern X" \
  --details "Pattern X caused retries and unstable behavior."
```

Recall relevant memory before exploring:

```bash
graphiti recall "How does this repo expect tests to run?"
```

Index the project:

```bash
graphiti index
graphiti index --changed
```

Inspect local status:

```bash
graphiti doctor
```

Run the stdio MCP server for Codex:

```bash
graphiti mcp --transport stdio
```

Run the deterministic benchmark and reward gate:

```bash
python -m graphiti_core.memory.benchmark run
make benchmark-memory
python -m graphiti_core.memory.benchmark run --mode dogfood --tier smoke
make benchmark-memory-dogfood
```

Useful benchmark commands:

```bash
python -m graphiti_core.memory.benchmark list
python -m graphiti_core.memory.benchmark inspect artifact-test-command
python -m graphiti_core.memory.benchmark doctor
python -m graphiti_core.memory.benchmark compare baseline.json candidate.json
```

Recommended Codex usage:

- register `graphiti mcp --transport stdio`
- let the agent call `init_project` during onboarding
- let the agent inspect prior local sessions with `discover_history`, `list_history_sessions`, and `read_history_session`
- let the agent write curated durable memory through `store_memory`
- use the CLI only as a local dev and smoke-test path
- keep the managed `AGENTS.md` block applied so explicit recall and write triggers stay visible to the agent

## Local State

Each project gets its own memory directory:

- `.graphiti/memory.kuzu`
- `.graphiti/config.toml`
- `.graphiti/index_state.json`
- `.graphiti/agent_instructions.md`

`graphiti init` writes fallback agent instructions under `.graphiti/agent_instructions.md` and can apply a managed Graphiti block directly into `AGENTS.md`.
That managed block now includes concrete recall triggers and durable-memory write triggers so the agent is nudged to use Graphiti proactively.

For normal Codex usage, prefer the MCP server and let the agent handle reasoning, summarization, and memory curation. Graphiti is responsible for local storage, provenance, retrieval, indexing, and managed project state.

## Indexing Model

`graphiti index` is explicit and incremental. It prioritizes high-signal project artifacts instead of dumping raw code:

- `AGENTS.md`
- `README*`
- manifests and tool configuration
- top-level package/module inventory
- selected docs
- optional recent git history summary

During onboarding, Graphiti can also register project-matching Codex and Claude conversations stored on disk as source evidence and distill durable memory from them. That bootstrap is scoped to the current repo, defaults to the last 90 days, and can take several minutes on larger histories.

## Fallback Modes

- Default runtime: Kuzu-backed local memory
- Structured graph extraction can still be enabled when an OpenAI-compatible LLM and embedder are configured
- Onboarding and transcript handling do not require a separate model endpoint when used through Codex MCP
- Neo4j: retained as a lower-level fallback backend, not the primary turnkey path

## Development

Install dev dependencies:

```bash
make install
```

Run formatting:

```bash
make format
```

Run lint and type checks:

```bash
make lint
```

Run the default test suite:

```bash
make test
```

Bring up Neo4j locally if you need to exercise the fallback backend:

```bash
docker-compose -f docker-compose.test.yml up
```

## Benchmarking

Graphiti now includes a deterministic benchmark intended to act as a verifiable reward for autoresearch loops. See [docs/benchmarking.md](docs/benchmarking.md) for the fixture schema, result schema, and authoring guidance.

- Primary entrypoint: `python -m graphiti_core.memory.benchmark run`
- Default suite: `deterministic_core`
- Default tier: `smoke`
- Modes:
  - `synthetic`: deterministic seeded corpus for stable regression tracking
  - `dogfood`: current repo artifacts plus local Codex history distilled into benchmark tasks
- Exit codes:
  - `0`: gate passed
  - `1`: gate failed
  - `2`: invalid setup, unsupported runtime, or corrupted fixtures

The benchmark runs paired execution:

- control: a deterministic naive source scan over synthetic artifacts, seeded memories, and synthetic Codex history
- treatment: `MemoryEngine.recall()` over the same seeded corpus

Both modes now emit the same staged metrics:

- `retrieval_score`
- `attribution_score`
- `answer_score`
- `capability_score`
- `efficiency_score`
- `task_score`

Dogfood mode keeps the same task/result schema, but builds the corpus from the current repo:

- copies high-signal local artifacts into a temporary benchmark project
- imports local Codex history for the current repo as source evidence
- deterministically distills a small set of durable memories from those sessions
- generates artifact, history, and multi-hop tasks from that local material

Task score reaches `100` only when the benchmark retrieves the right support, attaches explicit provenance IDs, answers correctly, and stays within hard budgets for retrieval calls and returned context size.

The default reward is only emitted when hard gates pass and is the mean normalized task score across the suite.

Legacy fixtures are still accepted and upgraded at load time, but new tasks should be authored directly against the v2 schema documented in [docs/benchmarking.md](docs/benchmarking.md).

The default deterministic suite does not require an external judge model. The optional `hybrid_extended` suite reserves room for future free-form judge-backed tasks.
