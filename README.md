# Graphiti

Graphiti is a project-local memory engine for coding agents.

The supported product surface in this repository is intentionally narrow:

- `graphiti init`
- `graphiti recall "<task or question>"`
- `graphiti remember --kind ... --summary ...`
- `graphiti index`
- `graphiti doctor`
- `graphiti mcp --transport stdio`

It stores repo-scoped memory under `.graphiti/` and uses Kuzu as the default local backend. Neo4j is retained as a fallback backend for the underlying graph engine, but the turnkey workflow is local-first and CLI-first.

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
- an OpenAI-compatible endpoint if you want full structured extraction

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

`graphiti init` now acts as the onboarding flow. It creates `.graphiti/`, detects an OpenAI-compatible setup from the environment, optionally bootstraps project-local Codex or Claude transcript history from disk, and can update `AGENTS.md` with a managed Graphiti block.

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

## Local State

Each project gets its own memory directory:

- `.graphiti/memory.kuzu`
- `.graphiti/config.toml`
- `.graphiti/index_state.json`
- `.graphiti/agent_instructions.md`

`graphiti init` writes fallback agent instructions under `.graphiti/agent_instructions.md` and can apply a managed Graphiti block directly into `AGENTS.md`.

## Indexing Model

`graphiti index` is explicit and incremental. It prioritizes high-signal project artifacts instead of dumping raw code:

- `AGENTS.md`
- `README*`
- manifests and tool configuration
- top-level package/module inventory
- selected docs
- optional recent git history summary

During onboarding, Graphiti can also perform a one-time bootstrap from project-matching Codex and Claude conversations stored on disk. That bootstrap is scoped to the current repo and defaults to the last 90 days.

## Fallback Modes

- Default runtime: Kuzu-backed local memory
- Structured mode: enabled when an OpenAI-compatible LLM and embedder are configured
- Fallback mode: episode-only memory when no model endpoint is configured
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
