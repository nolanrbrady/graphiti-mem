# Repository Guidelines

## Project Structure & Module Organization
Graphiti's supported product surface is now a CLI-first, project-local memory engine. The main entrypoint lives in `graphiti_core/memory/` (`cli.py`, `engine.py`, `config.py`, `models.py`). The retained core graph library lives under `graphiti_core/` with domain modules such as `nodes.py`, `edges.py`, `models/`, `search/`, and `utils/maintenance/` for ingestion, resolution, and retrieval. Database drivers in `graphiti_core/driver/` support Kuzu as the default local backend and Neo4j as a fallback backend. Additional retained modules include `cross_encoder/` (OpenAI reranker only), `namespaces/`, and `migrations/`. Tests now focus on the turnkey memory workflow under `tests/memory/`. Tooling manifests live at the repo root, including `pyproject.toml`, `Makefile`, `pytest.ini`, and `docker-compose.test.yml`.

## Build, Test, and Development Commands
- Runtime environment: use Miniconda and activate the `graphiti` environment before running repo commands (`source ~/miniconda3/bin/activate graphiti` or `conda activate graphiti` once conda is on `PATH`).
- `make install`: install the dev environment (`uv sync --extra dev`).
- `make format`: run `ruff` to sort imports and apply the canonical formatter.
- `make lint`: execute `ruff` plus `pyright` type checks against `graphiti_core`.
- `make test`: run the default local-memory test suite (`uv run pytest tests/memory -m "not integration"`).
- `make check`: run format, lint, and test in sequence.
- `uv run pytest tests/path/test_file.py`: target a specific module or test selection.
- `docker-compose -f docker-compose.test.yml up`: provision a local Neo4j instance for fallback-backend work.

## Coding Style & Naming Conventions
Python code uses 4-space indentation, 100-character lines, and prefers single quotes as configured in `pyproject.toml`. Modules, files, and functions stay snake_case; Pydantic models in `graphiti_core/models` use PascalCase with explicit type hints. Keep side-effectful code inside drivers or adapters (`graphiti_core/driver`, `graphiti_core/memory`, `graphiti_core/utils`) and rely on pure helpers elsewhere. Run `make format` before committing to normalize imports and docstring formatting.

## Testing Guidelines
Author tests under `tests/memory/`, naming files `test_<feature>.py` and functions `test_<behavior>`. Use `@pytest.mark.integration` only for cases that require real external services such as a configured OpenAI-compatible endpoint; `make test` excludes these by default. Async tests run automatically via `asyncio_mode = auto` in `pytest.ini`. Reproduce regressions with a failing test first and validate fixes via `uv run pytest -k "pattern"` after activating the Miniconda `graphiti` environment. Start Neo4j through `docker-compose.test.yml` only when you are exercising the retained fallback backend.

## Commit & Pull Request Guidelines
Commits use an imperative, present-tense summary (for example, `simplify kuzu recall path`) optionally suffixed with the PR number as seen in history (`(#927)`). Always prefer atomic commits with discrete, self-contained units of work. Squash fixups and keep unrelated changes isolated. Pull requests should include: a concise description, linked tracking issue, notes about CLI or storage impacts, and logs when behavior changes. Confirm `make lint` and `make test` pass locally, and update docs when public interfaces shift.
