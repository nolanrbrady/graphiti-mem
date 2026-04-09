.PHONY: install format lint test all check benchmark-memory benchmark-memory-dogfood

# Define variables
PYTHON = python3
UV = uv
PYTEST = $(UV) run pytest
RUFF = $(UV) run ruff
PYRIGHT = $(UV) run pyright

# Default target
all: format lint test

# Install dependencies
install:
	$(UV) sync --extra dev

# Format code
format:
	$(RUFF) check --select I --fix
	$(RUFF) format

# Lint code
lint:
	$(RUFF) check
	$(PYRIGHT) ./graphiti_core 

# Run tests
test:
	$(PYTEST) tests/memory -m "not integration"

# Run the deterministic Graphiti memory benchmark
benchmark-memory:
	$(UV) run python -m graphiti_core.memory.benchmark run --suite deterministic_core --tier smoke

benchmark-memory-dogfood:
	$(UV) run python -m graphiti_core.memory.benchmark run --mode dogfood --tier smoke

# Run format, lint, and test
check: format lint test
