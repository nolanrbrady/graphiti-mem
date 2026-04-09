from __future__ import annotations

import os
from pathlib import Path

import pytest

from graphiti_core.memory.engine import MemoryEngine
from graphiti_core.memory.models import MemoryKind

pytest.importorskip('kuzu')
pytest.importorskip('openai')


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.getenv('OPENAI_API_KEY'),
    reason='requires an OpenAI-compatible API key or local endpoint configured via OPENAI_API_KEY',
)
@pytest.mark.asyncio
async def test_conflicting_memories_surface_active_and_historical_facts(tmp_path: Path) -> None:
    (tmp_path / 'README.md').write_text('# Demo Project\n')
    (tmp_path / 'pyproject.toml').write_text(
        '[project]\nname = "demo-project"\nversion = "0.1.0"\nrequires-python = ">=3.10"\n'
    )
    MemoryEngine.init_project(tmp_path)

    async with await MemoryEngine.open(tmp_path) as engine:
        await engine.remember(
            kind=MemoryKind.decision,
            summary='Use Pattern X for ingestion',
            details='We currently use Pattern X for ingestion.',
        )
        await engine.remember(
            kind=MemoryKind.decision,
            summary='Use Pattern Y for ingestion',
            details='Pattern Y replaces Pattern X for ingestion because it is more reliable.',
        )
        recall = await engine.recall('current ingestion pattern')

    assert 'Active Facts' in recall
    assert 'Historical Facts' in recall
