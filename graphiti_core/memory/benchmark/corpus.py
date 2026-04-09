from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from graphiti_core.memory.models import MemoryKind
from graphiti_core.memory.models import BootstrapDiscovery, BootstrapSession


@dataclass(frozen=True, slots=True)
class BenchmarkHistorySession:
    session_id: str
    title: str
    user_message: str
    assistant_message: str
    tokens_used: int


@dataclass(frozen=True, slots=True)
class BenchmarkMemorySeed:
    kind: MemoryKind
    summary: str
    details: str
    source_agent: str
    session_id: str
    thread_title: str
    tags: tuple[str, ...] = ()
    artifact_path: str = ''


@dataclass(frozen=True, slots=True)
class BaselineSource:
    key: str
    source_type: str
    content: str


DOGFOOD_PATTERNS: tuple[str, ...] = (
    'AGENTS.md',
    'README*',
    'pyproject.toml',
    'Makefile',
    'docker-compose*.yml',
    'docker-compose*.yaml',
)


ARTIFACT_FILES: dict[str, str] = {
    'README.md': """# Benchmark Demo Repo

Use `make test` to run tests.
Use `make benchmark-memory` to run the deterministic Graphiti benchmark.
Graphiti is expected to reduce token-hungry broad search by recalling durable memory first.
The default benchmark must remain deterministic and offline-capable.
""",
    'AGENTS.md': """# Agent Guidance

Run Graphiti recall before broad file search.
Prefer concise evidence-first answers.
Do not rely on transcript import alone; store durable memories for decisions and pitfalls.
""",
    'pyproject.toml': """[project]
name = "benchmark-demo"
version = "0.1.0"
requires-python = ">=3.10"

[tool.graphiti]
default_benchmark_suite = "deterministic_core"
default_benchmark_tier = "smoke"
""",
    'Makefile': """.PHONY: test benchmark-memory

test:
\tpython3 -m pytest tests/memory -m "not integration"

benchmark-memory:
\tpython3 -m graphiti_core.memory.benchmark run --suite deterministic_core --tier smoke
""",
    'docs/benchmarking.md': """# Benchmarking Graphiti

The benchmark uses paired execution:
- control scans synthetic corpus sources directly
- treatment calls Graphiti recall before exploration

The deterministic, offline-capable reward path is safe for unattended optimization.
Reward is only returned when hard gates pass.
Hard gates block wins that lower tokens while hurting accuracy or evidence coverage.
""",
    'docs/history.md': """# History Bootstrap Notes

Importing transcript sessions stores source evidence.
Source evidence alone is not enough for deterministic recall in the offline benchmark.
Durable memories should capture decisions, workflows, pitfalls, and implementation notes with provenance.
""",
    'docs/reward-loop.md': """# Reward Loop

The smoke tier should stay fast enough for every autoresearch iteration.
The full tier is for keep-discard decisions and final acceptance.
Search reduction is measured with a stable source-scan proxy in the deterministic suite.
""",
}


HISTORY_SESSIONS: tuple[BenchmarkHistorySession, ...] = (
    BenchmarkHistorySession(
        session_id='session-pattern-y',
        title='Pattern Y migration',
        user_message='Pattern X caused retries during memory ingestion. What should we keep?',
        assistant_message='Prefer pattern Y over pattern X because it keeps ingestion deterministic.',
        tokens_used=1400,
    ),
    BenchmarkHistorySession(
        session_id='session-search-first',
        title='Recall before search',
        user_message='How should agents explore this repo without wasting tokens?',
        assistant_message='Run Graphiti recall before broad file search and keep the returned context concise.',
        tokens_used=1200,
    ),
    BenchmarkHistorySession(
        session_id='session-history-gap',
        title='History import gap',
        user_message='Is transcript import enough for recall quality?',
        assistant_message='No. Import raw sessions for provenance, then store durable memories for decisions and pitfalls.',
        tokens_used=1350,
    ),
    BenchmarkHistorySession(
        session_id='session-benchmark-loop',
        title='Autoresearch reward loop',
        user_message='What makes the benchmark safe for unattended optimization?',
        assistant_message='Keep the default suite deterministic, CLI-first, and guarded by hard pass-fail thresholds before reward.',
        tokens_used=1500,
    ),
)


MEMORY_SEEDS: tuple[BenchmarkMemorySeed, ...] = (
    BenchmarkMemorySeed(
        kind=MemoryKind.decision,
        summary='Prefer pattern Y over pattern X',
        details='Pattern X caused retries. Pattern Y keeps ingestion deterministic and is the current default.',
        source_agent='codex',
        session_id='session-pattern-y',
        thread_title='Pattern Y migration',
        tags=('ingestion', 'determinism'),
    ),
    BenchmarkMemorySeed(
        kind=MemoryKind.workflow,
        summary='Run Graphiti recall before broad file search',
        details='Use recall first so the agent can avoid broad token-hungry repository search when durable memory already covers the task.',
        source_agent='codex',
        session_id='session-search-first',
        thread_title='Recall before search',
        tags=('workflow', 'search'),
    ),
    BenchmarkMemorySeed(
        kind=MemoryKind.pitfall,
        summary='Transcript import alone is insufficient for deterministic recall',
        details='Import history sessions as source evidence, but also store durable memories for decisions, workflows, and pitfalls.',
        source_agent='codex',
        session_id='session-history-gap',
        thread_title='History import gap',
        tags=('history', 'pitfall'),
    ),
    BenchmarkMemorySeed(
        kind=MemoryKind.constraint,
        summary='Default benchmark must remain deterministic and offline-capable',
        details='The autoresearch reward path must not require an external model judge.',
        source_agent='codex',
        session_id='session-benchmark-loop',
        thread_title='Autoresearch reward loop',
        tags=('benchmark', 'constraint'),
    ),
    BenchmarkMemorySeed(
        kind=MemoryKind.implementation_note,
        summary='Reward is returned only after hard gates pass',
        details='Hard gates prevent false wins where token usage drops but answer accuracy or evidence coverage regresses. This keeps the reward path safe for unattended optimization.',
        source_agent='codex',
        session_id='session-benchmark-loop',
        thread_title='Autoresearch reward loop',
        tags=('benchmark', 'reward'),
    ),
    BenchmarkMemorySeed(
        kind=MemoryKind.pattern,
        summary='Pair naive source scan control against Graphiti recall treatment',
        details='Use a deterministic control retriever that scans synthetic artifacts, seeded memories, and session text directly. Search reduction is measured with a stable source-scan proxy in the deterministic suite.',
        source_agent='codex',
        session_id='session-benchmark-loop',
        thread_title='Autoresearch reward loop',
        tags=('benchmark', 'control'),
    ),
)


def materialize_project(root: Path) -> list[BaselineSource]:
    sources: list[BaselineSource] = []
    for relative_path, content in ARTIFACT_FILES.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        sources.append(BaselineSource(key=relative_path, source_type='artifact', content=content))

    for seed in MEMORY_SEEDS:
        sources.append(
            BaselineSource(
                key=f'memory:{seed.kind.value}:{seed.summary}',
                source_type='memory',
                content=f'{seed.summary}\n{seed.details}\n{seed.thread_title}\n{seed.session_id}',
            )
        )

    for session in HISTORY_SESSIONS:
        transcript = '\n'.join(
            [
                f'User: {session.user_message}',
                f'Assistant: {session.assistant_message}',
            ]
        )
        sources.append(
            BaselineSource(
                key=f'history:{session.session_id}',
                source_type='history',
                content=f'{session.title}\n{transcript}',
            )
        )
    return sources


def write_codex_history(home: Path, project_root: Path) -> None:
    codex_dir = home / '.codex'
    session_dir = codex_dir / 'sessions' / '2026' / '04' / '09'
    session_dir.mkdir(parents=True, exist_ok=True)

    db_path = codex_dir / 'state_5.sqlite'
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS threads (
            id TEXT PRIMARY KEY,
            rollout_path TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            source TEXT NOT NULL,
            model_provider TEXT NOT NULL,
            cwd TEXT NOT NULL,
            title TEXT NOT NULL,
            sandbox_policy TEXT NOT NULL,
            approval_mode TEXT NOT NULL,
            tokens_used INTEGER NOT NULL DEFAULT 0,
            has_user_event INTEGER NOT NULL DEFAULT 0,
            archived INTEGER NOT NULL DEFAULT 0,
            archived_at INTEGER,
            git_sha TEXT,
            git_branch TEXT,
            git_origin_url TEXT,
            cli_version TEXT NOT NULL DEFAULT '',
            first_user_message TEXT NOT NULL DEFAULT '',
            agent_nickname TEXT,
            agent_role TEXT,
            memory_mode TEXT NOT NULL DEFAULT 'enabled',
            model TEXT,
            reasoning_effort TEXT,
            agent_path TEXT
        )
        """
    )

    timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
    for session in HISTORY_SESSIONS:
        rollout_path = session_dir / f'rollout-{session.session_id}.jsonl'
        rollout_records = [
            {
                'type': 'event_msg',
                'payload': {'type': 'user_message', 'message': session.user_message},
            },
            {
                'type': 'event_msg',
                'payload': {
                    'type': 'agent_message',
                    'message': session.assistant_message,
                    'phase': 'final',
                },
            },
        ]
        rollout_path.write_text(''.join(json.dumps(record) + '\n' for record in rollout_records))
        connection.execute(
            """
            INSERT OR REPLACE INTO threads (
                id, rollout_path, created_at, updated_at, source, model_provider, cwd, title,
                sandbox_policy, approval_mode, tokens_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.session_id,
                str(rollout_path),
                timestamp,
                timestamp,
                'vscode',
                'openai',
                str(project_root),
                session.title,
                'workspace-write',
                'never',
                session.tokens_used,
            ),
        )

    connection.commit()
    connection.close()


def seeded_memory_provenance(seed: BenchmarkMemorySeed) -> dict[str, str]:
    return {
        'Source Agent': seed.source_agent,
        'Session ID': seed.session_id,
        'Thread Title': seed.thread_title,
        'Captured From': 'benchmark fixture history',
    }


def collect_dogfood_artifacts(project_root: Path, max_files: int = 16) -> list[BaselineSource]:
    files: list[Path] = []
    for pattern in DOGFOOD_PATTERNS:
        files.extend(sorted(project_root.glob(pattern)))
    for subdir in ('docs', 'spec'):
        directory = project_root / subdir
        if directory.exists():
            files.extend(sorted(directory.rglob('*.md')))

    sources: list[BaselineSource] = []
    seen: set[str] = set()
    for path in files:
        if not path.is_file():
            continue
        try:
            relative = str(path.relative_to(project_root))
        except ValueError:
            continue
        if relative in seen or '.graphiti' in path.parts:
            continue
        seen.add(relative)
        sources.append(
            BaselineSource(
                key=relative,
                source_type='artifact',
                content=path.read_text(errors='ignore')[:12_000],
            )
        )
        if len(sources) >= max_files:
            break
    return sources


def session_excerpt(session: BootstrapSession) -> tuple[str, str]:
    user = ''
    assistant = ''
    for line in session.content.splitlines():
        if line.startswith('User: ') and not user:
            user = line.removeprefix('User: ').strip()
        elif line.startswith('Assistant: ') and not assistant:
            assistant = line.removeprefix('Assistant: ').strip()
        if user and assistant:
            break
    return user, assistant


def build_dogfood_history_sources(
    discovery: BootstrapDiscovery,
    limit: int = 4,
) -> list[BaselineSource]:
    sources: list[BaselineSource] = []
    for session in discovery.all_sessions()[:limit]:
        sources.append(
            BaselineSource(
                key=f'history:{session.title[:80]}',
                source_type='history',
                content=session.content,
            )
        )
    return sources


def _history_memory_kind(text: str) -> MemoryKind:
    normalized = text.lower()
    if 'must' in normalized or 'required' in normalized or 'constraint' in normalized:
        return MemoryKind.constraint
    if 'prefer' in normalized or 'default' in normalized:
        return MemoryKind.decision
    if 'avoid' in normalized or 'not enough' in normalized or 'pitfall' in normalized:
        return MemoryKind.pitfall
    if 'run ' in normalized or 'use ' in normalized or 'inspect' in normalized:
        return MemoryKind.workflow
    return MemoryKind.implementation_note


def _first_sentence(text: str, fallback: str) -> str:
    parts = [part.strip() for part in re.split(r'(?<=[.!?])\s+', text.strip()) if part.strip()]
    if parts:
        return parts[0][:160]
    return fallback[:160]


def distill_history_seeds(
    discovery: BootstrapDiscovery,
    limit: int = 4,
) -> list[BenchmarkMemorySeed]:
    seeds: list[BenchmarkMemorySeed] = []
    for session in discovery.all_sessions()[:limit]:
        user, assistant = session_excerpt(session)
        summary_source = assistant or user or session.title
        summary = _first_sentence(summary_source, session.title)
        details_parts = [part for part in (user, assistant) if part]
        details = '\n'.join(details_parts) or session.content[:800]
        seeds.append(
            BenchmarkMemorySeed(
                kind=_history_memory_kind(summary_source),
                summary=summary,
                details=details,
                source_agent=session.source_agent,
                session_id=session.session_id,
                thread_title=session.title[:120],
                tags=('dogfood', 'history'),
            )
        )
    return seeds


class HomeOverride:
    def __init__(self, home: Path):
        self.home = home
        self.previous_home = os.environ.get('HOME')

    def __enter__(self) -> None:
        os.environ['HOME'] = str(self.home)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.previous_home is None:
            os.environ.pop('HOME', None)
        else:
            os.environ['HOME'] = self.previous_home
