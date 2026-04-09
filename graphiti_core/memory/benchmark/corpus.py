from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from graphiti_core.memory.models import BootstrapDiscovery, BootstrapSession, MemoryKind

from .models import (
    BenchmarkScenarioEvent,
    BenchmarkScenarioEventKind,
    artifact_source_id,
    memory_source_id,
    session_source_id,
    thread_source_id,
)

BENCHMARK_NOW = datetime(2026, 4, 9, 12, 0, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class BenchmarkHistorySession:
    session_id: str
    title: str
    user_message: str
    assistant_message: str
    tokens_used: int
    created_at: datetime = BENCHMARK_NOW


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
    captured_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class BaselineSource:
    key: str
    source_type: str
    content: str
    provenance_ids: tuple[str, ...] = ()


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

# Blessed command for the deterministic memory benchmark: make benchmark-memory
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
The default reward path must not require an external model judge.
Each task is scored in stages: retrieval, attribution, answer, and efficiency.
Task score reaches 100 only when the right support is retrieved, provenance is attached, the answer is correct, and hard budgets are respected.
Reward is only returned when hard gates pass.
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
    'docs/legacy-benchmark.md': """# Legacy Benchmark Notes

The retired v1 benchmark used `make benchmark-memory-v1`.
That command is obsolete and should not be treated as the current target.
""",
    'docs/legacy-runtime.md': """# Legacy Runtime Notes

An earlier prototype targeted `>=3.9`.
Current project metadata should override this stale note.
""",
    'docs/legacy-judge.md': """# Legacy Judge Notes

An older experiment relied on an external model judge.
The default deterministic reward path should not use that networked design.
""",
}


HISTORY_SESSIONS: tuple[BenchmarkHistorySession, ...] = (
    BenchmarkHistorySession(
        session_id='session-pattern-y',
        title='Pattern Y migration',
        user_message='Pattern X caused retries during memory ingestion. What should we keep?',
        assistant_message='Prefer pattern Y over pattern X because it keeps ingestion deterministic.',
        tokens_used=1400,
        created_at=BENCHMARK_NOW - timedelta(days=10),
    ),
    BenchmarkHistorySession(
        session_id='session-search-first',
        title='Recall before search',
        user_message='How should agents explore this repo without wasting tokens?',
        assistant_message='Run Graphiti recall before broad file search and keep the returned context concise.',
        tokens_used=1200,
        created_at=BENCHMARK_NOW - timedelta(days=8),
    ),
    BenchmarkHistorySession(
        session_id='session-history-gap',
        title='History import gap',
        user_message='Is transcript import enough for recall quality?',
        assistant_message='No. Import raw sessions for provenance, then store durable memories for decisions and pitfalls.',
        tokens_used=1350,
        created_at=BENCHMARK_NOW - timedelta(days=6),
    ),
    BenchmarkHistorySession(
        session_id='session-benchmark-loop',
        title='Autoresearch reward loop',
        user_message='What makes the benchmark safe for unattended optimization?',
        assistant_message='Keep the default suite deterministic, CLI-first, and guarded by hard pass-fail thresholds before reward.',
        tokens_used=1500,
        created_at=BENCHMARK_NOW - timedelta(days=4),
    ),
    BenchmarkHistorySession(
        session_id='session-search-antipattern',
        title='Legacy search playbook',
        user_message='Should the agent start with broad repository search every time?',
        assistant_message='No. That old search-first playbook is stale and wastes tokens when durable memory already covers the task.',
        tokens_used=900,
        created_at=BENCHMARK_NOW - timedelta(days=2),
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
        captured_at=BENCHMARK_NOW - timedelta(days=10),
    ),
    BenchmarkMemorySeed(
        kind=MemoryKind.workflow,
        summary='Run Graphiti recall before broad file search',
        details='Use recall first so the agent can avoid broad token-hungry repository search when durable memory already covers the task.',
        source_agent='codex',
        session_id='session-search-first',
        thread_title='Recall before search',
        tags=('workflow', 'search'),
        captured_at=BENCHMARK_NOW - timedelta(days=8),
    ),
    BenchmarkMemorySeed(
        kind=MemoryKind.pitfall,
        summary='Transcript import alone is insufficient for deterministic recall',
        details='Import history sessions as source evidence, but also store durable memories for decisions, workflows, and pitfalls.',
        source_agent='codex',
        session_id='session-history-gap',
        thread_title='History import gap',
        tags=('history', 'pitfall'),
        captured_at=BENCHMARK_NOW - timedelta(days=6),
    ),
    BenchmarkMemorySeed(
        kind=MemoryKind.constraint,
        summary='Default benchmark must remain deterministic and offline-capable',
        details='The autoresearch reward path must not require an external model judge.',
        source_agent='codex',
        session_id='session-benchmark-loop',
        thread_title='Autoresearch reward loop',
        tags=('benchmark', 'constraint'),
        captured_at=BENCHMARK_NOW - timedelta(days=4),
    ),
    BenchmarkMemorySeed(
        kind=MemoryKind.implementation_note,
        summary='Reward is returned only after hard gates pass',
        details='Hard gates prevent false wins where token usage drops but answer accuracy or evidence coverage regresses. This keeps the reward path safe for unattended optimization.',
        source_agent='codex',
        session_id='session-benchmark-loop',
        thread_title='Autoresearch reward loop',
        tags=('benchmark', 'reward'),
        captured_at=BENCHMARK_NOW - timedelta(days=4),
    ),
    BenchmarkMemorySeed(
        kind=MemoryKind.pattern,
        summary='Pair naive source scan control against Graphiti recall treatment',
        details='Use a deterministic control retriever that scans synthetic artifacts, seeded memories, and session text directly. Search reduction is measured with a stable source-scan proxy in the deterministic suite.',
        source_agent='codex',
        session_id='session-benchmark-loop',
        thread_title='Autoresearch reward loop',
        tags=('benchmark', 'control'),
        captured_at=BENCHMARK_NOW - timedelta(days=4),
    ),
    BenchmarkMemorySeed(
        kind=MemoryKind.pitfall,
        summary='Legacy search-first guidance is stale',
        details='Do not start with broad repository search when durable memory already covers the task.',
        source_agent='codex',
        session_id='session-search-antipattern',
        thread_title='Legacy search playbook',
        tags=('search', 'stale'),
        captured_at=BENCHMARK_NOW - timedelta(days=2),
    ),
)


def materialize_project(root: Path) -> list[BaselineSource]:
    sources: list[BaselineSource] = []
    for relative_path, content in ARTIFACT_FILES.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        sources.append(
            BaselineSource(
                key=relative_path,
                source_type='artifact',
                content=content,
                provenance_ids=(artifact_source_id(relative_path),),
            )
        )

    for seed in MEMORY_SEEDS:
        provenance_ids = [memory_source_id(seed.kind.value, seed.summary)]
        if seed.thread_title:
            provenance_ids.append(thread_source_id(seed.thread_title))
        if seed.session_id:
            provenance_ids.append(session_source_id(seed.session_id))
        if seed.artifact_path:
            provenance_ids.append(artifact_source_id(seed.artifact_path))
        sources.append(
            BaselineSource(
                key=f'memory:{seed.kind.value}:{seed.summary}',
                source_type='memory',
                content=f'{seed.summary}\n{seed.details}\n{seed.thread_title}\n{seed.session_id}',
                provenance_ids=tuple(provenance_ids),
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
                provenance_ids=(
                    thread_source_id(session.title),
                    session_source_id(session.session_id),
                ),
            )
        )
    return sources


def write_codex_history(home: Path, project_root: Path) -> None:
    write_codex_history_sessions(home, project_root, HISTORY_SESSIONS)


def write_codex_history_sessions(
    home: Path,
    project_root: Path,
    sessions: tuple[BenchmarkHistorySession, ...] | list[BenchmarkHistorySession],
) -> None:
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

    for session in sessions:
        timestamp = int(session.created_at.timestamp() * 1000)
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


def temporal_event_provenance(event: BenchmarkScenarioEvent) -> dict[str, str]:
    provenance = {'Captured From': f'benchmark replay:{event.kind.value}'}
    if event.source_agent:
        provenance['Source Agent'] = event.source_agent
    if event.session_id:
        provenance['Session ID'] = event.session_id
    if event.thread_title:
        provenance['Thread Title'] = event.thread_title
    if event.artifact_path:
        provenance['Artifact Path'] = event.artifact_path
    return provenance


def temporal_event_support_ids(event: BenchmarkScenarioEvent) -> tuple[str, ...]:
    ids: list[str] = []
    if event.artifact_path:
        ids.append(artifact_source_id(event.artifact_path))
    if event.session_id:
        ids.append(session_source_id(event.session_id))
    if event.thread_title:
        ids.append(thread_source_id(event.thread_title))
    if event.summary:
        ids.append(
            memory_source_id(
                temporal_event_memory_kind(event).value,
                event.summary,
            )
        )
    return tuple(dict.fromkeys(identifier for identifier in ids if identifier))


def temporal_event_memory_kind(event: BenchmarkScenarioEvent) -> MemoryKind:
    if event.kind is BenchmarkScenarioEventKind.decision_update:
        return MemoryKind.decision
    if event.kind is BenchmarkScenarioEventKind.pitfall_update:
        return MemoryKind.pitfall
    if event.kind is BenchmarkScenarioEventKind.constraint_update:
        return MemoryKind.constraint
    return event.memory_kind or MemoryKind.implementation_note


def temporal_event_to_history_session(event: BenchmarkScenarioEvent) -> BenchmarkHistorySession:
    return BenchmarkHistorySession(
        session_id=event.session_id or event.event_id,
        title=event.thread_title or event.summary or event.event_id,
        user_message=event.user_message or 'What changed?',
        assistant_message=event.assistant_message or event.details or event.summary,
        tokens_used=900,
        created_at=event.timestamp,
    )


def temporal_event_to_baseline_source(event: BenchmarkScenarioEvent) -> BaselineSource:
    if event.kind is BenchmarkScenarioEventKind.artifact_snapshot:
        return BaselineSource(
            key=f'{event.event_id}:{event.artifact_path}',
            source_type='artifact',
            content=event.content,
            provenance_ids=temporal_event_support_ids(event),
        )

    if event.kind is BenchmarkScenarioEventKind.history_turn:
        content = '\n'.join(
            [
                event.thread_title or event.summary or event.event_id,
                f'User: {event.user_message}',
                f'Assistant: {event.assistant_message}',
            ]
        ).strip()
        return BaselineSource(
            key=f'history:{event.session_id or event.event_id}',
            source_type='history',
            content=content,
            provenance_ids=temporal_event_support_ids(event),
        )

    content = '\n'.join(
        part
        for part in (
            event.summary,
            event.details,
            event.thread_title,
            event.session_id,
        )
        if part
    )
    return BaselineSource(
        key=f'memory:{event.event_id}',
        source_type='memory',
        content=content,
        provenance_ids=temporal_event_support_ids(event),
    )


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
                provenance_ids=(artifact_source_id(relative),),
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
                provenance_ids=(
                    thread_source_id(session.title[:80]),
                    session_source_id(session.session_id),
                ),
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
                captured_at=session.created_at,
            )
        )
    return seeds


def build_dogfood_temporal_events(
    artifact_sources: list[BaselineSource],
    history_seeds: list[BenchmarkMemorySeed],
) -> list[BenchmarkScenarioEvent]:
    events: list[BenchmarkScenarioEvent] = []
    base_time = BENCHMARK_NOW - timedelta(days=max(2, len(artifact_sources) + len(history_seeds)))

    for index, source in enumerate(artifact_sources, start=1):
        events.append(
            BenchmarkScenarioEvent(
                event_id=f'dogfood-artifact-{index}',
                timestamp=base_time + timedelta(hours=index),
                kind=BenchmarkScenarioEventKind.artifact_snapshot,
                artifact_path=source.key,
                content=source.content,
                summary=f'Artifact snapshot for {source.key}',
                details=source.content[:400],
            )
        )

    for index, seed in enumerate(history_seeds, start=1):
        session_time = seed.captured_at or (base_time + timedelta(days=1, hours=index))
        events.append(
            BenchmarkScenarioEvent(
                event_id=f'dogfood-history-{index}',
                timestamp=session_time,
                kind=BenchmarkScenarioEventKind.history_turn,
                session_id=seed.session_id,
                thread_title=seed.thread_title,
                user_message=seed.summary,
                assistant_message=seed.details,
                summary=seed.summary,
                details=seed.details,
                source_agent=seed.source_agent,
            )
        )
        events.append(
            BenchmarkScenarioEvent(
                event_id=f'dogfood-memory-{index}',
                timestamp=session_time + timedelta(minutes=5),
                kind=BenchmarkScenarioEventKind.memory_seed,
                memory_kind=seed.kind,
                summary=seed.summary,
                details=seed.details,
                source_agent=seed.source_agent,
                session_id=seed.session_id,
                thread_title=seed.thread_title,
                tags=list(seed.tags),
            )
        )

    stale_candidates = [
        event
        for event in events
        if 'stale' in event.summary.lower()
        or 'legacy' in event.summary.lower()
        or 'obsolete' in event.summary.lower()
        or 'deprecated' in event.summary.lower()
    ]
    if stale_candidates and history_seeds:
        latest = max(
            history_seeds,
            key=lambda seed: seed.captured_at or BENCHMARK_NOW,
        )
        stale_event = stale_candidates[0]
        events.append(
            BenchmarkScenarioEvent(
                event_id='dogfood-supersession',
                timestamp=(latest.captured_at or BENCHMARK_NOW) + timedelta(minutes=10),
                kind=BenchmarkScenarioEventKind.pitfall_update,
                summary=f'Avoid stale guidance from {stale_event.summary}',
                details='Newer local history superseded this older guidance.',
                source_agent=latest.source_agent,
                session_id=latest.session_id,
                thread_title=latest.thread_title,
                supersedes=[stale_event.event_id],
            )
        )

    return sorted(events, key=lambda event: (event.timestamp, event.event_id))


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
