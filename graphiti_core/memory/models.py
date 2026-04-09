from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


class MemoryKind(str, Enum):
    decision = 'decision'
    constraint = 'constraint'
    pattern = 'pattern'
    implementation_note = 'implementation_note'
    workflow = 'workflow'
    pitfall = 'pitfall'
    index_artifact = 'index_artifact'


class BackendType(str, Enum):
    kuzu = 'kuzu'
    neo4j = 'neo4j'


MEMORY_KIND_PRIORITY: dict[MemoryKind, int] = {
    MemoryKind.decision: 0,
    MemoryKind.constraint: 1,
    MemoryKind.pattern: 2,
    MemoryKind.workflow: 3,
    MemoryKind.pitfall: 4,
    MemoryKind.implementation_note: 5,
    MemoryKind.index_artifact: 6,
}


@dataclass(slots=True)
class ProjectPaths:
    root: Path
    state_dir: Path
    database_path: Path
    config_path: Path
    index_state_path: Path
    agent_instructions_path: Path


@dataclass(slots=True)
class RuntimeConfig:
    project_name: str
    project_id: str
    backend: BackendType = BackendType.kuzu
    database_path: str
    llm_model: str = 'gpt-4.1-mini'
    llm_small_model: str = 'gpt-4.1-nano'
    llm_base_url: str = ''
    llm_api_key_env: str = 'OPENAI_API_KEY'
    embedder_model: str = 'text-embedding-3-small'
    embedder_base_url: str = ''
    embedder_api_key_env: str = 'OPENAI_API_KEY'
    neo4j_uri_env: str = 'NEO4J_URI'
    neo4j_user_env: str = 'NEO4J_USER'
    neo4j_password_env: str = 'NEO4J_PASSWORD'
    neo4j_database: str = 'neo4j'


@dataclass(slots=True)
class ParsedMemoryEpisode:
    uuid: str
    kind: MemoryKind
    summary: str
    details: str
    source: str
    tags: list[str] = field(default_factory=list)
    artifact_path: str = ''
    created_at: datetime | None = None
    raw_name: str = ''
    source_agent: str = ''
    session_id: str = ''
    thread_title: str = ''
    captured_from: str = ''


@dataclass(slots=True)
class BootstrapSession:
    source_agent: str
    session_id: str
    title: str
    created_at: datetime
    fingerprint: str
    content: str
    source_path: str

    def content_chunks(self, max_chars: int) -> list[str]:
        if len(self.content) <= max_chars:
            return [self.content]

        lines = self.content.splitlines()
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for line in lines:
            line_len = len(line) + 1
            if current and current_len + line_len > max_chars:
                chunks.append('\n'.join(current).strip())
                current = [line]
                current_len = line_len
                continue
            current.append(line)
            current_len += line_len
        if current:
            chunks.append('\n'.join(current).strip())
        return [chunk for chunk in chunks if chunk]


@dataclass(slots=True)
class BootstrapDiscovery:
    codex_sessions: list[BootstrapSession] = field(default_factory=list)
    claude_sessions: list[BootstrapSession] = field(default_factory=list)

    @property
    def total_sessions(self) -> int:
        return len(self.codex_sessions) + len(self.claude_sessions)

    def all_sessions(self) -> list[BootstrapSession]:
        return sorted(
            [*self.codex_sessions, *self.claude_sessions],
            key=lambda session: session.created_at,
            reverse=True,
        )


@dataclass(slots=True)
class OnboardingDecisions:
    apply_agents_update: bool
    import_history: bool
    backend: BackendType
    history_days: int = 90


def default_index_state() -> dict:
    return {
        'artifacts': {},
        'history_bootstrap': {
            'sessions': {},
            'last_bootstrap_at': '',
        },
    }


def ensure_index_state_shape(state: dict | None) -> dict:
    current = default_index_state()
    if not state:
        return current
    current['artifacts'].update(state.get('artifacts', {}))
    history_state = current['history_bootstrap']
    history_state.update(state.get('history_bootstrap', {}))
    history_state.setdefault('sessions', {})
    history_state.setdefault('last_bootstrap_at', '')
    return current


def hash_text_parts(parts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode('utf-8'))
    return digest.hexdigest()


def build_project_paths(root: Path) -> ProjectPaths:
    state_dir = root / '.graphiti'
    return ProjectPaths(
        root=root,
        state_dir=state_dir,
        database_path=state_dir / 'memory.kuzu',
        config_path=state_dir / 'config.toml',
        index_state_path=state_dir / 'index_state.json',
        agent_instructions_path=state_dir / 'agent_instructions.md',
    )


def build_project_id(root: Path) -> str:
    digest = hashlib.sha256(str(root.resolve()).encode('utf-8')).hexdigest()
    return digest[:16]
