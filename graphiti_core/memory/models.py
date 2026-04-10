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
    database_path: str
    backend: BackendType = BackendType.kuzu
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
class BootstrapArtifactCandidate:
    artifact_path: str
    title: str
    artifact_type: str
    fingerprint: str
    content: str
    reasons: list[str] = field(default_factory=list)

    @property
    def content_length(self) -> int:
        return len(self.content)

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
class OnboardingDecisions:
    apply_agents_update: bool
    import_history: bool
    backend: BackendType
    history_days: int = 90


def default_structured_graph_refs() -> dict:
    return {
        'episode_uuids': [],
        'episodic_edge_uuids': [],
        'node_uuids': [],
        'edge_uuids': [],
        'community_uuids': [],
        'community_edge_uuids': [],
    }


def default_bootstrap_session_state() -> dict:
    return {
        'fingerprint': '',
        'status': '',
        'processed_at': '',
        'history_days': 90,
        'source_agent': '',
        'thread_title': '',
        'created_at': '',
        'source_path': '',
        'source_episode_uuids': [],
        'durable_memory_uuids': [],
        'structured_graph_refs': default_structured_graph_refs(),
    }


def default_bootstrap_artifact_state() -> dict:
    return {
        'artifact_path': '',
        'artifact_type': '',
        'reasons': [],
        'content_length': 0,
        'fingerprint': '',
        'indexed_fingerprint': '',
        'distilled_fingerprint': '',
        'source_episode_uuids': [],
        'index_episode_uuid': '',
        'durable_memory_uuids': [],
        'structured_graph_refs': default_structured_graph_refs(),
        'indexed_at': '',
        'processed_at': '',
    }


def default_semantic_bootstrap_state() -> dict:
    return {
        'bootstrap_pending': False,
        'bootstrap_completed_at': '',
        'bootstrap_history_days': 90,
        'last_checked_at': '',
        'eligible_sessions': 0,
        'artifact_candidate_count': 0,
        'artifact_completed_at': '',
        'structured_graph_available': False,
        'sessions': {},
        'artifacts': {},
    }


def default_index_state() -> dict:
    return {
        'artifacts': {},
        'semantic_bootstrap': default_semantic_bootstrap_state(),
    }


def ensure_index_state_shape(state: dict | None) -> dict:
    current = default_index_state()
    if not state:
        return current
    current['artifacts'].update(state.get('artifacts', {}))

    bootstrap_state = current['semantic_bootstrap']
    bootstrap_state.update(state.get('semantic_bootstrap', {}))
    bootstrap_state.setdefault('sessions', {})
    bootstrap_state.setdefault('bootstrap_pending', False)
    bootstrap_state.setdefault('bootstrap_completed_at', '')
    bootstrap_state.setdefault('bootstrap_history_days', 90)
    bootstrap_state.setdefault('last_checked_at', '')
    bootstrap_state.setdefault('eligible_sessions', 0)
    bootstrap_state.setdefault('artifact_candidate_count', 0)
    bootstrap_state.setdefault('artifact_completed_at', '')
    bootstrap_state.setdefault('structured_graph_available', False)
    bootstrap_state.setdefault('artifacts', {})

    legacy_state = state.get('history_bootstrap', {})
    if legacy_state:
        bootstrap_state['bootstrap_completed_at'] = bootstrap_state.get(
            'bootstrap_completed_at'
        ) or legacy_state.get('last_bootstrap_at', '')
        for session_id, raw_session in legacy_state.get('sessions', {}).items():
            session_state = default_bootstrap_session_state()
            session_state.update(raw_session)
            session_state['durable_memory_uuids'] = list(
                raw_session.get('durable_memory_uuids', raw_session.get('memory_episode_uuids', []))
            )
            session_state['status'] = session_state.get('status') or 'processed'
            session_state['processed_at'] = session_state.get('processed_at') or legacy_state.get(
                'last_bootstrap_at', ''
            )
            refs = default_structured_graph_refs()
            refs.update(raw_session.get('structured_graph_refs', {}))
            session_state['structured_graph_refs'] = refs
            bootstrap_state['sessions'].setdefault(session_id, session_state)

    for session_id, raw_session in list(bootstrap_state['sessions'].items()):
        session_state = default_bootstrap_session_state()
        session_state.update(raw_session)
        session_state['source_episode_uuids'] = list(raw_session.get('source_episode_uuids', []))
        session_state['durable_memory_uuids'] = list(
            raw_session.get('durable_memory_uuids', raw_session.get('memory_episode_uuids', []))
        )
        refs = default_structured_graph_refs()
        refs.update(raw_session.get('structured_graph_refs', {}))
        session_state['structured_graph_refs'] = refs
        bootstrap_state['sessions'][session_id] = session_state

    for artifact_path, raw_artifact in list(bootstrap_state['artifacts'].items()):
        artifact_state = default_bootstrap_artifact_state()
        artifact_state.update(raw_artifact)
        artifact_state['reasons'] = list(raw_artifact.get('reasons', []))
        artifact_state['source_episode_uuids'] = list(raw_artifact.get('source_episode_uuids', []))
        artifact_state['durable_memory_uuids'] = list(raw_artifact.get('durable_memory_uuids', []))
        refs = default_structured_graph_refs()
        refs.update(raw_artifact.get('structured_graph_refs', {}))
        artifact_state['structured_graph_refs'] = refs
        bootstrap_state['artifacts'][artifact_path] = artifact_state
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
