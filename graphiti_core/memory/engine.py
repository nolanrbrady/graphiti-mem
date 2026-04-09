from __future__ import annotations

import asyncio
import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search import search
from graphiti_core.search.search_config import (
    DEFAULT_SEARCH_LIMIT,
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    EpisodeReranker,
    EpisodeSearchConfig,
    EpisodeSearchMethod,
    NodeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
    SearchResults,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.tracer import NoOpTracer
from graphiti_core.utils.datetime_utils import utc_now

from .config import (
    apply_agent_instructions,
    detect_project_root,
    initialize_project_files,
    load_index_state,
    load_runtime_config,
    save_index_state,
)
from .history import discover_project_history
from .lock import project_lock
from .models import (
    MEMORY_KIND_PRIORITY,
    BackendType,
    BootstrapDiscovery,
    MemoryKind,
    ParsedMemoryEpisode,
    ProjectPaths,
    RuntimeConfig,
    build_project_paths,
)

RECALL_LIMIT = 8
ARTIFACT_CONTENT_LIMIT = 12000
BOOTSTRAP_CHUNK_LIMIT = 10_000
BOOTSTRAP_WARN_THRESHOLD = 25


class NullLLMClient(LLMClient):
    def __init__(self):
        super().__init__(config=None, cache=False)

    async def _generate_response(self, *args, **kwargs):
        raise RuntimeError('LLM client is not configured for this Graphiti project.')


class NullEmbedder(EmbedderClient):
    async def create(self, input_data):
        return [0.0] * 16


class NullCrossEncoder(CrossEncoderClient):
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return [(passage, 0.0) for passage in passages]


class ArtifactSummary(BaseModel):
    summary: str
    key_points: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class IndexedArtifact:
    key: str
    title: str
    body: str
    fingerprint: str
    artifact_path: str = ''


class MemoryEngine:
    def __init__(
        self,
        project: ProjectPaths,
        config: RuntimeConfig,
        driver: GraphDriver,
        clients: GraphitiClients,
        graphiti: Any | None,
        structured_memory_enabled: bool,
    ):
        self.project = project
        self.config = config
        self.driver = driver
        self.clients = clients
        self.graphiti = graphiti
        self.structured_memory_enabled = structured_memory_enabled
        self.lock_path = self.project.state_dir / 'memory.lock'

    @classmethod
    async def open(cls, start: Path | None = None) -> MemoryEngine:
        root = detect_project_root((start or Path.cwd()).resolve())
        project = build_project_paths(root)
        config = load_runtime_config(project.config_path)
        database_path = Path(config.database_path)
        if not database_path.is_absolute():
            database_path = root / database_path
        project = ProjectPaths(
            root=project.root,
            state_dir=project.state_dir,
            database_path=database_path,
            config_path=project.config_path,
            index_state_path=project.index_state_path,
            agent_instructions_path=project.agent_instructions_path,
        )

        driver = await cls._open_driver_with_retry(config, project)
        await driver.build_indices_and_constraints()

        llm_client: LLMClient = NullLLMClient()
        embedder: EmbedderClient = NullEmbedder()
        graphiti = None
        structured_memory_enabled = False

        llm_available = bool(config.llm_base_url or os.getenv(config.llm_api_key_env))
        embedder_available = bool(
            config.embedder_base_url
            or config.llm_base_url
            or os.getenv(config.embedder_api_key_env)
            or os.getenv(config.llm_api_key_env)
        )

        if llm_available and embedder_available:
            try:
                from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
                from graphiti_core.graphiti import Graphiti
                from graphiti_core.llm_client import LLMConfig as OpenAILLMConfig
                from graphiti_core.llm_client import OpenAIClient

                llm_api_key = os.getenv(config.llm_api_key_env) or 'graphiti-local'
                embedder_api_key = os.getenv(config.embedder_api_key_env) or llm_api_key

                llm_client = OpenAIClient(
                    config=OpenAILLMConfig(
                        api_key=llm_api_key,
                        model=config.llm_model,
                        small_model=config.llm_small_model,
                        base_url=config.llm_base_url or None,
                        temperature=0,
                    ),
                    reasoning=None,
                    verbosity=None,
                )
                embedder = OpenAIEmbedder(
                    config=OpenAIEmbedderConfig(
                        api_key=embedder_api_key,
                        embedding_model=config.embedder_model,
                        base_url=(config.embedder_base_url or config.llm_base_url or None),
                    )
                )
                graphiti = Graphiti(
                    graph_driver=driver,
                    llm_client=llm_client,
                    embedder=embedder,
                    cross_encoder=NullCrossEncoder(),
                    trace_span_prefix='graphiti.memory',
                )
                structured_memory_enabled = True
            except ImportError:
                graphiti = None
                structured_memory_enabled = False

        clients = GraphitiClients(
            driver=driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=NullCrossEncoder(),
            tracer=NoOpTracer(),
        )

        return cls(project, config, driver, clients, graphiti, structured_memory_enabled)

    @classmethod
    async def _open_driver_with_retry(
        cls,
        config: RuntimeConfig,
        project: ProjectPaths,
    ) -> GraphDriver:
        if config.backend is not BackendType.kuzu:
            return cls._build_driver(config, project)

        last_error: Exception | None = None
        for _ in range(50):
            try:
                return cls._build_driver(config, project)
            except Exception as exc:
                if 'Could not set lock on file' not in str(exc):
                    raise
                last_error = exc
                await asyncio.sleep(0.1)

        raise RuntimeError(
            'Graphiti database is busy. Another Graphiti process is holding the Kuzu lock.'
        ) from last_error

    @staticmethod
    def _build_driver(config: RuntimeConfig, project: ProjectPaths) -> GraphDriver:
        if config.backend is BackendType.neo4j:
            from graphiti_core.driver.neo4j_driver import Neo4jDriver

            uri = os.getenv(config.neo4j_uri_env, '')
            if not uri:
                raise RuntimeError(
                    f'Neo4j backend requires {config.neo4j_uri_env} to be set in the environment.'
                )
            return Neo4jDriver(
                uri=uri,
                user=os.getenv(config.neo4j_user_env),
                password=os.getenv(config.neo4j_password_env),
                database=config.neo4j_database,
            )

        from graphiti_core.driver.kuzu_driver import KuzuDriver

        return KuzuDriver(db=str(project.database_path))

    @classmethod
    def can_use_kuzu(cls) -> bool:
        try:
            from graphiti_core.driver.kuzu_driver import KuzuDriver

            KuzuDriver()
        except Exception:
            return False
        return True

    @classmethod
    def bootstrap_warning_threshold(cls) -> int:
        return BOOTSTRAP_WARN_THRESHOLD

    @classmethod
    def discover_history(cls, root: Path, history_days: int = 90) -> BootstrapDiscovery:
        return discover_project_history(root, max_age_days=history_days)

    @classmethod
    def choose_backend(
        cls,
        root: Path,
        *,
        requested_backend: BackendType | None = None,
    ) -> BackendType:
        if requested_backend is not None:
            return requested_backend

        config_path = build_project_paths(root).config_path
        if config_path.exists():
            return load_runtime_config(config_path).backend

        if cls.can_use_kuzu():
            return BackendType.kuzu
        return BackendType.neo4j

    @classmethod
    def default_runtime_config(
        cls,
        root: Path,
        *,
        backend: BackendType | None = None,
    ) -> RuntimeConfig:
        from .config import default_runtime_config as _default_runtime_config

        return _default_runtime_config(root, backend=backend or cls.choose_backend(root))

    async def __aenter__(self) -> MemoryEngine:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        await self.driver.close()

    @staticmethod
    def init_project(
        start: Path | None = None,
        force: bool = False,
        config: RuntimeConfig | None = None,
    ) -> tuple[ProjectPaths, RuntimeConfig]:
        root = (start or Path.cwd()).resolve()
        if config is None:
            config = MemoryEngine.default_runtime_config(root)
        return initialize_project_files(root, force=force, config=config)

    @staticmethod
    def apply_managed_agents_block(start: Path | None = None) -> Path:
        return apply_agent_instructions((start or Path.cwd()).resolve())

    @classmethod
    def detect_onboarding_state(
        cls,
        start: Path | None = None,
        *,
        history_days: int = 90,
        requested_backend: BackendType | None = None,
    ) -> dict[str, Any]:
        root = detect_project_root((start or Path.cwd()).resolve())
        project = build_project_paths(root)
        history = cls.discover_history(root, history_days=history_days)
        backend = cls.choose_backend(root, requested_backend=requested_backend)
        return {
            'root': str(root),
            'state_dir': str(project.state_dir),
            'configured': project.config_path.exists(),
            'backend': backend.value,
            'history_days': history_days,
            'history_sessions_detected': history.total_sessions,
            'history_warning_threshold': cls.bootstrap_warning_threshold(),
            'codex_sessions_detected': len(history.codex_sessions),
            'claude_sessions_detected': len(history.claude_sessions),
            'agents_path': str(root / 'AGENTS.md'),
            'agent_instructions_path': str(project.agent_instructions_path),
            'mcp_command': 'graphiti mcp --transport stdio',
        }

    def _memory_name(self, kind: MemoryKind, summary: str) -> str:
        return f'{kind.value}: {summary[:120]}'

    def _render_memory_content(
        self,
        kind: MemoryKind,
        summary: str,
        details: str,
        source: str,
        tags: list[str] | None = None,
        artifact_path: str = '',
        provenance: dict[str, str] | None = None,
        captured_at: datetime | None = None,
    ) -> str:
        tag_text = ', '.join(tags or [])
        lines = [
            f'Kind: {kind.value}',
            f'Summary: {summary}',
            f'Source: {source}',
            f'Captured At: {(captured_at or utc_now()).isoformat()}',
        ]
        if artifact_path:
            lines.append(f'Artifact Path: {artifact_path}')
        if tag_text:
            lines.append(f'Tags: {tag_text}')
        for key, value in (provenance or {}).items():
            if value:
                lines.append(f'{key}: {value}')
        lines.extend(['', 'Details:', details.strip() or summary.strip()])
        return '\n'.join(lines)

    async def _save_plain_episode(
        self,
        *,
        name: str,
        content: str,
        source_description: str,
        created_at: datetime | None = None,
        existing_uuid: str = '',
    ) -> EpisodicNode:
        timestamp = created_at or utc_now()
        episode_kwargs: dict[str, Any] = {}
        if existing_uuid:
            episode_kwargs['uuid'] = existing_uuid

        episode = EpisodicNode(
            name=name,
            group_id=self.config.project_id,
            labels=[],
            source=EpisodeType.text,
            content=content,
            source_description=source_description,
            created_at=timestamp,
            valid_at=timestamp,
            **episode_kwargs,
        )
        await episode.save(self.driver)
        return episode

    async def _save_source_episode(
        self,
        *,
        name: str,
        content: str,
        source_description: str,
        created_at: datetime | None = None,
        existing_uuid: str = '',
    ) -> EpisodicNode:
        timestamp = created_at or utc_now()
        if self.structured_memory_enabled and self.graphiti is not None:
            result = await self.graphiti.add_episode(
                name=name,
                episode_body=content,
                source_description=source_description,
                reference_time=timestamp,
                source=EpisodeType.text,
                group_id=self.config.project_id,
            )
            return result.episode
        return await self._save_plain_episode(
            name=name,
            content=content,
            source_description=source_description,
            created_at=timestamp,
            existing_uuid=existing_uuid,
        )

    def list_history_sessions(
        self,
        *,
        history_days: int = 90,
        limit: int | None = None,
    ) -> list[dict[str, str]]:
        discovery = self.discover_history(self.project.root, history_days)
        sessions = discovery.all_sessions()
        if limit is not None:
            sessions = sessions[:limit]
        return [
            {
                'session_id': session.session_id,
                'source_agent': session.source_agent,
                'title': session.title,
                'created_at': session.created_at.isoformat(),
                'source_path': session.source_path,
                'content_length': str(len(session.content)),
            }
            for session in sessions
        ]

    def read_history_session(
        self,
        session_id: str,
        *,
        history_days: int = 90,
        offset: int = 0,
        max_chars: int = 6000,
    ) -> dict[str, str | int | bool]:
        discovery = self.discover_history(self.project.root, history_days)
        for session in discovery.all_sessions():
            if session.session_id != session_id:
                continue
            content = session.content
            start = max(0, offset)
            end = min(len(content), start + max_chars)
            return {
                'session_id': session.session_id,
                'source_agent': session.source_agent,
                'title': session.title,
                'created_at': session.created_at.isoformat(),
                'source_path': session.source_path,
                'offset': start,
                'returned_chars': end - start,
                'has_more': end < len(content),
                'content': content[start:end],
            }
        raise ValueError(f'History session not found: {session_id}')

    async def remember(
        self,
        *,
        kind: MemoryKind,
        summary: str,
        details: str = '',
        source: str = 'agent',
        tags: list[str] | None = None,
        artifact_path: str = '',
        provenance: dict[str, str] | None = None,
        captured_at: datetime | None = None,
        prefer_plain_episode: bool = False,
    ) -> dict[str, str]:
        with project_lock(self.lock_path):
            return await self._remember_unlocked(
                kind=kind,
                summary=summary,
                details=details,
                source=source,
                tags=tags,
                artifact_path=artifact_path,
                provenance=provenance,
                captured_at=captured_at,
                prefer_plain_episode=prefer_plain_episode,
            )

    async def _remember_unlocked(
        self,
        *,
        kind: MemoryKind,
        summary: str,
        details: str = '',
        source: str = 'agent',
        tags: list[str] | None = None,
        artifact_path: str = '',
        provenance: dict[str, str] | None = None,
        captured_at: datetime | None = None,
        prefer_plain_episode: bool = False,
        existing_uuid: str = '',
    ) -> dict[str, str]:
        content = self._render_memory_content(
            kind,
            summary,
            details,
            source,
            tags,
            artifact_path,
            provenance=provenance,
            captured_at=captured_at,
        )
        name = self._memory_name(kind, summary)
        source_description = f'memory:{kind.value}'

        if (
            self.structured_memory_enabled
            and kind is not MemoryKind.index_artifact
            and not prefer_plain_episode
        ):
            result = await self.graphiti.add_episode(
                name=name,
                episode_body=content,
                source_description=source_description,
                reference_time=captured_at or utc_now(),
                source=EpisodeType.text,
                group_id=self.config.project_id,
            )
            episode = result.episode
            mode = 'structured'
        else:
            episode = await self._save_plain_episode(
                name=name,
                content=content,
                source_description=source_description,
                created_at=captured_at,
                existing_uuid=existing_uuid,
            )
            mode = 'episode'

        return {'uuid': episode.uuid, 'mode': mode}

    def _recall_search_config(self, include_graph: bool, limit: int) -> SearchConfig:
        if include_graph:
            return SearchConfig(
                edge_config=EdgeSearchConfig(
                    search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
                    reranker=EdgeReranker.rrf,
                ),
                node_config=NodeSearchConfig(
                    search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
                    reranker=NodeReranker.rrf,
                ),
                episode_config=EpisodeSearchConfig(
                    search_methods=[EpisodeSearchMethod.bm25],
                    reranker=EpisodeReranker.rrf,
                ),
                limit=limit,
            )

        return SearchConfig(
            episode_config=EpisodeSearchConfig(
                search_methods=[EpisodeSearchMethod.bm25],
                reranker=EpisodeReranker.rrf,
            ),
            limit=limit,
        )

    async def _search(self, query: str, limit: int = RECALL_LIMIT) -> SearchResults:
        config = self._recall_search_config(self.structured_memory_enabled, limit)
        return await search(
            self.clients,
            query,
            [self.config.project_id],
            config,
            SearchFilters(),
            driver=self.driver,
        )

    def _parse_memory_episode(self, episode: EpisodicNode) -> ParsedMemoryEpisode | None:
        values: dict[str, str] = {}
        details_started = False
        detail_lines: list[str] = []

        for raw_line in episode.content.splitlines():
            line = raw_line.rstrip()
            if details_started:
                detail_lines.append(line)
                continue
            if line == 'Details:':
                details_started = True
                continue
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            values[key.strip()] = value.strip()

        kind_value = values.get('Kind')
        if not kind_value:
            return None

        try:
            kind = MemoryKind(kind_value)
        except ValueError:
            return None

        tags = [tag.strip() for tag in values.get('Tags', '').split(',') if tag.strip()]
        details = '\n'.join(detail_lines).strip()

        return ParsedMemoryEpisode(
            uuid=episode.uuid,
            kind=kind,
            summary=values.get('Summary', episode.name),
            details=details,
            source=values.get('Source', episode.source_description),
            tags=tags,
            artifact_path=values.get('Artifact Path', ''),
            created_at=episode.created_at,
            raw_name=episode.name,
            source_agent=values.get('Source Agent', ''),
            session_id=values.get('Session ID', ''),
            thread_title=values.get('Thread Title', ''),
            captured_from=values.get('Captured From', ''),
        )

    async def _edge_evidence(self, edges: list[EntityEdge]) -> dict[str, str]:
        episode_ids: list[str] = []
        for edge in edges:
            if edge.episodes:
                episode_ids.extend(edge.episodes[:1])

        if not episode_ids:
            return {}

        seen: set[str] = set()
        ordered_ids: list[str] = []
        for episode_id in episode_ids:
            if episode_id not in seen:
                seen.add(episode_id)
                ordered_ids.append(episode_id)

        episodes = await EpisodicNode.get_by_uuids(self.driver, ordered_ids)
        return {episode.uuid: episode.name for episode in episodes}

    def _memory_line(self, memory: ParsedMemoryEpisode) -> str:
        created_at = memory.created_at.isoformat() if memory.created_at else 'unknown'
        details = memory.details.replace('\n', ' ')[:220]
        parts = [f'[{memory.kind.value}] {memory.summary}']
        if memory.source_agent:
            parts.append(f'agent={memory.source_agent}')
        if memory.thread_title:
            parts.append(f'thread={memory.thread_title[:80]}')
        parts.append(f'source={memory.source}')
        parts.append(f'captured_at={created_at}')
        if details:
            parts.append(f'details={details}')
        return '- ' + ' | '.join(parts)

    def _tokenize_query(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r'[a-z0-9_]+', text.lower())
            if len(token) >= 3
        }

    def _memory_overlap_score(self, memory: ParsedMemoryEpisode, query: str) -> int:
        haystack = ' '.join(
            [
                memory.summary,
                memory.details,
                memory.source,
                memory.artifact_path,
                memory.raw_name,
                memory.thread_title,
            ]
        ).lower()
        tokens = self._tokenize_query(query)
        if not tokens:
            return 0

        overlap = sum(1 for token in tokens if token in haystack)
        if query.lower() in haystack:
            overlap += max(2, len(tokens))
        return overlap

    async def _fallback_memory_episodes(
        self,
        query: str,
        *,
        limit: int,
        exclude_ids: set[str] | None = None,
    ) -> list[ParsedMemoryEpisode]:
        exclude = exclude_ids or set()
        episodes = await EpisodicNode.get_by_group_ids(
            self.driver,
            [self.config.project_id],
            limit=max(limit * 8, 64),
        )

        ranked: list[tuple[int, ParsedMemoryEpisode]] = []
        for episode in episodes:
            if episode.uuid in exclude:
                continue
            parsed = self._parse_memory_episode(episode)
            if parsed is None:
                continue
            score = self._memory_overlap_score(parsed, query)
            if score <= 0:
                continue
            ranked.append((score, parsed))

        ranked.sort(
            key=lambda item: (
                -item[0],
                -(
                    item[1].created_at.timestamp()
                    if item[1].created_at is not None
                    else 0
                ),
            )
        )
        return [parsed for _, parsed in ranked[:limit]]

    def _matching_snippet(self, text: str, query: str, *, max_len: int = 220) -> str:
        tokens = self._tokenize_query(query)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        best_line = ''
        best_score = 0
        for line in lines:
            normalized = line.lower()
            score = sum(1 for token in tokens if token in normalized)
            if query.lower() in normalized:
                score += max(2, len(tokens))
            if score > best_score:
                best_score = score
                best_line = line
        if best_line:
            return best_line[:max_len]
        return text.replace('\n', ' ')[:max_len]

    async def recall(self, query: str, limit: int = RECALL_LIMIT) -> str:
        results = await self._search(query, limit=max(limit, DEFAULT_SEARCH_LIMIT))

        seen_edges: set[str] = set()
        edges: list[EntityEdge] = []
        for edge in results.edges:
            if edge.uuid not in seen_edges:
                seen_edges.add(edge.uuid)
                edges.append(edge)

        edge_evidence = await self._edge_evidence(edges)

        now = datetime.now(timezone.utc)
        active_edges = [
            edge
            for edge in edges
            if edge.invalid_at is None or edge.invalid_at.astimezone(timezone.utc) > now
        ]
        historical_edges = [edge for edge in edges if edge not in active_edges]

        parsed_episodes: list[ParsedMemoryEpisode] = []
        seen_episodes: set[str] = set()
        for episode in results.episodes:
            if episode.uuid in seen_episodes:
                continue
            seen_episodes.add(episode.uuid)
            parsed = self._parse_memory_episode(episode)
            if parsed is not None:
                parsed_episodes.append(parsed)

        if len(parsed_episodes) < limit:
            fallback_episodes = await self._fallback_memory_episodes(
                query,
                limit=limit * 2,
                exclude_ids=seen_episodes,
            )
            for episode in fallback_episodes:
                if episode.uuid in seen_episodes:
                    continue
                seen_episodes.add(episode.uuid)
                parsed_episodes.append(episode)

        parsed_episodes.sort(
            key=lambda episode: (
                MEMORY_KIND_PRIORITY[episode.kind],
                -(episode.created_at.timestamp() if episode.created_at else 0),
            )
        )

        decisions = [episode for episode in parsed_episodes if episode.kind is MemoryKind.decision]
        constraints = [
            episode for episode in parsed_episodes if episode.kind is MemoryKind.constraint
        ]
        pitfalls = [episode for episode in parsed_episodes if episode.kind is MemoryKind.pitfall]
        other_memories = [
            episode
            for episode in parsed_episodes
            if episode.kind
            not in {
                MemoryKind.decision,
                MemoryKind.constraint,
                MemoryKind.pitfall,
                MemoryKind.index_artifact,
            }
        ]
        indexed_artifacts = [
            episode for episode in parsed_episodes if episode.kind is MemoryKind.index_artifact
        ]
        unique_indexed_artifacts: list[ParsedMemoryEpisode] = []
        seen_artifact_paths: set[str] = set()
        for artifact in indexed_artifacts:
            key = artifact.artifact_path or artifact.raw_name or artifact.summary
            if key in seen_artifact_paths:
                continue
            seen_artifact_paths.add(key)
            unique_indexed_artifacts.append(artifact)
        indexed_artifacts = unique_indexed_artifacts

        lines: list[str] = []

        for header, items in (
            ('Relevant Decisions', decisions),
            ('Relevant Constraints', constraints),
            ('Relevant Pitfalls', pitfalls),
            ('Patterns And Workflows', other_memories),
        ):
            if items:
                if lines:
                    lines.append('')
                lines.append(header)
                for memory in items[:limit]:
                    lines.append(self._memory_line(memory))

        if active_edges:
            if lines:
                lines.append('')
            lines.append('Active Facts')
            for edge in active_edges[:limit]:
                evidence = (
                    edge_evidence.get(edge.episodes[0], 'unknown source')
                    if edge.episodes
                    else 'unknown source'
                )
                validity = edge.valid_at.isoformat() if edge.valid_at else 'unknown'
                lines.append(f'- {edge.fact} | valid_at={validity} | evidence={evidence}')

        if historical_edges:
            if lines:
                lines.append('')
            lines.append('Historical Facts')
            for edge in historical_edges[:limit]:
                evidence = (
                    edge_evidence.get(edge.episodes[0], 'unknown source')
                    if edge.episodes
                    else 'unknown source'
                )
                valid_at = edge.valid_at.isoformat() if edge.valid_at else 'unknown'
                invalid_at = edge.invalid_at.isoformat() if edge.invalid_at else 'unknown'
                lines.append(
                    f'- {edge.fact} | valid_at={valid_at} | invalid_at={invalid_at} | evidence={evidence}'
                )

        if indexed_artifacts:
            if lines:
                lines.append('')
            lines.append('Supporting Artifacts')
            for artifact in indexed_artifacts[:limit]:
                created_at = artifact.created_at.isoformat() if artifact.created_at else 'unknown'
                details = self._matching_snippet(artifact.details, query)
                path = artifact.artifact_path or artifact.raw_name
                lines.append(
                    f'- {path} | captured_at={created_at} | summary={artifact.summary} | details={details}'
                )

        if not lines:
            return 'No relevant memory found.'

        return '\n'.join(lines)

    def _candidate_files(self, max_files: int = 24) -> list[Path]:
        root = self.project.root
        candidates: list[Path] = []
        patterns = [
            'AGENTS.md',
            'README*',
            'pyproject.toml',
            'package.json',
            'Makefile',
            'requirements*.txt',
            'Cargo.toml',
            'go.mod',
            'docker-compose*.yml',
            'docker-compose*.yaml',
        ]

        for pattern in patterns:
            candidates.extend(sorted(root.glob(pattern)))

        for subdir in ['docs', 'spec']:
            directory = root / subdir
            if directory.exists():
                candidates.extend(sorted(directory.rglob('*.md')))

        unique_candidates: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            if candidate.is_file() and candidate not in seen and '.graphiti' not in candidate.parts:
                seen.add(candidate)
                unique_candidates.append(candidate)
            if len(unique_candidates) >= max_files:
                break

        return unique_candidates

    def _inventory_artifact(self) -> IndexedArtifact:
        root = self.project.root
        top_level_lines = [f'Project Root: {root.name}', '', 'Top-level inventory:']
        for child in sorted(root.iterdir(), key=lambda path: path.name):
            if child.name == '.graphiti':
                continue
            if child.is_dir():
                py_count = sum(1 for _ in child.rglob('*.py'))
                top_level_lines.append(f'- {child.name}/ ({py_count} python files)')
            else:
                top_level_lines.append(f'- {child.name}')

        body = '\n'.join(top_level_lines)
        fingerprint = hashlib.sha256(body.encode('utf-8')).hexdigest()
        return IndexedArtifact(
            key='__inventory__',
            title='Project inventory',
            body=body,
            fingerprint=fingerprint,
            artifact_path='__inventory__',
        )

    def _git_artifact(self) -> IndexedArtifact | None:
        git_dir = self.project.root / '.git'
        if not git_dir.exists():
            return None

        try:
            result = subprocess.run(
                ['git', 'log', '--oneline', '--max-count=5'],
                cwd=self.project.root,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None

        if result.returncode != 0 or not result.stdout.strip():
            return None

        body = f'Recent git history:\n{result.stdout.strip()}'
        fingerprint = hashlib.sha256(body.encode('utf-8')).hexdigest()
        return IndexedArtifact(
            key='__git_recent__',
            title='Recent git history',
            body=body,
            fingerprint=fingerprint,
            artifact_path='__git_recent__',
        )

    async def _summarize_artifact(self, title: str, content: str) -> ArtifactSummary:
        commands: list[str] = []
        key_points: list[str] = []
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        summary = title
        for line in lines[:40]:
            if line.startswith('#'):
                key_points.append(line.lstrip('# ').strip())
                continue
            normalized = line.lower()
            if summary == title and len(line) > 8:
                summary = line[:200]
            if line.startswith('- '):
                key_points.append(line[2:].strip())
            if any(token in normalized for token in ('default', 'kuzu', 'neo4j', 'mcp', 'cli')):
                key_points.append(line[:180])
            if any(
                marker in normalized
                for marker in ('make ', 'uv ', 'pip ', 'python ', 'docker ', 'pytest ')
            ):
                commands.append(line)
        return ArtifactSummary(
            summary=summary,
            key_points=key_points[:5],
            commands=commands[:5],
        )

    def _artifact_details(self, artifact: IndexedArtifact, summary: ArtifactSummary) -> str:
        lines = [summary.summary]
        if summary.key_points:
            lines.append('Key points:')
            lines.extend(f'- {point}' for point in summary.key_points[:5])
        if summary.commands:
            lines.append('Commands:')
            lines.extend(f'- {command}' for command in summary.commands[:5])
        lines.append('')
        lines.append('Excerpt:')
        lines.append(artifact.body[:1600])
        return '\n'.join(lines)

    async def _delete_episode_if_present(self, episode_uuid: str) -> None:
        if not episode_uuid:
            return
        try:
            episode = await EpisodicNode.get_by_uuid(self.driver, episode_uuid)
        except Exception:
            return
        await episode.delete(self.driver)

    async def _delete_episodes(self, episode_uuids: list[str]) -> None:
        for episode_uuid in episode_uuids:
            await self._delete_episode_if_present(episode_uuid)

    async def index(
        self, *, changed_only: bool = False, max_files: int = 24
    ) -> list[dict[str, str]]:
        with project_lock(self.lock_path):
            state = load_index_state(self.project.index_state_path)
            artifacts_state = state.setdefault('artifacts', {})
            indexed: list[dict[str, str]] = []

            artifacts: list[IndexedArtifact] = [self._inventory_artifact()]
            git_artifact = self._git_artifact()
            if git_artifact is not None:
                artifacts.append(git_artifact)

            for path in self._candidate_files(max_files=max_files):
                raw_content = path.read_text(errors='ignore')
                body = raw_content[:ARTIFACT_CONTENT_LIMIT]
                fingerprint = hashlib.sha256(body.encode('utf-8')).hexdigest()
                artifacts.append(
                    IndexedArtifact(
                        key=str(path.relative_to(self.project.root)),
                        title=str(path.relative_to(self.project.root)),
                        body=body,
                        fingerprint=fingerprint,
                        artifact_path=str(path.relative_to(self.project.root)),
                    )
                )

            for artifact in artifacts:
                previous = artifacts_state.get(artifact.key, {})
                if changed_only and previous.get('fingerprint') == artifact.fingerprint:
                    continue

                summary = await self._summarize_artifact(artifact.title, artifact.body)
                details = self._artifact_details(artifact, summary)
                remembered = await self._remember_unlocked(
                    kind=MemoryKind.index_artifact,
                    summary=summary.summary,
                    details=details,
                    source='indexer',
                    artifact_path=artifact.artifact_path,
                    provenance={'Captured From': 'artifact index'},
                    prefer_plain_episode=True,
                )

                artifacts_state[artifact.key] = {
                    'fingerprint': artifact.fingerprint,
                    'episode_uuid': remembered['uuid'],
                    'memory_episode_uuids': [],
                }
                indexed.append({'artifact': artifact.key, 'episode_uuid': remembered['uuid']})

            save_index_state(self.project.index_state_path, state)
            return indexed

    async def import_history_sessions(
        self,
        *,
        session_ids: list[str] | None = None,
        history_days: int = 90,
        discovery: BootstrapDiscovery | None = None,
    ) -> list[dict[str, str]]:
        discovery = discovery or self.discover_history(self.project.root, history_days)
        with project_lock(self.lock_path):
            state = load_index_state(self.project.index_state_path)
            history_state = state.setdefault(
                'history_bootstrap', {'sessions': {}, 'last_bootstrap_at': ''}
            )
            sessions_state = history_state.setdefault('sessions', {})
            imported: list[dict[str, str]] = []
            selected_session_ids = set(session_ids or [])

            for session in discovery.all_sessions():
                if selected_session_ids and session.session_id not in selected_session_ids:
                    continue
                previous = sessions_state.get(session.session_id, {})
                if previous.get('fingerprint') == session.fingerprint:
                    continue

                await self._delete_episodes(previous.get('source_episode_uuids', []))
                await self._delete_episodes(previous.get('memory_episode_uuids', []))

                source_episode_uuids: list[str] = []
                session_header = '\n'.join(
                    [
                        f'Source Agent: {session.source_agent}',
                        f'Session ID: {session.session_id}',
                        f'Thread Title: {session.title}',
                        f'Session Timestamp: {session.created_at.isoformat()}',
                        f'Source Path: {session.source_path}',
                        '',
                        'Conversation:',
                    ]
                )
                for index, chunk in enumerate(
                    session.content_chunks(BOOTSTRAP_CHUNK_LIMIT), start=1
                ):
                    episode = await self._save_source_episode(
                        name=f'bootstrap:{session.source_agent}:{session.title[:96]} (part {index})',
                        content=f'{session_header}\n{chunk}',
                        source_description=f'bootstrap:{session.source_agent}',
                        created_at=session.created_at,
                    )
                    source_episode_uuids.append(episode.uuid)

                sessions_state[session.session_id] = {
                    'fingerprint': session.fingerprint,
                    'source_agent': session.source_agent,
                    'thread_title': session.title,
                    'created_at': session.created_at.isoformat(),
                    'source_episode_uuids': source_episode_uuids,
                    'memory_episode_uuids': [],
                    'source_path': session.source_path,
                }
                imported.append(
                    {
                        'session_id': session.session_id,
                        'source_agent': session.source_agent,
                        'thread_title': session.title,
                    }
                )

            history_state['last_bootstrap_at'] = utc_now().isoformat()
            save_index_state(self.project.index_state_path, state)
            return imported

    async def bootstrap_history(
        self,
        *,
        history_days: int = 90,
        discovery: BootstrapDiscovery | None = None,
    ) -> list[dict[str, str]]:
        return await self.import_history_sessions(
            history_days=history_days,
            discovery=discovery,
        )

    async def doctor(self) -> str:
        records = await self._query_records(
            """
            MATCH (e:Episodic)
            WHERE e.group_id = $group_id
            RETURN count(e) AS episode_count
            """,
            group_id=self.config.project_id,
        )
        episode_count = records[0]['episode_count'] if records else 0
        state = load_index_state(self.project.index_state_path)
        artifact_count = len(state.get('artifacts', {}))
        history_state = state.get('history_bootstrap', {})
        imported_session_count = len(history_state.get('sessions', {}))
        structured_status = (
            'enabled'
            if self.structured_memory_enabled
            else 'disabled (agent-driven MCP workflow does not require it)'
        )
        lines = [
            f'Project: {self.config.project_name}',
            f'Project ID: {self.config.project_id}',
            f'Backend: {self.config.backend.value}',
            f'Database: {self.project.database_path}',
            f'State dir: {self.project.state_dir}',
            f'Structured graph extraction: {structured_status}',
            'MCP: available via `graphiti mcp --transport stdio`',
            f'Episodes: {episode_count}',
            f'Indexed artifacts: {artifact_count}',
            f'History bootstrap sessions: {imported_session_count}',
            f'Last bootstrap: {history_state.get("last_bootstrap_at", "") or "not run"}',
            f'Agent instructions: {self.project.agent_instructions_path}',
        ]
        return '\n'.join(lines)

    async def _query_records(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        if self.config.backend is BackendType.neo4j:
            result = await self.driver.execute_query(query, params=kwargs)
        else:
            result = await self.driver.execute_query(query, **kwargs)
        if isinstance(result, tuple):
            return result[0]
        if hasattr(result, 'records'):
            return [record.data() for record in result.records]
        return result  # type: ignore[return-value]
