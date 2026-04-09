from __future__ import annotations

import argparse
import asyncio
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip('kuzu')

from graphiti_core.driver.kuzu_driver import KuzuDriver, KuzuDriverSession
from graphiti_core.memory import cli, config, history, mcp
from graphiti_core.memory.engine import ArtifactSummary, IndexedArtifact, MemoryEngine
from graphiti_core.memory.models import (
    BackendType,
    BootstrapDiscovery,
    BootstrapSession,
    MemoryKind,
    RuntimeConfig,
    build_project_id,
    build_project_paths,
    ensure_index_state_shape,
)


class FakeContext:
    def __init__(self, engine: Any):
        self.engine = engine

    async def __aenter__(self):
        return self.engine

    async def __aexit__(self, exc_type, exc, tb):
        return None


class FakeEngine:
    def __init__(self):
        self.remember_calls: list[dict[str, Any]] = []
        self.index_calls: list[dict[str, Any]] = []
        self.recall_calls: list[dict[str, Any]] = []

    async def recall(self, query: str, limit: int = 8) -> str:
        self.recall_calls.append({'query': query, 'limit': limit})
        return f'recall:{query}:{limit}'

    async def remember(self, **kwargs: Any) -> dict[str, str]:
        self.remember_calls.append(kwargs)
        return {'uuid': 'memory-1', 'mode': 'episode'}

    async def index(
        self, *, changed_only: bool = False, max_files: int = 24
    ) -> list[dict[str, str]]:
        self.index_calls.append({'changed_only': changed_only, 'max_files': max_files})
        if changed_only:
            return []
        return [{'artifact': 'README.md', 'episode_uuid': 'episode-1'}]

    async def doctor(self) -> str:
        return 'doctor-ok'

    def list_history_sessions(self, *, history_days: int = 90, limit: int | None = None):
        return [{'session_id': 'session-1', 'history_days': history_days, 'limit': limit}]

    def read_history_session(
        self,
        session_id: str,
        *,
        history_days: int = 90,
        offset: int = 0,
        max_chars: int = 6000,
    ):
        return {
            'session_id': session_id,
            'history_days': history_days,
            'offset': offset,
            'returned_chars': min(max_chars, 12),
            'has_more': False,
            'content': 'bootstrap text',
        }

    async def import_history_sessions(
        self,
        *,
        session_ids: list[str] | None = None,
        history_days: int = 90,
        discovery: BootstrapDiscovery | None = None,
    ) -> list[dict[str, str]]:
        del discovery
        return [
            {
                'session_id': session_id,
                'source_agent': 'codex',
                'thread_title': f'thread:{history_days}',
            }
            for session_id in (session_ids or ['default-session'])
        ]


def build_engine(root: Path) -> MemoryEngine:
    paths = build_project_paths(root)
    config_obj = RuntimeConfig(
        project_name=root.name,
        project_id=build_project_id(root),
        database_path=str(paths.database_path.relative_to(root)),
        backend=BackendType.kuzu,
    )
    return MemoryEngine(
        project=paths,
        config=config_obj,
        driver=SimpleNamespace(close=lambda: None),
        clients=SimpleNamespace(),
        graphiti=None,
        structured_memory_enabled=False,
    )


def test_parse_scalar_and_config_round_trip(tmp_path: Path) -> None:
    runtime = config.default_runtime_config(tmp_path, backend=BackendType.neo4j)
    runtime.llm_base_url = 'http://localhost:11434/v1'
    runtime.neo4j_database = 'graphiti'
    config_path = tmp_path / 'config.toml'

    config.write_runtime_config(config_path, runtime)
    loaded = config.load_runtime_config(config_path)

    assert config._parse_scalar('"quoted \\"value\\""') == 'quoted "value"'
    assert config._parse_scalar('plain') == 'plain'
    assert loaded.backend is BackendType.neo4j
    assert loaded.llm_base_url == 'http://localhost:11434/v1'
    assert loaded.neo4j_database == 'graphiti'


def test_detect_project_root_prefers_graphiti_then_pyproject_then_git(tmp_path: Path) -> None:
    root = tmp_path / 'repo'
    nested = root / 'app' / 'core'
    nested.mkdir(parents=True)
    assert config.detect_project_root(nested) == nested.resolve()

    (root / '.git').mkdir()
    assert config.detect_project_root(nested) == root.resolve()

    (root / 'AGENTS.md').write_text('# Agents\n')
    assert config.detect_project_root(nested) == root.resolve()

    (root / 'pyproject.toml').write_text('[project]\nname = "demo"\n')
    assert config.detect_project_root(nested) == root.resolve()

    (root / '.graphiti').mkdir()
    assert config.detect_project_root(nested) == root.resolve()


def test_apply_agent_instructions_creates_and_replaces_managed_block(tmp_path: Path) -> None:
    agents_path = config.apply_agent_instructions(tmp_path)
    created = agents_path.read_text()
    assert config.GRAPHITI_BLOCK_START in created
    assert 'MCP recall triggers:' in created
    assert 'MCP write triggers:' in created

    agents_path.write_text(
        '# Existing\n\n'
        f'{config.GRAPHITI_BLOCK_START}\nold\n{config.GRAPHITI_BLOCK_END}\n\n'
        'Tail note.\n'
    )
    config.apply_agent_instructions(tmp_path)
    updated = agents_path.read_text()

    assert updated.count(config.GRAPHITI_BLOCK_START) == 1
    assert 'Tail note.' in updated
    assert 'old' not in updated


def test_install_codex_mcp_server_creates_and_updates_config(tmp_path: Path) -> None:
    home = tmp_path / 'home'

    config_path, changed = config.install_codex_mcp_server(
        home,
        python_executable='/tmp/venv/bin/python',
    )
    assert changed is True
    assert config_path == home / '.codex' / 'config.toml'
    contents = config_path.read_text()
    assert '[mcp_servers.graphiti]' in contents
    assert 'command = "/tmp/venv/bin/python"' in contents
    assert 'graphiti_core.memory.cli' in contents
    assert config.codex_mcp_server_installed(home) is True

    config_path.write_text(contents + '\n[mcp_servers.other]\ncommand = "npx"\n')
    _, changed = config.install_codex_mcp_server(
        home,
        python_executable='/tmp/venv/bin/python3.12',
    )
    assert changed is True
    updated = config_path.read_text()
    assert updated.count('[mcp_servers.graphiti]') == 1
    assert 'command = "/tmp/venv/bin/python3.12"' in updated
    assert '[mcp_servers.other]' in updated


def test_load_index_state_defaults_and_normalization(tmp_path: Path) -> None:
    state_path = tmp_path / 'index_state.json'

    assert config.load_index_state(state_path) == ensure_index_state_shape(None)

    config.save_index_state(
        state_path,
        {'artifacts': {'README.md': {'fingerprint': 'abc'}}, 'history_bootstrap': {}},
    )
    loaded = config.load_index_state(state_path)

    assert loaded['artifacts']['README.md']['fingerprint'] == 'abc'
    assert loaded['history_bootstrap']['sessions'] == {}
    assert loaded['history_bootstrap']['last_bootstrap_at'] == ''


def test_load_runtime_config_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        config.load_runtime_config(tmp_path / 'missing.toml')


def test_load_runtime_config_ignores_malformed_lines(tmp_path: Path) -> None:
    config_path = tmp_path / 'config.toml'
    config_path.write_text(
        '\n'.join(
            [
                'name = "ignored outside section"',
                '[project]',
                'name = "demo"',
                'backend = "kuzu"',
                'not-a-pair',
                '[storage]',
                'database_path = ".graphiti/memory.kuzu"',
            ]
        )
    )
    loaded = config.load_runtime_config(config_path)
    assert loaded.project_name == 'demo'
    assert loaded.database_path == '.graphiti/memory.kuzu'


def test_history_parsers_cover_invalid_and_mixed_formats(tmp_path: Path) -> None:
    assert history._parse_timestamp(None) is None
    assert history._parse_timestamp('not-a-date') is None
    assert history._parse_unix_timestamp(None) is None
    assert history._parse_unix_timestamp(1_700_000_000) is not None
    assert history._parse_unix_timestamp(1_700_000_000_000) is not None
    assert history._parse_unix_timestamp(float('inf')) is None

    rollout = tmp_path / 'rollout.jsonl'
    rollout.write_text(
        '\n'.join(
            [
                '{"bad": ',
                json.dumps({'type': 'event_msg', 'payload': {'type': 'tool_call', 'message': 'x'}}),
                json.dumps(
                    {
                        'type': 'event_msg',
                        'payload': {'type': 'user_message', 'message': 'User asks'},
                    }
                ),
                json.dumps(
                    {
                        'type': 'event_msg',
                        'payload': {'type': 'agent_message', 'message': 'Assistant answers'},
                    }
                ),
            ]
        )
        + '\n'
    )
    parsed_rollout = history._parse_codex_rollout(rollout)
    assert 'User: User asks' in parsed_rollout
    assert 'Assistant: Assistant answers' in parsed_rollout

    assert history._extract_claude_text('plain text') == 'plain text'
    assert (
        history._extract_claude_text([{'type': 'text', 'text': 'hello'}, {'type': 'tool_use'}])
        == 'hello'
    )
    assert history._extract_claude_text({'bad': 'shape'}) == ''

    claude_path = tmp_path / 'claude.jsonl'
    claude_path.write_text(
        '\n'.join(
            [
                '{"bad": ',
                json.dumps({'type': 'summary', 'message': {'content': 'ignore'}}),
                json.dumps(
                    {
                        'type': 'user',
                        'timestamp': '2026-04-09T12:00:00Z',
                        'sessionId': 'claude-1',
                        'message': {'content': 'How do tests run?'},
                    }
                ),
                json.dumps(
                    {
                        'type': 'assistant',
                        'timestamp': '2026-04-09T12:00:01Z',
                        'sessionId': 'claude-1',
                        'message': {'content': [{'type': 'text', 'text': 'Run pytest.'}]},
                    }
                ),
            ]
        )
        + '\n'
    )
    content, session_id, title, created_at = history._parse_claude_transcript(claude_path)
    assert 'Run pytest.' in content
    assert session_id == 'claude-1'
    assert title == 'How do tests run?'
    assert created_at is not None


def test_discover_codex_sessions_handles_sqlite_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenSqliteError(Exception):
        pass

    def broken_connect(_path: str):
        raise sqlite3.Error('boom')  # type: ignore[name-defined]

    import sqlite3

    monkeypatch.setattr(history.sqlite3, 'connect', broken_connect)
    sessions = history._discover_codex_sessions(
        tmp_path,
        tmp_path,
        datetime.now(timezone.utc) - timedelta(days=1),
    )
    assert sessions == []


def test_discover_codex_sessions_skips_invalid_rows(tmp_path: Path) -> None:
    codex_dir = tmp_path / '.codex'
    codex_dir.mkdir()
    db_path = codex_dir / 'state_1.sqlite'
    rollout_path = codex_dir / 'missing-rollout.jsonl'
    connection = history.sqlite3.connect(db_path)
    connection.execute(
        'CREATE TABLE threads (id TEXT, title TEXT, cwd TEXT, rollout_path TEXT, updated_at INTEGER)'
    )
    connection.execute(
        'INSERT INTO threads VALUES (?, ?, ?, ?, ?)',
        ('bad-row', 'Bad row', str(tmp_path), str(rollout_path), 0),
    )
    connection.commit()
    connection.close()

    sessions = history._discover_codex_sessions(
        codex_dir,
        tmp_path,
        datetime.now(timezone.utc) - timedelta(days=30),
    )
    assert sessions == []


def test_bootstrap_models_and_index_state_helpers() -> None:
    created = datetime(2026, 4, 9, tzinfo=timezone.utc)
    session = BootstrapSession(
        source_agent='codex',
        session_id='abc',
        title='Chunk me',
        created_at=created,
        fingerprint='fp',
        content='line1\nline2\nline3\nline4',
        source_path='/tmp/source',
    )
    chunks = session.content_chunks(8)
    assert len(chunks) >= 2

    older = BootstrapSession(
        source_agent='claude',
        session_id='older',
        title='Older',
        created_at=created - timedelta(days=1),
        fingerprint='older',
        content='older',
        source_path='/tmp/older',
    )
    discovery = BootstrapDiscovery(codex_sessions=[older], claude_sessions=[session])
    assert discovery.total_sessions == 2
    assert discovery.all_sessions()[0].session_id == 'abc'


@pytest.mark.asyncio
async def test_engine_helper_behaviors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = build_engine(tmp_path)
    content = engine._render_memory_content(
        MemoryKind.workflow,
        'Run recall first',
        'Use recall before broad repo search.',
        'agent',
        tags=['memory', 'workflow'],
        artifact_path='AGENTS.md',
        provenance={'Source Agent': 'codex'},
        captured_at=datetime(2026, 4, 9, tzinfo=timezone.utc),
    )
    parsed = engine._parse_memory_episode(
        SimpleNamespace(
            uuid='episode-1',
            content=content,
            name='workflow: Run recall first',
            source_description='memory:workflow',
            created_at=datetime(2026, 4, 9, tzinfo=timezone.utc),
        )
    )
    assert parsed is not None
    assert parsed.kind is MemoryKind.workflow
    assert parsed.source_agent == 'codex'

    assert engine._tokenize_query('Try Pattern_X twice? no.') == {'try', 'pattern_x', 'twice'}
    assert engine._memory_overlap_score(parsed, 'run recall workflow') > 0
    assert engine._matching_snippet('alpha\npattern y wins\nomega', 'pattern y') == 'pattern y wins'

    summary = await engine._summarize_artifact(
        'README.md',
        '# Heading\nUse `graphiti mcp --transport stdio`\n- default backend is kuzu\nmake test\n',
    )
    details = engine._artifact_details(
        IndexedArtifact(key='README.md', title='README.md', body='body text', fingerprint='fp'),
        summary,
    )
    assert isinstance(summary, ArtifactSummary)
    assert 'make test' in details

    artifact = engine._inventory_artifact()
    assert artifact.key == '__inventory__'

    monkeypatch.setattr(
        'graphiti_core.memory.engine.subprocess.run',
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout='abc123 message\n'),
    )
    (tmp_path / '.git').mkdir()
    git_artifact = engine._git_artifact()
    assert git_artifact is not None
    assert git_artifact.key == '__git_recent__'


@pytest.mark.asyncio
async def test_engine_choose_backend_detect_state_and_query_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    paths = build_project_paths(root)
    paths.state_dir.mkdir()
    config.write_runtime_config(
        paths.config_path,
        RuntimeConfig(
            project_name='demo',
            project_id='pid',
            database_path='.graphiti/memory.kuzu',
            backend=BackendType.neo4j,
        ),
    )

    assert MemoryEngine.choose_backend(root) is BackendType.neo4j
    assert MemoryEngine.choose_backend(root, requested_backend=BackendType.kuzu) is BackendType.kuzu

    monkeypatch.setattr(MemoryEngine, 'can_use_kuzu', classmethod(lambda cls: False))
    other_root = tmp_path / 'other'
    other_root.mkdir()
    assert MemoryEngine.choose_backend(other_root) is BackendType.neo4j

    history_payload = BootstrapDiscovery(
        codex_sessions=[
            BootstrapSession(
                source_agent='codex',
                session_id='codex-1',
                title='Recent',
                created_at=datetime.now(timezone.utc),
                fingerprint='fp',
                content='content',
                source_path='/tmp/rollout',
            )
        ]
    )
    monkeypatch.setattr(
        MemoryEngine,
        'discover_history',
        classmethod(lambda cls, root, history_days=90: history_payload),
    )
    monkeypatch.setenv('HOME', str(root))
    state = MemoryEngine.detect_onboarding_state(root, history_days=30)
    assert state['history_sessions_detected'] == 1
    assert state['backend'] == 'neo4j'
    assert state['codex_config_path'].endswith('.codex/config.toml')
    assert state['codex_mcp_installed'] is False

    engine = build_engine(root)

    async def execute_tuple(*args: Any, **kwargs: Any):
        return [{'episode_count': 3}], None, None

    engine.driver = SimpleNamespace(execute_query=execute_tuple)
    assert await engine._query_records('MATCH ...') == [{'episode_count': 3}]

    record_obj = SimpleNamespace(records=[SimpleNamespace(data=lambda: {'episode_count': 4})])

    async def execute_records(*args: Any, **kwargs: Any):
        return record_obj

    engine.driver = SimpleNamespace(execute_query=execute_records)
    assert await engine._query_records('MATCH ...') == [{'episode_count': 4}]

    async def execute_list(*args: Any, **kwargs: Any):
        return [{'episode_count': 5}]

    engine.driver = SimpleNamespace(execute_query=execute_list)
    assert await engine._query_records('MATCH ...') == [{'episode_count': 5}]


@pytest.mark.asyncio
async def test_open_driver_with_retry_handles_lock_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = build_project_paths(tmp_path)
    config_obj = RuntimeConfig(
        project_name='demo',
        project_id='pid',
        database_path='.graphiti/memory.kuzu',
        backend=BackendType.kuzu,
    )
    attempts = {'count': 0}

    def fake_build_driver(_config: RuntimeConfig, _project: Any) -> str:
        attempts['count'] += 1
        if attempts['count'] < 3:
            raise RuntimeError('Could not set lock on file')
        return 'driver-ok'

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(MemoryEngine, '_build_driver', staticmethod(fake_build_driver))
    monkeypatch.setattr('graphiti_core.memory.engine.asyncio.sleep', fake_sleep)

    assert await MemoryEngine._open_driver_with_retry(config_obj, project) == 'driver-ok'
    assert attempts['count'] == 3


@pytest.mark.asyncio
async def test_open_driver_retry_non_kuzu_and_busy_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = build_project_paths(tmp_path)
    neo4j_config = RuntimeConfig(
        project_name='demo',
        project_id='pid',
        database_path='.graphiti/memory.kuzu',
        backend=BackendType.neo4j,
    )
    kuzu_config = RuntimeConfig(
        project_name='demo',
        project_id='pid',
        database_path='.graphiti/memory.kuzu',
        backend=BackendType.kuzu,
    )

    monkeypatch.setattr(
        MemoryEngine, '_build_driver', staticmethod(lambda _config, _project: 'neo4j-driver')
    )
    assert await MemoryEngine._open_driver_with_retry(neo4j_config, project) == 'neo4j-driver'

    def raise_other(_config: RuntimeConfig, _project: Any):
        raise RuntimeError('different failure')

    monkeypatch.setattr(MemoryEngine, '_build_driver', staticmethod(raise_other))
    with pytest.raises(RuntimeError, match='different failure'):
        await MemoryEngine._open_driver_with_retry(kuzu_config, project)

    attempts = {'count': 0}

    def raise_lock(_config: RuntimeConfig, _project: Any):
        attempts['count'] += 1
        raise RuntimeError('Could not set lock on file')

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(MemoryEngine, '_build_driver', staticmethod(raise_lock))
    monkeypatch.setattr('graphiti_core.memory.engine.asyncio.sleep', fake_sleep)
    with pytest.raises(RuntimeError, match='database is busy'):
        await MemoryEngine._open_driver_with_retry(kuzu_config, project)
    assert attempts['count'] == 50


@pytest.mark.asyncio
async def test_cli_command_paths_and_error_handling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    fake_engine = FakeEngine()

    async def fake_open(_cwd: Path):
        return FakeContext(fake_engine)

    async def fake_run_mcp_server(_cwd: Path) -> None:
        print('mcp-server-started')

    monkeypatch.setattr(cli.MemoryEngine, 'open', fake_open)
    monkeypatch.setattr(cli, 'run_stdio_mcp_server', fake_run_mcp_server)
    monkeypatch.chdir(tmp_path)

    assert await cli._run_async(argparse.Namespace(command='mcp', transport='stdio')) == 0
    assert 'mcp-server-started' in capsys.readouterr().out

    assert await cli._run_async(argparse.Namespace(command='recall', query='task', limit=3)) == 0
    assert 'recall:task:3' in capsys.readouterr().out

    assert (
        await cli._run_async(
            argparse.Namespace(
                command='remember',
                kind='decision',
                summary='Prefer Y',
                details='Because X failed',
                source='agent',
                tag=['one'],
                path='README.md',
            )
        )
        == 0
    )
    remember_out = capsys.readouterr().out
    assert 'Stored decision memory as memory-1 (episode).' in remember_out

    assert await cli._run_async(argparse.Namespace(command='index', changed=True, max_files=5)) == 0
    assert 'No artifacts needed re-indexing.' in capsys.readouterr().out

    assert (
        await cli._run_async(argparse.Namespace(command='index', changed=False, max_files=5)) == 0
    )
    assert 'Indexed 1 artifact(s).' in capsys.readouterr().out

    assert await cli._run_async(argparse.Namespace(command='doctor')) == 0
    assert 'doctor-ok' in capsys.readouterr().out

    assert cli._prompt_yes_no('prompt', default=False, interactive=False) is False
    inputs = iter(['maybe', 'Y'])
    monkeypatch.setattr('builtins.input', lambda _prompt: next(inputs))
    assert cli._prompt_yes_no('prompt', default=True, interactive=True) is True
    prompt_out = capsys.readouterr().out
    assert 'Please answer yes or no.' in prompt_out


@pytest.mark.asyncio
async def test_cli_init_prompt_and_large_history_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    paths = build_project_paths(tmp_path)
    config_obj = RuntimeConfig(
        project_name='demo',
        project_id='pid',
        database_path='.graphiti/memory.kuzu',
    )
    discovery = BootstrapDiscovery(
        codex_sessions=[
            BootstrapSession(
                source_agent='codex',
                session_id='session-1',
                title='Import me',
                created_at=datetime.now(timezone.utc),
                fingerprint='fp',
                content='transcript',
                source_path='/tmp/rollout',
            )
        ]
    )
    fake_engine = FakeEngine()
    prompts: list[str] = []
    answers = iter([False, True])

    async def fake_open(_cwd: Path):
        return FakeContext(fake_engine)

    monkeypatch.setattr(cli, 'detect_project_root', lambda path: tmp_path)
    monkeypatch.setattr(
        cli.MemoryEngine,
        'choose_backend',
        classmethod(lambda cls, root, requested_backend=None: BackendType.kuzu),
    )
    monkeypatch.setattr(
        cli.MemoryEngine,
        'default_runtime_config',
        classmethod(lambda cls, root, backend=None: config_obj),
    )
    monkeypatch.setattr(
        cli.MemoryEngine,
        'init_project',
        staticmethod(lambda root, force=False, config=None: (paths, config_obj)),
    )
    monkeypatch.setattr(
        cli.MemoryEngine,
        'discover_history',
        classmethod(lambda cls, root, history_days=90: discovery),
    )
    monkeypatch.setattr(
        cli.MemoryEngine, 'bootstrap_warning_threshold', classmethod(lambda cls: 10)
    )
    monkeypatch.setattr(cli.MemoryEngine, 'open', fake_open)
    monkeypatch.setattr(cli, 'apply_agent_instructions', lambda root: root / 'AGENTS.md')
    monkeypatch.setattr(
        cli,
        '_prompt_yes_no',
        lambda prompt, default=True, interactive=True: prompts.append(prompt) or next(answers),
    )
    monkeypatch.setattr(cli.sys, 'stdin', SimpleNamespace(isatty=lambda: True))
    monkeypatch.chdir(tmp_path)

    rc = await cli._run_init(
        argparse.Namespace(
            force=False,
            yes=False,
            import_history=False,
            skip_history=False,
            apply_agents=False,
            no_apply_agents=False,
            install_mcp=False,
            no_install_mcp=False,
            backend=None,
            history_days=90,
        )
    )
    output = capsys.readouterr().out

    assert rc == 0
    assert prompts[0].startswith('Apply the managed Graphiti block')
    assert prompts[1].startswith('Import 1 matching Codex/Claude project sessions')
    assert '- Updated agent instructions:' not in output
    assert '- Left Codex MCP config unchanged' in output
    assert '- Imported transcript source sessions: 1' in output

    large_discovery = BootstrapDiscovery(
        codex_sessions=[
            BootstrapSession(
                source_agent='codex',
                session_id=f'session-{index}',
                title='Large import',
                created_at=datetime.now(timezone.utc),
                fingerprint=f'fp-{index}',
                content='transcript',
                source_path='/tmp/rollout',
            )
            for index in range(12)
        ]
    )
    monkeypatch.setattr(
        cli.MemoryEngine,
        'discover_history',
        classmethod(lambda cls, root, history_days=90: large_discovery),
    )
    monkeypatch.setattr(cli.sys, 'stdin', SimpleNamespace(isatty=lambda: False))
    rc = await cli._run_init(
        argparse.Namespace(
            force=False,
            yes=False,
            import_history=False,
            skip_history=False,
            apply_agents=False,
            no_apply_agents=False,
            install_mcp=False,
            no_install_mcp=False,
            backend=None,
            history_days=90,
        )
    )
    output = capsys.readouterr().out

    assert rc == 0
    assert 'Transcript source import skipped by default' in output


def test_cli_main_error_handling(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    def raise_missing(coroutine):
        coroutine.close()
        raise FileNotFoundError('missing config')

    def raise_runtime(coroutine):
        coroutine.close()
        raise RuntimeError('boom')

    monkeypatch.setattr(cli.asyncio, 'run', raise_missing)
    assert cli.main(['doctor']) == 1
    assert 'Run `graphiti init` from the project root first.' in capsys.readouterr().out

    monkeypatch.setattr(cli.asyncio, 'run', raise_runtime)
    assert cli.main(['doctor']) == 1
    assert 'boom' in capsys.readouterr().out


@pytest.mark.asyncio
async def test_engine_structured_and_recall_helper_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved_payloads: list[dict[str, Any]] = []

    class FakeEpisodeNode:
        def __init__(self, **kwargs: Any):
            self.__dict__.update(kwargs)
            self.uuid = kwargs.get('uuid', 'episode-generated')

        async def save(self, driver: Any) -> None:
            del driver
            saved_payloads.append(self.__dict__)

    @staticmethod
    async def fake_get_by_uuids(driver: Any, uuids: list[str]):
        del driver
        return [SimpleNamespace(uuid=uuid, name=f'name:{uuid}') for uuid in uuids]

    @staticmethod
    async def fake_get_by_group_ids(driver: Any, group_ids: list[str], limit: int):
        del driver, group_ids, limit
        return [
            SimpleNamespace(
                uuid='excluded',
                content='Kind: decision\nSummary: Ignore me\nSource: agent\n\nDetails:\ntext',
                name='ignore',
                source_description='memory:decision',
                created_at=datetime(2026, 4, 7, tzinfo=timezone.utc),
            ),
            SimpleNamespace(
                uuid='score-zero',
                content='Kind: workflow\nSummary: Unrelated\nSource: agent\n\nDetails:\nnone',
                name='unrelated',
                source_description='memory:workflow',
                created_at=datetime(2026, 4, 8, tzinfo=timezone.utc),
            ),
            SimpleNamespace(
                uuid='scored',
                content='Kind: decision\nSummary: Pattern Y wins\nSource: agent\n\nDetails:\nPattern Y replaced X',
                name='decision',
                source_description='memory:decision',
                created_at=datetime(2026, 4, 9, tzinfo=timezone.utc),
            ),
            SimpleNamespace(
                uuid='invalid',
                content='Kind: nope\nSummary: invalid\n\nDetails:\ninvalid',
                name='invalid',
                source_description='memory:invalid',
                created_at=datetime(2026, 4, 6, tzinfo=timezone.utc),
            ),
        ]

    @staticmethod
    async def fake_get_by_uuid(driver: Any, uuid: str):
        del driver
        if uuid == 'missing':
            raise RuntimeError('missing')

        class Deletable:
            async def delete(self, driver: Any) -> None:
                del driver
                saved_payloads.append({'deleted': uuid})

        return Deletable()

    FakeEpisodeNode.get_by_uuids = fake_get_by_uuids
    FakeEpisodeNode.get_by_group_ids = fake_get_by_group_ids
    FakeEpisodeNode.get_by_uuid = fake_get_by_uuid
    monkeypatch.setattr('graphiti_core.memory.engine.EpisodicNode', FakeEpisodeNode)

    engine = build_engine(tmp_path)
    fake_graphiti = SimpleNamespace(
        add_episode=lambda **kwargs: asyncio.sleep(
            0, result=SimpleNamespace(episode=SimpleNamespace(uuid='structured-episode'))
        )
    )
    engine.graphiti = fake_graphiti
    engine.structured_memory_enabled = True

    remembered = await engine._remember_unlocked(
        kind=MemoryKind.decision,
        summary='Prefer pattern Y',
        details='Pattern X failed repeatedly.',
    )
    assert remembered == {'uuid': 'structured-episode', 'mode': 'structured'}

    source_episode = await engine._save_source_episode(
        name='bootstrap',
        content='transcript chunk',
        source_description='bootstrap:codex',
    )
    assert source_episode.uuid == 'structured-episode'

    engine.structured_memory_enabled = False
    plain_episode = await engine._save_plain_episode(
        name='plain',
        content='body',
        source_description='memory:workflow',
        existing_uuid='existing-1',
    )
    assert plain_episode.uuid == 'existing-1'

    evidence = await engine._edge_evidence(
        [
            SimpleNamespace(episodes=['ep-1', 'ep-1']),
            SimpleNamespace(episodes=[]),
            SimpleNamespace(episodes=['ep-2']),
        ]
    )
    assert evidence == {'ep-1': 'name:ep-1', 'ep-2': 'name:ep-2'}

    scored = await engine._fallback_memory_episodes(
        'pattern y',
        limit=2,
        exclude_ids={'excluded'},
    )
    assert [item.uuid for item in scored] == ['scored']

    assert engine._matching_snippet('alpha\nbeta', 'zzz') == 'alpha beta'
    overlap_memory = SimpleNamespace(
        summary='Prefer pattern Y',
        details='Detailed context',
        source='agent',
        artifact_path='README.md',
        raw_name='decision',
        thread_title='Migrate memory runtime',
    )
    assert engine._memory_overlap_score(overlap_memory, '??') == 0

    rendered = engine._memory_line(
        SimpleNamespace(
            kind=MemoryKind.decision,
            summary='Prefer pattern Y',
            details='Detailed context',
            source='agent',
            created_at=datetime(2026, 4, 9, tzinfo=timezone.utc),
            source_agent='codex',
            thread_title='Migrate memory runtime',
        )
    )
    assert 'agent=codex' in rendered
    assert 'thread=Migrate memory runtime' in rendered

    assert (
        engine._parse_memory_episode(
            SimpleNamespace(
                content='Summary: no kind',
                uuid='u',
                name='n',
                source_description='s',
                created_at=None,
            )
        )
        is None
    )
    assert (
        engine._parse_memory_episode(
            SimpleNamespace(
                content='Kind: unknown\nSummary: bad',
                uuid='u',
                name='n',
                source_description='s',
                created_at=None,
            )
        )
        is None
    )

    await engine._delete_episode_if_present('')
    await engine._delete_episode_if_present('missing')
    await engine._delete_episodes(['delete-me'])
    assert {'deleted': 'delete-me'} in saved_payloads
    assert engine._recall_search_config(True, 4).edge_config is not None


@pytest.mark.asyncio
async def test_engine_recall_index_import_and_doctor_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = build_engine(tmp_path)
    engine.project.state_dir.mkdir(parents=True, exist_ok=True)
    config.save_index_state(engine.project.index_state_path, ensure_index_state_shape(None))

    async def fake_search(_query: str, limit: int = 8):
        del limit
        now = datetime.now(timezone.utc)
        return SimpleNamespace(
            edges=[
                SimpleNamespace(
                    uuid='edge-1',
                    fact='Pattern Y is current',
                    episodes=['source-1'],
                    valid_at=now,
                    invalid_at=None,
                ),
                SimpleNamespace(
                    uuid='edge-1',
                    fact='duplicate',
                    episodes=['source-1'],
                    valid_at=now,
                    invalid_at=None,
                ),
                SimpleNamespace(
                    uuid='edge-2',
                    fact='Pattern X was replaced',
                    episodes=['source-2'],
                    valid_at=now - timedelta(days=2),
                    invalid_at=now - timedelta(days=1),
                ),
            ],
            episodes=[
                SimpleNamespace(
                    uuid='decision-1',
                    content='Kind: decision\nSummary: Prefer pattern Y\nSource: agent\n\nDetails:\nUse pattern Y.',
                    name='decision',
                    source_description='memory:decision',
                    created_at=now,
                ),
                SimpleNamespace(
                    uuid='artifact-1',
                    content='Kind: index_artifact\nSummary: README summary\nSource: indexer\nArtifact Path: README.md\n\nDetails:\nPattern Y docs',
                    name='artifact',
                    source_description='memory:index_artifact',
                    created_at=now,
                ),
                SimpleNamespace(
                    uuid='artifact-dup',
                    content='Kind: index_artifact\nSummary: README summary\nSource: indexer\nArtifact Path: README.md\n\nDetails:\nPattern Y docs',
                    name='artifact-dup',
                    source_description='memory:index_artifact',
                    created_at=now,
                ),
            ],
        )

    monkeypatch.setattr(engine, '_search', fake_search)
    monkeypatch.setattr(
        engine,
        '_edge_evidence',
        lambda edges: asyncio.sleep(
            0, result={'source-1': 'bootstrap source', 'source-2': 'old source'}
        ),
    )
    monkeypatch.setattr(
        engine,
        '_fallback_memory_episodes',
        lambda query, limit, exclude_ids=None: asyncio.sleep(
            0,
            result=[
                SimpleNamespace(
                    uuid='workflow-1',
                    kind=MemoryKind.workflow,
                    summary='Run recall first',
                    details='Avoid broad search first.',
                    source='agent',
                    tags=[],
                    artifact_path='',
                    created_at=datetime.now(timezone.utc),
                    raw_name='workflow',
                    source_agent='',
                    session_id='',
                    thread_title='',
                    captured_from='',
                )
            ],
        ),
    )

    recall = await engine.recall('pattern y', limit=4)
    assert 'Relevant Decisions' in recall
    assert 'Patterns And Workflows' in recall
    assert 'Active Facts' in recall
    assert 'Historical Facts' in recall
    assert 'Supporting Artifacts' in recall
    assert recall.count('README.md') == 1

    monkeypatch.setattr(
        engine,
        '_search',
        lambda query, limit=8: asyncio.sleep(0, result=SimpleNamespace(edges=[], episodes=[])),
    )
    monkeypatch.setattr(engine, '_edge_evidence', lambda edges: asyncio.sleep(0, result={}))
    monkeypatch.setattr(
        engine,
        '_fallback_memory_episodes',
        lambda query, limit, exclude_ids=None: asyncio.sleep(0, result=[]),
    )
    assert await engine.recall('nothing useful', limit=1) == 'No relevant memory found.'

    (tmp_path / 'AGENTS.md').write_text('Agent guidance')
    (tmp_path / 'README.md').write_text('# Demo\n')
    (tmp_path / '.git').mkdir(exist_ok=True)
    monkeypatch.setattr(
        engine,
        '_summarize_artifact',
        lambda title, content: asyncio.sleep(
            0, result=ArtifactSummary(summary=title, key_points=[], commands=[])
        ),
    )
    monkeypatch.setattr(
        engine,
        '_remember_unlocked',
        lambda **kwargs: asyncio.sleep(
            0,
            result={
                'uuid': f'episode-{kwargs["artifact_path"] or kwargs["summary"]}',
                'mode': 'episode',
            },
        ),
    )
    monkeypatch.setattr(engine, '_git_artifact', lambda: None)
    indexed = await engine.index(changed_only=False, max_files=4)
    changed = await engine.index(changed_only=True, max_files=4)
    assert indexed
    assert changed == []

    now = datetime.now(timezone.utc)
    discovery = BootstrapDiscovery(
        codex_sessions=[
            BootstrapSession(
                source_agent='codex',
                session_id='import-1',
                title='Import me',
                created_at=now,
                fingerprint='fp-1',
                content='one\ntwo',
                source_path='/tmp/rollout',
            )
        ]
    )
    monkeypatch.setattr(
        engine,
        '_save_source_episode',
        lambda **kwargs: asyncio.sleep(
            0, result=SimpleNamespace(uuid=f'source-{len(kwargs["content"])}')
        ),
    )
    imported = await engine.import_history_sessions(discovery=discovery)
    skipped = await engine.import_history_sessions(discovery=discovery, session_ids=['other'])
    bootstrapped = await engine.bootstrap_history(discovery=discovery)
    assert imported[0]['session_id'] == 'import-1'
    assert skipped == []
    assert bootstrapped == []

    monkeypatch.setattr(
        engine,
        '_query_records',
        lambda query, **kwargs: asyncio.sleep(0, result=[{'episode_count': 7}]),
    )
    doctor = await engine.doctor()
    assert 'Episodes: 7' in doctor
    assert 'History bootstrap sessions: 1' in doctor
    assert 'Codex MCP installed:' in doctor


@pytest.mark.asyncio
async def test_mcp_tool_definitions_and_stdio_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = {tool['name']: tool for tool in mcp._tool_definitions()}
    assert 'remember' in tools
    assert tools['init_project']['inputSchema']['properties']['backend']['enum'] == [
        'kuzu',
        'neo4j',
    ]
    assert tools['init_project']['inputSchema']['properties']['install_mcp']['default'] is False

    body = json.dumps({'jsonrpc': '2.0', 'method': 'ping'}).encode('utf-8')
    fake_stdin = SimpleNamespace(
        buffer=io.BytesIO(f'Content-Length: {len(body)}\r\n\r\n'.encode() + body)
    )
    fake_stdout_buffer = io.BytesIO()
    fake_stdout = SimpleNamespace(buffer=fake_stdout_buffer)
    monkeypatch.setattr(mcp.sys, 'stdin', fake_stdin)
    monkeypatch.setattr(mcp.sys, 'stdout', fake_stdout)

    assert mcp._read_message()['method'] == 'ping'
    mcp._write_message({'jsonrpc': '2.0', 'id': 1, 'result': {'ok': True}})
    written = fake_stdout_buffer.getvalue()
    assert b'Content-Length:' in written
    assert b'"ok": true' in written
    monkeypatch.setattr(mcp.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO()))
    assert mcp._read_message() is None
    monkeypatch.setattr(
        mcp.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(b'Content-Length: 1\r\n\r\n'))
    )
    assert mcp._read_message() is None

    fake_engine = FakeEngine()

    async def fake_open(_root: Path):
        return FakeContext(fake_engine)

    monkeypatch.setattr(mcp.MemoryEngine, 'open', fake_open)
    monkeypatch.setattr(
        mcp.MemoryEngine,
        'choose_backend',
        classmethod(
            lambda cls, root, requested_backend=None: requested_backend or BackendType.kuzu
        ),
    )
    monkeypatch.setattr(
        mcp.MemoryEngine,
        'default_runtime_config',
        classmethod(
            lambda cls, root, backend=None: RuntimeConfig(
                project_name='demo',
                project_id='pid',
                database_path='.graphiti/memory.kuzu',
                backend=backend or BackendType.kuzu,
            )
        ),
    )
    monkeypatch.setattr(
        mcp.MemoryEngine,
        'init_project',
        staticmethod(
            lambda root, force=False, config=None: (
                build_project_paths(root),
                config
                or RuntimeConfig(
                    project_name='demo', project_id='pid', database_path='.graphiti/memory.kuzu'
                ),
            )
        ),
    )
    monkeypatch.setattr(
        mcp.MemoryEngine,
        'apply_managed_agents_block',
        staticmethod(lambda root: root / 'AGENTS.md'),
    )
    monkeypatch.setattr(
        mcp.MemoryEngine,
        'detect_onboarding_state',
        classmethod(
            lambda cls, root, history_days=90, requested_backend=None: {
                'configured': True,
                'backend': (requested_backend or BackendType.kuzu).value,
                'history_sessions_detected': 0,
                'mcp_command': 'graphiti mcp --transport stdio',
                'codex_mcp_installed': True,
            }
        ),
    )
    init_payload = json.loads(
        await mcp._call_tool(tmp_path, 'init_project', {'apply_agents': True})
    )
    assert init_payload['agents_updated'] is True
    assert init_payload['configured_backend'] == 'kuzu'
    assert 'codex_mcp_updated' in init_payload
    assert init_payload['codex_mcp_updated'] is False

    monkeypatch.setattr(
        mcp,
        'install_codex_mcp_server',
        lambda **kwargs: (tmp_path / '.codex' / 'config.toml', True),
    )
    init_payload = json.loads(
        await mcp._call_tool(tmp_path, 'init_project', {'apply_agents': True, 'install_mcp': True})
    )
    assert init_payload['codex_mcp_updated'] is True

    apply_payload = json.loads(await mcp._call_tool(tmp_path, 'apply_agents_instructions', {}))
    assert apply_payload['agents_path'].endswith('AGENTS.md')

    assert (
        json.loads(await mcp._call_tool(tmp_path, 'list_history_sessions', {'limit': 2}))[0][
            'limit'
        ]
        == 2
    )
    assert (
        json.loads(
            await mcp._call_tool(tmp_path, 'read_history_session', {'session_id': 'session-1'})
        )['session_id']
        == 'session-1'
    )
    assert (
        json.loads(
            await mcp._call_tool(tmp_path, 'import_history_sessions', {'session_ids': ['s1']})
        )[0]['session_id']
        == 's1'
    )
    assert (
        json.loads(
            await mcp._call_tool(tmp_path, 'remember', {'kind': 'decision', 'summary': 'Prefer Y'})
        )['uuid']
        == 'memory-1'
    )
    assert await mcp._call_tool(tmp_path, 'recall', {'query': 'task'}) == 'recall:task:8'
    assert (
        json.loads(await mcp._call_tool(tmp_path, 'index', {'changed_only': False}))[0]['artifact']
        == 'README.md'
    )
    assert await mcp._call_tool(tmp_path, 'doctor', {}) == 'doctor-ok'

    with pytest.raises(ValueError):
        await mcp._call_tool(tmp_path, 'unsupported', {})

    messages = iter(
        [
            {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {}},
            {'jsonrpc': '2.0', 'method': 'notifications/initialized', 'params': {}},
            {'jsonrpc': '2.0', 'id': 2, 'method': 'ping', 'params': {}},
            {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/list', 'params': {}},
            {
                'jsonrpc': '2.0',
                'id': 4,
                'method': 'tools/call',
                'params': {'name': 'doctor', 'arguments': {}},
            },
            {
                'jsonrpc': '2.0',
                'id': 5,
                'method': 'tools/call',
                'params': {'name': 'doctor', 'arguments': {}},
            },
            {'jsonrpc': '2.0', 'id': 6, 'method': 'unknown', 'params': {}},
            None,
        ]
    )
    written_messages: list[dict[str, Any]] = []
    tool_calls = iter(['doctor-ok', RuntimeError('bad tool')])

    def fake_read() -> dict[str, Any] | None:
        return next(messages)

    def fake_write(message: dict[str, Any]) -> None:
        written_messages.append(message)

    async def fake_call_tool(_root: Path, _name: str, _arguments: dict[str, Any]) -> str:
        result = next(tool_calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(mcp, '_read_message', fake_read)
    monkeypatch.setattr(mcp, '_write_message', fake_write)
    monkeypatch.setattr(mcp, '_call_tool', fake_call_tool)

    await mcp.run_stdio_mcp_server(tmp_path)

    assert written_messages[0]['result']['serverInfo']['name'] == 'graphiti'
    assert written_messages[1]['result'] == {}
    assert 'tools' in written_messages[2]['result']
    assert written_messages[3]['result']['isError'] is False
    assert written_messages[4]['result']['isError'] is True
    assert written_messages[5]['error']['code'] == -32601


@pytest.mark.asyncio
async def test_kuzu_driver_query_and_session_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    connection_calls: list[Any] = []

    class FakeResult:
        def __init__(self, rows: list[dict[str, Any]]):
            self._rows = rows

        def rows_as_dict(self):
            return iter(self._rows)

    class FakeConnection:
        def __init__(self, db: Any):
            self.db = db
            self.closed = False
            self.executed: list[tuple[str, dict[str, Any] | None]] = []
            connection_calls.append(self)

        def execute(self, query: str, parameters: dict[str, Any] | None = None):
            self.executed.append((query, parameters))
            if query == 'SCHEMA':
                return None
            return FakeResult([{'value': 1}])

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr('graphiti_core.driver.kuzu_driver.kuzu.Database', lambda db: f'db:{db}')
    monkeypatch.setattr('graphiti_core.driver.kuzu_driver.kuzu.Connection', FakeConnection)
    monkeypatch.setattr('graphiti_core.driver.kuzu_driver.SCHEMA_QUERIES', 'SCHEMA')

    driver = KuzuDriver(db=':memory:')
    assert driver.entity_node_ops is not None
    assert driver.episode_node_ops is not None
    assert driver.community_node_ops is not None
    assert driver.saga_node_ops is not None
    assert driver.entity_edge_ops is not None
    assert driver.episodic_edge_ops is not None
    assert driver.community_edge_ops is not None
    assert driver.has_episode_edge_ops is not None
    assert driver.next_episode_edge_ops is not None
    assert driver.search_ops is not None
    assert driver.graph_ops is not None
    result, _, _ = await driver.execute_query(
        'MATCH (n) RETURN n',
        keep='ok',
        database_='ignored',
        routing_='ignored',
    )
    assert result == [{'value': 1}]

    list_result = FakeResult([{'value': 2}])
    monkeypatch.setattr(driver, '_execute_query_sync', lambda query, params: [list_result])
    result, _, _ = await driver.execute_query('MATCH (n) RETURN n', keep='ok')
    assert result == [[{'value': 2}]]

    monkeypatch.setattr(driver, '_execute_query_sync', lambda query, params: [])
    result, _, _ = await driver.execute_query('MATCH (n) RETURN n')
    assert result == []

    logged: list[str] = []
    monkeypatch.setattr(
        'graphiti_core.driver.kuzu_driver.logger.error', lambda message: logged.append(message)
    )

    def raise_exists(query, params):
        raise RuntimeError('index already exists')

    monkeypatch.setattr(driver, '_execute_query_sync', raise_exists)
    with pytest.raises(RuntimeError):
        await driver.execute_query('MATCH (n) RETURN n')
    assert logged == []

    def raise_other(query, params):
        raise RuntimeError('kaboom')

    monkeypatch.setattr(driver, '_execute_query_sync', raise_other)
    with pytest.raises(RuntimeError):
        await driver.execute_query('MATCH (n) RETURN n', items=list(range(10)))
    assert logged

    session = KuzuDriverSession(driver)
    assert driver.session() is not None
    await driver.close()
    driver.delete_all_indexes('local')
    seen_queries: list[Any] = []

    async def fake_execute_query(query: str, **kwargs: Any):
        seen_queries.append((query, kwargs))
        return [], None, None

    monkeypatch.setattr(driver, 'execute_query', fake_execute_query)

    async with session as active_session:
        assert active_session is session
    await session.close()

    async def run_write(tx, value):
        return ('write-ok', tx is session, value)

    assert await session.execute_write(run_write, 7) == ('write-ok', True, 7)
    assert await session.run([('MATCH 1', {'a': 1}), ('MATCH 2', {'b': 2})]) is None
    assert await session.run('MATCH 3', c=3) is None
    assert seen_queries == [('MATCH 1', {'a': 1}), ('MATCH 2', {'b': 2}), ('MATCH 3', {'c': 3})]
    assert connection_calls[0].closed is True
