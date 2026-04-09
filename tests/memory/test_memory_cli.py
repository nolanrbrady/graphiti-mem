from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip('kuzu')

from graphiti_core.memory.cli import main
from graphiti_core.memory.config import GRAPHITI_BLOCK_END, GRAPHITI_BLOCK_START
from graphiti_core.memory.engine import MemoryEngine
from graphiti_core.memory.models import MemoryKind


def write_project_files(root: Path) -> None:
    (root / 'README.md').write_text(
        '# Demo Project\n\nUse `make test` to run tests.\n\nThis project uses local memory.\n'
    )
    (root / 'AGENTS.md').write_text('# Agent Notes\n\nPrefer concise output.\n')
    (root / 'pyproject.toml').write_text(
        '[project]\nname = "demo-project"\nversion = "0.1.0"\nrequires-python = ">=3.10"\n'
    )
    (root / 'app').mkdir()
    (root / 'app' / '__init__.py').write_text('')
    (root / 'app' / 'service.py').write_text('def greet() -> str:\n    return "hello"\n')


def write_codex_history(
    home: Path,
    project_root: Path,
    *,
    thread_id: str,
    title: str,
    user_message: str,
    agent_message: str,
    age_days: int = 0,
    cwd: Path | None = None,
) -> None:
    codex_dir = home / '.codex'
    session_dir = codex_dir / 'sessions' / '2026' / '04' / '09'
    session_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = session_dir / f'rollout-{thread_id}.jsonl'
    rollout_records = [
        {
            'type': 'event_msg',
            'payload': {'type': 'user_message', 'message': user_message},
        },
        {
            'type': 'event_msg',
            'payload': {'type': 'agent_message', 'message': agent_message, 'phase': 'final'},
        },
    ]
    rollout_path.write_text(''.join(json.dumps(record) + '\n' for record in rollout_records))

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
    timestamp = int((datetime.now(timezone.utc) - timedelta(days=age_days)).timestamp() * 1000)
    connection.execute(
        """
        INSERT OR REPLACE INTO threads (
            id, rollout_path, created_at, updated_at, source, model_provider, cwd, title,
            sandbox_policy, approval_mode
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            thread_id,
            str(rollout_path),
            timestamp,
            timestamp,
            'vscode',
            'openai',
            str(cwd or project_root),
            title,
            'workspace-write',
            'never',
        ),
    )
    connection.commit()
    connection.close()


def write_claude_history(
    home: Path,
    project_root: Path,
    *,
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    escaped = str(project_root).replace(os.sep, '-')
    project_dir = home / '.claude' / 'projects' / escaped
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f'{session_id}.jsonl'
    records = [
        {'type': 'file-history-snapshot', 'messageId': 'ignore-me'},
        {
            'type': 'user',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'sessionId': session_id,
            'cwd': str(project_root),
            'message': {'role': 'user', 'content': user_message},
        },
        {
            'type': 'assistant',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'sessionId': session_id,
            'cwd': str(project_root),
            'message': {
                'role': 'assistant',
                'content': [{'type': 'tool_use', 'name': 'Read'}],
            },
        },
        {
            'type': 'assistant',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'sessionId': session_id,
            'cwd': str(project_root),
            'message': {
                'role': 'assistant',
                'content': [{'type': 'text', 'text': assistant_message}],
            },
        },
    ]
    path.write_text(''.join(json.dumps(record) + '\n' for record in records))


def send_mcp_message(stdin, message: dict) -> None:
    body = json.dumps(message).encode('utf-8')
    stdin.write(f'Content-Length: {len(body)}\r\n\r\n'.encode())
    stdin.write(body)
    stdin.flush()


def read_mcp_message(stdout) -> dict:
    content_length = None
    while True:
        line = stdout.readline()
        assert line
        decoded = line.decode('utf-8').strip()
        if not decoded:
            break
        if decoded.lower().startswith('content-length:'):
            content_length = int(decoded.split(':', 1)[1].strip())
    assert content_length is not None
    payload = stdout.read(content_length)
    return json.loads(payload.decode('utf-8'))


@pytest.mark.asyncio
async def test_init_creates_local_state(tmp_path: Path) -> None:
    write_project_files(tmp_path)

    paths, config = MemoryEngine.init_project(tmp_path)

    assert paths.state_dir.exists()
    assert paths.config_path.exists()
    assert paths.index_state_path.exists()
    assert paths.agent_instructions_path.exists()
    assert config.project_name == tmp_path.name
    assert 'graphiti recall "<current task>"' in paths.agent_instructions_path.read_text()


def test_cli_init_applies_managed_agents_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    write_project_files(tmp_path)
    monkeypatch.chdir(tmp_path)

    rc = main(['init', '--yes', '--skip-history', '--apply-agents'])
    out = capsys.readouterr().out
    agents_text = (tmp_path / 'AGENTS.md').read_text()

    assert rc == 0
    assert 'Initialized Graphiti local memory' in out
    assert 'graphiti mcp --transport stdio' in out
    assert GRAPHITI_BLOCK_START in agents_text
    assert GRAPHITI_BLOCK_END in agents_text


def test_cli_init_can_leave_agents_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    write_project_files(tmp_path)
    original_agents = (tmp_path / 'AGENTS.md').read_text()
    monkeypatch.chdir(tmp_path)

    rc = main(['init', '--yes', '--skip-history', '--no-apply-agents'])
    out = capsys.readouterr().out

    assert rc == 0
    assert 'Left AGENTS.md unchanged' in out
    assert (tmp_path / 'AGENTS.md').read_text() == original_agents
    assert (tmp_path / '.graphiti' / 'agent_instructions.md').exists()


def test_cli_init_is_idempotent_for_managed_agents_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_project_files(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert main(['init', '--yes', '--skip-history', '--apply-agents']) == 0
    assert main(['init', '--yes', '--skip-history', '--apply-agents']) == 0

    agents_text = (tmp_path / 'AGENTS.md').read_text()
    assert agents_text.count(GRAPHITI_BLOCK_START) == 1
    assert 'Prefer concise output.' in agents_text


@pytest.mark.asyncio
async def test_remember_then_recall_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_project_files(tmp_path)
    MemoryEngine.init_project(tmp_path)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)

    async with await MemoryEngine.open(tmp_path) as engine:
        remembered = await engine.remember(
            kind=MemoryKind.decision,
            summary='Prefer pattern Y over pattern X',
            details='Pattern X caused unnecessary retries; use pattern Y for deterministic behavior.',
        )
        recall = await engine.recall('pattern Y deterministic behavior')

    assert remembered['uuid']
    assert 'Prefer pattern Y over pattern X' in recall
    assert 'Relevant Decisions' in recall


@pytest.mark.asyncio
async def test_index_stores_high_signal_artifacts_and_recall_finds_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_project_files(tmp_path)
    MemoryEngine.init_project(tmp_path)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)

    async with await MemoryEngine.open(tmp_path) as engine:
        indexed = await engine.index(changed_only=False, max_files=8)
        recall = await engine.recall('make test local memory')

    assert indexed
    assert any(item['artifact'] == 'README.md' for item in indexed)
    assert 'Supporting Artifacts' in recall or 'Patterns And Workflows' in recall
    assert 'README.md' in recall or '__inventory__' in recall


@pytest.mark.asyncio
async def test_persistence_survives_reopen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_project_files(tmp_path)
    MemoryEngine.init_project(tmp_path)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)

    async with await MemoryEngine.open(tmp_path) as engine:
        await engine.remember(
            kind=MemoryKind.workflow,
            summary='Run recall before exploring the codebase',
            details='This repository expects agents to call recall before broad file search.',
        )

    async with await MemoryEngine.open(tmp_path) as engine:
        recall = await engine.recall('exploring the codebase')
        doctor = await engine.doctor()

    assert 'Run recall before exploring the codebase' in recall
    assert 'Episodes:' in doctor


@pytest.mark.asyncio
async def test_changed_only_reindex_updates_only_modified_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_project_files(tmp_path)
    MemoryEngine.init_project(tmp_path)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)

    async with await MemoryEngine.open(tmp_path) as engine:
        first_index = await engine.index(changed_only=False, max_files=8)
        no_changes = await engine.index(changed_only=True, max_files=8)

    state_before = json.loads((tmp_path / '.graphiti' / 'index_state.json').read_text())
    readme_episode_before = state_before['artifacts']['README.md']['episode_uuid']

    (tmp_path / 'README.md').write_text(
        '# Demo Project\n\nUse `make test` to run tests.\n\nUpdated workflow summary for agents.\n'
    )

    async with await MemoryEngine.open(tmp_path) as engine:
        changed = await engine.index(changed_only=True, max_files=8)

    state_after = json.loads((tmp_path / '.graphiti' / 'index_state.json').read_text())
    readme_episode_after = state_after['artifacts']['README.md']['episode_uuid']

    assert first_index
    assert no_changes == []
    assert any(item['artifact'] == 'README.md' for item in changed)
    assert readme_episode_before != readme_episode_after


def test_init_bootstraps_recent_project_codex_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_project_files(tmp_path)
    fake_home = tmp_path / 'home'
    other_project = tmp_path / 'other-project'
    other_project.mkdir()
    monkeypatch.setenv('HOME', str(fake_home))
    monkeypatch.chdir(tmp_path)

    write_codex_history(
        fake_home,
        tmp_path,
        thread_id='recent-project',
        title='Use pattern Y',
        user_message='We tried pattern X already.',
        agent_message='Prefer pattern Y over pattern X because it avoids retries.',
        age_days=2,
    )
    write_codex_history(
        fake_home,
        tmp_path,
        thread_id='old-project',
        title='Old session',
        user_message='Old transcript',
        agent_message='Prefer pattern Z.',
        age_days=120,
    )
    write_codex_history(
        fake_home,
        tmp_path,
        thread_id='other-project',
        title='Different cwd',
        user_message='Other project transcript',
        agent_message='Use a different workflow.',
        age_days=1,
        cwd=other_project,
    )

    assert main(['init', '--yes', '--import-history', '--apply-agents']) == 0

    async def _assert_history() -> None:
        async with await MemoryEngine.open(tmp_path) as engine:
            recall = await engine.recall('pattern Y retries')
            doctor = await engine.doctor()
        assert 'Prefer pattern Y over pattern X because it avoids retries.' in recall
        assert 'agent=codex' in recall
        assert 'History bootstrap sessions: 1' in doctor

    asyncio.run(_assert_history())


def test_init_bootstraps_claude_history_and_ignores_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_project_files(tmp_path)
    fake_home = tmp_path / 'home'
    monkeypatch.setenv('HOME', str(fake_home))
    monkeypatch.chdir(tmp_path)

    write_claude_history(
        fake_home,
        tmp_path,
        session_id='claude-session',
        user_message='How should tests run in this repo?',
        assistant_message='Run `make test` before broad code search and keep output concise.',
    )

    assert main(['init', '--yes', '--import-history', '--apply-agents']) == 0

    async def _assert_history() -> None:
        async with await MemoryEngine.open(tmp_path) as engine:
            recall = await engine.recall('how should tests run')
        assert 'Run `make test` before broad code search' in recall
        assert 'agent=claude' in recall

    asyncio.run(_assert_history())


def test_mcp_stdio_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_project_files(tmp_path)
    MemoryEngine.init_project(tmp_path)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)

    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
    process = subprocess.Popen(
        [sys.executable, '-m', 'graphiti_core.memory.cli', 'mcp', '--transport', 'stdio'],
        cwd=tmp_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    assert process.stdin is not None
    assert process.stdout is not None

    try:
        send_mcp_message(
            process.stdin,
            {'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {}},
        )
        init_response = read_mcp_message(process.stdout)
        assert init_response['result']['serverInfo']['name'] == 'graphiti'

        send_mcp_message(
            process.stdin,
            {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}},
        )
        list_response = read_mcp_message(process.stdout)
        tool_names = {tool['name'] for tool in list_response['result']['tools']}
        assert {'recall', 'remember', 'index', 'doctor'} <= tool_names

        send_mcp_message(
            process.stdin,
            {
                'jsonrpc': '2.0',
                'id': 3,
                'method': 'tools/call',
                'params': {
                    'name': 'remember',
                    'arguments': {
                        'kind': 'decision',
                        'summary': 'Prefer pattern Y',
                        'details': 'Pattern Y is the current choice.',
                    },
                },
            },
        )
        remember_response = read_mcp_message(process.stdout)
        assert remember_response['result']['isError'] is False

        send_mcp_message(
            process.stdin,
            {
                'jsonrpc': '2.0',
                'id': 4,
                'method': 'tools/call',
                'params': {'name': 'recall', 'arguments': {'query': 'pattern Y', 'limit': 4}},
            },
        )
        recall_response = read_mcp_message(process.stdout)
        assert 'Prefer pattern Y' in recall_response['result']['content'][0]['text']
    finally:
        process.terminate()
        process.wait(timeout=5)
