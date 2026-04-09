from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from graphiti_core.memory.benchmark.__main__ import main as benchmark_main
from graphiti_core.memory.benchmark.fixtures import get_fixture, get_suite_tasks, list_fixture_catalog
from graphiti_core.memory.benchmark.models import BENCHMARK_SCHEMA_VERSION, BenchmarkResult
from graphiti_core.memory.benchmark.runner import benchmark_doctor, compare_results, run_benchmark
from graphiti_core.memory.benchmark.telemetry import (
    count_search_actions_from_rollout,
    read_codex_thread_metrics,
)


def write_codex_history(
    home: Path,
    project_root: Path,
    *,
    thread_id: str,
    title: str,
    user_message: str,
    assistant_message: str,
) -> None:
    codex_dir = home / '.codex'
    session_dir = codex_dir / 'sessions' / '2026' / '04' / '09'
    session_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = session_dir / f'rollout-{thread_id}.jsonl'
    rollout_path.write_text(
        ''.join(
            json.dumps(record) + '\n'
            for record in [
                {
                    'type': 'event_msg',
                    'payload': {'type': 'user_message', 'message': user_message},
                },
                {
                    'type': 'event_msg',
                    'payload': {
                        'type': 'agent_message',
                        'message': assistant_message,
                        'phase': 'final',
                    },
                },
            ]
        )
    )

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
    connection.execute(
        """
        INSERT OR REPLACE INTO threads (
            id, rollout_path, created_at, updated_at, source, model_provider, cwd, title,
            sandbox_policy, approval_mode, tokens_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            thread_id,
            str(rollout_path),
            timestamp,
            timestamp,
            'vscode',
            'openai',
            str(project_root),
            title,
            'workspace-write',
            'never',
            1234,
        ),
    )
    connection.commit()
    connection.close()


def test_fixture_catalog_and_suite_expansion() -> None:
    catalog = list_fixture_catalog()
    assert catalog.suites['deterministic_core']['smoke'] >= 8
    assert len(get_suite_tasks('deterministic_core', 'full')) >= 24

    task = get_fixture('artifact-test-command')
    assert task.suite == 'deterministic_core'
    assert task.task_id == 'artifact-test-command'
    assert task.max_recall_chars > 0


def test_telemetry_rollout_and_sqlite_metrics(tmp_path: Path) -> None:
    rollout_path = tmp_path / 'rollout.jsonl'
    rollout_path.write_text(
        '\n'.join(
            [
                json.dumps(
                    {
                        'type': 'event_msg',
                        'payload': {
                            'type': 'tool_call',
                            'tool_name': 'exec_command',
                            'cmd': 'rg --files graphiti_core',
                        },
                    }
                ),
                json.dumps(
                    {
                        'type': 'event_msg',
                        'payload': {
                            'type': 'tool_call',
                            'tool_name': 'exec_command',
                            'cmd': 'ls graphiti_core',
                        },
                    }
                ),
                json.dumps(
                    {
                        'type': 'event_msg',
                        'payload': {
                            'type': 'tool_call',
                            'tool_name': 'exec_command',
                            'cmd': 'echo done',
                        },
                    }
                ),
            ]
        )
        + '\n'
    )
    assert count_search_actions_from_rollout(rollout_path) == 2

    db_path = tmp_path / 'state_5.sqlite'
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE threads (
            id TEXT PRIMARY KEY,
            rollout_path TEXT NOT NULL,
            tokens_used INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    connection.execute(
        'INSERT INTO threads (id, rollout_path, tokens_used) VALUES (?, ?, ?)',
        ('thread-1', str(rollout_path), 321),
    )
    connection.commit()
    connection.close()

    metrics = read_codex_thread_metrics(db_path, 'thread-1')
    assert metrics['tokens_used'] == 321
    assert metrics['search_actions'] == 2
    assert str(rollout_path) == metrics['rollout_path']


def test_doctor_and_compare_commands(tmp_path: Path, capsys) -> None:
    doctor = benchmark_doctor()
    assert 'deterministic_core' in doctor.suites
    assert doctor.fixtures_valid is True

    first = BenchmarkResult.model_validate(
        {
            'schema_version': BENCHMARK_SCHEMA_VERSION,
            'suite': 'deterministic_core',
            'tier': 'smoke',
            'config': {},
            'timestamp': '2026-04-09T00:00:00Z',
            'tasks': [],
            'aggregate': {
                'task_count': 0,
                'accuracy_score': 0.8,
                'evidence_coverage': 0.8,
                'token_efficiency_score': 0.2,
                'search_efficiency_score': 0.1,
                'artifact_accuracy': 0.8,
                'history_accuracy': 0.8,
                'multi_hop_accuracy': 0.8,
                'mean_recall_chars': 700.0,
                'mean_control_chars': 1200.0,
                'gate_thresholds': {},
            },
            'gate_passed': True,
            'reward': 0.55,
            'failure_reasons': [],
        }
    )
    second = first.model_copy(
        update={
            'aggregate': first.aggregate.model_copy(
                update={
                    'accuracy_score': 0.85,
                    'evidence_coverage': 0.82,
                    'token_efficiency_score': 0.3,
                    'search_efficiency_score': 0.15,
                }
            ),
            'reward': 0.61,
        }
    )
    comparison = compare_results(first, second)
    assert comparison.reward_delta == pytest.approx(0.06)
    assert comparison.aggregate_deltas['accuracy_score'] == pytest.approx(0.05)

    baseline_path = tmp_path / 'baseline.json'
    candidate_path = tmp_path / 'candidate.json'
    baseline_path.write_text(json.dumps(first.model_dump(mode='json')))
    candidate_path.write_text(json.dumps(second.model_dump(mode='json')))
    assert benchmark_main(['compare', str(baseline_path), str(candidate_path)]) == 0
    assert 'reward_delta' in capsys.readouterr().out


@pytest.mark.asyncio
async def test_run_benchmark_smoke_suite() -> None:
    pytest.importorskip('kuzu')

    result = await run_benchmark('deterministic_core', 'smoke')

    assert result.schema_version == BENCHMARK_SCHEMA_VERSION
    assert result.suite == 'deterministic_core'
    assert result.tier == 'smoke'
    assert result.aggregate.task_count >= 8
    assert result.gate_passed is True
    assert result.reward is not None
    assert all(task.treatment.context for task in result.tasks)


@pytest.mark.asyncio
async def test_run_benchmark_dogfood_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip('kuzu')

    project_root = tmp_path / 'repo'
    project_root.mkdir()
    (project_root / 'README.md').write_text(
        '# Demo Repo\n\nUse `make test` to run tests.\nRun `graphiti mcp --transport stdio` for Codex.\n'
    )
    (project_root / 'AGENTS.md').write_text('Run Graphiti recall before broad file search.\n')
    (project_root / 'pyproject.toml').write_text(
        '[project]\nname = "demo"\nrequires-python = ">=3.10"\n'
    )
    home = tmp_path / 'home'
    write_codex_history(
        home,
        project_root,
        thread_id='dogfood-1',
        title='Benchmark dogfood session',
        user_message='How should we approach the benchmark work?',
        assistant_message='Inspect the memory engine and existing tests first so the benchmark stays grounded in the repo.',
    )

    monkeypatch.setenv('HOME', str(home))
    monkeypatch.chdir(project_root)

    result = await run_benchmark('deterministic_core', 'smoke', mode='dogfood')

    assert result.suite == 'dogfood_local'
    assert result.config['mode'] == 'dogfood'
    assert result.aggregate.task_count >= 4
    assert any(task.task_type.value == 'history_recall' for task in result.tasks)
