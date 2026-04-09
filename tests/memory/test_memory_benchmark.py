from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from graphiti_core.memory.benchmark.__main__ import main as benchmark_main
from graphiti_core.memory.benchmark.fixtures import get_fixture, get_suite_tasks, list_fixture_catalog
from graphiti_core.memory.benchmark.models import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkBaselineExpectation,
    BenchmarkBudget,
    BenchmarkFactMatch,
    BenchmarkGoldFact,
    BenchmarkHardFailRule,
    BenchmarkResult,
    BenchmarkRetrievalTrace,
    BenchmarkSupportSet,
    BenchmarkTaskFixture,
    BenchmarkTaskType,
)
from graphiti_core.memory.benchmark.runner import (
    _history_trace_candidate_ids,
    _memory_candidate_ids,
    _multi_hop_trace_candidate_ids,
    _channel_result,
    benchmark_doctor,
    compare_results,
    run_benchmark,
)
from graphiti_core.memory.models import MemoryKind, ParsedMemoryEpisode
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
    assert task.budgets.max_returned_context_chars > 0
    assert task.gold_facts
    assert task.acceptable_support_sets


def test_memory_candidate_ids_prioritize_support_sources_by_task_type() -> None:
    memory = ParsedMemoryEpisode(
        uuid='episode-1',
        kind=MemoryKind.workflow,
        summary='Run Graphiti recall before broad file search',
        details='Use recall first.',
        source='agent',
        artifact_path='AGENTS.md',
        source_agent='codex',
        session_id='session-search-first',
        thread_title='Recall before search',
    )

    artifact_ids = _memory_candidate_ids(memory, task_type=BenchmarkTaskType.artifact_recall)
    history_ids = _memory_candidate_ids(memory, task_type=BenchmarkTaskType.history_recall)
    multihop_ids = _memory_candidate_ids(memory, task_type=BenchmarkTaskType.multi_hop_recall)

    assert artifact_ids[:2] == ['artifact:AGENTS.md', 'thread:Recall before search']
    assert history_ids[:2] == ['thread:Recall before search', 'artifact:AGENTS.md']
    assert multihop_ids[:2] == ['thread:Recall before search', 'artifact:AGENTS.md']
    assert artifact_ids[-1] == 'session:session-search-first'
    assert history_ids[-1] == 'session:session-search-first'
    assert multihop_ids[-1] == 'session:session-search-first'


def test_history_trace_candidate_ids_prioritize_selected_history_match() -> None:
    selected_pitfall = ParsedMemoryEpisode(
        uuid='episode-pitfall',
        kind=MemoryKind.pitfall,
        summary='Transcript import alone is insufficient for deterministic recall',
        details='Store durable memories too.',
        source='agent',
        source_agent='codex',
        session_id='session-history-gap',
        thread_title='History import gap',
    )
    selected_constraint = ParsedMemoryEpisode(
        uuid='episode-constraint',
        kind=MemoryKind.constraint,
        summary='Default benchmark must remain deterministic and offline-capable',
        details='Avoid external judges.',
        source='agent',
        source_agent='codex',
        session_id='session-benchmark-loop',
        thread_title='Autoresearch reward loop',
    )
    leftover_artifact = ParsedMemoryEpisode(
        uuid='episode-artifact',
        kind=MemoryKind.index_artifact,
        summary='Importing transcript sessions stores source evidence',
        details='docs/history.md',
        source='artifact',
        artifact_path='docs/history.md',
    )

    candidate_ids = _history_trace_candidate_ids(
        [selected_constraint, selected_pitfall],
        [leftover_artifact, selected_constraint, selected_pitfall],
        query='Why is transcript import alone not enough for the offline benchmark?',
    )

    assert candidate_ids[:2] == [
        'thread:History import gap',
        'thread:Autoresearch reward loop',
    ]
    assert candidate_ids[-1] == 'artifact:docs/history.md'


def test_multi_hop_trace_candidate_ids_prioritize_selected_thread_and_artifact_sources() -> None:
    selected_pattern = ParsedMemoryEpisode(
        uuid='episode-pattern',
        kind=MemoryKind.pattern,
        summary='Pair naive source scan control against Graphiti recall treatment',
        details='Use paired execution.',
        source='agent',
        source_agent='codex',
        session_id='session-benchmark-loop',
        thread_title='Autoresearch reward loop',
    )
    selected_doc = ParsedMemoryEpisode(
        uuid='episode-doc',
        kind=MemoryKind.index_artifact,
        summary='The benchmark uses paired execution',
        details='docs/benchmarking.md',
        source='artifact',
        artifact_path='docs/benchmarking.md',
    )
    leftover_artifact = ParsedMemoryEpisode(
        uuid='episode-leftover',
        kind=MemoryKind.index_artifact,
        summary='Use make test to run tests',
        details='README.md',
        source='artifact',
        artifact_path='README.md',
    )
    leftover_thread = ParsedMemoryEpisode(
        uuid='episode-leftover-thread',
        kind=MemoryKind.decision,
        summary='Prefer pattern Y over pattern X',
        details='Keeps ingestion deterministic.',
        source='agent',
        source_agent='codex',
        session_id='session-pattern-y',
        thread_title='Pattern Y migration',
    )

    candidate_ids = _multi_hop_trace_candidate_ids(
        [selected_pattern, selected_doc],
        [selected_doc, selected_pattern, leftover_thread, leftover_artifact],
    )

    assert candidate_ids[:6] == [
        'thread:Autoresearch reward loop',
        'artifact:docs/benchmarking.md',
        'memory:pattern:pair-naive-source-scan-control-against-graphiti-recall-treatment',
        'memory:index_artifact:the-benchmark-uses-paired-execution',
        'session:session-benchmark-loop',
        'thread:Pattern Y migration',
    ]


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
                'mean_task_score': 80.0,
                'mean_task_score_normalized': 0.8,
                'mean_retrieval_score': 0.8,
                'mean_attribution_score': 0.8,
                'mean_answer_score': 0.8,
                'mean_capability_score': 0.8,
                'mean_efficiency_score': 1.0,
                'artifact_task_score': 80.0,
                'history_task_score': 80.0,
                'multi_hop_task_score': 80.0,
                'budget_failure_count': 0,
                'provenance_failure_count': 0,
                'support_failure_count': 0,
                'unsupported_claim_failure_count': 0,
                'mean_returned_context_chars': 700.0,
                'mean_control_context_chars': 1200.0,
                'gate_thresholds': {},
            },
            'gate_passed': True,
            'reward': 0.8,
            'failure_reasons': [],
        }
    )
    second = first.model_copy(
        update={
            'aggregate': first.aggregate.model_copy(
                update={
                    'mean_task_score': 86.0,
                    'mean_task_score_normalized': 0.86,
                    'mean_retrieval_score': 0.85,
                    'mean_attribution_score': 0.82,
                    'mean_answer_score': 0.84,
                    'mean_capability_score': 0.84,
                }
            ),
            'reward': 0.86,
        }
    )
    comparison = compare_results(first, second)
    assert comparison.reward_delta == pytest.approx(0.06)
    assert comparison.aggregate_deltas['mean_task_score'] == pytest.approx(6.0)

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
    assert all(task.treatment.retrieval_trace.candidate_ids for task in result.tasks)


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


def test_channel_result_requires_provenance_for_hard_failures() -> None:
    fixture = BenchmarkTaskFixture(
        task_id='missing-provenance',
        suite='deterministic_core',
        tier='smoke',
        query='What command should run?',
        task_type=BenchmarkTaskType.artifact_recall,
        difficulty='easy',
        gold_facts=[BenchmarkGoldFact(key='command', values=['make test'])],
        acceptable_support_sets=[BenchmarkSupportSet(source_ids=['artifact:README.md'])],
        budgets=BenchmarkBudget(max_retrieval_calls=2, max_returned_context_chars=200, max_selected_items=1),
        hard_fail_rules=[BenchmarkHardFailRule.missing_provenance],
        baseline_expectation=BenchmarkBaselineExpectation(
            minimum_treatment_accuracy=0.0,
            minimum_evidence_coverage=0.0,
            minimum_retrieval_score=0.0,
            minimum_answer_score=0.0,
            minimum_task_score=0.0,
        ),
    )
    trace = BenchmarkRetrievalTrace(
        retrieval_calls=1,
        retrieval_queries=['What command should run?'],
        candidate_ids=['artifact:README.md'],
        selected_evidence_ids=['artifact:README.md'],
        provenance_ids=[],
        selected_item_count=1,
        candidate_count=1,
    )
    result = _channel_result(fixture, '- Use `make test`.', trace)
    assert result.answer_score == pytest.approx(1.0)
    assert result.capability_score == pytest.approx(0.0)
    assert 'missing_provenance' in result.hard_failures


def test_channel_result_distractor_zeroes_attribution() -> None:
    fixture = BenchmarkTaskFixture(
        task_id='wrong-support',
        suite='deterministic_core',
        tier='smoke',
        query='What is current?',
        task_type=BenchmarkTaskType.artifact_recall,
        difficulty='easy',
        gold_facts=[BenchmarkGoldFact(key='target', values=['make benchmark-memory'])],
        acceptable_support_sets=[BenchmarkSupportSet(source_ids=['artifact:Makefile'])],
        distractor_source_ids=['artifact:docs/legacy-benchmark.md'],
        budgets=BenchmarkBudget(max_retrieval_calls=2, max_returned_context_chars=300, max_selected_items=1),
        hard_fail_rules=[BenchmarkHardFailRule.wrong_support],
        baseline_expectation=BenchmarkBaselineExpectation(
            minimum_treatment_accuracy=0.0,
            minimum_evidence_coverage=0.0,
            minimum_retrieval_score=0.0,
            minimum_answer_score=0.0,
            minimum_task_score=0.0,
        ),
    )
    trace = BenchmarkRetrievalTrace(
        retrieval_calls=1,
        retrieval_queries=['What is current?'],
        candidate_ids=['artifact:Makefile', 'artifact:docs/legacy-benchmark.md'],
        selected_evidence_ids=['artifact:docs/legacy-benchmark.md'],
        provenance_ids=['artifact:docs/legacy-benchmark.md'],
        selected_item_count=1,
        candidate_count=2,
    )
    result = _channel_result(fixture, '- legacy target', trace)
    assert result.attribution_score == pytest.approx(0.0)
    assert result.capability_score == pytest.approx(0.0)
    assert 'wrong_support' in result.hard_failures


def test_channel_result_paraphrase_fact_matching() -> None:
    fixture = BenchmarkTaskFixture(
        task_id='paraphrase',
        suite='deterministic_core',
        tier='smoke',
        query='What search should memory prevent?',
        task_type=BenchmarkTaskType.artifact_recall,
        difficulty='easy',
        gold_facts=[
            BenchmarkGoldFact(
                key='search_cost',
                values=['token-hungry broad search', 'reduce token-hungry broad search'],
                match=BenchmarkFactMatch.any_contains,
            )
        ],
        acceptable_support_sets=[BenchmarkSupportSet(source_ids=['artifact:README.md'])],
        budgets=BenchmarkBudget(max_retrieval_calls=2, max_returned_context_chars=300, max_selected_items=1),
        hard_fail_rules=[],
        baseline_expectation=BenchmarkBaselineExpectation(
            minimum_treatment_accuracy=0.0,
            minimum_evidence_coverage=0.0,
            minimum_retrieval_score=0.0,
            minimum_answer_score=0.0,
            minimum_task_score=0.0,
        ),
    )
    trace = BenchmarkRetrievalTrace(
        retrieval_calls=1,
        retrieval_queries=['What search should memory prevent?'],
        candidate_ids=['artifact:README.md'],
        selected_evidence_ids=['artifact:README.md'],
        provenance_ids=['artifact:README.md'],
        selected_item_count=1,
        candidate_count=1,
    )
    result = _channel_result(
        fixture,
        '- Graphiti should reduce token-hungry broad search before broad repo spelunking.',
        trace,
    )
    assert result.answer_score == pytest.approx(1.0)
    assert result.capability_score > 0


def test_channel_result_budget_overrun_zeroes_task_score() -> None:
    fixture = BenchmarkTaskFixture(
        task_id='budget-overrun',
        suite='deterministic_core',
        tier='smoke',
        query='How should answers be formatted?',
        task_type=BenchmarkTaskType.artifact_recall,
        difficulty='easy',
        gold_facts=[BenchmarkGoldFact(key='style', values=['concise evidence-first answers'])],
        acceptable_support_sets=[BenchmarkSupportSet(source_ids=['artifact:AGENTS.md'])],
        budgets=BenchmarkBudget(max_retrieval_calls=1, max_returned_context_chars=40, max_selected_items=1),
        hard_fail_rules=[BenchmarkHardFailRule.budget_overrun],
        baseline_expectation=BenchmarkBaselineExpectation(
            minimum_treatment_accuracy=0.0,
            minimum_evidence_coverage=0.0,
            minimum_retrieval_score=0.0,
            minimum_answer_score=0.0,
            minimum_task_score=0.0,
        ),
    )
    trace = BenchmarkRetrievalTrace(
        retrieval_calls=2,
        retrieval_queries=['How should answers be formatted?', 'fallback'],
        candidate_ids=['artifact:AGENTS.md'],
        selected_evidence_ids=['artifact:AGENTS.md'],
        provenance_ids=['artifact:AGENTS.md'],
        selected_item_count=1,
        candidate_count=1,
    )
    result = _channel_result(fixture, '- concise evidence-first answers', trace)
    assert result.efficiency_score == pytest.approx(0.0)
    assert result.task_score == pytest.approx(0.0)
