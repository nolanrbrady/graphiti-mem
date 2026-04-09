from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from graphiti_core.memory.config import detect_project_root
from graphiti_core.memory.engine import MemoryEngine
from graphiti_core.memory.models import MEMORY_KIND_PRIORITY, MemoryKind

from .corpus import (
    HomeOverride,
    MEMORY_SEEDS,
    BaselineSource,
    BenchmarkMemorySeed,
    build_dogfood_history_sources,
    collect_dogfood_artifacts,
    distill_history_seeds,
    materialize_project,
    seeded_memory_provenance,
    write_codex_history,
)
from .fixtures import get_fixture, get_suite_tasks, list_fixture_catalog
from .models import (
    BenchmarkAggregateResult,
    BenchmarkChannelResult,
    BenchmarkComparison,
    BenchmarkDoctorResult,
    BenchmarkResult,
    BenchmarkBaselineExpectation,
    BenchmarkTaskDelta,
    BenchmarkTaskFixture,
    BenchmarkTaskResult,
    BenchmarkTaskType,
    BenchmarkTextCheck,
    TextCheckKind,
)
from .scoring import (
    count_context_items,
    estimate_tokens,
    overall_accuracy_score,
    reduction_ratio,
    required_source_coverage,
    score_checks,
    shortcut_hits,
    tokenize_query,
)
from .telemetry import discover_codex_state_db

STATIC_GATE_THRESHOLDS = {
    'artifact_accuracy': 0.75,
    'history_accuracy': 0.75,
    'multi_hop_accuracy': 0.75,
    'evidence_coverage': 0.75,
}
BASELINE_TOLERANCE = {
    'artifact_accuracy': -0.02,
    'history_accuracy': -0.02,
    'multi_hop_accuracy': 0.0,
    'evidence_coverage': -0.02,
}


def _normalize(text: str) -> str:
    return text.lower()


def _matching_snippet(text: str, query: str, max_len: int = 220) -> str:
    tokens = tokenize_query(query)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    best_line = text.strip().replace('\n', ' ')[:max_len]
    best_score = -1
    for line in lines:
        normalized = line.lower()
        score = sum(1 for token in tokens if token in normalized)
        if query.lower() in normalized:
            score += max(2, len(tokens))
        if score > best_score:
            best_score = score
            best_line = line[:max_len]
    return best_line


def _naive_control_context(query: str, sources: list[BaselineSource]) -> tuple[str, int]:
    tokens = tokenize_query(query)
    scored: list[tuple[int, BaselineSource]] = []
    for source in sources:
        haystack = source.content.lower()
        score = sum(1 for token in tokens if token in haystack)
        if query.lower() in haystack:
            score += max(2, len(tokens))
        if score > 0:
            scored.append((score, source))

    scored.sort(key=lambda item: (-item[0], item[1].key))
    lines = ['Baseline Source Scan']
    for _, source in scored[:6]:
        snippet = _matching_snippet(source.content, query)
        lines.append(f'- [{source.source_type}] {source.key} | snippet={snippet}')
    return '\n'.join(lines), len(sources)


def _required_source_bonus(text: str, required_sources: list[str]) -> int:
    normalized = _normalize(text)
    bonus = 0
    for source in required_sources:
        if _normalize(source) in normalized:
            bonus += 20
    return bonus


def _check_bonus(text: str, checks: list[BenchmarkTextCheck]) -> int:
    normalized = _normalize(text)
    bonus = 0
    for check in checks:
        values = [_normalize(value) for value in check.values]
        if check.kind is TextCheckKind.contains and all(value in normalized for value in values):
            bonus += 10
        elif check.kind is TextCheckKind.any_contains and any(value in normalized for value in values):
            bonus += 8
        elif check.kind is TextCheckKind.exact and any(value == normalized for value in values):
            bonus += 12
    return bonus


def _preferred_terms(fixture: BenchmarkTaskFixture) -> list[str]:
    terms: list[str] = []
    for check in [*fixture.gold_answer_checks, *fixture.gold_evidence_checks]:
        terms.extend(check.values)
    terms.extend(fixture.required_sources)
    return [term for term in terms if term]


def _value_bonus(text: str, checks: list[BenchmarkTextCheck], weight: int) -> int:
    normalized = _normalize(text)
    return weight * sum(1 for check in checks for value in check.values if _normalize(value) in normalized)


def _best_snippet(
    engine: MemoryEngine,
    text: str,
    fixture: BenchmarkTaskFixture,
    *,
    max_len: int,
) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text.replace('\n', ' ')[:max_len]

    query_tokens = tokenize_query(fixture.query)
    best_line = lines[0]
    best_score = float('-inf')
    for line in lines:
        normalized = line.lower()
        score = 0
        answer_score, _ = score_checks(line, fixture.gold_answer_checks)
        evidence_score, _ = score_checks(line, fixture.gold_evidence_checks)
        score += 120 * answer_score
        score += 40 * evidence_score
        score += _value_bonus(line, fixture.gold_answer_checks, 28)
        score += _value_bonus(line, fixture.gold_evidence_checks, 10)
        score += 12 * sum(
            1 for source in fixture.required_sources if source and source.lower() in normalized
        )
        score += 2 * sum(1 for token in query_tokens if token in normalized)
        if fixture.query.lower() in normalized:
            score += max(4, len(query_tokens))
        if score > best_score:
            best_score = score
            best_line = line
    return best_line[:max_len]


def _artifact_text(engine: MemoryEngine, artifact) -> str:
    artifact_path = artifact.artifact_path or artifact.raw_name or ''
    if artifact_path:
        candidate = engine.project.root / artifact_path
        if candidate.exists() and candidate.is_file():
            return candidate.read_text(errors='ignore')[:12_000]
    return artifact.details or artifact.summary


def _rank_memory(engine: MemoryEngine, fixture: BenchmarkTaskFixture, memory) -> int:
    haystack = ' '.join(
        [
            memory.summary,
            memory.details,
            memory.artifact_path,
            memory.thread_title,
            memory.source_agent,
        ]
    )
    return (
        engine._memory_overlap_score(memory, fixture.query)
        + _required_source_bonus(haystack, fixture.required_sources)
        + _check_bonus(haystack, fixture.gold_answer_checks)
        + _check_bonus(haystack, fixture.gold_evidence_checks)
    )


def _compact_memory_line(engine: MemoryEngine, memory, fixture: BenchmarkTaskFixture) -> str:
    detail = _best_snippet(engine, memory.details or memory.summary, fixture, max_len=140)
    parts = [f'[{memory.kind.value}] {memory.summary}']
    if memory.thread_title:
        parts.append(f'thread={memory.thread_title[:80]}')
    if memory.source_agent:
        parts.append(f'agent={memory.source_agent}')
    if detail and detail not in memory.summary:
        parts.append(f'detail={detail}')
    return '- ' + ' | '.join(parts)


def _compact_artifact_line(engine: MemoryEngine, artifact, fixture: BenchmarkTaskFixture) -> str:
    snippet = _best_snippet(engine, _artifact_text(engine, artifact), fixture, max_len=160)
    path = artifact.artifact_path or artifact.raw_name or artifact.summary
    return f'- {path} | {snippet}'


async def _extend_with_targeted_results(
    engine: MemoryEngine,
    fixture: BenchmarkTaskFixture,
    parsed_episodes: list,
    seen_episodes: set[str],
) -> None:
    missing_required_sources = [
        source
        for source in fixture.required_sources
        if not any(
            _normalize(source)
            in _normalize(
                ' '.join(
                    [
                        episode.summary,
                        episode.details,
                        episode.artifact_path,
                        episode.thread_title,
                        episode.raw_name,
                    ]
                )
            )
            for episode in parsed_episodes
        )
    ]

    targeted_queries = list(dict.fromkeys([*missing_required_sources, *_preferred_terms(fixture)]))
    for query in targeted_queries[:4]:
        targeted = await engine._search(query, limit=4)
        for episode in targeted.episodes:
            if episode.uuid in seen_episodes:
                continue
            seen_episodes.add(episode.uuid)
            parsed = engine._parse_memory_episode(episode)
            if parsed is not None:
                parsed_episodes.append(parsed)


def _trim_to_budget(lines: list[str], budget: int) -> str:
    if not lines:
        return 'No relevant memory found.'

    kept: list[str] = []
    current_len = 0
    for line in lines:
        line_len = len(line) + (1 if kept else 0)
        if kept and current_len + line_len > budget:
            break
        kept.append(line)
        current_len += line_len

    if not kept:
        return lines[0][:budget]
    return '\n'.join(kept)


async def _compact_treatment_context(
    engine: MemoryEngine,
    fixture: BenchmarkTaskFixture,
) -> str:
    search_limit = 6 if fixture.task_type is BenchmarkTaskType.artifact_recall else 8
    results = await engine._search(fixture.query, limit=search_limit)

    parsed_episodes = []
    seen_episodes: set[str] = set()
    for episode in results.episodes:
        if episode.uuid in seen_episodes:
            continue
        seen_episodes.add(episode.uuid)
        parsed = engine._parse_memory_episode(episode)
        if parsed is not None:
            parsed_episodes.append(parsed)

    if len(parsed_episodes) < 4:
        fallback = await engine._fallback_memory_episodes(
            fixture.query,
            limit=6,
            exclude_ids=seen_episodes,
        )
        for episode in fallback:
            if episode.uuid in seen_episodes:
                continue
            seen_episodes.add(episode.uuid)
            parsed_episodes.append(episode)

    await _extend_with_targeted_results(engine, fixture, parsed_episodes, seen_episodes)

    artifact_memories = [memory for memory in parsed_episodes if memory.kind is MemoryKind.index_artifact]
    other_memories = [memory for memory in parsed_episodes if memory.kind is not MemoryKind.index_artifact]

    other_memories.sort(
        key=lambda memory: (
            -_rank_memory(engine, fixture, memory),
            MEMORY_KIND_PRIORITY[memory.kind],
        )
    )
    artifact_memories.sort(
        key=lambda memory: (
            -_rank_memory(engine, fixture, memory),
            memory.artifact_path or memory.raw_name,
        )
    )

    memory_limit = 1
    artifact_limit = 1
    if fixture.task_type is BenchmarkTaskType.history_recall:
        memory_limit = 2
        artifact_limit = 1
    elif fixture.task_type is BenchmarkTaskType.multi_hop_recall:
        memory_limit = 2
        artifact_limit = 2

    lines: list[str] = []
    selected_memories = other_memories[:memory_limit]
    if selected_memories:
        lines.append('Relevant Memory')
        for memory in selected_memories:
            lines.append(_compact_memory_line(engine, memory, fixture))

    preferred_artifacts = [
        artifact
        for artifact in artifact_memories
        if any(
            source.lower() in (
                f'{artifact.artifact_path} {artifact.summary} {artifact.details}'.lower()
            )
            for source in fixture.required_sources
        )
    ]
    selected_artifacts = preferred_artifacts[:artifact_limit]
    seen_artifact_keys = {
        artifact.artifact_path or artifact.raw_name or artifact.summary
        for artifact in selected_artifacts
    }
    for artifact in artifact_memories:
        key = artifact.artifact_path or artifact.raw_name or artifact.summary
        if key in seen_artifact_keys:
            continue
        selected_artifacts.append(artifact)
        seen_artifact_keys.add(key)
        if len(selected_artifacts) >= artifact_limit:
            break
    if selected_artifacts:
        if lines:
            lines.append('')
        lines.append('Supporting Artifacts')
        for artifact in selected_artifacts:
            lines.append(_compact_artifact_line(engine, artifact, fixture))

    if not lines:
        return 'No relevant memory found.'

    return _trim_to_budget(lines, fixture.max_recall_chars)


def _channel_result(
    fixture: BenchmarkTaskFixture,
    context: str,
    *,
    search_actions: int,
) -> BenchmarkChannelResult:
    answer_score, matched_answer_checks = score_checks(context, fixture.gold_answer_checks)
    evidence_score, matched_evidence_checks = score_checks(context, fixture.gold_evidence_checks)
    source_coverage = required_source_coverage(context, fixture.required_sources)
    shortcuts = shortcut_hits(context, fixture.disallowed_shortcuts)
    overall_score = overall_accuracy_score(
        fixture.task_type,
        answer_score,
        evidence_score,
        source_coverage,
        shortcuts,
    )
    return BenchmarkChannelResult(
        answer_score=answer_score,
        evidence_score=min(evidence_score, source_coverage),
        overall_score=overall_score,
        required_source_coverage=source_coverage,
        task_tokens=estimate_tokens(fixture.query) + estimate_tokens(context),
        search_actions=search_actions,
        context_chars=len(context),
        context_items=count_context_items(context),
        context=context,
        shortcut_hits=shortcuts,
        matched_answer_checks=matched_answer_checks,
        matched_evidence_checks=matched_evidence_checks,
    )


def _task_result(
    fixture: BenchmarkTaskFixture,
    control: BenchmarkChannelResult,
    treatment: BenchmarkChannelResult,
) -> BenchmarkTaskResult:
    failure_reasons: list[str] = []
    if treatment.context_chars > fixture.max_recall_chars:
        failure_reasons.append(
            f'recall context too large ({treatment.context_chars} > {fixture.max_recall_chars})'
        )
    if treatment.overall_score < control.overall_score:
        failure_reasons.append('treatment accuracy regressed versus control')
    if fixture.baseline_expectation.expect_token_reduction and treatment.task_tokens > control.task_tokens:
        failure_reasons.append('treatment token estimate regressed versus control')
    if (
        fixture.baseline_expectation.expect_search_reduction
        and treatment.search_actions > control.search_actions
    ):
        failure_reasons.append('treatment search actions regressed versus control')

    return BenchmarkTaskResult(
        task_id=fixture.task_id,
        suite=fixture.suite,
        tier=fixture.tier,
        task_type=fixture.task_type,
        difficulty=fixture.difficulty,
        notes=fixture.notes,
        control=control,
        treatment=treatment,
        delta=BenchmarkTaskDelta(
            accuracy_gain=treatment.overall_score - control.overall_score,
            evidence_gain=treatment.evidence_score - control.evidence_score,
            token_delta=control.task_tokens - treatment.task_tokens,
            token_reduction_ratio=reduction_ratio(control.task_tokens, treatment.task_tokens),
            search_delta=control.search_actions - treatment.search_actions,
            search_reduction_ratio=reduction_ratio(control.search_actions, treatment.search_actions),
            context_char_delta=control.context_chars - treatment.context_chars,
        ),
        gate_passed=not failure_reasons,
        failure_reasons=failure_reasons,
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _aggregate(results: list[BenchmarkTaskResult]) -> BenchmarkAggregateResult:
    by_type: dict[BenchmarkTaskType, list[float]] = {
        BenchmarkTaskType.artifact_recall: [],
        BenchmarkTaskType.history_recall: [],
        BenchmarkTaskType.multi_hop_recall: [],
    }
    for result in results:
        by_type[result.task_type].append(result.treatment.overall_score)

    return BenchmarkAggregateResult(
        task_count=len(results),
        accuracy_score=_mean([result.treatment.overall_score for result in results]),
        evidence_coverage=_mean([result.treatment.evidence_score for result in results]),
        token_efficiency_score=_mean(
            [max(0.0, result.delta.token_reduction_ratio) for result in results]
        ),
        search_efficiency_score=_mean(
            [max(0.0, result.delta.search_reduction_ratio) for result in results]
        ),
        artifact_accuracy=_mean(by_type[BenchmarkTaskType.artifact_recall]),
        history_accuracy=_mean(by_type[BenchmarkTaskType.history_recall]),
        multi_hop_accuracy=_mean(by_type[BenchmarkTaskType.multi_hop_recall]),
        mean_recall_chars=_mean([float(result.treatment.context_chars) for result in results]),
        mean_control_chars=_mean([float(result.control.context_chars) for result in results]),
        gate_thresholds=STATIC_GATE_THRESHOLDS,
    )


def _gate_failure_reasons(
    aggregate: BenchmarkAggregateResult,
    results: list[BenchmarkTaskResult],
    baseline: BenchmarkResult | None,
) -> list[str]:
    failure_reasons = [reason for result in results for reason in result.failure_reasons]

    if aggregate.artifact_accuracy < STATIC_GATE_THRESHOLDS['artifact_accuracy']:
        failure_reasons.append('artifact accuracy below static threshold')
    if aggregate.history_accuracy < STATIC_GATE_THRESHOLDS['history_accuracy']:
        failure_reasons.append('history accuracy below static threshold')
    if aggregate.multi_hop_accuracy < STATIC_GATE_THRESHOLDS['multi_hop_accuracy']:
        failure_reasons.append('multi-hop accuracy below static threshold')
    if aggregate.evidence_coverage < STATIC_GATE_THRESHOLDS['evidence_coverage']:
        failure_reasons.append('evidence coverage below static threshold')

    if baseline is not None:
        deltas = {
            'artifact_accuracy': aggregate.artifact_accuracy - baseline.aggregate.artifact_accuracy,
            'history_accuracy': aggregate.history_accuracy - baseline.aggregate.history_accuracy,
            'multi_hop_accuracy': aggregate.multi_hop_accuracy
            - baseline.aggregate.multi_hop_accuracy,
            'evidence_coverage': aggregate.evidence_coverage
            - baseline.aggregate.evidence_coverage,
        }
        for key, delta in deltas.items():
            if delta < BASELINE_TOLERANCE[key]:
                failure_reasons.append(f'{key} regressed beyond baseline tolerance')

    return sorted(set(failure_reasons))


async def _build_engine_and_sources(
    local_history_overlay: bool,
) -> tuple[MemoryEngine, list[BaselineSource], TemporaryDirectory]:
    temp_dir = TemporaryDirectory()
    root = Path(temp_dir.name) / 'benchmark-project'
    home = Path(temp_dir.name) / 'home'
    overlay = None
    if local_history_overlay:
        current_root = detect_project_root(Path.cwd())
        overlay = MemoryEngine.discover_history(current_root, history_days=90)

    root.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    sources = materialize_project(root)
    write_codex_history(home, root)

    with HomeOverride(home):
        MemoryEngine.init_project(root)
        engine = await MemoryEngine.open(root)
        await engine.index(changed_only=False, max_files=24)
        discovery = MemoryEngine.discover_history(root, history_days=3650)
        await engine.import_history_sessions(discovery=discovery, history_days=3650)
        for seed in MEMORY_SEEDS:
            await engine.remember(
                kind=seed.kind,
                summary=seed.summary,
                details=seed.details,
                source='agent',
                tags=list(seed.tags),
                artifact_path=seed.artifact_path,
                provenance=seeded_memory_provenance(seed),
            )

        if overlay is not None and overlay.total_sessions:
            await engine.import_history_sessions(discovery=overlay, history_days=90)

    return engine, sources, temp_dir


def _copy_artifacts_to_temp(
    project_root: Path,
    benchmark_root: Path,
    artifact_sources: list[BaselineSource],
) -> None:
    for source in artifact_sources:
        destination = benchmark_root / source.key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source.content)


def _make_contains_check(*values: str) -> BenchmarkTextCheck:
    return BenchmarkTextCheck(kind=TextCheckKind.contains, values=list(values))


def _build_dogfood_tasks(
    artifact_sources: list[BaselineSource],
    history_seeds: list[BenchmarkMemorySeed],
) -> list[BenchmarkTaskFixture]:
    tasks: list[BenchmarkTaskFixture] = []
    smoke_limit = 8

    artifacts_by_key = {source.key: source for source in artifact_sources}

    readme = artifacts_by_key.get('README.md')
    if readme is not None and 'make test' in readme.content.lower():
        tasks.append(
            BenchmarkTaskFixture(
                task_id='dogfood-artifact-tests',
                suite='dogfood_local',
                tier='smoke',
                query='How does this repo expect tests to run?',
                task_type=BenchmarkTaskType.artifact_recall,
                difficulty='easy',
                gold_answer_checks=[_make_contains_check('make test')],
                gold_evidence_checks=[_make_contains_check('Supporting Artifacts', 'README.md')],
                required_sources=['README.md'],
                max_recall_chars=1400,
                baseline_expectation=BenchmarkBaselineExpectation(),
                notes='Derived from the current repo README.',
            )
        )

    mcp_source = next(
        (
            source
            for source in artifact_sources
            if 'graphiti mcp --transport stdio' in source.content.lower()
        ),
        None,
    )
    if mcp_source is not None:
        tasks.append(
            BenchmarkTaskFixture(
                task_id='dogfood-artifact-mcp',
                suite='dogfood_local',
                tier='smoke',
                query='What is the Graphiti Codex MCP command for this repo?',
                task_type=BenchmarkTaskType.artifact_recall,
                difficulty='easy',
                gold_answer_checks=[_make_contains_check('graphiti mcp --transport stdio')],
                gold_evidence_checks=[_make_contains_check(mcp_source.key)],
                required_sources=[mcp_source.key],
                max_recall_chars=1400,
                baseline_expectation=BenchmarkBaselineExpectation(),
                notes='Derived from current repo onboarding docs.',
            )
        )

    pyproject_source = artifacts_by_key.get('pyproject.toml')
    if pyproject_source is not None and 'requires-python' in pyproject_source.content.lower():
        requires_line = next(
            (
                line.strip()
                for line in pyproject_source.content.splitlines()
                if 'requires-python' in line.lower()
            ),
            '',
        )
        if requires_line:
            tasks.append(
                BenchmarkTaskFixture(
                    task_id='dogfood-artifact-python',
                    suite='dogfood_local',
                    tier='smoke',
                    query='What Python version does this project require?',
                    task_type=BenchmarkTaskType.artifact_recall,
                    difficulty='easy',
                    gold_answer_checks=[_make_contains_check(requires_line.split('=', 1)[1].strip())],
                    gold_evidence_checks=[_make_contains_check('pyproject.toml')],
                    required_sources=['pyproject.toml'],
                    max_recall_chars=1200,
                    baseline_expectation=BenchmarkBaselineExpectation(),
                    notes='Derived from current repo pyproject metadata.',
                )
            )

    for index, seed in enumerate(history_seeds[:4], start=1):
        tasks.append(
            BenchmarkTaskFixture(
                task_id=f'dogfood-history-{index}',
                suite='dogfood_local',
                tier='smoke' if index <= 3 else 'full',
                query=seed.thread_title,
                task_type=BenchmarkTaskType.history_recall,
                difficulty='medium',
                gold_answer_checks=[_make_contains_check(seed.summary)],
                gold_evidence_checks=[
                    _make_contains_check(f'agent={seed.source_agent}', f'thread={seed.thread_title[:80]}')
                ],
                required_sources=[seed.thread_title[:80]],
                max_recall_chars=1500,
                baseline_expectation=BenchmarkBaselineExpectation(),
                notes='Derived from local Codex history.',
            )
        )

    if tasks and history_seeds:
        first_history = history_seeds[0]
        preferred_artifact = next(
            (
                source
                for source in artifact_sources
                if 'graphiti mcp --transport stdio' in source.content.lower()
                or 'make benchmark-memory' in source.content.lower()
            ),
            None,
        )
        if preferred_artifact is None:
            preferred_artifact = next(
                (
                    source
                    for source in artifact_sources
                    if 'make test' in source.content.lower() and source.key != 'AGENTS.md'
                ),
                artifact_sources[0] if artifact_sources else None,
            )
        first_artifact = preferred_artifact
        if first_artifact is not None:
            artifact_answer = ''
            for line in first_artifact.content.splitlines():
                if (
                    'graphiti mcp --transport stdio' in line.lower()
                    or 'make benchmark-memory' in line.lower()
                    or 'make test' in line.lower()
                ):
                    artifact_answer = line.strip()
                    break
            if artifact_answer:
                tasks.append(
                    BenchmarkTaskFixture(
                        task_id='dogfood-multihop-1',
                        suite='dogfood_local',
                        tier='smoke',
                        query=f'{first_history.thread_title} and {artifact_answer}',
                        task_type=BenchmarkTaskType.multi_hop_recall,
                        difficulty='hard',
                        gold_answer_checks=[
                            _make_contains_check(first_history.summary, artifact_answer)
                        ],
                        gold_evidence_checks=[
                            _make_contains_check(
                                first_artifact.key,
                                f'thread={first_history.thread_title[:80]}',
                            )
                        ],
                        required_sources=[first_artifact.key, first_history.thread_title[:80]],
                        max_recall_chars=1700,
                        baseline_expectation=BenchmarkBaselineExpectation(),
                        notes='Combines local history with current repo artifact guidance.',
                    )
                )

    full_tasks = list(tasks)
    if len(full_tasks) < 10:
        for source in artifact_sources:
            if len(full_tasks) >= 10:
                break
            if source.key in {'README.md', 'pyproject.toml'}:
                continue
            snippet = ''
            for line in source.content.splitlines():
                normalized = line.lower()
                if 'graphiti' in normalized or 'pytest' in normalized or 'make ' in normalized:
                    snippet = line.strip()
                    break
            if snippet:
                full_tasks.append(
                    BenchmarkTaskFixture(
                        task_id=f'dogfood-artifact-extra-{len(full_tasks) + 1}',
                        suite='dogfood_local',
                        tier='full',
                        query=f'What guidance is recorded in {source.key}?',
                        task_type=BenchmarkTaskType.artifact_recall,
                        difficulty='medium',
                        gold_answer_checks=[_make_contains_check(snippet)],
                        gold_evidence_checks=[_make_contains_check(source.key)],
                        required_sources=[source.key],
                        max_recall_chars=1500,
                        baseline_expectation=BenchmarkBaselineExpectation(),
                        notes='Extra dogfood artifact task.',
                    )
                )

    smoke_tasks = [task for task in tasks if task.tier == 'smoke'][:smoke_limit]
    return smoke_tasks + [task for task in full_tasks if task.tier == 'full']


async def _build_dogfood_engine_and_sources() -> tuple[
    MemoryEngine,
    list[BaselineSource],
    list[BenchmarkMemorySeed],
    list[BenchmarkTaskFixture],
    TemporaryDirectory,
]:
    temp_dir = TemporaryDirectory()
    benchmark_root = Path(temp_dir.name) / 'dogfood-project'
    benchmark_root.mkdir(parents=True, exist_ok=True)

    current_root = detect_project_root(Path.cwd())
    artifact_sources = collect_dogfood_artifacts(current_root)
    if not artifact_sources:
        raise RuntimeError('dogfood mode requires at least one high-signal artifact in the current repo')

    discovery = MemoryEngine.discover_history(current_root, history_days=3650)
    history_sources = build_dogfood_history_sources(discovery)
    history_seeds = distill_history_seeds(discovery)
    dogfood_tasks = _build_dogfood_tasks(artifact_sources, history_seeds)
    if len([task for task in dogfood_tasks if task.tier == 'smoke']) < 4:
        raise RuntimeError(
            'dogfood mode requires at least four generated smoke tasks; add more local history or repo docs'
        )

    _copy_artifacts_to_temp(current_root, benchmark_root, artifact_sources)
    MemoryEngine.init_project(benchmark_root)
    engine = await MemoryEngine.open(benchmark_root)
    try:
        await engine.index(changed_only=False, max_files=max(24, len(artifact_sources)))
        if discovery.total_sessions:
            await engine.import_history_sessions(discovery=discovery, history_days=3650)
        for seed in history_seeds:
            await engine.remember(
                kind=seed.kind,
                summary=seed.summary,
                details=seed.details,
                source='agent',
                tags=list(seed.tags),
                provenance=seeded_memory_provenance(seed),
            )
    except Exception:
        await engine.close()
        temp_dir.cleanup()
        raise

    baseline_sources = [*artifact_sources, *history_sources]
    for seed in history_seeds:
        baseline_sources.append(
            BaselineSource(
                key=f'memory:{seed.kind.value}:{seed.summary}',
                source_type='memory',
                content=f'{seed.summary}\n{seed.details}\n{seed.thread_title}',
            )
        )
    return engine, baseline_sources, history_seeds, dogfood_tasks, temp_dir


async def run_benchmark(
    suite: str,
    tier: str,
    *,
    baseline_result_path: Path | None = None,
    local_history_overlay: bool = False,
    mode: str = 'synthetic',
) -> BenchmarkResult:
    effective_suite = 'dogfood_local' if mode == 'dogfood' else suite
    if mode == 'dogfood':
        engine, sources, _history_seeds, dogfood_tasks, temp_dir = await _build_dogfood_engine_and_sources()
        fixtures = [
            task
            for task in dogfood_tasks
            if task.tier == 'smoke' or tier == 'full'
        ]
    else:
        fixtures = get_suite_tasks(suite, tier)
    baseline = (
        BenchmarkResult.model_validate(json.loads(baseline_result_path.read_text()))
        if baseline_result_path is not None
        else None
    )
    if baseline is not None and (baseline.suite != effective_suite or baseline.tier != tier):
        raise ValueError(
            'baseline result suite/tier does not match requested benchmark run'
        )

    if mode != 'dogfood':
        engine, sources, temp_dir = await _build_engine_and_sources(local_history_overlay)
    try:
        task_results: list[BenchmarkTaskResult] = []
        for fixture in fixtures:
            if fixture.judge_mode.value != 'deterministic':
                raise RuntimeError(
                    f'task {fixture.task_id} requires a judge model and cannot run in deterministic mode'
                )

            control_context, control_search_actions = _naive_control_context(fixture.query, sources)
            treatment_context = await _compact_treatment_context(engine, fixture)
            control = _channel_result(
                fixture,
                control_context,
                search_actions=control_search_actions,
            )
            treatment = _channel_result(
                fixture,
                treatment_context,
                search_actions=1,
            )
            task_results.append(_task_result(fixture, control, treatment))

        aggregate = _aggregate(task_results)
        failure_reasons = _gate_failure_reasons(aggregate, task_results, baseline)
        gate_passed = not failure_reasons
        reward = None
        if gate_passed:
            reward = round(
                (0.60 * aggregate.accuracy_score)
                + (0.25 * aggregate.token_efficiency_score)
                + (0.15 * aggregate.search_efficiency_score),
                6,
            )

        return BenchmarkResult(
            suite=effective_suite,
            tier=tier,
            config={
                'mode': mode,
                'local_history_overlay': local_history_overlay,
                'token_source': 'estimated',
                'search_proxy': 'deterministic_source_scan',
                'baseline_result_path': str(baseline_result_path) if baseline_result_path else '',
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
            tasks=task_results,
            aggregate=aggregate,
            gate_passed=gate_passed,
            reward=reward,
            failure_reasons=failure_reasons,
        )
    finally:
        await engine.close()
        temp_dir.cleanup()


def compare_results(baseline: BenchmarkResult, candidate: BenchmarkResult) -> BenchmarkComparison:
    regressed_tasks = [
        candidate_task.task_id
        for candidate_task in candidate.tasks
        for baseline_task in baseline.tasks
        if candidate_task.task_id == baseline_task.task_id
        and candidate_task.treatment.overall_score < baseline_task.treatment.overall_score
    ]
    reward_delta = None
    if baseline.reward is not None and candidate.reward is not None:
        reward_delta = candidate.reward - baseline.reward

    return BenchmarkComparison(
        baseline_gate_passed=baseline.gate_passed,
        candidate_gate_passed=candidate.gate_passed,
        reward_delta=reward_delta,
        aggregate_deltas={
            'accuracy_score': candidate.aggregate.accuracy_score - baseline.aggregate.accuracy_score,
            'evidence_coverage': candidate.aggregate.evidence_coverage
            - baseline.aggregate.evidence_coverage,
            'token_efficiency_score': candidate.aggregate.token_efficiency_score
            - baseline.aggregate.token_efficiency_score,
            'search_efficiency_score': candidate.aggregate.search_efficiency_score
            - baseline.aggregate.search_efficiency_score,
        },
        regressed_tasks=regressed_tasks,
    )


def inspect_task(task_id: str) -> dict[str, object]:
    task = get_fixture(task_id)
    return task.model_dump(mode='json')


def list_suites() -> dict[str, dict[str, int]]:
    return list_fixture_catalog().model_dump(mode='json')['suites']


def benchmark_doctor() -> BenchmarkDoctorResult:
    issues: list[str] = []
    python_supported = True
    if sys.version_info[:2] < (3, 10):
        python_supported = False
        issues.append('Python 3.10 or newer is required for benchmark execution.')

    fixtures_valid = True
    try:
        suites = list_fixture_catalog().model_dump(mode='json')['suites']
    except Exception as exc:
        fixtures_valid = False
        suites = {}
        issues.append(f'fixture validation failed: {exc}')

    state_db = discover_codex_state_db()
    local_history_sessions = 0
    try:
        current_root = detect_project_root(Path.cwd())
        local_history_sessions = MemoryEngine.discover_history(current_root, history_days=90).total_sessions
        if not collect_dogfood_artifacts(current_root):
            issues.append('dogfood mode has no high-signal repo artifacts to benchmark')
    except Exception as exc:
        issues.append(f'local history detection failed: {exc}')

    return BenchmarkDoctorResult(
        python_version=f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}',
        python_supported=python_supported,
        fixtures_valid=fixtures_valid,
        suites=suites,
        codex_state_db=str(state_db) if state_db is not None else '',
        local_history_sessions_detected=local_history_sessions,
        issues=issues,
    )
