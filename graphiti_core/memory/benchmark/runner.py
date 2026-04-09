from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from graphiti_core.memory.config import detect_project_root
from graphiti_core.memory.engine import MemoryEngine
from graphiti_core.memory.models import MEMORY_KIND_PRIORITY, MemoryKind, ParsedMemoryEpisode

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
    BenchmarkBaselineExpectation,
    BenchmarkBudget,
    BenchmarkChannelResult,
    BenchmarkComparison,
    BenchmarkDoctorResult,
    BenchmarkFactMatch,
    BenchmarkGoldFact,
    BenchmarkHardFailRule,
    BenchmarkResult,
    BenchmarkRetrievalTrace,
    BenchmarkSupportSet,
    BenchmarkTaskDelta,
    BenchmarkTaskFixture,
    BenchmarkTaskResult,
    BenchmarkTaskType,
    artifact_source_id,
    coerce_source_id,
    memory_source_id,
    session_source_id,
    thread_source_id,
)
from .scoring import (
    budget_overruns,
    count_context_items,
    estimate_tokens,
    score_attribution,
    score_facts,
    score_retrieval,
    tokenize_query,
)
from .telemetry import discover_codex_state_db

STATIC_GATE_THRESHOLDS = {
    'artifact_task_score': 75.0,
    'history_task_score': 75.0,
    'multi_hop_task_score': 75.0,
    'mean_retrieval_score': 0.75,
    'mean_attribution_score': 0.75,
    'mean_answer_score': 0.75,
}
BASELINE_TOLERANCE = {
    'mean_task_score_normalized': -0.02,
    'mean_retrieval_score': -0.02,
    'mean_attribution_score': -0.02,
    'mean_answer_score': -0.02,
}


def _normalize(text: str) -> str:
    return text.lower()


def _dedupe_preserve(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _token_overlap(text: str, query: str) -> int:
    normalized = text.lower()
    tokens = tokenize_query(query)
    score = sum(1 for token in tokens if token in normalized)
    if query.lower() in normalized:
        score += max(2, len(tokens))
    return score


def _matching_snippet(text: str, query: str, *, max_len: int = 220) -> str:
    tokens = tokenize_query(query)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text.replace('\n', ' ')[:max_len]

    best_line = lines[0]
    best_score = -1
    normalized_query = query.lower()
    for line in lines:
        normalized_line = line.lower()
        score = sum(1 for token in tokens if token in normalized_line)
        if normalized_query in normalized_line:
            score += max(2, len(tokens))
        if score > best_score:
            best_score = score
            best_line = line
    return best_line[:max_len]


def _artifact_text(engine: MemoryEngine, artifact: ParsedMemoryEpisode) -> str:
    artifact_path = artifact.artifact_path or artifact.raw_name or ''
    if artifact_path:
        candidate = engine.project.root / artifact_path
        if candidate.exists() and candidate.is_file():
            return candidate.read_text(errors='ignore')[:12_000]
    return artifact.details or artifact.summary


def _memory_primary_item_id(memory: ParsedMemoryEpisode) -> str:
    if memory.kind is MemoryKind.index_artifact and memory.artifact_path:
        return artifact_source_id(memory.artifact_path)
    return f'episode:{memory.uuid}'


def _memory_provenance_ids(memory: ParsedMemoryEpisode) -> list[str]:
    provenance_ids = [memory_source_id(memory.kind.value, memory.summary)]
    if memory.artifact_path:
        provenance_ids.append(artifact_source_id(memory.artifact_path))
    if memory.thread_title:
        provenance_ids.append(thread_source_id(memory.thread_title))
    if memory.session_id:
        provenance_ids.append(session_source_id(memory.session_id))
    return _dedupe_preserve(provenance_ids)


def _memory_candidate_ids(
    memory: ParsedMemoryEpisode,
    *,
    task_type: BenchmarkTaskType,
) -> list[str]:
    provenance_ids = _memory_provenance_ids(memory)
    preferred_ids: list[str] = []

    if task_type is BenchmarkTaskType.artifact_recall and memory.artifact_path:
        preferred_ids.append(artifact_source_id(memory.artifact_path))
    elif task_type is BenchmarkTaskType.history_recall and memory.thread_title:
        preferred_ids.append(thread_source_id(memory.thread_title))
    else:
        if memory.thread_title:
            preferred_ids.append(thread_source_id(memory.thread_title))
        if memory.artifact_path:
            preferred_ids.append(artifact_source_id(memory.artifact_path))

    preferred_ids.extend(
        provenance_id
        for provenance_id in provenance_ids
        if not provenance_id.startswith(('memory:', 'session:'))
    )
    preferred_ids.extend(
        provenance_id for provenance_id in provenance_ids if provenance_id.startswith('memory:')
    )
    preferred_ids.extend(
        provenance_id for provenance_id in provenance_ids if provenance_id.startswith('session:')
    )
    return _dedupe_preserve(preferred_ids)


def _history_trace_candidate_ids(
    selected_memories: list[ParsedMemoryEpisode],
    ranked_memories: list[ParsedMemoryEpisode],
    *,
    query: str,
) -> list[str]:
    selected_ids = {memory.uuid for memory in selected_memories}
    selected_history = sorted(
        [memory for memory in selected_memories if memory.kind is not MemoryKind.index_artifact],
        key=lambda memory: (
            _token_overlap(' '.join([memory.summary, memory.thread_title]), query),
            _token_overlap(memory.details, query),
            memory.created_at.timestamp() if memory.created_at is not None else 0.0,
        ),
        reverse=True,
    )
    remaining_history = [
        memory
        for memory in ranked_memories
        if memory.uuid not in selected_ids and memory.kind is not MemoryKind.index_artifact
    ]
    remaining_artifacts = [
        memory
        for memory in ranked_memories
        if memory.uuid not in selected_ids and memory.kind is MemoryKind.index_artifact
    ]
    ordered_memories = [*selected_history, *remaining_history, *remaining_artifacts]

    candidate_ids: list[str] = []
    for memory in ordered_memories:
        if memory.thread_title:
            candidate_ids.append(thread_source_id(memory.thread_title))
    for memory in ordered_memories:
        candidate_ids.extend(
            provenance_id
            for provenance_id in _memory_provenance_ids(memory)
            if provenance_id.startswith('memory:')
        )
    for memory in ordered_memories:
        if memory.session_id:
            candidate_ids.append(session_source_id(memory.session_id))
    for memory in remaining_artifacts:
        if memory.artifact_path:
            candidate_ids.append(artifact_source_id(memory.artifact_path))
    return _dedupe_preserve(candidate_ids)


def _multi_hop_trace_candidate_ids(
    selected_memories: list[ParsedMemoryEpisode],
    ranked_memories: list[ParsedMemoryEpisode],
) -> list[str]:
    selected_ids = {memory.uuid for memory in selected_memories}
    remaining_memories = [memory for memory in ranked_memories if memory.uuid not in selected_ids]
    candidate_ids: list[str] = []
    for prefix in ('thread:', 'artifact:', 'memory:', 'session:'):
        for memory in selected_memories:
            candidate_ids.extend(
                candidate_id
                for candidate_id in _memory_candidate_ids(
                    memory,
                    task_type=BenchmarkTaskType.multi_hop_recall,
                )
                if candidate_id.startswith(prefix)
            )
        for memory in remaining_memories:
            candidate_ids.extend(
                candidate_id
                for candidate_id in _memory_candidate_ids(
                    memory,
                    task_type=BenchmarkTaskType.multi_hop_recall,
                )
                if candidate_id.startswith(prefix)
            )
    return _dedupe_preserve(candidate_ids)


def _memory_rank(engine: MemoryEngine, memory: ParsedMemoryEpisode, query: str) -> tuple[int, int, float, str]:
    return (
        engine._memory_overlap_score(memory, query),
        -MEMORY_KIND_PRIORITY[memory.kind],
        memory.created_at.timestamp() if memory.created_at is not None else 0.0,
        _memory_primary_item_id(memory),
    )


def _default_selected_items(task_type: BenchmarkTaskType) -> int:
    if task_type is BenchmarkTaskType.artifact_recall:
        return 1
    if task_type is BenchmarkTaskType.history_recall:
        return 2
    return 4


def _selection_cap(task_type: BenchmarkTaskType, budget: BenchmarkBudget) -> int:
    if budget.max_selected_items > 0:
        return budget.max_selected_items
    return _default_selected_items(task_type)


def _select_memories(
    ranked_memories: list[ParsedMemoryEpisode],
    *,
    task_type: BenchmarkTaskType,
    budget: BenchmarkBudget,
) -> list[ParsedMemoryEpisode]:
    cap = _selection_cap(task_type, budget)
    if not ranked_memories or cap <= 0:
        return []

    artifacts = [memory for memory in ranked_memories if memory.kind is MemoryKind.index_artifact]
    non_artifacts = [memory for memory in ranked_memories if memory.kind is not MemoryKind.index_artifact]
    selected: list[ParsedMemoryEpisode] = []

    if task_type is BenchmarkTaskType.artifact_recall:
        selected.extend(artifacts[:cap] if artifacts else ranked_memories[:cap])
        return selected[:cap]

    if task_type is BenchmarkTaskType.history_recall:
        selected.extend(non_artifacts[:cap] if non_artifacts else ranked_memories[:cap])
        return selected[:cap]

    memory_cap = min(2, cap)
    artifact_cap = min(2, max(0, cap - memory_cap))
    selected.extend(non_artifacts[:memory_cap])
    selected.extend(artifacts[:artifact_cap])
    seen = {memory.uuid for memory in selected}
    if len(selected) < cap:
        for memory in ranked_memories:
            if memory.uuid in seen:
                continue
            selected.append(memory)
            seen.add(memory.uuid)
            if len(selected) >= cap:
                break
    return selected[:cap]


def _source_rank(query: str, source: BaselineSource) -> tuple[int, str]:
    tokens = tokenize_query(query)
    haystack = source.content.lower()
    score = sum(1 for token in tokens if token in haystack)
    if query.lower() in haystack:
        score += max(2, len(tokens))
    return score, source.key


def _source_item_id(source: BaselineSource) -> str:
    return f'{source.source_type}:{source.key}'


def _source_line(source: BaselineSource, query: str) -> str:
    snippet = _matching_snippet(source.content, query, max_len=180)
    provenance = ','.join(source.provenance_ids) or 'none'
    return (
        f'- [{source.source_type}] {source.key} | provenance={provenance} | snippet={snippet}'
    )


def _memory_line(engine: MemoryEngine, memory: ParsedMemoryEpisode, query: str) -> str:
    provenance = ','.join(_memory_provenance_ids(memory))
    if memory.kind is MemoryKind.index_artifact:
        snippet = _matching_snippet(_artifact_text(engine, memory), query, max_len=180)
        path = memory.artifact_path or memory.raw_name or memory.summary
        return f'- {path} | provenance={provenance} | snippet={snippet}'

    detail = _matching_snippet(memory.details or memory.summary, query, max_len=160)
    parts = [f'[{memory.kind.value}] {memory.summary}', f'provenance={provenance}']
    if memory.source_agent:
        parts.append(f'agent={memory.source_agent}')
    if memory.thread_title:
        parts.append(f'thread={memory.thread_title[:80]}')
    if detail and detail not in memory.summary:
        parts.append(f'detail={detail}')
    return '- ' + ' | '.join(parts)


def _fit_sections_to_budget(
    sections: list[tuple[str, list[tuple[str, str, list[str]]]]],
    budget: int,
) -> tuple[str, list[str], list[str]]:
    kept_lines: list[str] = []
    selected_evidence_ids: list[str] = []
    provenance_ids: list[str] = []

    for header, entries in sections:
        section_lines: list[str] = []
        section_selected: list[str] = []
        section_provenance: list[str] = []
        for line, item_id, line_provenance in entries:
            candidate_lines = [
                *kept_lines,
                *([header] if section_lines == [] else []),
                *section_lines,
                line,
            ]
            if len('\n'.join(candidate_lines)) > budget:
                break
            if section_lines == []:
                section_lines.append(header)
            section_lines.append(line)
            section_selected.append(item_id)
            section_provenance.extend(line_provenance)
        if section_lines:
            if kept_lines:
                candidate_lines = [*kept_lines, '', *section_lines]
                if len('\n'.join(candidate_lines)) > budget:
                    break
                kept_lines.append('')
            kept_lines.extend(section_lines)
            selected_evidence_ids.extend(section_selected)
            provenance_ids.extend(section_provenance)

    if not kept_lines:
        return 'No relevant memory found.', [], []
    return '\n'.join(kept_lines), selected_evidence_ids, _dedupe_preserve(provenance_ids)


def _build_treatment_context(
    engine: MemoryEngine,
    selected_memories: list[ParsedMemoryEpisode],
    fixture: BenchmarkTaskFixture,
) -> tuple[str, list[str], list[str]]:
    non_artifacts = [memory for memory in selected_memories if memory.kind is not MemoryKind.index_artifact]
    artifacts = [memory for memory in selected_memories if memory.kind is MemoryKind.index_artifact]
    sections: list[tuple[str, list[tuple[str, str, list[str]]]]] = []

    if non_artifacts:
        sections.append(
            (
                'Relevant Memory',
                [
                    (
                        _memory_line(engine, memory, fixture.query),
                        _memory_primary_item_id(memory),
                        _memory_provenance_ids(memory),
                    )
                    for memory in non_artifacts
                ],
            )
        )
    if artifacts:
        sections.append(
            (
                'Supporting Artifacts',
                [
                    (
                        _memory_line(engine, memory, fixture.query),
                        _memory_primary_item_id(memory),
                        _memory_provenance_ids(memory),
                    )
                    for memory in artifacts
                ],
            )
        )

    return _fit_sections_to_budget(sections, fixture.budgets.max_returned_context_chars)


async def _collect_treatment_channel(
    engine: MemoryEngine,
    fixture: BenchmarkTaskFixture,
) -> tuple[str, BenchmarkRetrievalTrace]:
    search_limit = 6 if fixture.task_type is BenchmarkTaskType.artifact_recall else 8
    results = await engine._search(fixture.query, limit=search_limit)
    retrieval_calls = 1
    retrieval_queries = [fixture.query]

    parsed_memories: list[ParsedMemoryEpisode] = []
    seen_episodes: set[str] = set()
    for episode in results.episodes:
        if episode.uuid in seen_episodes:
            continue
        seen_episodes.add(episode.uuid)
        parsed = engine._parse_memory_episode(episode)
        if parsed is not None:
            parsed_memories.append(parsed)

    if len(parsed_memories) < _default_selected_items(fixture.task_type):
        fallback = await engine._fallback_memory_episodes(
            fixture.query,
            limit=max(search_limit, _default_selected_items(fixture.task_type) * 2),
            exclude_ids=seen_episodes,
        )
        retrieval_calls += 1
        retrieval_queries.append(f'fallback:{fixture.query}')
        for episode in fallback:
            if episode.uuid in seen_episodes:
                continue
            seen_episodes.add(episode.uuid)
            parsed_memories.append(episode)

    ranked_memories = sorted(
        parsed_memories,
        key=lambda memory: _memory_rank(engine, memory, fixture.query),
        reverse=True,
    )
    selected_memories = _select_memories(
        ranked_memories,
        task_type=fixture.task_type,
        budget=fixture.budgets,
    )
    context, selected_evidence_ids, provenance_ids = _build_treatment_context(
        engine,
        selected_memories,
        fixture,
    )
    if fixture.task_type is BenchmarkTaskType.history_recall:
        candidate_ids = _history_trace_candidate_ids(
            selected_memories,
            ranked_memories,
            query=fixture.query,
        )
    elif fixture.task_type is BenchmarkTaskType.multi_hop_recall:
        candidate_ids = _multi_hop_trace_candidate_ids(
            selected_memories,
            ranked_memories,
        )
    else:
        candidate_ids = _dedupe_preserve(
            [
                candidate_id
                for memory in ranked_memories
                for candidate_id in _memory_candidate_ids(memory, task_type=fixture.task_type)
            ]
        )
    trace = BenchmarkRetrievalTrace(
        retrieval_calls=retrieval_calls,
        retrieval_queries=retrieval_queries,
        candidate_ids=candidate_ids,
        selected_evidence_ids=selected_evidence_ids,
        provenance_ids=provenance_ids,
        selected_item_count=len(selected_evidence_ids),
        candidate_count=len(candidate_ids),
    )
    return context, trace


def _build_control_context(
    sources: list[BaselineSource],
    fixture: BenchmarkTaskFixture,
) -> tuple[str, BenchmarkRetrievalTrace]:
    ranked_sources = [
        source
        for score, source in sorted(
            (
                (_source_rank(fixture.query, source), source)
                for source in sources
                if _source_rank(fixture.query, source)[0] > 0
            ),
            key=lambda item: (item[0][0], item[0][1]),
            reverse=True,
        )
    ]
    if not ranked_sources:
        ranked_sources = sources[:]

    artifacts = [source for source in ranked_sources if source.source_type == 'artifact']
    non_artifacts = [source for source in ranked_sources if source.source_type != 'artifact']
    selected_sources: list[BaselineSource] = []
    cap = _selection_cap(fixture.task_type, fixture.budgets)
    if fixture.task_type is BenchmarkTaskType.artifact_recall:
        selected_sources.extend(artifacts[:cap] if artifacts else ranked_sources[:cap])
    elif fixture.task_type is BenchmarkTaskType.history_recall:
        selected_sources.extend(non_artifacts[:cap] if non_artifacts else ranked_sources[:cap])
    else:
        selected_sources.extend(non_artifacts[: min(2, cap)])
        selected_sources.extend(artifacts[: max(0, cap - len(selected_sources))])
        seen = {source.key for source in selected_sources}
        for source in ranked_sources:
            if source.key in seen:
                continue
            selected_sources.append(source)
            seen.add(source.key)
            if len(selected_sources) >= cap:
                break

    sections: list[tuple[str, list[tuple[str, str, list[str]]]]] = []
    non_artifact_entries = [
        (_source_line(source, fixture.query), _source_item_id(source), list(source.provenance_ids))
        for source in selected_sources
        if source.source_type != 'artifact'
    ]
    artifact_entries = [
        (_source_line(source, fixture.query), _source_item_id(source), list(source.provenance_ids))
        for source in selected_sources
        if source.source_type == 'artifact'
    ]
    if non_artifact_entries:
        sections.append(('Baseline Memory Scan', non_artifact_entries))
    if artifact_entries:
        sections.append(('Baseline Artifact Scan', artifact_entries))
    context, selected_evidence_ids, provenance_ids = _fit_sections_to_budget(
        sections,
        fixture.budgets.max_returned_context_chars,
    )
    trace = BenchmarkRetrievalTrace(
        retrieval_calls=1,
        retrieval_queries=[fixture.query],
        candidate_ids=_dedupe_preserve(
            [provenance_id for source in ranked_sources for provenance_id in source.provenance_ids]
        ),
        selected_evidence_ids=selected_evidence_ids,
        provenance_ids=provenance_ids,
        selected_item_count=len(selected_evidence_ids),
        candidate_count=len(ranked_sources),
    )
    return context, trace


def _channel_result(
    fixture: BenchmarkTaskFixture,
    context: str,
    trace: BenchmarkRetrievalTrace,
) -> BenchmarkChannelResult:
    answer_score, matched_fact_count = score_facts(context, fixture.gold_facts)
    retrieval_score = score_retrieval(trace.candidate_ids, fixture.acceptable_support_sets)
    attribution_score = score_attribution(
        trace.provenance_ids,
        fixture.acceptable_support_sets,
        fixture.distractor_source_ids,
    )
    overruns = budget_overruns(
        fixture.budgets,
        retrieval_calls=trace.retrieval_calls,
        context_chars=len(context),
        selected_item_count=trace.selected_item_count,
    )
    hard_failures: list[str] = []
    if BenchmarkHardFailRule.missing_provenance in fixture.hard_fail_rules and not trace.provenance_ids:
        hard_failures.append('missing_provenance')
    if BenchmarkHardFailRule.budget_overrun in fixture.hard_fail_rules and overruns:
        hard_failures.extend(overruns)
    if BenchmarkHardFailRule.wrong_support in fixture.hard_fail_rules and attribution_score < 1.0:
        hard_failures.append('wrong_support')
    if BenchmarkHardFailRule.unsupported_claim in fixture.hard_fail_rules and answer_score < 1.0:
        hard_failures.append('unsupported_claim')

    efficiency_score = 1.0 if not overruns else 0.0
    capability_score = retrieval_score * attribution_score * answer_score
    if hard_failures:
        capability_score = 0.0
    task_score = round(100.0 * capability_score * efficiency_score, 6)
    normalized_task_score = task_score / 100.0
    return BenchmarkChannelResult(
        retrieval_score=retrieval_score,
        attribution_score=attribution_score,
        answer_score=answer_score,
        capability_score=capability_score,
        efficiency_score=efficiency_score,
        task_score=task_score,
        normalized_task_score=normalized_task_score,
        task_tokens=estimate_tokens(fixture.query) + estimate_tokens(context),
        retrieval_calls=trace.retrieval_calls,
        context_chars=len(context),
        context_items=count_context_items(context),
        context=context,
        retrieval_trace=trace,
        selected_evidence_ids=trace.selected_evidence_ids,
        provenance_ids=trace.provenance_ids,
        hard_failures=_dedupe_preserve(hard_failures),
        matched_fact_count=matched_fact_count,
        budget_passed=not overruns,
    )


def _task_failure_reasons(
    fixture: BenchmarkTaskFixture,
    treatment: BenchmarkChannelResult,
) -> list[str]:
    expectation: BenchmarkBaselineExpectation = fixture.baseline_expectation
    failure_reasons = list(treatment.hard_failures)
    if treatment.retrieval_score < expectation.minimum_retrieval_score:
        failure_reasons.append('retrieval score below task minimum')
    if treatment.attribution_score < expectation.minimum_evidence_coverage:
        failure_reasons.append('attribution score below task minimum')
    if treatment.answer_score < expectation.minimum_answer_score:
        failure_reasons.append('answer score below task minimum')
    if treatment.normalized_task_score < expectation.minimum_task_score:
        failure_reasons.append('task score below task minimum')
    return _dedupe_preserve(failure_reasons)


def _task_result(
    fixture: BenchmarkTaskFixture,
    control: BenchmarkChannelResult,
    treatment: BenchmarkChannelResult,
) -> BenchmarkTaskResult:
    failure_reasons = _task_failure_reasons(fixture, treatment)
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
            retrieval_gain=treatment.retrieval_score - control.retrieval_score,
            attribution_gain=treatment.attribution_score - control.attribution_score,
            answer_gain=treatment.answer_score - control.answer_score,
            capability_gain=treatment.capability_score - control.capability_score,
            task_score_gain=treatment.task_score - control.task_score,
            retrieval_call_delta=control.retrieval_calls - treatment.retrieval_calls,
            context_char_delta=control.context_chars - treatment.context_chars,
        ),
        gate_passed=not failure_reasons,
        failure_reasons=failure_reasons,
    )


def _aggregate(results: list[BenchmarkTaskResult]) -> BenchmarkAggregateResult:
    by_type: dict[BenchmarkTaskType, list[float]] = {
        BenchmarkTaskType.artifact_recall: [],
        BenchmarkTaskType.history_recall: [],
        BenchmarkTaskType.multi_hop_recall: [],
    }
    budget_failure_count = 0
    provenance_failure_count = 0
    support_failure_count = 0
    unsupported_claim_failure_count = 0
    for result in results:
        by_type[result.task_type].append(result.treatment.task_score)
        for failure in result.treatment.hard_failures:
            if failure.startswith('budget_overrun'):
                budget_failure_count += 1
            elif failure == 'missing_provenance':
                provenance_failure_count += 1
            elif failure == 'wrong_support':
                support_failure_count += 1
            elif failure == 'unsupported_claim':
                unsupported_claim_failure_count += 1

    mean_task_score = _mean([result.treatment.task_score for result in results])
    return BenchmarkAggregateResult(
        task_count=len(results),
        mean_task_score=mean_task_score,
        mean_task_score_normalized=mean_task_score / 100.0,
        mean_retrieval_score=_mean([result.treatment.retrieval_score for result in results]),
        mean_attribution_score=_mean([result.treatment.attribution_score for result in results]),
        mean_answer_score=_mean([result.treatment.answer_score for result in results]),
        mean_capability_score=_mean([result.treatment.capability_score for result in results]),
        mean_efficiency_score=_mean([result.treatment.efficiency_score for result in results]),
        artifact_task_score=_mean(by_type[BenchmarkTaskType.artifact_recall]),
        history_task_score=_mean(by_type[BenchmarkTaskType.history_recall]),
        multi_hop_task_score=_mean(by_type[BenchmarkTaskType.multi_hop_recall]),
        budget_failure_count=budget_failure_count,
        provenance_failure_count=provenance_failure_count,
        support_failure_count=support_failure_count,
        unsupported_claim_failure_count=unsupported_claim_failure_count,
        mean_returned_context_chars=_mean(
            [float(result.treatment.context_chars) for result in results]
        ),
        mean_control_context_chars=_mean([float(result.control.context_chars) for result in results]),
        gate_thresholds=STATIC_GATE_THRESHOLDS,
    )


def _gate_failure_reasons(
    aggregate: BenchmarkAggregateResult,
    results: list[BenchmarkTaskResult],
    baseline: BenchmarkResult | None,
) -> list[str]:
    failure_reasons = [reason for result in results for reason in result.failure_reasons]
    if aggregate.artifact_task_score < STATIC_GATE_THRESHOLDS['artifact_task_score']:
        failure_reasons.append('artifact task score below static threshold')
    if aggregate.history_task_score < STATIC_GATE_THRESHOLDS['history_task_score']:
        failure_reasons.append('history task score below static threshold')
    if aggregate.multi_hop_task_score < STATIC_GATE_THRESHOLDS['multi_hop_task_score']:
        failure_reasons.append('multi-hop task score below static threshold')
    if aggregate.mean_retrieval_score < STATIC_GATE_THRESHOLDS['mean_retrieval_score']:
        failure_reasons.append('mean retrieval score below static threshold')
    if aggregate.mean_attribution_score < STATIC_GATE_THRESHOLDS['mean_attribution_score']:
        failure_reasons.append('mean attribution score below static threshold')
    if aggregate.mean_answer_score < STATIC_GATE_THRESHOLDS['mean_answer_score']:
        failure_reasons.append('mean answer score below static threshold')

    if baseline is not None:
        deltas = {
            'mean_task_score_normalized': aggregate.mean_task_score_normalized
            - baseline.aggregate.mean_task_score_normalized,
            'mean_retrieval_score': aggregate.mean_retrieval_score
            - baseline.aggregate.mean_retrieval_score,
            'mean_attribution_score': aggregate.mean_attribution_score
            - baseline.aggregate.mean_attribution_score,
            'mean_answer_score': aggregate.mean_answer_score - baseline.aggregate.mean_answer_score,
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


def _default_budget(max_chars: int, task_type: BenchmarkTaskType) -> BenchmarkBudget:
    return BenchmarkBudget(
        max_retrieval_calls=2,
        max_returned_context_chars=max_chars,
        max_selected_items=_default_selected_items(task_type),
    )


def _default_hard_fail_rules() -> list[BenchmarkHardFailRule]:
    return [
        BenchmarkHardFailRule.missing_provenance,
        BenchmarkHardFailRule.budget_overrun,
    ]


def _all_fact(key: str, *values: str) -> BenchmarkGoldFact:
    return BenchmarkGoldFact(key=key, values=list(values), match=BenchmarkFactMatch.all_contains)


def _any_fact(key: str, *values: str) -> BenchmarkGoldFact:
    return BenchmarkGoldFact(key=key, values=list(values), match=BenchmarkFactMatch.any_contains)


def _support(*references: str) -> list[BenchmarkSupportSet]:
    return [BenchmarkSupportSet(source_ids=[coerce_source_id(reference) for reference in references])]


def _build_dogfood_fixture(
    *,
    task_id: str,
    tier: str,
    query: str,
    task_type: BenchmarkTaskType,
    difficulty: str,
    facts: list[BenchmarkGoldFact],
    support_refs: list[str],
    max_chars: int,
    notes: str,
    distractor_refs: list[str] | None = None,
    baseline_expectation: BenchmarkBaselineExpectation | None = None,
) -> BenchmarkTaskFixture:
    return BenchmarkTaskFixture(
        task_id=task_id,
        suite='dogfood_local',
        tier=tier,
        query=query,
        task_type=task_type,
        difficulty=difficulty,
        gold_facts=facts,
        acceptable_support_sets=_support(*support_refs),
        distractor_source_ids=[coerce_source_id(reference) for reference in (distractor_refs or [])],
        budgets=_default_budget(max_chars, task_type),
        hard_fail_rules=_default_hard_fail_rules(),
        baseline_expectation=baseline_expectation or BenchmarkBaselineExpectation(),
        notes=notes,
    )


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
            _build_dogfood_fixture(
                task_id='dogfood-artifact-tests',
                tier='smoke',
                query='How does this repo expect tests to run?',
                task_type=BenchmarkTaskType.artifact_recall,
                difficulty='easy',
                facts=[_all_fact('test_command', 'make test')],
                support_refs=['README.md'],
                max_chars=1400,
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
            _build_dogfood_fixture(
                task_id='dogfood-artifact-mcp',
                tier='smoke',
                query='What is the Graphiti Codex MCP command for this repo?',
                task_type=BenchmarkTaskType.artifact_recall,
                difficulty='easy',
                facts=[_all_fact('mcp_command', 'graphiti mcp --transport stdio')],
                support_refs=[mcp_source.key],
                max_chars=1400,
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
                _build_dogfood_fixture(
                    task_id='dogfood-artifact-python',
                    tier='smoke',
                    query='What Python version does this project require?',
                    task_type=BenchmarkTaskType.artifact_recall,
                    difficulty='easy',
                    facts=[_all_fact('python_version', requires_line.split('=', 1)[1].strip())],
                    support_refs=['pyproject.toml'],
                    max_chars=1200,
                    notes='Derived from current repo pyproject metadata.',
                )
            )

    for index, seed in enumerate(history_seeds[:4], start=1):
        tasks.append(
            _build_dogfood_fixture(
                task_id=f'dogfood-history-{index}',
                tier='smoke' if index <= 3 else 'full',
                query=seed.thread_title,
                task_type=BenchmarkTaskType.history_recall,
                difficulty='medium',
                facts=[_all_fact(f'history_{index}', seed.summary)],
                support_refs=[seed.thread_title[:80]],
                max_chars=1500,
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
        if preferred_artifact is not None:
            artifact_answer = ''
            for line in preferred_artifact.content.splitlines():
                if (
                    'graphiti mcp --transport stdio' in line.lower()
                    or 'make benchmark-memory' in line.lower()
                    or 'make test' in line.lower()
                ):
                    artifact_answer = line.strip()
                    break
            if artifact_answer:
                tasks.append(
                    _build_dogfood_fixture(
                        task_id='dogfood-multihop-1',
                        tier='smoke',
                        query=f'{first_history.thread_title} and {artifact_answer}',
                        task_type=BenchmarkTaskType.multi_hop_recall,
                        difficulty='hard',
                        facts=[
                            _all_fact('history_summary', first_history.summary),
                            _all_fact('artifact_answer', artifact_answer),
                        ],
                        support_refs=[preferred_artifact.key, first_history.thread_title[:80]],
                        max_chars=1700,
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
                    _build_dogfood_fixture(
                        task_id=f'dogfood-artifact-extra-{len(full_tasks) + 1}',
                        tier='full',
                        query=f'What guidance is recorded in {source.key}?',
                        task_type=BenchmarkTaskType.artifact_recall,
                        difficulty='medium',
                        facts=[_all_fact(f'guidance_{len(full_tasks) + 1}', snippet)],
                        support_refs=[source.key],
                        max_chars=1500,
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
        provenance_ids = [
            memory_source_id(seed.kind.value, seed.summary),
            thread_source_id(seed.thread_title),
            session_source_id(seed.session_id),
        ]
        baseline_sources.append(
            BaselineSource(
                key=f'memory:{seed.kind.value}:{seed.summary}',
                source_type='memory',
                content=f'{seed.summary}\n{seed.details}\n{seed.thread_title}',
                provenance_ids=tuple(_dedupe_preserve(provenance_ids)),
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
        fixtures = [task for task in dogfood_tasks if task.tier == 'smoke' or tier == 'full']
    else:
        fixtures = get_suite_tasks(suite, tier)
        engine, sources, temp_dir = await _build_engine_and_sources(local_history_overlay)

    baseline = (
        BenchmarkResult.model_validate(json.loads(baseline_result_path.read_text()))
        if baseline_result_path is not None
        else None
    )
    if baseline is not None and (baseline.suite != effective_suite or baseline.tier != tier):
        raise ValueError('baseline result suite/tier does not match requested benchmark run')

    try:
        task_results: list[BenchmarkTaskResult] = []
        for fixture in fixtures:
            if fixture.judge_mode.value != 'deterministic':
                raise RuntimeError(
                    f'task {fixture.task_id} requires a judge model and cannot run in deterministic mode'
                )

            control_context, control_trace = _build_control_context(sources, fixture)
            treatment_context, treatment_trace = await _collect_treatment_channel(engine, fixture)
            control = _channel_result(fixture, control_context, control_trace)
            treatment = _channel_result(fixture, treatment_context, treatment_trace)
            task_results.append(_task_result(fixture, control, treatment))

        aggregate = _aggregate(task_results)
        failure_reasons = _gate_failure_reasons(aggregate, task_results, baseline)
        gate_passed = not failure_reasons
        reward = round(aggregate.mean_task_score_normalized, 6) if gate_passed else None
        return BenchmarkResult(
            suite=effective_suite,
            tier=tier,
            config={
                'mode': mode,
                'local_history_overlay': local_history_overlay,
                'reward_contract': 'mean_normalized_task_score',
                'hard_budgets': ['retrieval_calls', 'returned_context_chars'],
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
        and candidate_task.treatment.task_score < baseline_task.treatment.task_score
    ]
    reward_delta = None
    if baseline.reward is not None and candidate.reward is not None:
        reward_delta = candidate.reward - baseline.reward

    return BenchmarkComparison(
        baseline_gate_passed=baseline.gate_passed,
        candidate_gate_passed=candidate.gate_passed,
        reward_delta=reward_delta,
        aggregate_deltas={
            'mean_task_score': candidate.aggregate.mean_task_score - baseline.aggregate.mean_task_score,
            'mean_retrieval_score': candidate.aggregate.mean_retrieval_score
            - baseline.aggregate.mean_retrieval_score,
            'mean_attribution_score': candidate.aggregate.mean_attribution_score
            - baseline.aggregate.mean_attribution_score,
            'mean_answer_score': candidate.aggregate.mean_answer_score
            - baseline.aggregate.mean_answer_score,
            'mean_capability_score': candidate.aggregate.mean_capability_score
            - baseline.aggregate.mean_capability_score,
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
