from __future__ import annotations

import math
import re

from .models import (
    BenchmarkBudget,
    BenchmarkFactMatch,
    BenchmarkGoldFact,
    BenchmarkSupportSet,
    BenchmarkTextCheck,
)


def normalize_text(text: str) -> str:
    return ' '.join(text.lower().split())


def estimate_tokens(text: str) -> int:
    if not text.strip():
        return 0
    return max(1, math.ceil(len(text) / 4))


def count_context_items(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.startswith('- '))


def tokenize_query(text: str) -> set[str]:
    return {token for token in re.findall(r'[a-z0-9_]+', text.lower()) if len(token) >= 3}


def _fact_match(
    text: str,
    values: list[str],
    *,
    match: BenchmarkFactMatch,
    case_sensitive: bool,
) -> bool:
    haystack = text if case_sensitive else normalize_text(text)
    normalized_values = values if case_sensitive else [normalize_text(value) for value in values]
    if match is BenchmarkFactMatch.exact:
        return any(haystack == value for value in normalized_values)
    if match is BenchmarkFactMatch.any_contains:
        return any(value in haystack for value in normalized_values)
    return all(value in haystack for value in normalized_values)


def score_checks(text: str, checks: list[BenchmarkTextCheck]) -> tuple[float, int]:
    facts: list[BenchmarkGoldFact] = []
    for index, check in enumerate(checks, start=1):
        if check.kind.value == 'absent':
            continue
        match = BenchmarkFactMatch.all_contains
        if check.kind.value == 'any_contains':
            match = BenchmarkFactMatch.any_contains
        elif check.kind.value == 'exact':
            match = BenchmarkFactMatch.exact
        facts.append(
            BenchmarkGoldFact(
                key=f'legacy_check_{index}',
                values=check.values,
                match=match,
                case_sensitive=check.case_sensitive,
            )
        )
    return score_facts(text, facts)


def score_facts(text: str, facts: list[BenchmarkGoldFact]) -> tuple[float, int]:
    if not facts:
        return 1.0, 0

    total_weight = sum(max(fact.weight, 0.0) for fact in facts)
    if total_weight <= 0:
        return 1.0, len(facts)

    matched = 0
    matched_weight = 0.0
    for fact in facts:
        if _fact_match(
            text,
            fact.values,
            match=fact.match,
            case_sensitive=fact.case_sensitive,
        ):
            matched += 1
            matched_weight += max(fact.weight, 0.0)

    return matched_weight / total_weight, matched


def score_retrieval(candidate_ids: list[str], acceptable_support_sets: list[BenchmarkSupportSet]) -> float:
    if not acceptable_support_sets:
        return 1.0

    ordered_ids = list(dict.fromkeys(candidate_ids))
    best_score = 0.0
    for support_set in acceptable_support_sets:
        expected = [source_id for source_id in dict.fromkeys(support_set.source_ids) if source_id]
        if not expected:
            continue

        matched = [source_id for source_id in expected if source_id in ordered_ids]
        support_recall = len(matched) / len(expected)
        if not matched:
            continue

        actual_rr_sum = sum(1 / (ordered_ids.index(source_id) + 1) for source_id in matched)
        ideal_rr_sum = sum(1 / (index + 1) for index in range(len(matched)))
        rank_quality = actual_rr_sum / ideal_rr_sum if ideal_rr_sum else 0.0
        best_score = max(best_score, min(1.0, (0.7 * support_recall) + (0.3 * rank_quality)))

    return best_score


def score_attribution(
    provenance_ids: list[str],
    acceptable_support_sets: list[BenchmarkSupportSet],
    distractor_source_ids: list[str],
) -> float:
    selected = set(provenance_ids)
    if distractor_source_ids and any(source_id in selected for source_id in distractor_source_ids):
        return 0.0
    if not acceptable_support_sets:
        return 1.0

    best_score = 0.0
    for support_set in acceptable_support_sets:
        expected = [source_id for source_id in dict.fromkeys(support_set.source_ids) if source_id]
        if not expected:
            continue
        matched = sum(1 for source_id in expected if source_id in selected)
        best_score = max(best_score, matched / len(expected))
    return best_score


def budget_overruns(
    budget: BenchmarkBudget,
    *,
    retrieval_calls: int,
    context_chars: int,
    selected_item_count: int,
) -> list[str]:
    failures: list[str] = []
    if retrieval_calls > budget.max_retrieval_calls:
        failures.append('budget_overrun:retrieval_calls')
    if context_chars > budget.max_returned_context_chars:
        failures.append('budget_overrun:returned_context_chars')
    if budget.max_selected_items > 0 and selected_item_count > budget.max_selected_items:
        failures.append('budget_overrun:selected_items')
    return failures


def reduction_ratio(control_value: int, treatment_value: int) -> float:
    if control_value <= 0:
        return 0.0
    return max(-1.0, min(1.0, (control_value - treatment_value) / control_value))
