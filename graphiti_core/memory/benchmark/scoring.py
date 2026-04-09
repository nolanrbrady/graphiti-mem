from __future__ import annotations

import math
import re

from .models import BenchmarkTextCheck, BenchmarkTaskType


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


def score_checks(text: str, checks: list[BenchmarkTextCheck]) -> tuple[float, int]:
    if not checks:
        return 1.0, 0

    matched = 0
    for check in checks:
        haystack = text if check.case_sensitive else normalize_text(text)
        values = check.values if check.case_sensitive else [normalize_text(value) for value in check.values]
        if check.kind.value == 'exact':
            if any(haystack == value for value in values):
                matched += 1
        elif check.kind.value == 'contains':
            if all(value in haystack for value in values):
                matched += 1
        elif check.kind.value == 'any_contains':
            if any(value in haystack for value in values):
                matched += 1
        elif check.kind.value == 'absent':
            if all(value not in haystack for value in values):
                matched += 1
    return matched / len(checks), matched


def required_source_coverage(text: str, required_sources: list[str]) -> float:
    if not required_sources:
        return 1.0
    haystack = normalize_text(text)
    matched = sum(1 for source in required_sources if normalize_text(source) in haystack)
    return matched / len(required_sources)


def shortcut_hits(text: str, shortcuts: list[str]) -> list[str]:
    haystack = normalize_text(text)
    return [shortcut for shortcut in shortcuts if normalize_text(shortcut) in haystack]


def overall_accuracy_score(
    task_type: BenchmarkTaskType,
    answer_score: float,
    evidence_score: float,
    source_coverage: float,
    shortcut_matches: list[str],
) -> float:
    adjusted_evidence = min(evidence_score, source_coverage)
    if shortcut_matches and source_coverage < 1.0:
        adjusted_evidence = 0.0

    if task_type is BenchmarkTaskType.multi_hop_recall:
        return min(answer_score, adjusted_evidence)
    return (0.7 * answer_score) + (0.3 * adjusted_evidence)


def reduction_ratio(control_value: int, treatment_value: int) -> float:
    if control_value <= 0:
        return 0.0
    return max(-1.0, min(1.0, (control_value - treatment_value) / control_value))
