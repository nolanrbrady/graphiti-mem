from __future__ import annotations

import json
from importlib import resources

from .models import (
    BenchmarkBaselineExpectation,
    BenchmarkBudget,
    BenchmarkFactMatch,
    BenchmarkGoldFact,
    BenchmarkHardFailRule,
    BenchmarkSuiteCatalog,
    BenchmarkSupportSet,
    BenchmarkTaskFixture,
    BenchmarkTaskType,
    TextCheckKind,
    coerce_source_id,
)


def _fixture_resource_names() -> list[str]:
    fixtures_dir = resources.files(__package__) / 'fixtures'
    return sorted(
        entry.name for entry in fixtures_dir.iterdir() if entry.is_file() and entry.name.endswith('.json')
    )


def _legacy_max_selected_items(task_type: BenchmarkTaskType) -> int:
    if task_type is BenchmarkTaskType.artifact_recall:
        return 2
    if task_type is BenchmarkTaskType.history_recall:
        return 2
    return 4


def _legacy_check_to_fact(raw_check: dict, *, index: int) -> BenchmarkGoldFact | None:
    values = list(raw_check.get('values', []))
    if not values:
        return None

    match = BenchmarkFactMatch.all_contains
    kind = raw_check.get('kind')
    if kind == TextCheckKind.any_contains.value:
        match = BenchmarkFactMatch.any_contains
    elif kind == TextCheckKind.exact.value:
        match = BenchmarkFactMatch.exact
    elif kind == TextCheckKind.absent.value:
        return None

    return BenchmarkGoldFact(
        key=f'legacy_fact_{index}',
        values=values,
        match=match,
        case_sensitive=bool(raw_check.get('case_sensitive', False)),
    )


def _upgrade_legacy_fixture(raw_task: dict) -> dict:
    upgraded = dict(raw_task)
    task_type = BenchmarkTaskType(upgraded['task_type'])
    legacy_expectation = BenchmarkBaselineExpectation.model_validate(
        upgraded.get('baseline_expectation', {})
    )
    facts = [
        fact
        for index, raw_check in enumerate(upgraded.get('gold_answer_checks', []), start=1)
        if (fact := _legacy_check_to_fact(raw_check, index=index)) is not None
    ]
    upgraded['gold_facts'] = [fact.model_dump(mode='json') for fact in facts]
    upgraded['acceptable_support_sets'] = [
        BenchmarkSupportSet(
            source_ids=[coerce_source_id(source) for source in upgraded.get('required_sources', [])]
        ).model_dump(mode='json')
    ]
    upgraded['distractor_source_ids'] = upgraded.get('distractor_source_ids', [])
    upgraded['budgets'] = BenchmarkBudget(
        max_retrieval_calls=2,
        max_returned_context_chars=int(upgraded.get('max_recall_chars', 1200)),
        max_selected_items=_legacy_max_selected_items(task_type),
    ).model_dump(mode='json')
    upgraded['baseline_expectation'] = legacy_expectation.model_copy(
        update={
            'minimum_answer_score': legacy_expectation.minimum_treatment_accuracy,
            'minimum_task_score': min(legacy_expectation.minimum_treatment_accuracy, 0.7),
            'minimum_retrieval_score': min(legacy_expectation.minimum_evidence_coverage, 0.7),
        }
    ).model_dump(mode='json')
    upgraded['hard_fail_rules'] = [
        BenchmarkHardFailRule.missing_provenance.value,
        BenchmarkHardFailRule.budget_overrun.value,
    ]
    return upgraded


def _normalize_fixture_payload(raw_task: dict) -> dict:
    if raw_task.get('gold_facts') is not None and raw_task.get('acceptable_support_sets') is not None:
        return raw_task
    return _upgrade_legacy_fixture(raw_task)


def load_fixtures() -> list[BenchmarkTaskFixture]:
    tasks: list[BenchmarkTaskFixture] = []
    fixtures_dir = resources.files(__package__) / 'fixtures'
    for name in _fixture_resource_names():
        payload = json.loads((fixtures_dir / name).read_text())
        tasks.extend(
            BenchmarkTaskFixture.model_validate(_normalize_fixture_payload(item))
            for item in payload['tasks']
        )
    seen_ids: set[str] = set()
    for task in tasks:
        if task.task_id in seen_ids:
            raise ValueError(f'duplicate benchmark task id: {task.task_id}')
        seen_ids.add(task.task_id)
    return tasks


def list_fixture_catalog() -> BenchmarkSuiteCatalog:
    catalog: dict[str, dict[str, int]] = {}
    for task in load_fixtures():
        suite_entry = catalog.setdefault(task.suite, {})
        suite_entry[task.tier] = suite_entry.get(task.tier, 0) + 1
    return BenchmarkSuiteCatalog(suites=catalog)


def get_fixture(task_id: str) -> BenchmarkTaskFixture:
    for task in load_fixtures():
        if task.task_id == task_id:
            return task
    raise KeyError(f'unknown benchmark task: {task_id}')


def get_suite_tasks(suite: str, tier: str) -> list[BenchmarkTaskFixture]:
    if tier == 'full':
        tiers = {'smoke', 'full'}
    else:
        tiers = {tier}
    tasks = [task for task in load_fixtures() if task.suite == suite and task.tier in tiers]
    if not tasks:
        raise ValueError(f'no benchmark tasks found for suite={suite!r} tier={tier!r}')
    return tasks
