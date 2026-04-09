from __future__ import annotations

import json
from importlib import resources

from .models import BenchmarkSuiteCatalog, BenchmarkTaskFixture


def _fixture_resource_names() -> list[str]:
    fixtures_dir = resources.files(__package__) / 'fixtures'
    return sorted(
        entry.name for entry in fixtures_dir.iterdir() if entry.is_file() and entry.name.endswith('.json')
    )


def load_fixtures() -> list[BenchmarkTaskFixture]:
    tasks: list[BenchmarkTaskFixture] = []
    fixtures_dir = resources.files(__package__) / 'fixtures'
    for name in _fixture_resource_names():
        payload = json.loads((fixtures_dir / name).read_text())
        tasks.extend(BenchmarkTaskFixture.model_validate(item) for item in payload['tasks'])
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
