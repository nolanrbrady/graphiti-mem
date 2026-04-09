from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from graphiti_core.memory.models import MemoryKind

BENCHMARK_SCHEMA_VERSION = '2.1'

_ARTIFACT_LIKE_REFERENCES = {
    'agents.md',
    'makefile',
    'pyproject.toml',
    'readme.md',
}


def benchmark_slug(value: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-')
    return slug or 'item'


def artifact_source_id(path: str) -> str:
    return f'artifact:{path}'


def thread_source_id(title: str) -> str:
    return f'thread:{title}'


def session_source_id(session_id: str) -> str:
    return f'session:{session_id}'


def memory_source_id(kind: str, summary: str) -> str:
    return f'memory:{kind}:{benchmark_slug(summary)}'


def coerce_source_id(reference: str) -> str:
    if reference.startswith(('artifact:', 'thread:', 'session:', 'memory:')):
        return reference

    normalized = reference.lower()
    if (
        '/' in reference
        or normalized in _ARTIFACT_LIKE_REFERENCES
        or normalized.endswith(('.md', '.toml', '.yaml', '.yml', '.json', '.txt'))
    ):
        return artifact_source_id(reference)
    return thread_source_id(reference)


class BenchmarkTaskType(str, Enum):
    artifact_recall = 'artifact_recall'
    history_recall = 'history_recall'
    multi_hop_recall = 'multi_hop_recall'


class BenchmarkJudgeMode(str, Enum):
    deterministic = 'deterministic'
    free_form = 'free_form'


class TextCheckKind(str, Enum):
    exact = 'exact'
    contains = 'contains'
    any_contains = 'any_contains'
    absent = 'absent'


class BenchmarkFactMatch(str, Enum):
    exact = 'exact'
    all_contains = 'all_contains'
    any_contains = 'any_contains'


class BenchmarkHardFailRule(str, Enum):
    missing_provenance = 'missing_provenance'
    budget_overrun = 'budget_overrun'
    wrong_support = 'wrong_support'
    unsupported_claim = 'unsupported_claim'
    stale_support = 'stale_support'


class BenchmarkScenarioEventKind(str, Enum):
    artifact_snapshot = 'artifact_snapshot'
    history_turn = 'history_turn'
    memory_seed = 'memory_seed'
    decision_update = 'decision_update'
    pitfall_update = 'pitfall_update'
    constraint_update = 'constraint_update'


class BenchmarkTextCheck(BaseModel):
    kind: TextCheckKind
    values: list[str] = Field(default_factory=list)
    case_sensitive: bool = False


class BenchmarkGoldFact(BaseModel):
    key: str
    values: list[str] = Field(default_factory=list)
    match: BenchmarkFactMatch = BenchmarkFactMatch.all_contains
    case_sensitive: bool = False
    weight: float = 1.0


class BenchmarkSupportSet(BaseModel):
    source_ids: list[str] = Field(default_factory=list)


class BenchmarkScenarioEvent(BaseModel):
    event_id: str
    timestamp: datetime
    kind: BenchmarkScenarioEventKind
    artifact_path: str = ''
    content: str = ''
    user_message: str = ''
    assistant_message: str = ''
    memory_kind: MemoryKind | None = None
    summary: str = ''
    details: str = ''
    source_agent: str = 'codex'
    session_id: str = ''
    thread_title: str = ''
    tags: list[str] = Field(default_factory=list)
    supersedes: list[str] = Field(default_factory=list)
    active_until: datetime | None = None


class BenchmarkScenarioFixture(BaseModel):
    scenario_id: str
    suite: str = ''
    events: list[BenchmarkScenarioEvent] = Field(default_factory=list)


class BenchmarkBudget(BaseModel):
    max_retrieval_calls: int = 2
    max_returned_context_chars: int = 1200
    max_selected_items: int = 0


class BenchmarkBaselineExpectation(BaseModel):
    minimum_treatment_accuracy: float = 0.75
    minimum_evidence_coverage: float = 0.75
    minimum_retrieval_score: float = 0.75
    minimum_answer_score: float = 0.75
    minimum_task_score: float = 0.75
    expect_token_reduction: bool = False
    expect_search_reduction: bool = False


class BenchmarkTaskFixture(BaseModel):
    task_id: str
    suite: str
    tier: str
    query: str
    task_type: BenchmarkTaskType
    difficulty: str
    gold_facts: list[BenchmarkGoldFact] = Field(default_factory=list)
    acceptable_support_sets: list[BenchmarkSupportSet] = Field(default_factory=list)
    forbidden_support_sets: list[BenchmarkSupportSet] = Field(default_factory=list)
    distractor_source_ids: list[str] = Field(default_factory=list)
    scenario_id: str = ''
    query_time: datetime | None = None
    events: list[BenchmarkScenarioEvent] = Field(default_factory=list)
    budgets: BenchmarkBudget = Field(default_factory=BenchmarkBudget)
    hard_fail_rules: list[BenchmarkHardFailRule] = Field(default_factory=list)
    gold_answer_checks: list[BenchmarkTextCheck] = Field(default_factory=list)
    gold_evidence_checks: list[BenchmarkTextCheck] = Field(default_factory=list)
    required_sources: list[str] = Field(default_factory=list)
    disallowed_shortcuts: list[str] = Field(default_factory=list)
    max_recall_chars: int = 1200
    baseline_expectation: BenchmarkBaselineExpectation = Field(
        default_factory=BenchmarkBaselineExpectation
    )
    notes: str = ''
    judge_mode: BenchmarkJudgeMode = BenchmarkJudgeMode.deterministic


class BenchmarkSuiteCatalog(BaseModel):
    suites: dict[str, dict[str, int]]


class BenchmarkRetrievalTrace(BaseModel):
    retrieval_calls: int
    retrieval_queries: list[str] = Field(default_factory=list)
    candidate_ids: list[str] = Field(default_factory=list)
    selected_evidence_ids: list[str] = Field(default_factory=list)
    provenance_ids: list[str] = Field(default_factory=list)
    selected_item_count: int = 0
    candidate_count: int = 0


class BenchmarkChannelResult(BaseModel):
    retrieval_score: float
    attribution_score: float
    answer_score: float
    temporal_score: float = 1.0
    capability_score: float
    efficiency_score: float
    task_score: float
    normalized_task_score: float
    task_tokens: int
    retrieval_calls: int
    context_chars: int
    context_items: int
    context: str
    retrieval_trace: BenchmarkRetrievalTrace
    selected_evidence_ids: list[str] = Field(default_factory=list)
    provenance_ids: list[str] = Field(default_factory=list)
    hard_failures: list[str] = Field(default_factory=list)
    matched_fact_count: int = 0
    budget_passed: bool = True


class BenchmarkTaskDelta(BaseModel):
    retrieval_gain: float
    attribution_gain: float
    answer_gain: float
    capability_gain: float
    task_score_gain: float
    retrieval_call_delta: int
    context_char_delta: int


class BenchmarkTaskResult(BaseModel):
    task_id: str
    suite: str
    tier: str
    task_type: BenchmarkTaskType
    difficulty: str
    notes: str = ''
    control: BenchmarkChannelResult
    treatment: BenchmarkChannelResult
    delta: BenchmarkTaskDelta
    gate_passed: bool
    failure_reasons: list[str] = Field(default_factory=list)


class BenchmarkAggregateResult(BaseModel):
    task_count: int
    mean_task_score: float
    mean_task_score_normalized: float
    mean_retrieval_score: float
    mean_attribution_score: float
    mean_answer_score: float
    mean_temporal_score: float
    mean_capability_score: float
    mean_efficiency_score: float
    artifact_task_score: float
    history_task_score: float
    multi_hop_task_score: float
    budget_failure_count: int
    provenance_failure_count: int
    support_failure_count: int
    unsupported_claim_failure_count: int
    stale_support_failure_count: int
    mean_returned_context_chars: float
    mean_control_context_chars: float
    gate_thresholds: dict[str, float] = Field(default_factory=dict)


class BenchmarkResult(BaseModel):
    schema_version: str = BENCHMARK_SCHEMA_VERSION
    suite: str
    tier: str
    config: dict[str, Any]
    timestamp: str
    tasks: list[BenchmarkTaskResult]
    aggregate: BenchmarkAggregateResult
    gate_passed: bool
    reward: float | None
    failure_reasons: list[str] = Field(default_factory=list)


class BenchmarkComparison(BaseModel):
    baseline_gate_passed: bool
    candidate_gate_passed: bool
    reward_delta: float | None
    aggregate_deltas: dict[str, float] = Field(default_factory=dict)
    regressed_tasks: list[str] = Field(default_factory=list)


class BenchmarkDoctorResult(BaseModel):
    python_version: str
    python_supported: bool
    fixtures_valid: bool
    suites: dict[str, dict[str, int]]
    codex_state_db: str = ''
    local_history_sessions_detected: int = 0
    issues: list[str] = Field(default_factory=list)
