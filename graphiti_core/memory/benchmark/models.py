from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

BENCHMARK_SCHEMA_VERSION = '1.0'


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


class BenchmarkTextCheck(BaseModel):
    kind: TextCheckKind
    values: list[str] = Field(default_factory=list)
    case_sensitive: bool = False


class BenchmarkBaselineExpectation(BaseModel):
    minimum_treatment_accuracy: float = 0.75
    minimum_evidence_coverage: float = 0.75
    expect_token_reduction: bool = True
    expect_search_reduction: bool = True


class BenchmarkTaskFixture(BaseModel):
    task_id: str
    suite: str
    tier: str
    query: str
    task_type: BenchmarkTaskType
    difficulty: str
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


class BenchmarkChannelResult(BaseModel):
    answer_score: float
    evidence_score: float
    overall_score: float
    required_source_coverage: float
    task_tokens: int
    search_actions: int
    context_chars: int
    context_items: int
    context: str
    shortcut_hits: list[str] = Field(default_factory=list)
    matched_answer_checks: int = 0
    matched_evidence_checks: int = 0


class BenchmarkTaskDelta(BaseModel):
    accuracy_gain: float
    evidence_gain: float
    token_delta: int
    token_reduction_ratio: float
    search_delta: int
    search_reduction_ratio: float
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
    accuracy_score: float
    evidence_coverage: float
    token_efficiency_score: float
    search_efficiency_score: float
    artifact_accuracy: float
    history_accuracy: float
    multi_hop_accuracy: float
    mean_recall_chars: float
    mean_control_chars: float
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
