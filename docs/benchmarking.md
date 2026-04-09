# Benchmarking Graphiti Memory

Graphiti's deterministic benchmark is a staged evaluation of:

1. `retrieval`
2. `attribution`
3. `answer`
4. `efficiency`

The default optimization target is the frozen synthetic suite, `deterministic_core`. Dogfood mode uses the same task and result schema, but sources its artifacts and history from the local repo and local Codex history.

## What A Score Of 100 Means

A task reaches `100` only when all of the following are true:

- the retrieved candidate set contains the required support
- the returned answer/context carries explicit provenance IDs for the selected evidence
- the returned answer satisfies every required fact slot
- the hard budgets are respected

If the task breaks a hard budget, the task score becomes `0`.

The suite reward is the mean normalized task score and is emitted only when suite-level gates pass.

## Fixture Schema

Benchmark task fixtures are now structured around explicit answer facts, support sets, distractors, budgets, and hard-fail rules.

```json
{
  "task_id": "artifact-current-benchmark-target",
  "suite": "deterministic_core",
  "tier": "smoke",
  "query": "What is the current benchmark target, not the legacy benchmark-memory-v1 alias?",
  "task_type": "artifact_recall",
  "difficulty": "medium",
  "gold_facts": [
    {
      "key": "current_target",
      "match": "any_contains",
      "values": [
        "make benchmark-memory",
        "python3 -m graphiti_core.memory.benchmark run --suite deterministic_core --tier smoke"
      ]
    }
  ],
  "acceptable_support_sets": [
    {
      "source_ids": ["artifact:Makefile"]
    }
  ],
  "distractor_source_ids": ["artifact:docs/legacy-benchmark.md"],
  "budgets": {
    "max_retrieval_calls": 2,
    "max_returned_context_chars": 320,
    "max_selected_items": 1
  },
  "hard_fail_rules": ["missing_provenance", "budget_overrun", "wrong_support"]
}
```

Field meanings:

- `gold_facts`: answer slots to satisfy. `match` can be `all_contains`, `any_contains`, or `exact`.
- `acceptable_support_sets`: minimal provenance/source ID sets that justify the answer.
- `distractor_source_ids`: confusable but wrong evidence. Selecting these drops attribution to `0`.
- `budgets`: hard limits for retrieval calls and returned context size. `max_selected_items` is also enforced when set.
- `hard_fail_rules`: failures that zero capability for the task.

## Result Schema

Each channel result now exposes:

- `retrieval_trace`: retrieval calls, retrieval queries, candidate IDs, selected evidence IDs, and attached provenance IDs
- `retrieval_score`
- `attribution_score`
- `answer_score`
- `capability_score`
- `efficiency_score`
- `task_score`

Aggregate results expose:

- `mean_task_score`
- `mean_retrieval_score`
- `mean_attribution_score`
- `mean_answer_score`
- category task scores for artifact/history/multi-hop
- hard-failure counts for budget, provenance, support, and unsupported-claim failures

`compare` reports deltas on the staged metrics rather than only a single blended accuracy proxy.

## Authoring Guidance

When adding tasks:

- keep support sets minimal and unambiguous
- use explicit provenance IDs such as `artifact:README.md` or `thread:Recall before search`
- add distractors when the task is supposed to catch stale or confusable evidence
- avoid gold leakage: task hints must not be required by the runner to rank or expand treatment results
- prefer facts that describe meaning or canonical values, not only one exact surface form
- keep smoke and full tiers semantically identical; smoke is the fast optimization loop and full is the acceptance gate

## Legacy Fixture Compatibility

Legacy fixtures that only define `gold_answer_checks`, `required_sources`, and `max_recall_chars` are still accepted. They are upgraded at load time into:

- `gold_facts`
- `acceptable_support_sets`
- `budgets`
- default hard-fail rules for missing provenance and budget overrun

This compatibility path exists to keep older suites runnable while new tasks are authored directly against the v2 schema.
