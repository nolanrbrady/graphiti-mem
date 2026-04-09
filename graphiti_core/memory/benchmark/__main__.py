from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .models import BenchmarkResult
from .runner import benchmark_doctor, compare_results, inspect_task, list_suites, run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m graphiti_core.memory.benchmark',
        description='Deterministic Graphiti benchmark and reward runner',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    run_parser = subparsers.add_parser('run', help='Run the benchmark suite')
    run_parser.add_argument('--mode', choices=['synthetic', 'dogfood'], default='synthetic')
    run_parser.add_argument('--suite', default='deterministic_core')
    run_parser.add_argument('--tier', default='smoke', choices=['smoke', 'full'])
    run_parser.add_argument('--baseline-result', type=Path)
    run_parser.add_argument('--output', type=Path)
    run_parser.add_argument('--summary-path', type=Path)
    run_parser.add_argument(
        '--local-history-overlay',
        action='store_true',
        help='Import current repo local history as extra distractor/source evidence',
    )

    subparsers.add_parser('list', help='List benchmark suites and tiers')

    inspect_parser = subparsers.add_parser('inspect', help='Inspect a benchmark task fixture')
    inspect_parser.add_argument('task_id')

    compare_parser = subparsers.add_parser('compare', help='Compare two benchmark result files')
    compare_parser.add_argument('baseline', type=Path)
    compare_parser.add_argument('candidate', type=Path)

    subparsers.add_parser('doctor', help='Validate benchmark prerequisites and fixtures')
    return parser


def _require_python_310() -> bool:
    return sys.version_info >= (3, 10)


def _markdown_summary(result: BenchmarkResult) -> str:
    return '\n'.join(
        [
            f'# Graphiti Benchmark: {result.suite}/{result.tier}',
            '',
            f'- Gate passed: `{result.gate_passed}`',
            f'- Reward: `{result.reward}`',
            f'- Mean task score: `{result.aggregate.mean_task_score:.2f}`',
            f'- Mean retrieval score: `{result.aggregate.mean_retrieval_score:.3f}`',
            f'- Mean attribution score: `{result.aggregate.mean_attribution_score:.3f}`',
            f'- Mean answer score: `{result.aggregate.mean_answer_score:.3f}`',
            f'- Failure reasons: `{", ".join(result.failure_reasons) or "none"}`',
        ]
    )


async def _run_async(args: argparse.Namespace) -> int:
    if args.command == 'list':
        print(json.dumps({'suites': list_suites()}, indent=2, sort_keys=True))
        return 0

    if args.command == 'inspect':
        print(json.dumps(inspect_task(args.task_id), indent=2, sort_keys=True))
        return 0

    if args.command == 'compare':
        baseline = BenchmarkResult.model_validate(json.loads(args.baseline.read_text()))
        candidate = BenchmarkResult.model_validate(json.loads(args.candidate.read_text()))
        print(
            json.dumps(
                compare_results(baseline, candidate).model_dump(mode='json'),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == 'doctor':
        doctor = benchmark_doctor()
        print(json.dumps(doctor.model_dump(mode='json'), indent=2, sort_keys=True))
        return 0 if doctor.python_supported and doctor.fixtures_valid else 2

    if not _require_python_310():
        print(
            json.dumps(
                {
                    'error': 'Python 3.10 or newer is required for benchmark execution.',
                    'python_version': sys.version.split()[0],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    try:
        result = await run_benchmark(
            args.suite,
            args.tier,
            baseline_result_path=args.baseline_result,
            local_history_overlay=bool(args.local_history_overlay),
            mode=args.mode,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(json.dumps({'error': str(exc)}, indent=2, sort_keys=True))
        return 2

    payload = json.dumps(result.model_dump(mode='json'), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(payload + '\n')
    if args.summary_path is not None:
        args.summary_path.write_text(_markdown_summary(result) + '\n')
    print(payload)
    return 0 if result.gate_passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run_async(args))


if __name__ == '__main__':
    raise SystemExit(main())
