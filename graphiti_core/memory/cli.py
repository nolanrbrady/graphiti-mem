from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .config import apply_agent_instructions, detect_project_root, install_codex_mcp_server
from .engine import MemoryEngine
from .mcp import run_stdio_mcp_server
from .models import BackendType, MemoryKind


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='graphiti', description='Local project memory for coding agents'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    init_parser = subparsers.add_parser(
        'init', help='Initialize Graphiti local memory for this project'
    )
    init_parser.add_argument(
        '--force', action='store_true', help='Overwrite generated Graphiti state files'
    )
    init_parser.add_argument('--yes', action='store_true', help='Accept default onboarding choices')
    init_parser.add_argument(
        '--apply-agents', action='store_true', help='Apply the managed Graphiti block to AGENTS.md'
    )
    init_parser.add_argument(
        '--no-apply-agents', action='store_true', help='Do not modify AGENTS.md during setup'
    )
    init_parser.add_argument(
        '--install-mcp',
        action='store_true',
        help='Install Graphiti into the local Codex MCP config',
    )
    init_parser.add_argument(
        '--no-install-mcp',
        action='store_true',
        help='Do not modify the local Codex MCP config during setup',
    )
    init_parser.add_argument(
        '--backend',
        choices=[backend.value for backend in BackendType],
        help='Override the configured storage backend for this project',
    )
    init_parser.add_argument(
        '--history-days',
        type=int,
        default=90,
        help='Maximum age of project transcript history to inspect for semantic bootstrap',
    )

    bootstrap_parser = subparsers.add_parser(
        'bootstrap', help='Run semantic bootstrap from recent project transcript history'
    )
    bootstrap_parser.add_argument(
        '--history-days',
        type=int,
        default=90,
        help='Maximum age of project transcript history to process',
    )
    bootstrap_parser.add_argument(
        '--force',
        action='store_true',
        help='Reprocess matching sessions even when fingerprints are unchanged',
    )

    recall_parser = subparsers.add_parser(
        'recall', help='Recall relevant memory for a task or question'
    )
    recall_parser.add_argument('query', help='Task or question to recall against')
    recall_parser.add_argument(
        '--limit', type=int, default=8, help='Maximum items per recall section'
    )

    remember_parser = subparsers.add_parser('remember', help='Persist durable project memory')
    remember_parser.add_argument(
        '--kind', choices=[kind.value for kind in MemoryKind], required=True
    )
    remember_parser.add_argument('--summary', required=True, help='Short durable summary')
    remember_parser.add_argument('--details', default='', help='Longer details to persist')
    remember_parser.add_argument('--source', default='agent', help='Source label for the memory')
    remember_parser.add_argument(
        '--tag', action='append', default=[], help='Optional tag; repeat as needed'
    )
    remember_parser.add_argument(
        '--path', default='', help='Optional file or artifact path for this memory'
    )

    index_parser = subparsers.add_parser(
        'index', help='Index high-signal project artifacts into memory'
    )
    index_parser.add_argument(
        '--changed', action='store_true', help='Only re-index artifacts whose fingerprint changed'
    )
    index_parser.add_argument(
        '--max-files', type=int, default=24, help='Maximum files to consider during indexing'
    )

    subparsers.add_parser('doctor', help='Inspect local Graphiti memory health')

    mcp_parser = subparsers.add_parser('mcp', help='Run the Graphiti stdio MCP server')
    mcp_parser.add_argument(
        '--transport',
        choices=['stdio'],
        default='stdio',
        help='Transport for the MCP server',
    )
    return parser


def _prompt_yes_no(prompt: str, *, default: bool = True, interactive: bool = True) -> bool:
    if not interactive:
        return default

    suffix = '[Y/n]' if default else '[y/N]'
    while True:
        response = input(f'{prompt} {suffix} ').strip().lower()
        if not response:
            return default
        if response in {'y', 'yes'}:
            return True
        if response in {'n', 'no'}:
            return False
        print('Please answer yes or no.')


async def _run_init(args: argparse.Namespace) -> int:
    root = detect_project_root(Path.cwd())
    interactive = sys.stdin.isatty() and not args.yes
    backend = MemoryEngine.choose_backend(
        root,
        requested_backend=BackendType(args.backend) if args.backend else None,
    )
    config = MemoryEngine.default_runtime_config(root, backend=backend)
    paths, config = MemoryEngine.init_project(
        root,
        force=args.force,
        config=config,
        history_days=args.history_days,
    )

    history_discovery = MemoryEngine.discover_history(paths.root, history_days=args.history_days)
    onboarding_state = MemoryEngine.sync_semantic_bootstrap_state(
        paths.root,
        history_days=args.history_days,
        discovery=history_discovery,
        requested_backend=backend,
    )
    history_count = history_discovery.total_sessions

    if args.apply_agents:
        apply_agents = True
    elif args.no_apply_agents:
        apply_agents = False
    elif args.yes or not interactive:
        apply_agents = True
    else:
        apply_agents = _prompt_yes_no(
            'Apply the managed Graphiti block to AGENTS.md?',
            default=True,
            interactive=interactive,
        )

    if args.install_mcp:
        install_mcp = True
    elif args.no_install_mcp:
        install_mcp = False
    else:
        install_mcp = False

    agents_path = apply_agent_instructions(paths.root) if apply_agents else None
    codex_config = None
    codex_config_changed = False
    if install_mcp:
        codex_config, codex_config_changed = install_codex_mcp_server(
            python_executable=sys.executable
        )

    print(f'Initialized Graphiti local memory in {paths.state_dir}')
    print(f'- Backend: {config.backend.value}')
    print(
        '- Agent-driven onboarding: enabled (no separate OpenAI-compatible endpoint required for init/history inspection)'
    )
    if history_count:
        print(f'- Matching transcript sessions detected: {history_count}')
        if onboarding_state['bootstrap_pending']:
            print(
                '- Semantic bootstrap: pending approval '
                f'({onboarding_state["bootstrap_processed_sessions"]}/{history_count} current sessions already processed)'
            )
        else:
            print(
                '- Semantic bootstrap: current '
                f'({onboarding_state["bootstrap_processed_sessions"]}/{history_count} current sessions processed)'
            )
    else:
        print('- Matching transcript sessions detected: 0')
        print('- Semantic bootstrap: not needed')
    if agents_path is not None:
        print(f'- Updated agent instructions: {agents_path}')
    else:
        print(
            f'- Left AGENTS.md unchanged; fallback instructions are in {paths.agent_instructions_path}'
        )
    if codex_config is not None:
        state = 'updated' if codex_config_changed else 'already current'
        print(f'- Codex MCP config: {codex_config} ({state})')
    else:
        print('- Left Codex MCP config unchanged (use `--install-mcp` to register globally)')
    print('')
    print('Codex MCP command:')
    print('- `graphiti mcp --transport stdio`')
    print('')
    print('Recommended next steps:')
    print('- Run `graphiti doctor` to verify backend and bootstrap status.')
    print('- If semantic bootstrap is pending, ask the user before running `graphiti bootstrap`.')
    print('- Prefer MCP tools for onboarding and memory management inside Codex.')
    print(
        '- Use `graphiti recall "<current task>"` and `graphiti remember ...` as local dev/test equivalents.'
    )
    print(
        '- Use `graphiti index --changed` to register project artifacts without automatic distillation.'
    )
    return 0


async def _run_async(args: argparse.Namespace) -> int:
    if args.command == 'init':
        return await _run_init(args)

    if args.command == 'bootstrap':
        async with await MemoryEngine.open(Path.cwd()) as engine:
            discovery = MemoryEngine.discover_history(Path.cwd(), history_days=args.history_days)
            processed = await engine.bootstrap_history(
                history_days=args.history_days,
                discovery=discovery,
                force=bool(args.force),
            )
            state = MemoryEngine.sync_semantic_bootstrap_state(
                Path.cwd(),
                history_days=args.history_days,
                discovery=discovery,
                requested_backend=getattr(getattr(engine, 'config', None), 'backend', None),
            )

        print(
            f'Semantic bootstrap processed {len(processed)} session(s) '
            f'out of {state["bootstrap_eligible_sessions"]} eligible session(s).'
        )
        print(
            '- Durable memories created: '
            + str(sum(int(item.get('memory_count', '0')) for item in processed))
        )
        print(
            '- Structured graph extraction: '
            + ('available' if state['bootstrap_structured_graph_available'] else 'unavailable')
        )
        print(
            '- Bootstrap status: '
            + ('pending' if state['bootstrap_pending'] else state['bootstrap_status'])
        )
        return 0

    if args.command == 'mcp':
        await run_stdio_mcp_server(Path.cwd())
        return 0

    async with await MemoryEngine.open(Path.cwd()) as engine:
        if args.command == 'recall':
            print(await engine.recall(args.query, limit=args.limit))
            return 0

        if args.command == 'remember':
            remembered = await engine.remember(
                kind=MemoryKind(args.kind),
                summary=args.summary,
                details=args.details,
                source=args.source,
                tags=args.tag,
                artifact_path=args.path,
            )
            print(f'Stored {args.kind} memory as {remembered["uuid"]} ({remembered["mode"]}).')
            return 0

        if args.command == 'index':
            indexed = await engine.index(changed_only=args.changed, max_files=args.max_files)
            if not indexed:
                print('No artifacts needed re-indexing.')
                return 0
            print(f'Indexed {len(indexed)} artifact(s).')
            for artifact in indexed[:10]:
                print(f'- {artifact["artifact"]} -> {artifact["episode_uuid"]}')
            return 0

        if args.command == 'doctor':
            print(await engine.doctor())
            return 0

    return 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run_async(args))
    except FileNotFoundError as exc:
        print(f'{exc}\nRun `graphiti init` from the project root first.')
        return 1
    except RuntimeError as exc:
        print(str(exc))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
