from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .config import install_codex_mcp_server
from .engine import MemoryEngine
from .models import BackendType, MemoryKind

PROTOCOL_VERSION = '2024-11-05'


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _tool_definitions() -> list[dict[str, Any]]:
    memory_kind_values = [kind.value for kind in MemoryKind]
    backend_values = [backend.value for backend in BackendType]
    return [
        {
            'name': 'init_project',
            'description': 'Initialize Graphiti for the current project and return onboarding state.',
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'force': {'type': 'boolean', 'default': False},
                    'backend': {'type': 'string', 'enum': backend_values},
                    'apply_agents': {'type': 'boolean', 'default': False},
                    'install_mcp': {'type': 'boolean', 'default': False},
                    'history_days': {'type': 'integer', 'default': 90},
                },
            },
        },
        {
            'name': 'discover_history',
            'description': 'Inspect whether Codex or Claude project history is available locally.',
            'inputSchema': {
                'type': 'object',
                'properties': {'history_days': {'type': 'integer', 'default': 90}},
            },
        },
        {
            'name': 'list_history_sessions',
            'description': 'List project-matching Codex and Claude history sessions.',
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'history_days': {'type': 'integer', 'default': 90},
                    'limit': {'type': 'integer'},
                },
            },
        },
        {
            'name': 'read_history_session',
            'description': 'Read a chunk of one project-matching history session.',
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'session_id': {'type': 'string'},
                    'history_days': {'type': 'integer', 'default': 90},
                    'offset': {'type': 'integer', 'default': 0},
                    'max_chars': {'type': 'integer', 'default': 6000},
                },
                'required': ['session_id'],
            },
        },
        {
            'name': 'import_history_sessions',
            'description': 'Store selected history sessions as source episodes with provenance, without distilling memory automatically.',
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'session_ids': {'type': 'array', 'items': {'type': 'string'}},
                    'history_days': {'type': 'integer', 'default': 90},
                },
            },
        },
        {
            'name': 'apply_agents_instructions',
            'description': 'Apply or refresh the managed Graphiti block in AGENTS.md.',
            'inputSchema': {'type': 'object', 'properties': {}},
        },
        {
            'name': 'store_memory',
            'description': 'Persist durable project memory.',
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'kind': {'type': 'string', 'enum': memory_kind_values},
                    'summary': {'type': 'string'},
                    'details': {'type': 'string'},
                    'source': {'type': 'string'},
                    'tags': {'type': 'array', 'items': {'type': 'string'}},
                    'artifact_path': {'type': 'string'},
                    'provenance': {'type': 'object'},
                },
                'required': ['kind', 'summary'],
            },
        },
        {
            'name': 'recall_memory',
            'description': 'Recall relevant project memory for a task or question.',
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'limit': {'type': 'integer', 'default': 8},
                },
                'required': ['query'],
            },
        },
        {
            'name': 'index_project',
            'description': 'Index high-signal project artifacts without distilling durable memory automatically.',
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'changed_only': {'type': 'boolean', 'default': True},
                    'max_files': {'type': 'integer', 'default': 24},
                },
            },
        },
        {
            'name': 'doctor',
            'description': 'Inspect local Graphiti memory health.',
            'inputSchema': {'type': 'object', 'properties': {}},
        },
        {
            'name': 'remember',
            'description': 'Alias for store_memory.',
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'kind': {'type': 'string', 'enum': memory_kind_values},
                    'summary': {'type': 'string'},
                    'details': {'type': 'string'},
                    'source': {'type': 'string'},
                    'tags': {'type': 'array', 'items': {'type': 'string'}},
                    'artifact_path': {'type': 'string'},
                    'provenance': {'type': 'object'},
                },
                'required': ['kind', 'summary'],
            },
        },
        {
            'name': 'recall',
            'description': 'Alias for recall_memory.',
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string'},
                    'limit': {'type': 'integer', 'default': 8},
                },
                'required': ['query'],
            },
        },
        {
            'name': 'index',
            'description': 'Alias for index_project.',
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'changed_only': {'type': 'boolean', 'default': True},
                    'max_files': {'type': 'integer', 'default': 24},
                },
            },
        },
    ]


def _read_message() -> dict[str, Any] | None:
    content_length = None
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        decoded = line.decode('utf-8').strip()
        if not decoded:
            break
        if decoded.lower().startswith('content-length:'):
            content_length = int(decoded.split(':', 1)[1].strip())

    if content_length is None:
        return None

    payload = sys.stdin.buffer.read(content_length)
    if not payload:
        return None
    return json.loads(payload.decode('utf-8'))


def _write_message(message: dict[str, Any]) -> None:
    body = json.dumps(message).encode('utf-8')
    sys.stdout.buffer.write(f'Content-Length: {len(body)}\r\n\r\n'.encode())
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


async def _call_tool(root: Path, name: str, arguments: dict[str, Any]) -> str:
    if name == 'init_project':
        requested_backend = BackendType(arguments['backend']) if arguments.get('backend') else None
        config = MemoryEngine.default_runtime_config(
            root, backend=MemoryEngine.choose_backend(root, requested_backend=requested_backend)
        )
        paths, config = MemoryEngine.init_project(
            root,
            force=bool(arguments.get('force', False)),
            config=config,
        )
        agents_path = None
        if bool(arguments.get('apply_agents', False)):
            agents_path = MemoryEngine.apply_managed_agents_block(root)
        codex_config_path = None
        codex_config_updated = False
        if bool(arguments.get('install_mcp', False)):
            codex_config_path, codex_config_updated = install_codex_mcp_server(
                python_executable=sys.executable
            )
        payload = MemoryEngine.detect_onboarding_state(
            root,
            history_days=int(arguments.get('history_days', 90)),
            requested_backend=requested_backend,
        )
        payload['configured_backend'] = config.backend.value
        payload['config_path'] = str(paths.config_path)
        payload['agents_updated'] = bool(agents_path is not None)
        if agents_path is not None:
            payload['agents_path_updated'] = str(agents_path)
        payload['codex_mcp_updated'] = codex_config_updated
        if codex_config_path is not None:
            payload['codex_config_path_updated'] = str(codex_config_path)
        return _json_text(payload)

    if name == 'apply_agents_instructions':
        return _json_text({'agents_path': str(MemoryEngine.apply_managed_agents_block(root))})

    if name == 'discover_history':
        payload = MemoryEngine.detect_onboarding_state(
            root,
            history_days=int(arguments.get('history_days', 90)),
        )
        return _json_text(payload)

    async with await MemoryEngine.open(root) as engine:
        if name == 'list_history_sessions':
            payload = engine.list_history_sessions(
                history_days=int(arguments.get('history_days', 90)),
                limit=int(arguments['limit']) if arguments.get('limit') is not None else None,
            )
            return _json_text(payload)

        if name == 'read_history_session':
            payload = engine.read_history_session(
                arguments['session_id'],
                history_days=int(arguments.get('history_days', 90)),
                offset=int(arguments.get('offset', 0)),
                max_chars=int(arguments.get('max_chars', 6000)),
            )
            return _json_text(payload)

        if name == 'import_history_sessions':
            payload = await engine.import_history_sessions(
                session_ids=list(arguments.get('session_ids') or []),
                history_days=int(arguments.get('history_days', 90)),
            )
            return _json_text(payload)

        if name in {'store_memory', 'remember'}:
            result = await engine.remember(
                kind=MemoryKind(arguments['kind']),
                summary=arguments['summary'],
                details=arguments.get('details', ''),
                source=arguments.get('source', 'mcp'),
                tags=list(arguments.get('tags') or []),
                artifact_path=arguments.get('artifact_path', ''),
                provenance=dict(arguments.get('provenance') or {}),
            )
            return _json_text(result)

        if name in {'recall_memory', 'recall'}:
            return await engine.recall(arguments['query'], limit=int(arguments.get('limit', 8)))

        if name in {'index_project', 'index'}:
            indexed = await engine.index(
                changed_only=bool(arguments.get('changed_only', True)),
                max_files=int(arguments.get('max_files', 24)),
            )
            return _json_text(indexed)

        if name == 'doctor':
            return await engine.doctor()

    raise ValueError(f'Unsupported tool: {name}')


async def run_stdio_mcp_server(root: Path) -> None:
    tool_definitions = _tool_definitions()
    while True:
        message = await asyncio.to_thread(_read_message)
        if message is None:
            return

        method = message.get('method')
        message_id = message.get('id')

        if method == 'initialize':
            _write_message(
                {
                    'jsonrpc': '2.0',
                    'id': message_id,
                    'result': {
                        'protocolVersion': PROTOCOL_VERSION,
                        'capabilities': {'tools': {}},
                        'serverInfo': {'name': 'graphiti', 'version': '0.1.0'},
                    },
                }
            )
            continue

        if method == 'notifications/initialized':
            continue

        if method == 'ping':
            _write_message({'jsonrpc': '2.0', 'id': message_id, 'result': {}})
            continue

        if method == 'tools/list':
            _write_message(
                {'jsonrpc': '2.0', 'id': message_id, 'result': {'tools': tool_definitions}}
            )
            continue

        if method == 'tools/call':
            params = message.get('params', {})
            name = params.get('name')
            arguments = params.get('arguments', {})
            try:
                text = await _call_tool(root, name, arguments)
                _write_message(
                    {
                        'jsonrpc': '2.0',
                        'id': message_id,
                        'result': {'content': [{'type': 'text', 'text': text}], 'isError': False},
                    }
                )
            except Exception as exc:
                _write_message(
                    {
                        'jsonrpc': '2.0',
                        'id': message_id,
                        'result': {
                            'content': [{'type': 'text', 'text': str(exc)}],
                            'isError': True,
                        },
                    }
                )
            continue

        if message_id is not None:
            _write_message(
                {
                    'jsonrpc': '2.0',
                    'id': message_id,
                    'error': {'code': -32601, 'message': f'Unknown method: {method}'},
                }
            )
