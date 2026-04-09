from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .engine import MemoryEngine
from .models import MemoryKind

PROTOCOL_VERSION = '2024-11-05'


def _tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            'name': 'recall',
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
            'name': 'remember',
            'description': 'Persist durable project memory.',
            'inputSchema': {
                'type': 'object',
                'properties': {
                    'kind': {'type': 'string', 'enum': [kind.value for kind in MemoryKind]},
                    'summary': {'type': 'string'},
                    'details': {'type': 'string'},
                    'source': {'type': 'string'},
                    'tags': {'type': 'array', 'items': {'type': 'string'}},
                    'artifact_path': {'type': 'string'},
                },
                'required': ['kind', 'summary'],
            },
        },
        {
            'name': 'index',
            'description': 'Index high-signal project artifacts.',
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
    async with await MemoryEngine.open(root) as engine:
        if name == 'recall':
            return await engine.recall(arguments['query'], limit=int(arguments.get('limit', 8)))
        if name == 'remember':
            result = await engine.remember(
                kind=MemoryKind(arguments['kind']),
                summary=arguments['summary'],
                details=arguments.get('details', ''),
                source=arguments.get('source', 'mcp'),
                tags=arguments.get('tags') or [],
                artifact_path=arguments.get('artifact_path', ''),
            )
            return f'Stored memory as {result["uuid"]} ({result["mode"]}).'
        if name == 'index':
            indexed = await engine.index(
                changed_only=bool(arguments.get('changed_only', True)),
                max_files=int(arguments.get('max_files', 24)),
            )
            if not indexed:
                return 'No artifacts needed re-indexing.'
            return f'Indexed {len(indexed)} artifact(s).'
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
