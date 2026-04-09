from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SEARCH_TOOL_NAMES = {'exec_command'}
SEARCH_COMMAND_MARKERS = (
    'rg ',
    'rg --files',
    'grep ',
    'git grep',
    'find ',
    'fd ',
    'ls',
)


def discover_codex_state_db(home: Path | None = None) -> Path | None:
    base = (home or Path.home()) / '.codex'
    candidates = sorted(base.glob('state_*.sqlite'), reverse=True)
    return candidates[0] if candidates else None


def read_codex_thread_metrics(db_path: Path, thread_id: str) -> dict[str, str | int]:
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """
            SELECT id, rollout_path, tokens_used
            FROM threads
            WHERE id = ?
            """,
            (thread_id,),
        ).fetchone()
    finally:
        connection.close()

    if row is None:
        raise KeyError(f'unknown Codex thread id: {thread_id}')

    rollout_path = Path(row['rollout_path'])
    return {
        'thread_id': row['id'],
        'rollout_path': str(rollout_path),
        'tokens_used': int(row['tokens_used']),
        'search_actions': count_search_actions_from_rollout(rollout_path),
    }


def count_search_actions_from_rollout(path: Path) -> int:
    if not path.exists():
        return 0

    count = 0
    with path.open() as handle:
        for raw_line in handle:
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            payload = record.get('payload', {})
            if payload.get('type') != 'tool_call':
                continue

            tool_name = payload.get('tool_name') or payload.get('name') or ''
            if tool_name not in SEARCH_TOOL_NAMES:
                continue

            command = str(payload.get('cmd') or payload.get('message') or '').lower()
            if any(marker in command for marker in SEARCH_COMMAND_MARKERS):
                count += 1
    return count
