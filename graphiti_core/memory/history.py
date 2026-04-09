from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import BootstrapDiscovery, BootstrapSession, hash_text_parts

MAX_SESSION_CHARS = 24_000


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)
    except ValueError:
        return None


def _trim_content(content: str) -> str:
    return content.strip()[:MAX_SESSION_CHARS].strip()


def _codex_state_files(home: Path) -> list[Path]:
    return sorted(home.glob('state_*.sqlite'), reverse=True)


def discover_project_history(root: Path, max_age_days: int = 90) -> BootstrapDiscovery:
    root = root.resolve()
    since = _utc_now() - timedelta(days=max_age_days)
    home = Path.home()
    codex_sessions = _discover_codex_sessions(home / '.codex', root, since)
    claude_sessions = _discover_claude_sessions(home / '.claude', root, since)
    return BootstrapDiscovery(codex_sessions=codex_sessions, claude_sessions=claude_sessions)


def _discover_codex_sessions(base: Path, root: Path, since: datetime) -> list[BootstrapSession]:
    if not base.exists():
        return []

    sessions: list[BootstrapSession] = []
    for state_file in _codex_state_files(base):
        try:
            connection = sqlite3.connect(str(state_file))
            connection.row_factory = sqlite3.Row
        except sqlite3.Error:
            continue

        try:
            rows = connection.execute(
                """
                SELECT id, title, cwd, rollout_path, updated_at
                FROM threads
                WHERE cwd = ?
                ORDER BY updated_at DESC
                """,
                (str(root),),
            ).fetchall()
        except sqlite3.Error:
            connection.close()
            continue
        connection.close()

        for row in rows:
            created_at = datetime.fromtimestamp(row['updated_at'] / 1000, tz=timezone.utc)
            if created_at < since:
                continue

            rollout_path = Path(row['rollout_path'])
            if not rollout_path.exists():
                continue

            content = _trim_content(_parse_codex_rollout(rollout_path))
            if not content:
                continue

            sessions.append(
                BootstrapSession(
                    source_agent='codex',
                    session_id=row['id'],
                    title=row['title'] or 'Codex session',
                    created_at=created_at,
                    fingerprint=hash_text_parts([row['id'], str(created_at.timestamp()), content]),
                    content=content,
                    source_path=str(rollout_path),
                )
            )
        if sessions:
            break

    return sessions


def _parse_codex_rollout(path: Path) -> str:
    transcript_lines: list[str] = []
    with path.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get('type') != 'event_msg':
                continue

            payload = record.get('payload', {})
            payload_type = payload.get('type')
            if payload_type == 'user_message':
                message = payload.get('message', '').strip()
                if message:
                    transcript_lines.append(f'User: {message}')
            elif payload_type == 'agent_message':
                message = payload.get('message', '').strip()
                if message:
                    transcript_lines.append(f'Assistant: {message}')

    return '\n'.join(transcript_lines)


def _discover_claude_sessions(base: Path, root: Path, since: datetime) -> list[BootstrapSession]:
    projects_dir = base / 'projects'
    if not projects_dir.exists():
        return []

    escaped_project = str(root).replace(os.sep, '-')
    project_dir = projects_dir / escaped_project
    if not project_dir.exists():
        return []

    sessions: list[BootstrapSession] = []
    for path in sorted(project_dir.glob('*.jsonl'), reverse=True):
        content, session_id, title, created_at = _parse_claude_transcript(path)
        if not content or not session_id or created_at is None or created_at < since:
            continue

        sessions.append(
            BootstrapSession(
                source_agent='claude',
                session_id=session_id,
                title=title or path.stem,
                created_at=created_at,
                fingerprint=hash_text_parts([session_id, str(created_at.timestamp()), content]),
                content=content,
                source_path=str(path),
            )
        )

    return sessions


def _extract_claude_text(content: str | list[dict] | None) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ''

    parts: list[str] = []
    for block in content:
        if block.get('type') == 'text':
            text = str(block.get('text', '')).strip()
            if text:
                parts.append(text)
    return '\n'.join(parts).strip()


def _parse_claude_transcript(path: Path) -> tuple[str, str, str, datetime | None]:
    transcript_lines: list[str] = []
    session_id = ''
    title = ''
    created_at: datetime | None = None

    with path.open() as handle:
        for raw_line in handle:
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            record_type = record.get('type')
            if record_type == 'summary':
                continue
            if record_type == 'user':
                message = record.get('message', {})
                text = _extract_claude_text(message.get('content'))
                if text:
                    transcript_lines.append(f'User: {text}')
                    title = title or text.splitlines()[0][:120]
                session_id = session_id or record.get('sessionId', '')
                created_at = created_at or _parse_timestamp(record.get('timestamp'))
            elif record_type == 'assistant':
                message = record.get('message', {})
                text = _extract_claude_text(message.get('content'))
                if text:
                    transcript_lines.append(f'Assistant: {text}')
                session_id = session_id or record.get('sessionId', '')
                created_at = created_at or _parse_timestamp(record.get('timestamp'))

    return _trim_content('\n'.join(transcript_lines)), session_id, title, created_at
