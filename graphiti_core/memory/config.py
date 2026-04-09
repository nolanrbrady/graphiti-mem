from __future__ import annotations

import json
import os
from pathlib import Path

from .models import (
    BackendType,
    ProjectPaths,
    RuntimeConfig,
    build_project_id,
    build_project_paths,
    default_index_state,
    ensure_index_state_shape,
)

GRAPHITI_BLOCK_START = '<!-- graphiti:managed:start -->'
GRAPHITI_BLOCK_END = '<!-- graphiti:managed:end -->'


def detect_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / '.graphiti').exists():
            return candidate
        if (candidate / 'pyproject.toml').exists():
            return candidate
        if (candidate / 'AGENTS.md').exists():
            return candidate
        if (candidate / '.git').exists():
            return candidate
    return current


def default_runtime_config(root: Path, backend: BackendType = BackendType.kuzu) -> RuntimeConfig:
    paths = build_project_paths(root)
    return RuntimeConfig(
        project_name=root.name,
        project_id=build_project_id(root),
        backend=backend,
        database_path=str(paths.database_path.relative_to(root)),
        llm_base_url=os.getenv('OPENAI_BASE_URL', ''),
    )


def initialize_project_files(
    root: Path,
    force: bool = False,
    config: RuntimeConfig | None = None,
) -> tuple[ProjectPaths, RuntimeConfig]:
    root = detect_project_root(root)
    paths = build_project_paths(root)
    config = config or default_runtime_config(root)

    paths.state_dir.mkdir(parents=True, exist_ok=True)

    if force or not paths.config_path.exists():
        write_runtime_config(paths.config_path, config)
    else:
        config = load_runtime_config(paths.config_path)
    if force or not paths.index_state_path.exists():
        paths.index_state_path.write_text(json.dumps(default_index_state(), indent=2) + '\n')
    else:
        state = load_index_state(paths.index_state_path)
        save_index_state(paths.index_state_path, state)

    instructions = render_agent_instructions(root)
    if force or not paths.agent_instructions_path.exists():
        paths.agent_instructions_path.write_text(instructions)

    return paths, config


def _parse_scalar(raw_value: str) -> str:
    value = raw_value.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1].replace('\\"', '"').replace('\\\\', '\\')
    return value


def load_runtime_config(config_path: Path) -> RuntimeConfig:
    if not config_path.exists():
        raise FileNotFoundError(f'Graphiti config not found: {config_path}')

    data: dict[str, dict[str, str]] = {}
    current_section = ''

    for raw_line in config_path.read_text().splitlines():
        line = raw_line.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1].strip()
            data.setdefault(current_section, {})
            continue
        if '=' not in line or current_section == '':
            continue

        key, raw_value = line.split('=', 1)
        data[current_section][key.strip()] = _parse_scalar(raw_value)

    project = data.get('project', {})
    storage = data.get('storage', {})
    llm = data.get('llm', {})
    embedder = data.get('embedder', {})
    neo4j = data.get('neo4j', {})

    return RuntimeConfig(
        project_name=project.get('name', config_path.parent.parent.name),
        project_id=project.get('id', build_project_id(config_path.parent.parent)),
        backend=BackendType(project.get('backend', BackendType.kuzu.value)),
        database_path=storage.get('database_path', '.graphiti/memory.kuzu'),
        llm_model=llm.get('model', 'gpt-4.1-mini'),
        llm_small_model=llm.get('small_model', 'gpt-4.1-nano'),
        llm_base_url=llm.get('base_url', ''),
        llm_api_key_env=llm.get('api_key_env', 'OPENAI_API_KEY'),
        embedder_model=embedder.get('model', 'text-embedding-3-small'),
        embedder_base_url=embedder.get('base_url', ''),
        embedder_api_key_env=embedder.get('api_key_env', 'OPENAI_API_KEY'),
        neo4j_uri_env=neo4j.get('uri_env', 'NEO4J_URI'),
        neo4j_user_env=neo4j.get('user_env', 'NEO4J_USER'),
        neo4j_password_env=neo4j.get('password_env', 'NEO4J_PASSWORD'),
        neo4j_database=neo4j.get('database', 'neo4j'),
    )


def _quote(value: str) -> str:
    escaped = value.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'


def write_runtime_config(config_path: Path, config: RuntimeConfig) -> None:
    contents = '\n'.join(
        [
            '[project]',
            f'name = {_quote(config.project_name)}',
            f'id = {_quote(config.project_id)}',
            f'backend = {_quote(config.backend.value)}',
            '',
            '[storage]',
            f'database_path = {_quote(config.database_path)}',
            '',
            '[llm]',
            f'model = {_quote(config.llm_model)}',
            f'small_model = {_quote(config.llm_small_model)}',
            f'base_url = {_quote(config.llm_base_url)}',
            f'api_key_env = {_quote(config.llm_api_key_env)}',
            '',
            '[embedder]',
            f'model = {_quote(config.embedder_model)}',
            f'base_url = {_quote(config.embedder_base_url)}',
            f'api_key_env = {_quote(config.embedder_api_key_env)}',
            '',
            '[neo4j]',
            f'uri_env = {_quote(config.neo4j_uri_env)}',
            f'user_env = {_quote(config.neo4j_user_env)}',
            f'password_env = {_quote(config.neo4j_password_env)}',
            f'database = {_quote(config.neo4j_database)}',
            '',
        ]
    )
    config_path.write_text(contents)


def render_agent_instructions(root: Path) -> str:
    state_dir = (root / '.graphiti').relative_to(root)
    return '\n'.join(
        [
            GRAPHITI_BLOCK_START,
            '## Graphiti Local Memory',
            '',
            f'This repository uses Graphiti local memory at `{state_dir}`.',
            '',
            'Preferred interface:',
            '- If the local Graphiti MCP server is available, use `graphiti mcp --transport stdio` and call `init_project`, `discover_history`, `list_history_sessions`, `read_history_session`, `import_history_sessions`, `apply_agents_instructions`, `store_memory`, `recall_memory`, `index_project`, and `doctor` through MCP.',
            '- Use MCP as the product path. Use the CLI only as a fallback or local dev/test surface.',
            '',
            'MCP workflow:',
            '- If Graphiti is not initialized in this repo, call `init_project` first.',
            '- Before meaningful exploration, call `recall_memory` with the current task.',
            '- When you discover durable knowledge, call `store_memory` with an allowed kind and a concise summary.',
            '- After major changes or on first setup, call `index_project`.',
            '- During onboarding, use `discover_history`, `list_history_sessions`, `read_history_session`, and `import_history_sessions` to register prior Codex or Claude sessions as source evidence.',
            '- The agent should do the reasoning and decide what durable memory to write; Graphiti stores and retrieves local project memory.',
            '',
            'CLI fallback:',
            '- Before meaningful exploration, run `graphiti recall "<current task>"`.',
            '- When you discover durable knowledge, run `graphiti remember --kind <kind> --summary "<summary>" --details "<details>"`.',
            '- After major changes or on first setup, run `graphiti index --changed`.',
            '',
            'Allowed kinds:',
            '- decision',
            '- constraint',
            '- pattern',
            '- implementation_note',
            '- workflow',
            '- pitfall',
            '- index_artifact',
            '',
            'Prefer Graphiti recall over repeated broad bash search when you are orienting to a task.',
            GRAPHITI_BLOCK_END,
        ]
    )


def apply_agent_instructions(root: Path) -> Path:
    root = detect_project_root(root)
    instructions = render_agent_instructions(root)
    agents_path = root / 'AGENTS.md'
    if not agents_path.exists():
        agents_path.write_text(f'# Agent Instructions\n\n{instructions}\n')
        return agents_path

    current = agents_path.read_text()
    if GRAPHITI_BLOCK_START in current and GRAPHITI_BLOCK_END in current:
        start = current.index(GRAPHITI_BLOCK_START)
        end = current.index(GRAPHITI_BLOCK_END) + len(GRAPHITI_BLOCK_END)
        updated = current[:start].rstrip() + '\n\n' + instructions + '\n'
        if current[end:].strip():
            updated += '\n' + current[end:].lstrip('\n')
        agents_path.write_text(updated)
        return agents_path

    base = current.rstrip()
    if base:
        base += '\n\n'
    agents_path.write_text(base + instructions + '\n')
    return agents_path


def load_index_state(state_path: Path) -> dict:
    if not state_path.exists():
        return default_index_state()
    return ensure_index_state_shape(json.loads(state_path.read_text()))


def save_index_state(state_path: Path, state: dict) -> None:
    state_path.write_text(
        json.dumps(ensure_index_state_shape(state), indent=2, sort_keys=True) + '\n'
    )
