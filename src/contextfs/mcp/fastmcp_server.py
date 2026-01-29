"""
FastMCP Server for ContextFS.

A cleaner implementation using FastMCP's decorator-based API.
Supports SSE and Streamable HTTP transports for multi-client compatibility.

Usage:
    contextfs server start mcp --port 8003

Environment Variables:
    CONTEXTFS_MCP_HOST - Server host (default: 127.0.0.1)
    CONTEXTFS_MCP_PORT - Server port (default: 8003)
    CONTEXTFS_MCP_SSE_PATH - SSE endpoint path (default: /sse)

Client Configuration (Claude Code, Gemini CLI, Cursor, etc.):
    {"mcpServers": {"contextfs": {"type": "sse", "url": "http://localhost:8003/sse"}}}
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from contextfs.cli.utils import (
    create_repo_config,
    find_git_root,
    is_repo_initialized,
)
from contextfs.config import get_config
from contextfs.core import ContextFS
from contextfs.schemas import TYPE_SCHEMAS, MemoryType, get_memory_type_values, get_type_schema

# Disable tokenizers parallelism to avoid deadlocks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger(__name__)

# Memory type enum values for documentation
MEMORY_TYPE_ENUM = get_memory_type_values()

# Global state
_ctx: ContextFS | None = None
_source_tool: str | None = None
_session_started: bool = False


@dataclass
class IndexingState:
    """Track background indexing state."""

    running: bool = False
    repo_name: str = ""
    current_file: str = ""
    current: int = 0
    total: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None
    task: asyncio.Task | None = field(default=None, repr=False)


_indexing_state = IndexingState()


def get_ctx() -> ContextFS:
    """Get or create ContextFS instance."""
    global _ctx, _session_started
    if _ctx is None:
        logger.info("Initializing ContextFS instance...")
        _ctx = ContextFS(auto_load=True)
        logger.info("ContextFS initialized successfully")

    if not _session_started:
        tool = get_source_tool()
        logger.info(f"Starting session with tool: {tool}")
        _ctx.start_session(tool=tool)
        _session_started = True

    return _ctx


def get_source_tool() -> str:
    """Get the source tool name."""
    global _source_tool
    if _source_tool is None:
        _source_tool = os.environ.get("CONTEXTFS_SOURCE_TOOL", "claude-code")
    return _source_tool


def detect_current_repo() -> str | None:
    """Detect current repository from working directory."""
    try:
        import git

        repo = git.Repo(Path.cwd(), search_parent_directories=True)
        return Path(repo.working_tree_dir).name
    except Exception:
        return None


# Get MCP configuration from environment/config
_config = get_config()

# Create FastMCP server with configurable paths
# Default: /sse (FastMCP standard), configurable via CONTEXTFS_MCP_SSE_PATH
mcp = FastMCP(
    name="contextfs",
    instructions="ContextFS - Persistent memory for AI coding agents. Search, save, and recall memories across sessions.",
    host=_config.mcp_host,
    port=_config.mcp_port,
    sse_path=_config.mcp_sse_path,
    message_path=_config.mcp_message_path,
)


# Health check endpoint
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> Response:
    """Health check endpoint."""
    return JSONResponse({"status": "ok", "service": "contextfs-mcp"})


# Tool definitions using FastMCP decorators


@mcp.tool()
async def contextfs_save(
    content: str = "",
    type: str = "fact",
    tags: list[str] | None = None,
    summary: str | None = None,
    project: str | None = None,
    save_session: Literal["current", "previous"] | None = None,
    label: str | None = None,
    structured_data: dict | None = None,
) -> str:
    """Save a memory to ContextFS. Use for facts, decisions, procedures, or session summaries."""
    ctx = get_ctx()

    if save_session:
        session = ctx.get_current_session()
        if session:
            if label:
                session.label = label
            ctx.end_session(generate_summary=True)
            return f"Session saved.\nSession ID: {session.id}\nLabel: {session.label or 'none'}"
        return "No active session to save."

    if not content:
        return "Error: content is required"

    memory_type = MemoryType(type)
    memory = ctx.save(
        content=content,
        type=memory_type,
        tags=tags or [],
        summary=summary,
        source_tool=get_source_tool(),
        source_repo=detect_current_repo(),
        project=project,
        structured_data=structured_data,
    )

    response = f"Memory saved successfully.\nID: {memory.id}\nType: {memory.type.value}"
    if memory.source_repo:
        response += f"\nRepo: {memory.source_repo}"

    # Report auto-links if enabled
    if ctx.config.auto_link_enabled:
        related = ctx.get_related(memory.id, max_depth=1)
        if related:
            response += f"\nAuto-linked: {len(related)} related memories"

    return response


@mcp.tool()
def contextfs_get_type_schema(memory_type: str) -> str:
    """Get the JSON schema for a memory type. Use to understand what structured_data fields are available for each type."""
    if not memory_type:
        return "Error: memory_type is required"

    schema = get_type_schema(memory_type)
    if not schema:
        types_with_schemas = list(TYPE_SCHEMAS.keys())
        return f"No schema defined for type '{memory_type}'.\nTypes with schemas: {', '.join(types_with_schemas)}"

    return f"JSON Schema for type '{memory_type}':\n\n{json.dumps(schema, indent=2)}"


@mcp.tool()
def contextfs_search(
    query: str,
    limit: int = 5,
    type: str | None = None,
    cross_repo: bool = True,
    source_repo: str | None = None,
    source_tool: str | None = None,
    project: str | None = None,
) -> str:
    """Search memories using hybrid search (combines keyword + semantic). Supports cross-repo search."""
    ctx = get_ctx()
    type_filter = MemoryType(type) if type else None

    results = ctx.search(
        query,
        limit=limit,
        type=type_filter,
        cross_repo=cross_repo,
        source_tool=source_tool,
        source_repo=source_repo,
        project=project,
    )

    if not results:
        return "No memories found."

    output = []
    for r in results:
        line = f"[{r.memory.id}] ({r.score:.2f}) [{r.memory.type.value}]"
        if r.memory.project:
            line += f" [{r.memory.project}]"
        if r.memory.source_repo:
            line += f" @{r.memory.source_repo}"
        output.append(line)
        if r.memory.summary:
            output.append(f"  Summary: {r.memory.summary}")
        output.append(f"  {r.memory.content[:200]}...")
        output.append("")

    return "\n".join(output)


@mcp.tool()
def contextfs_list_repos() -> str:
    """List all repositories with saved memories"""
    ctx = get_ctx()
    repos = ctx.list_repos()
    indexes = ctx.list_indexes()

    output = []
    if repos:
        output.append("Repositories with memories:")
        for r in repos:
            output.append(f"  - {r['source_repo']} ({r['memory_count']} memories)")
    else:
        output.append("No repositories with memories found.")

    output.append("")
    if indexes:
        output.append("Indexed repositories:")
        for idx in indexes:
            repo_name = idx.repo_path.split("/")[-1] if idx.repo_path else idx.namespace_id
            output.append(
                f"  - {repo_name} ({idx.files_indexed} files, {idx.commits_indexed} commits)"
            )

    return "\n".join(output)


@mcp.tool()
def contextfs_list_tools() -> str:
    """List all source tools (Claude, Gemini, etc.) with saved memories"""
    ctx = get_ctx()
    tools = ctx.list_tools()
    if not tools:
        return "No source tools found."

    output = ["Source tools with memories:"]
    for t in tools:
        output.append(f"  - {t['source_tool']} ({t['memory_count']} memories)")
    return "\n".join(output)


@mcp.tool()
def contextfs_list_projects() -> str:
    """List all projects with saved memories (projects group memories across repos)"""
    ctx = get_ctx()
    projects = ctx.list_projects()
    if not projects:
        return "No projects found."

    output = ["Projects with memories:"]
    for p in projects:
        repos_str = ", ".join(p["repos"]) if p["repos"] else "no repos"
        output.append(f"  - {p['project']} ({p['memory_count']} memories)")
        output.append(f"    Repos: {repos_str}")
    return "\n".join(output)


@mcp.tool()
def contextfs_recall(id: str) -> str:
    """Recall a specific memory by ID"""
    ctx = get_ctx()
    memory = ctx.recall(id)

    if not memory:
        return f"Memory not found: {id}"

    output = [
        f"ID: {memory.id}",
        f"Type: {memory.type.value}",
        f"Created: {memory.created_at.isoformat()}",
    ]
    if memory.source_tool:
        output.append(f"Source: {memory.source_tool}")
    if memory.source_repo:
        output.append(f"Repo: {memory.source_repo}")
    if memory.project:
        output.append(f"Project: {memory.project}")
    if memory.summary:
        output.append(f"Summary: {memory.summary}")
    if memory.tags:
        output.append(f"Tags: {', '.join(memory.tags)}")
    output.append(f"\nContent:\n{memory.content}")

    if memory.structured_data:
        output.append("\nStructured Data:")
        output.append(json.dumps(memory.structured_data, indent=2))

    return "\n".join(output)


@mcp.tool()
def contextfs_list(
    limit: int = 10,
    type: str | None = None,
    source_tool: str | None = None,
    project: str | None = None,
) -> str:
    """List recent memories"""
    ctx = get_ctx()
    type_filter = MemoryType(type) if type else None

    memories = ctx.list_recent(
        limit=limit, type=type_filter, source_tool=source_tool, project=project
    )

    if not memories:
        return "No memories found."

    output = []
    for m in memories:
        line = f"[{m.id[:8]}] [{m.type.value}]"
        if m.project:
            line += f" [{m.project}]"
        if m.source_repo:
            line += f" @{m.source_repo}"
        output.append(line)
        if m.summary:
            output.append(f"  {m.summary}")
        else:
            output.append(f"  {m.content[:60]}...")
        output.append("")

    return "\n".join(output)


@mcp.tool()
def contextfs_sessions(
    limit: int = 10,
    label: str | None = None,
    tool: str | None = None,
) -> str:
    """List recent sessions"""
    ctx = get_ctx()
    sessions = ctx.list_sessions(limit=limit, label=label, tool=tool, all_namespaces=True)

    if not sessions:
        return "No sessions found."

    output = []
    for s in sessions:
        output.append(f"[{s.id[:8]}] {s.label or '(no label)'}")
        output.append(
            f"  Tool: {s.tool or 'unknown'}, Messages: {len(s.messages) if s.messages else 0}"
        )
        if s.summary:
            output.append(f"  {s.summary[:80]}...")
        output.append("")

    return "\n".join(output)


@mcp.tool()
def contextfs_load_session(
    session_id: str | None = None,
    label: str | None = None,
    max_messages: int = 20,
) -> str:
    """Load a session's messages into context"""
    ctx = get_ctx()
    session = ctx.load_session(session_id=session_id, label=label)
    if not session:
        return "Session not found."

    output = [
        f"Session: {session.id}",
        f"Label: {session.label or '(none)'}",
        f"Tool: {session.tool or 'unknown'}",
    ]
    if session.summary:
        output.append(f"Summary: {session.summary}")
    output.append("")

    if session.messages:
        output.append(
            f"Messages ({min(len(session.messages), max_messages)} of {len(session.messages)}):"
        )
        for msg in session.messages[:max_messages]:
            output.append(f"  [{msg.role}]: {msg.content[:100]}...")

    return "\n".join(output)


@mcp.tool()
def contextfs_message(role: str, content: str) -> str:
    """Add a message to the current session"""
    if not role or not content:
        return "Error: role and content are required"

    ctx = get_ctx()
    ctx.add_message(role=role, content=content)
    return "Message added to session."


@mcp.tool()
def contextfs_init(
    repo_path: str | None = None,
    auto_index: bool = True,
    run_index: bool = True,
    force: bool = False,
    max_commits: int = 100,
) -> str:
    """Initialize a repository for ContextFS indexing. Creates .contextfs/config.yaml marker file to opt-in this repo for automatic indexing."""
    # Determine repo path
    start_path = Path(repo_path) if repo_path else Path.cwd()
    git_root = find_git_root(start_path)

    if not git_root:
        return f"Error: Not a git repository: {start_path.resolve()}"

    # Check if already initialized
    if is_repo_initialized(git_root) and not force:
        return f"Repository already initialized: {git_root}\n(Use force=true to reinitialize)"

    # Create config
    config_path = create_repo_config(
        repo_path=git_root,
        auto_index=auto_index,
        created_by="mcp",
        max_commits=max_commits,
    )

    output = [f"Repository initialized: {git_root.name}", f"Config: {config_path}"]
    if auto_index:
        output.append("Auto-index: enabled")
    else:
        output.append("Auto-index: disabled")

    # Run index if requested
    if run_index:
        ctx = get_ctx()
        result = ctx.index_repository(repo_path=git_root, incremental=True)
        output.append(
            f"Indexed: {result.get('files_indexed', 0)} files, {result.get('commits_indexed', 0)} commits"
        )

    return "\n".join(output)


@mcp.tool()
async def contextfs_index(
    repo_path: str | None = None,
    incremental: bool = True,
    mode: Literal["all", "files_only", "commits_only"] = "all",
    force: bool = False,
) -> str:
    """Start indexing a repository's codebase in background. Defaults to current directory, or specify repo_path for any repository. Use contextfs_index_status to check progress."""
    global _indexing_state

    if _indexing_state.running:
        return f"Indexing already in progress: {_indexing_state.repo_name}\nProgress: {_indexing_state.current}/{_indexing_state.total}\nCurrent: {_indexing_state.current_file}"

    ctx = get_ctx()
    path = repo_path or str(Path.cwd())

    # Handle force flag: clear existing index and do full re-index
    if force:
        ctx.clear_index()
        incremental = False

    _indexing_state.running = True
    _indexing_state.repo_name = Path(path).name
    _indexing_state.current = 0
    _indexing_state.total = 0
    _indexing_state.current_file = ""
    _indexing_state.error = None
    _indexing_state.result = None

    def on_progress(current: int, total: int, filename: str) -> None:
        """Progress callback that updates global indexing state."""
        _indexing_state.current = current
        _indexing_state.total = total
        _indexing_state.current_file = filename

    async def do_index():
        global _indexing_state
        try:
            result = await asyncio.to_thread(
                ctx.index_repository,
                repo_path=Path(path),
                on_progress=on_progress,
                incremental=incremental,
                mode=mode,
            )
            _indexing_state.result = result
        except Exception as e:
            _indexing_state.error = str(e)
        finally:
            _indexing_state.running = False

    _indexing_state.task = asyncio.create_task(do_index())

    return f"Indexing started for {_indexing_state.repo_name}...\nUse contextfs_index_status to check progress."


@mcp.tool()
def contextfs_index_status(cancel: bool = False) -> str:
    """Check or cancel background indexing operation"""
    global _indexing_state

    if cancel and _indexing_state.task:
        _indexing_state.task.cancel()
        _indexing_state.running = False
        return "Indexing cancelled."

    if _indexing_state.running:
        return f"Indexing in progress: {_indexing_state.repo_name}\nProgress: {_indexing_state.current}/{_indexing_state.total}\nCurrent: {_indexing_state.current_file}"

    if _indexing_state.error:
        return f"Indexing failed: {_indexing_state.error}"

    if _indexing_state.result:
        r = _indexing_state.result
        return f"Indexing complete: {_indexing_state.repo_name}\nFiles: {r.get('files_indexed', 0)}\nCommits: {r.get('commits_indexed', 0)}\nMemories: {r.get('memories_created', 0)}"

    return "No indexing operation in progress."


@mcp.tool()
def contextfs_list_indexes() -> str:
    """List all indexed repositories with full status including files, commits, memories, and timestamps"""
    ctx = get_ctx()
    indexes = ctx.list_indexes()

    if not indexes:
        return "No indexed repositories found."

    output = ["Indexed repositories:"]
    for idx in indexes:
        repo_name = idx.repo_path.split("/")[-1] if idx.repo_path else idx.namespace_id
        output.append(f"\n{repo_name}:")
        output.append(f"  Path: {idx.repo_path}")
        output.append(f"  Files: {idx.files_indexed}, Commits: {idx.commits_indexed}")
        output.append(f"  Memories: {idx.memories_created}")
        if idx.indexed_at:
            output.append(f"  Last indexed: {idx.indexed_at}")

    return "\n".join(output)


@mcp.tool()
def contextfs_update(
    id: str,
    content: str | None = None,
    summary: str | None = None,
    tags: list[str] | None = None,
    type: str | None = None,
    project: str | None = None,
) -> str:
    """Update an existing memory"""
    ctx = get_ctx()

    updates: dict[str, Any] = {}
    if content:
        updates["content"] = content
    if summary:
        updates["summary"] = summary
    if tags:
        updates["tags"] = tags
    if type:
        updates["type"] = MemoryType(type)
    if project:
        updates["project"] = project

    memory = ctx.update(id, **updates)
    if not memory:
        return f"Memory not found: {id}"

    return f"Memory updated: {memory.id}"


@mcp.tool()
def contextfs_delete(id: str) -> str:
    """Delete a memory"""
    ctx = get_ctx()

    success = ctx.delete(id)
    if not success:
        return f"Memory not found: {id}"

    return f"Memory deleted: {id}"


@mcp.tool()
def contextfs_evolve(
    memory_id: str,
    new_content: str,
    summary: str | None = None,
    preserve_tags: bool = True,
    additional_tags: list[str] | None = None,
) -> str:
    """Update a memory with history tracking. Creates a new version while preserving the original. Use when knowledge evolves or needs correction."""
    ctx = get_ctx()

    new_memory = ctx.evolve(
        memory_id=memory_id,
        new_content=new_content,
        summary=summary,
        preserve_tags=preserve_tags,
        additional_tags=additional_tags,
    )

    if not new_memory:
        return f"Memory not found: {memory_id}"

    return f"Memory evolved.\nOriginal: {memory_id}\nNew: {new_memory.id}"


@mcp.tool()
def contextfs_link(
    from_id: str,
    to_id: str,
    relation: str,
    weight: float = 1.0,
    bidirectional: bool = False,
) -> str:
    """Create a relationship between two memories. Use for references, dependencies, contradictions, and other relationships."""
    ctx = get_ctx()

    ctx.link(
        from_memory_id=from_id,
        to_memory_id=to_id,
        relation=relation,
        weight=weight,
        bidirectional=bidirectional,
    )

    direction = "bidirectional" if bidirectional else "unidirectional"
    return f"Link created ({direction}): {from_id} --[{relation}]--> {to_id}"


def _get_cloud_config() -> dict:
    """Get cloud configuration from config file."""
    import yaml

    config_path = Path.home() / ".contextfs" / "config.yaml"
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    return config.get("cloud", {})


@mcp.tool()
async def contextfs_sync(
    direction: Literal["push", "pull", "both"] = "both",
    push_all: bool = False,
    force: bool = False,
) -> str:
    """Sync local memories with ContextFS Cloud. Requires cloud login (contextfs cloud login). Use for backup and cross-device access."""
    import time

    from contextfs.sync import SyncClient

    # Get cloud config for server URL and API key
    cloud_config = _get_cloud_config()
    if not cloud_config.get("enabled"):
        return "Cloud sync is disabled. Run: contextfs cloud configure --enabled"

    server_url = cloud_config.get("server_url", "https://api.contextfs.ai")
    api_key = cloud_config.get("api_key")

    if not api_key:
        return "No API key configured. Run: contextfs cloud login"

    ctx = get_ctx()
    start = time.time()
    try:
        async with SyncClient(server_url=server_url, ctx=ctx, api_key=api_key) as client:
            if direction == "push":
                result = await client.push(push_all=push_all, force=force)
                duration_ms = (time.time() - start) * 1000
                output = [
                    f"Push complete ({duration_ms:.0f}ms).",
                    f"Accepted: {result.accepted}",
                    f"Rejected: {result.rejected}",
                ]
                if result.conflicts:
                    output.append(f"Conflicts: {len(result.conflicts)}")
                if result.pushed_items:
                    output.append("")
                    output.append("Pushed:")
                    for item in result.pushed_items[:10]:
                        output.append(f"  [{item.type}] {item.summary or item.id[:8]}")
                    if len(result.pushed_items) > 10:
                        output.append(f"  ... and {len(result.pushed_items) - 10} more")
                return "\n".join(output)

            elif direction == "pull":
                result = await client.pull()
                duration_ms = (time.time() - start) * 1000
                output = [
                    f"Pull complete ({duration_ms:.0f}ms).",
                    f"Memories: {len(result.memories)}",
                    f"Sessions: {len(result.sessions)}",
                ]
                # Handle optional deleted_ids attribute
                deleted_ids = getattr(result, "deleted_ids", None)
                if deleted_ids:
                    output.append(f"Deleted: {len(deleted_ids)}")
                if result.memories:
                    output.append("")
                    output.append("Pulled:")
                    for m in result.memories[:10]:
                        summary = m.summary or (
                            m.content[:50] + "..." if len(m.content) > 50 else m.content
                        )
                        output.append(f"  [{m.type}] {summary}")
                    if len(result.memories) > 10:
                        output.append(f"  ... and {len(result.memories) - 10} more")
                return "\n".join(output)

            else:  # both
                push_result = await client.push(push_all=push_all, force=force)
                pull_result = await client.pull()
                duration_ms = (time.time() - start) * 1000

                output = [f"Sync complete ({duration_ms:.0f}ms).", ""]
                output.append(
                    f"Pushed: {push_result.accepted} accepted, {push_result.rejected} rejected"
                )
                if push_result.pushed_items:
                    for item in push_result.pushed_items[:5]:
                        output.append(f"    [{item.type}] {item.summary or item.id[:8]}")
                    if len(push_result.pushed_items) > 5:
                        output.append(f"    ... and {len(push_result.pushed_items) - 5} more")

                output.append(
                    f"Pulled: {len(pull_result.memories)} memories, {len(pull_result.sessions)} sessions"
                )
                if pull_result.memories:
                    for m in pull_result.memories[:5]:
                        summary = m.summary or (
                            m.content[:50] + "..." if len(m.content) > 50 else m.content
                        )
                        output.append(f"    [{m.type}] {summary}")
                    if len(pull_result.memories) > 5:
                        output.append(f"    ... and {len(pull_result.memories) - 5} more")

                return "\n".join(output)
    except Exception as e:
        logger.exception("Sync failed")
        error_type = type(e).__name__
        error_msg = str(e) or "(no message)"
        return f"Sync failed [{error_type}]: {error_msg}"


def run_mcp_server(host: str | None = None, port: int | None = None) -> None:
    """Run the MCP server with SSE transport."""
    import uvicorn

    config = get_config()

    # Use provided values or fall back to config
    host = host or config.mcp_host
    port = port or config.mcp_port
    sse_path = config.mcp_sse_path

    # Update FastMCP settings
    mcp.settings.host = host
    mcp.settings.port = port

    print("Starting ContextFS MCP Server (FastMCP)...")
    print(f"  URL: http://{host}:{port}")
    print(f"  SSE Endpoint: http://{host}:{port}{sse_path}")
    print(f"  Health Check: http://{host}:{port}/health")
    print()
    print("Configure Claude Code / Gemini CLI / Cursor with:")
    print(
        f'  {{"mcpServers": {{"contextfs": {{"type": "sse", "url": "http://{host}:{port}{sse_path}"}}}}}}'
    )
    print()

    # Create SSE app and run with uvicorn
    app = mcp.sse_app()

    uvicorn.run(app, host=host, port=port, log_level="info")


def create_mcp_app():
    """Create the MCP Starlette application for SSE transport."""
    return mcp.sse_app()


if __name__ == "__main__":
    import argparse

    config = get_config()

    parser = argparse.ArgumentParser(description="ContextFS MCP Server")
    parser.add_argument(
        "--host", default=None, help=f"Host to bind to (default: {config.mcp_host})"
    )
    parser.add_argument(
        "--port", type=int, default=None, help=f"Port to bind to (default: {config.mcp_port})"
    )
    args = parser.parse_args()

    run_mcp_server(host=args.host, port=args.port)
