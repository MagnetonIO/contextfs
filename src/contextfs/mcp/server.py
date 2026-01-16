"""
MCP Server for ContextFS using HTTP/SSE transport.

Provides a single shared MCP service that all Claude Code instances connect to.
This replaces the stdio-based server with a more robust HTTP/SSE approach.

Usage:
    contextfs mcp-server --port 8003

Claude Code Configuration:
    {"mcpServers": {"contextfs": {"type": "sse", "url": "http://localhost:8003/mcp/sse"}}}
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from contextfs.core import ContextFS
from contextfs.schemas import TYPE_SCHEMAS, MemoryType, get_memory_type_values, get_type_schema

# Disable tokenizers parallelism to avoid deadlocks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger(__name__)

# Memory type enum values
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


# Create MCP server
mcp_server = Server("contextfs")


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="contextfs_save",
            description="Save a memory to ContextFS. Use for facts, decisions, procedures, or session summaries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to save"},
                    "type": {
                        "type": "string",
                        "enum": MEMORY_TYPE_ENUM,
                        "description": "Memory type",
                        "default": "fact",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization",
                    },
                    "summary": {"type": "string", "description": "Brief summary"},
                    "project": {
                        "type": "string",
                        "description": "Project name for grouping memories across repos",
                    },
                    "save_session": {
                        "type": "string",
                        "enum": ["current", "previous"],
                        "description": "Save session instead of memory",
                    },
                    "label": {"type": "string", "description": "Label for session"},
                    "structured_data": {
                        "type": "object",
                        "description": "Optional structured data validated against the type's JSON schema.",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="contextfs_get_type_schema",
            description="Get the JSON schema for a memory type. Use to understand what structured_data fields are available for each type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "enum": MEMORY_TYPE_ENUM,
                        "description": "Memory type to get schema for",
                    },
                },
                "required": ["memory_type"],
            },
        ),
        Tool(
            name="contextfs_search",
            description="Search memories using hybrid search (combines keyword + semantic). Supports cross-repo search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "number", "description": "Maximum results", "default": 5},
                    "type": {
                        "type": "string",
                        "enum": MEMORY_TYPE_ENUM,
                        "description": "Filter by type",
                    },
                    "cross_repo": {
                        "type": "boolean",
                        "description": "Search across all repos (default: true)",
                        "default": True,
                    },
                    "source_tool": {"type": "string", "description": "Filter by source tool"},
                    "source_repo": {
                        "type": "string",
                        "description": "Filter by source repository name",
                    },
                    "project": {"type": "string", "description": "Filter by project name"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="contextfs_list_repos",
            description="List all repositories with saved memories",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="contextfs_list_tools",
            description="List all source tools (Claude, Gemini, etc.) with saved memories",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="contextfs_list_projects",
            description="List all projects with saved memories (projects group memories across repos)",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="contextfs_recall",
            description="Recall a specific memory by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Memory ID (can be partial, at least 8 chars)",
                    },
                },
                "required": ["id"],
            },
        ),
        Tool(
            name="contextfs_list",
            description="List recent memories",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "number", "description": "Maximum results", "default": 10},
                    "type": {
                        "type": "string",
                        "enum": MEMORY_TYPE_ENUM,
                        "description": "Filter by memory type",
                    },
                    "source_tool": {"type": "string", "description": "Filter by source tool"},
                    "project": {"type": "string", "description": "Filter by project name"},
                },
            },
        ),
        Tool(
            name="contextfs_sessions",
            description="List recent sessions",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "number", "description": "Maximum results", "default": 10},
                    "label": {"type": "string", "description": "Filter by label"},
                    "tool": {"type": "string", "description": "Filter by tool"},
                },
            },
        ),
        Tool(
            name="contextfs_load_session",
            description="Load a session's messages into context",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session ID (can be partial)"},
                    "label": {"type": "string", "description": "Session label"},
                    "max_messages": {
                        "type": "number",
                        "description": "Maximum messages to return",
                        "default": 20,
                    },
                },
            },
        ),
        Tool(
            name="contextfs_message",
            description="Add a message to the current session",
            inputSchema={
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["user", "assistant", "system"],
                        "description": "Message role",
                    },
                    "content": {"type": "string", "description": "Message content"},
                },
                "required": ["role", "content"],
            },
        ),
        Tool(
            name="contextfs_init",
            description="Initialize a repository for ContextFS indexing. Creates .contextfs/config.yaml marker file to opt-in this repo for automatic indexing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to repository to initialize. Defaults to current directory.",
                    },
                    "auto_index": {
                        "type": "boolean",
                        "description": "Enable automatic indexing on session start (default: true)",
                        "default": True,
                    },
                    "run_index": {
                        "type": "boolean",
                        "description": "Run indexing immediately after init (default: true)",
                        "default": True,
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Reinitialize even if already initialized",
                        "default": False,
                    },
                    "max_commits": {
                        "type": "number",
                        "description": "Maximum commits to index (default: 100)",
                        "default": 100,
                    },
                },
            },
        ),
        Tool(
            name="contextfs_index",
            description="Start indexing a repository's codebase in background. Defaults to current directory, or specify repo_path for any repository. Use contextfs_index_status to check progress.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Path to repository to index. Defaults to current directory.",
                    },
                    "incremental": {
                        "type": "boolean",
                        "description": "Only index new/changed files (default: true)",
                        "default": True,
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["all", "files_only", "commits_only"],
                        "description": "Index mode",
                        "default": "all",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Force re-index even if already indexed",
                        "default": False,
                    },
                },
            },
        ),
        Tool(
            name="contextfs_index_status",
            description="Check or cancel background indexing operation",
            inputSchema={
                "type": "object",
                "properties": {
                    "cancel": {
                        "type": "boolean",
                        "description": "Set to true to cancel the running indexing operation",
                        "default": False,
                    },
                },
            },
        ),
        Tool(
            name="contextfs_list_indexes",
            description="List all indexed repositories with full status including files, commits, memories, and timestamps",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="contextfs_update",
            description="Update an existing memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Memory ID (can be partial, at least 8 chars)",
                    },
                    "content": {"type": "string", "description": "New content (optional)"},
                    "summary": {"type": "string", "description": "New summary (optional)"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New tags (optional)",
                    },
                    "type": {
                        "type": "string",
                        "enum": MEMORY_TYPE_ENUM,
                        "description": "New memory type (optional)",
                    },
                    "project": {"type": "string", "description": "New project name (optional)"},
                },
                "required": ["id"],
            },
        ),
        Tool(
            name="contextfs_delete",
            description="Delete a memory",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "Memory ID (can be partial, at least 8 chars)",
                    },
                },
                "required": ["id"],
            },
        ),
        Tool(
            name="contextfs_evolve",
            description="Update a memory with history tracking. Creates a new version while preserving the original. Use when knowledge evolves or needs correction.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory to evolve (can be partial, at least 8 chars)",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "Updated content for the new version",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Optional summary for the new version",
                    },
                    "preserve_tags": {
                        "type": "boolean",
                        "description": "Whether to preserve original tags (default: true)",
                        "default": True,
                    },
                    "additional_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional tags to add to the new version",
                    },
                },
                "required": ["memory_id", "new_content"],
            },
        ),
        Tool(
            name="contextfs_link",
            description="Create a relationship between two memories. Use for references, dependencies, contradictions, and other relationships.",
            inputSchema={
                "type": "object",
                "properties": {
                    "from_id": {
                        "type": "string",
                        "description": "ID of the source memory (can be partial, at least 8 chars)",
                    },
                    "to_id": {
                        "type": "string",
                        "description": "ID of the target memory (can be partial, at least 8 chars)",
                    },
                    "relation": {
                        "type": "string",
                        "enum": [
                            "references",
                            "depends_on",
                            "contradicts",
                            "supports",
                            "supersedes",
                            "related_to",
                            "derived_from",
                            "example_of",
                            "part_of",
                            "implements",
                        ],
                        "description": "Type of relationship between the memories",
                    },
                    "weight": {
                        "type": "number",
                        "description": "Relationship strength (0.0-1.0, default: 1.0)",
                        "default": 1,
                    },
                    "bidirectional": {
                        "type": "boolean",
                        "description": "Create link in both directions (default: false)",
                        "default": False,
                    },
                },
                "required": ["from_id", "to_id", "relation"],
            },
        ),
        Tool(
            name="contextfs_sync",
            description="Sync local memories with ContextFS Cloud. Requires cloud login (contextfs cloud login). Use for backup and cross-device access.",
            inputSchema={
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["push", "pull", "both"],
                        "description": "Sync direction: push (local→cloud), pull (cloud→local), both (default)",
                        "default": "both",
                    },
                    "push_all": {
                        "type": "boolean",
                        "description": "Push all memories, not just changed ones (default: false)",
                        "default": False,
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Force overwrite server data regardless of vector clock state. Use to fix stale memories that keep getting rejected.",
                        "default": False,
                    },
                },
            },
        ),
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    global _indexing_state
    ctx = get_ctx()

    try:
        if name == "contextfs_save":
            return await _handle_save(ctx, arguments)
        elif name == "contextfs_get_type_schema":
            return _handle_get_type_schema(arguments)
        elif name == "contextfs_search":
            return _handle_search(ctx, arguments)
        elif name == "contextfs_list_repos":
            return _handle_list_repos(ctx)
        elif name == "contextfs_list_tools":
            return _handle_list_tools(ctx)
        elif name == "contextfs_list_projects":
            return _handle_list_projects(ctx)
        elif name == "contextfs_recall":
            return _handle_recall(ctx, arguments)
        elif name == "contextfs_list":
            return _handle_list(ctx, arguments)
        elif name == "contextfs_sessions":
            return _handle_sessions(ctx, arguments)
        elif name == "contextfs_load_session":
            return _handle_load_session(ctx, arguments)
        elif name == "contextfs_message":
            return _handle_message(ctx, arguments)
        elif name == "contextfs_init":
            return _handle_init(ctx, arguments)
        elif name == "contextfs_index":
            return await _handle_index(ctx, arguments)
        elif name == "contextfs_index_status":
            return _handle_index_status(arguments)
        elif name == "contextfs_list_indexes":
            return _handle_list_indexes(ctx)
        elif name == "contextfs_update":
            return _handle_update(ctx, arguments)
        elif name == "contextfs_delete":
            return _handle_delete(ctx, arguments)
        elif name == "contextfs_evolve":
            return _handle_evolve(ctx, arguments)
        elif name == "contextfs_link":
            return _handle_link(ctx, arguments)
        elif name == "contextfs_sync":
            return await _handle_sync(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


# Tool handlers
async def _handle_save(ctx: ContextFS, arguments: dict) -> list[TextContent]:
    """Handle contextfs_save tool."""
    if arguments.get("save_session"):
        session = ctx.get_current_session()
        if session:
            if arguments.get("label"):
                session.label = arguments["label"]
            ctx.end_session(generate_summary=True)
            return [
                TextContent(
                    type="text",
                    text=f"Session saved.\nSession ID: {session.id}\nLabel: {session.label or 'none'}",
                )
            ]
        return [TextContent(type="text", text="No active session to save.")]

    content = arguments.get("content", "")
    if not content:
        return [TextContent(type="text", text="Error: content is required")]

    memory_type = MemoryType(arguments.get("type", "fact"))
    tags = arguments.get("tags", [])
    summary = arguments.get("summary")
    structured_data = arguments.get("structured_data")
    project = arguments.get("project")
    source_repo = detect_current_repo()

    memory = ctx.save(
        content=content,
        type=memory_type,
        tags=tags,
        summary=summary,
        source_tool=get_source_tool(),
        source_repo=source_repo,
        project=project,
        structured_data=structured_data,
    )

    response = f"Memory saved successfully.\nID: {memory.id}\nType: {memory.type.value}"
    if source_repo:
        response += f"\nRepo: {source_repo}"

    # Report auto-links if enabled
    if ctx.config.auto_link_enabled:
        related = ctx.get_related(memory.id, max_depth=1)
        if related:
            response += f"\nAuto-linked: {len(related)} related memories"

    return [TextContent(type="text", text=response)]


def _handle_get_type_schema(arguments: dict) -> list[TextContent]:
    """Handle contextfs_get_type_schema tool."""
    import json

    memory_type = arguments.get("memory_type", "")
    if not memory_type:
        return [TextContent(type="text", text="Error: memory_type is required")]

    schema = get_type_schema(memory_type)
    if not schema:
        types_with_schemas = list(TYPE_SCHEMAS.keys())
        return [
            TextContent(
                type="text",
                text=f"No schema defined for type '{memory_type}'.\nTypes with schemas: {', '.join(types_with_schemas)}",
            )
        ]

    return [
        TextContent(
            type="text",
            text=f"JSON Schema for type '{memory_type}':\n\n{json.dumps(schema, indent=2)}",
        )
    ]


def _handle_search(ctx: ContextFS, arguments: dict) -> list[TextContent]:
    """Handle contextfs_search tool."""
    query = arguments.get("query", "")
    limit = arguments.get("limit", 5)
    type_filter = MemoryType(arguments["type"]) if arguments.get("type") else None
    cross_repo = arguments.get("cross_repo", True)
    source_tool = arguments.get("source_tool")
    source_repo = arguments.get("source_repo")
    project = arguments.get("project")

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
        return [TextContent(type="text", text="No memories found.")]

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

    return [TextContent(type="text", text="\n".join(output))]


def _handle_list_repos(ctx: ContextFS) -> list[TextContent]:
    """Handle contextfs_list_repos tool."""
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

    return [TextContent(type="text", text="\n".join(output))]


def _handle_list_tools(ctx: ContextFS) -> list[TextContent]:
    """Handle contextfs_list_tools tool."""
    tools = ctx.list_tools()
    if not tools:
        return [TextContent(type="text", text="No source tools found.")]

    output = ["Source tools with memories:"]
    for t in tools:
        output.append(f"  - {t['source_tool']} ({t['memory_count']} memories)")
    return [TextContent(type="text", text="\n".join(output))]


def _handle_list_projects(ctx: ContextFS) -> list[TextContent]:
    """Handle contextfs_list_projects tool."""
    projects = ctx.list_projects()
    if not projects:
        return [TextContent(type="text", text="No projects found.")]

    output = ["Projects with memories:"]
    for p in projects:
        repos_str = ", ".join(p["repos"]) if p["repos"] else "no repos"
        output.append(f"  - {p['project']} ({p['memory_count']} memories)")
        output.append(f"    Repos: {repos_str}")
    return [TextContent(type="text", text="\n".join(output))]


def _handle_recall(ctx: ContextFS, arguments: dict) -> list[TextContent]:
    """Handle contextfs_recall tool."""
    import json

    memory_id = arguments.get("id", "")
    memory = ctx.recall(memory_id)

    if not memory:
        return [TextContent(type="text", text=f"Memory not found: {memory_id}")]

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

    return [TextContent(type="text", text="\n".join(output))]


def _handle_list(ctx: ContextFS, arguments: dict) -> list[TextContent]:
    """Handle contextfs_list tool."""
    limit = arguments.get("limit", 10)
    type_filter = MemoryType(arguments["type"]) if arguments.get("type") else None
    source_tool = arguments.get("source_tool")
    project = arguments.get("project")

    memories = ctx.list_recent(
        limit=limit, type=type_filter, source_tool=source_tool, project=project
    )

    if not memories:
        return [TextContent(type="text", text="No memories found.")]

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

    return [TextContent(type="text", text="\n".join(output))]


def _handle_sessions(ctx: ContextFS, arguments: dict) -> list[TextContent]:
    """Handle contextfs_sessions tool."""
    limit = arguments.get("limit", 10)
    label = arguments.get("label")
    tool = arguments.get("tool")

    sessions = ctx.list_sessions(limit=limit, label=label, tool=tool, all_namespaces=True)

    if not sessions:
        return [TextContent(type="text", text="No sessions found.")]

    output = []
    for s in sessions:
        output.append(f"[{s.id[:8]}] {s.label or '(no label)'}")
        output.append(
            f"  Tool: {s.tool or 'unknown'}, Messages: {len(s.messages) if s.messages else 0}"
        )
        if s.summary:
            output.append(f"  {s.summary[:80]}...")
        output.append("")

    return [TextContent(type="text", text="\n".join(output))]


def _handle_load_session(ctx: ContextFS, arguments: dict) -> list[TextContent]:
    """Handle contextfs_load_session tool."""
    session_id = arguments.get("session_id")
    label = arguments.get("label")
    max_messages = arguments.get("max_messages", 20)

    session = ctx.get_session(session_id=session_id, label=label)
    if not session:
        return [TextContent(type="text", text="Session not found.")]

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

    return [TextContent(type="text", text="\n".join(output))]


def _handle_message(ctx: ContextFS, arguments: dict) -> list[TextContent]:
    """Handle contextfs_message tool."""
    role = arguments.get("role")
    content = arguments.get("content")

    if not role or not content:
        return [TextContent(type="text", text="Error: role and content are required")]

    ctx.add_message(role=role, content=content)
    return [TextContent(type="text", text="Message added to session.")]


def _handle_init(ctx: ContextFS, arguments: dict) -> list[TextContent]:
    """Handle contextfs_init tool."""
    repo_path = arguments.get("repo_path") or str(Path.cwd())
    auto_index = arguments.get("auto_index", True)
    run_index = arguments.get("run_index", True)
    force = arguments.get("force", False)
    max_commits = arguments.get("max_commits", 100)

    result = ctx.init_repository(
        repo_path=Path(repo_path),
        auto_index=auto_index,
        run_index=run_index,
        force=force,
        max_commits=max_commits,
    )

    output = [f"Repository initialized: {repo_path}"]
    if result.get("already_initialized") and not force:
        output.append("(Already initialized, use force=true to reinitialize)")
    if result.get("indexed"):
        output.append(
            f"Indexed: {result.get('files_indexed', 0)} files, {result.get('commits_indexed', 0)} commits"
        )

    return [TextContent(type="text", text="\n".join(output))]


async def _handle_index(ctx: ContextFS, arguments: dict) -> list[TextContent]:
    """Handle contextfs_index tool."""
    global _indexing_state

    if _indexing_state.running:
        return [
            TextContent(
                type="text",
                text=f"Indexing already in progress: {_indexing_state.repo_name}\nProgress: {_indexing_state.current}/{_indexing_state.total}\nCurrent: {_indexing_state.current_file}",
            )
        ]

    repo_path = arguments.get("repo_path") or str(Path.cwd())
    incremental = arguments.get("incremental", True)
    mode = arguments.get("mode", "all")
    force = arguments.get("force", False)

    _indexing_state.running = True
    _indexing_state.repo_name = Path(repo_path).name
    _indexing_state.current = 0
    _indexing_state.total = 0
    _indexing_state.error = None
    _indexing_state.result = None

    async def do_index():
        global _indexing_state
        try:
            result = await asyncio.to_thread(
                ctx.index_repository,
                repo_path=Path(repo_path),
                incremental=incremental,
                mode=mode,
                force=force,
            )
            _indexing_state.result = result
        except Exception as e:
            _indexing_state.error = str(e)
        finally:
            _indexing_state.running = False

    _indexing_state.task = asyncio.create_task(do_index())

    return [
        TextContent(
            type="text",
            text=f"Indexing started for {_indexing_state.repo_name}...\nUse contextfs_index_status to check progress.",
        )
    ]


def _handle_index_status(arguments: dict) -> list[TextContent]:
    """Handle contextfs_index_status tool."""
    global _indexing_state

    if arguments.get("cancel") and _indexing_state.task:
        _indexing_state.task.cancel()
        _indexing_state.running = False
        return [TextContent(type="text", text="Indexing cancelled.")]

    if _indexing_state.running:
        return [
            TextContent(
                type="text",
                text=f"Indexing in progress: {_indexing_state.repo_name}\nProgress: {_indexing_state.current}/{_indexing_state.total}\nCurrent: {_indexing_state.current_file}",
            )
        ]

    if _indexing_state.error:
        return [TextContent(type="text", text=f"Indexing failed: {_indexing_state.error}")]

    if _indexing_state.result:
        r = _indexing_state.result
        return [
            TextContent(
                type="text",
                text=f"Indexing complete: {_indexing_state.repo_name}\nFiles: {r.get('files_indexed', 0)}\nCommits: {r.get('commits_indexed', 0)}\nMemories: {r.get('memories_created', 0)}",
            )
        ]

    return [TextContent(type="text", text="No indexing operation in progress.")]


def _handle_list_indexes(ctx: ContextFS) -> list[TextContent]:
    """Handle contextfs_list_indexes tool."""
    indexes = ctx.list_indexes()

    if not indexes:
        return [TextContent(type="text", text="No indexed repositories found.")]

    output = ["Indexed repositories:"]
    for idx in indexes:
        repo_name = idx.repo_path.split("/")[-1] if idx.repo_path else idx.namespace_id
        output.append(f"\n{repo_name}:")
        output.append(f"  Path: {idx.repo_path}")
        output.append(f"  Files: {idx.files_indexed}, Commits: {idx.commits_indexed}")
        output.append(f"  Memories: {idx.memories_created}")
        if idx.indexed_at:
            output.append(f"  Last indexed: {idx.indexed_at}")

    return [TextContent(type="text", text="\n".join(output))]


def _handle_update(ctx: ContextFS, arguments: dict) -> list[TextContent]:
    """Handle contextfs_update tool."""
    memory_id = arguments.get("id")
    if not memory_id:
        return [TextContent(type="text", text="Error: id is required")]

    updates = {}
    if arguments.get("content"):
        updates["content"] = arguments["content"]
    if arguments.get("summary"):
        updates["summary"] = arguments["summary"]
    if arguments.get("tags"):
        updates["tags"] = arguments["tags"]
    if arguments.get("type"):
        updates["type"] = MemoryType(arguments["type"])
    if arguments.get("project"):
        updates["project"] = arguments["project"]

    memory = ctx.update(memory_id, **updates)
    if not memory:
        return [TextContent(type="text", text=f"Memory not found: {memory_id}")]

    return [TextContent(type="text", text=f"Memory updated: {memory.id}")]


def _handle_delete(ctx: ContextFS, arguments: dict) -> list[TextContent]:
    """Handle contextfs_delete tool."""
    memory_id = arguments.get("id")
    if not memory_id:
        return [TextContent(type="text", text="Error: id is required")]

    success = ctx.delete(memory_id)
    if not success:
        return [TextContent(type="text", text=f"Memory not found: {memory_id}")]

    return [TextContent(type="text", text=f"Memory deleted: {memory_id}")]


def _handle_evolve(ctx: ContextFS, arguments: dict) -> list[TextContent]:
    """Handle contextfs_evolve tool."""
    memory_id = arguments.get("memory_id")
    new_content = arguments.get("new_content")

    if not memory_id or not new_content:
        return [TextContent(type="text", text="Error: memory_id and new_content are required")]

    new_memory = ctx.evolve(
        memory_id=memory_id,
        new_content=new_content,
        summary=arguments.get("summary"),
        preserve_tags=arguments.get("preserve_tags", True),
        additional_tags=arguments.get("additional_tags"),
    )

    if not new_memory:
        return [TextContent(type="text", text=f"Memory not found: {memory_id}")]

    return [
        TextContent(
            type="text", text=f"Memory evolved.\nOriginal: {memory_id}\nNew: {new_memory.id}"
        )
    ]


def _handle_link(ctx: ContextFS, arguments: dict) -> list[TextContent]:
    """Handle contextfs_link tool."""
    from_id = arguments.get("from_id")
    to_id = arguments.get("to_id")
    relation = arguments.get("relation")

    if not from_id or not to_id or not relation:
        return [TextContent(type="text", text="Error: from_id, to_id, and relation are required")]

    ctx.link(
        from_memory_id=from_id,
        to_memory_id=to_id,
        relation=relation,
        weight=arguments.get("weight", 1.0),
        bidirectional=arguments.get("bidirectional", False),
    )

    direction = "bidirectional" if arguments.get("bidirectional") else "unidirectional"
    return [
        TextContent(
            type="text", text=f"Link created ({direction}): {from_id} --[{relation}]--> {to_id}"
        )
    ]


def _get_cloud_config() -> dict:
    """Get cloud configuration from config file."""
    import yaml

    config_path = Path.home() / ".contextfs" / "config.yaml"
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    return config.get("cloud", {})


async def _handle_sync(arguments: dict) -> list[TextContent]:
    """Handle contextfs_sync tool."""
    import time

    from contextfs.sync import SyncClient

    direction = arguments.get("direction", "both")
    push_all = arguments.get("push_all", False)
    force = arguments.get("force", False)

    # Get cloud config for server URL and API key
    cloud_config = _get_cloud_config()
    if not cloud_config.get("enabled"):
        return [
            TextContent(
                type="text", text="Cloud sync is disabled. Run: contextfs cloud configure --enabled"
            )
        ]

    server_url = cloud_config.get("server_url", "https://api.contextfs.ai")
    api_key = cloud_config.get("api_key")

    if not api_key:
        return [TextContent(type="text", text="No API key configured. Run: contextfs cloud login")]

    ctx = get_ctx()
    start = time.time()
    try:
        async with SyncClient(server_url=server_url, ctx=ctx, api_key=api_key) as client:
            if direction == "push":
                result = await client.push(push_all=push_all, force=force)
                duration_ms = (time.time() - start) * 1000
                return [
                    TextContent(
                        type="text",
                        text=(
                            f"Push complete ({duration_ms:.0f}ms).\n"
                            f"Accepted: {result.accepted}\n"
                            f"Rejected: {result.rejected}"
                            + (f"\nConflicts: {len(result.conflicts)}" if result.conflicts else "")
                        ),
                    )
                ]
            elif direction == "pull":
                result = await client.pull()
                duration_ms = (time.time() - start) * 1000
                return [
                    TextContent(
                        type="text",
                        text=(
                            f"Pull complete.\n"
                            f"Memories: {len(result.memories)}\n"
                            f"Sessions: {len(result.sessions)}"
                            + (
                                f"\nDeleted: {len(result.deleted_ids)}"
                                if result.deleted_ids
                                else ""
                            )
                        ),
                    )
                ]
            else:  # both
                # First push
                push_result = await client.push(push_all=push_all, force=force)
                # Then pull
                pull_result = await client.pull()
                duration_ms = (time.time() - start) * 1000

                return [
                    TextContent(
                        type="text",
                        text=(
                            f"Sync complete ({duration_ms:.0f}ms).\n"
                            f"Pushed: {push_result.accepted} accepted, {push_result.rejected} rejected\n"
                            f"Pulled: {len(pull_result.memories)} memories, {len(pull_result.sessions)} sessions"
                        ),
                    )
                ]
    except Exception as e:
        logger.exception("Sync failed")
        return [TextContent(type="text", text=f"Sync failed: {e}")]


# SSE Transport setup
sse_transport = SseServerTransport("/mcp/messages/")


async def handle_sse(request: Request) -> Response:
    """Handle SSE connection for MCP."""
    logger.info(f"New MCP SSE connection from {request.client}")
    async with sse_transport.connect_sse(request.scope, request.receive, request._send) as (
        read_stream,
        write_stream,
    ):
        await mcp_server.run(read_stream, write_stream, mcp_server.create_initialization_options())
    return Response()


async def handle_health(request: Request) -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({"status": "ok", "service": "contextfs-mcp"})


def create_mcp_app() -> Starlette:
    """Create the MCP Starlette application."""
    routes = [
        Route("/health", endpoint=handle_health, methods=["GET"]),
        Route("/mcp/sse", endpoint=handle_sse, methods=["GET"]),
        Mount("/mcp/messages/", app=sse_transport.handle_post_message),
    ]

    return Starlette(routes=routes)


def run_mcp_server(host: str = "127.0.0.1", port: int = 8003) -> None:
    """Run the MCP server."""
    print("Starting ContextFS MCP Server...")
    print(f"  URL: http://{host}:{port}")
    print(f"  SSE Endpoint: http://{host}:{port}/mcp/sse")
    print(f"  Health Check: http://{host}:{port}/health")
    print()
    print("Configure Claude Code with:")
    print(
        f'  {{"mcpServers": {{"contextfs": {{"type": "sse", "url": "http://{host}:{port}/mcp/sse"}}}}}}'
    )
    print()

    app = create_mcp_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_mcp_server()
