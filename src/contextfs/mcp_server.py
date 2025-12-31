"""
MCP Server for ContextFS.

Provides memory operations via Model Context Protocol.
Works with Claude Desktop, Claude Code, and any MCP client.
"""

import asyncio
import os
import signal

# Disable tokenizers parallelism to avoid deadlocks when using threads
# Must be set before any tokenizers are imported
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from dataclasses import dataclass, field
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import GetPromptResult, Prompt, PromptArgument, PromptMessage, TextContent, Tool

from contextfs.core import ContextFS
from contextfs.schemas import MemoryType, get_memory_type_values

# Memory type enum values - generated from schema (single source of truth)
MEMORY_TYPE_ENUM = get_memory_type_values()

# Global ContextFS instance
_ctx: ContextFS | None = None

# Source tool name (auto-detected or set by environment)
_source_tool: str | None = None  # Will be auto-detected on first use

# Session auto-started flag
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


# Global indexing state
_indexing_state = IndexingState()


def get_ctx() -> ContextFS:
    """Get or create ContextFS instance."""
    global _ctx, _session_started
    if _ctx is None:
        _ctx = ContextFS(auto_load=True)

    # Auto-start session for any tool that uses the MCP server
    if not _session_started:
        tool = get_source_tool()
        _ctx.start_session(tool=tool)
        _session_started = True

    return _ctx


def _detect_source_tool() -> str:
    """Auto-detect whether running under Claude Code or Claude Desktop.

    Detection strategy:
    1. Check explicit environment variable (highest priority)
    2. Check for Claude Code indicators (terminal environment, TTY)
    3. Default to claude-desktop (MCP is typically used via Desktop)
    """
    import os
    import sys

    # Explicit override always wins
    if env_tool := os.environ.get("CONTEXTFS_SOURCE_TOOL"):
        return env_tool

    # Claude Code indicators:
    # - Running in a terminal (has TTY)
    # - Has TERM environment variable
    # - Has typical shell environment vars
    # - Parent process is a shell or terminal

    # Check for terminal/TTY (Claude Code runs in terminal, Desktop doesn't)
    has_tty = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
    has_term = bool(os.environ.get("TERM"))
    has_shell = bool(os.environ.get("SHELL"))

    # Claude Code typically has all three; Claude Desktop has none
    if has_term and has_shell:
        return "claude-code"

    # Additional check: look for Claude Code specific patterns
    # Claude Code runs from a working directory, Desktop runs from app bundle
    cwd = os.getcwd()
    if "/Applications/" in cwd or "/Library/" in cwd:
        return "claude-desktop"

    # If we have terminal indicators but not all, still likely claude-code
    if has_tty or has_term:
        return "claude-code"

    return "claude-desktop"


def get_source_tool() -> str:
    """Get source tool name (cached after first detection)."""
    global _source_tool

    if _source_tool is None:
        _source_tool = _detect_source_tool()

    return _source_tool


def detect_current_repo() -> str | None:
    """Detect repo name from current working directory at runtime."""
    from pathlib import Path

    cwd = Path.cwd()
    # Walk up to find .git
    for parent in [cwd] + list(cwd.parents):
        if (parent / ".git").exists():
            return parent.name
    return None


# Create MCP server
server = Server("contextfs")


@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    """List available prompts for Claude Desktop."""
    return [
        Prompt(
            name="contextfs-session-guide",
            description="Instructions for capturing conversation context to ContextFS",
            arguments=[],
        ),
        Prompt(
            name="contextfs-save-session",
            description="Save the current conversation session with a summary",
            arguments=[
                PromptArgument(
                    name="summary",
                    description="Brief summary of what was discussed/accomplished",
                    required=True,
                ),
                PromptArgument(
                    name="label",
                    description="Optional label for the session (e.g., 'bug-fix', 'feature-planning')",
                    required=False,
                ),
            ],
        ),
        Prompt(
            name="contextfs-save-memory",
            description="Save important information to memory with guided categorization",
            arguments=[
                PromptArgument(
                    name="content",
                    description="The information to save",
                    required=True,
                ),
                PromptArgument(
                    name="type",
                    description="Memory type: fact, decision, procedural, code, error",
                    required=False,
                ),
            ],
        ),
        Prompt(
            name="contextfs-index",
            description="Index the current repository for semantic code search",
            arguments=[],
        ),
        Prompt(
            name="contextfs-init-repo",
            description="Initialize a repository for ContextFS indexing (opt-in)",
            arguments=[
                PromptArgument(
                    name="auto_index",
                    description="Enable automatic indexing on session start (default: true)",
                    required=False,
                ),
            ],
        ),
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None = None) -> GetPromptResult:
    """Get prompt content."""
    if name == "contextfs-session-guide":
        return GetPromptResult(
            description="ContextFS Session Capture Guide",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text="""# ContextFS Session Capture

Throughout this conversation, please use the ContextFS tools to capture important context:

1. **During conversation**: Use `contextfs_message` to log key exchanges:
   - User questions or requirements
   - Important decisions made
   - Code explanations or solutions provided

2. **Save important facts**: Use `contextfs_save` with appropriate types:
   - `fact` - Technical facts, configurations, dependencies
   - `decision` - Decisions made and their rationale
   - `procedural` - How-to procedures, workflows
   - `code` - Important code snippets or patterns
   - `error` - Error resolutions and fixes

3. **End of session**: Use `contextfs_save` with `save_session: "current"` to save the full session.

This ensures your insights and decisions are preserved for future conversations.""",
                    ),
                )
            ],
        )
    elif name == "contextfs-save-session":
        summary = (arguments or {}).get("summary", "")
        label = (arguments or {}).get("label", "")

        return GetPromptResult(
            description="Save Current Session",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"""Please save this conversation session to ContextFS:

Summary: {summary}
Label: {label or "(auto-generated)"}

Use the `contextfs_save` tool with:
- `save_session`: "current"
- `label`: "{label}" (if provided)

Then confirm the session was saved.""",
                    ),
                )
            ],
        )

    elif name == "contextfs-save-memory":
        content = (arguments or {}).get("content", "")
        mem_type = (arguments or {}).get("type", "")

        type_guidance = ""
        if not mem_type:
            type_guidance = """
First, determine the appropriate memory type:
- `fact` - Technical facts, configurations, dependencies, architecture
- `decision` - Decisions made and their rationale
- `procedural` - How-to procedures, workflows, deployment steps
- `code` - Important code snippets, patterns, algorithms
- `error` - Error resolutions, debugging solutions, fixes

"""

        return GetPromptResult(
            description="Save Memory to ContextFS",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"""Please save this information to ContextFS memory:

Content: {content}
{f"Type: {mem_type}" if mem_type else ""}
{type_guidance}
Use the `contextfs_save` tool with:
- `content`: A clear, searchable description of the information
- `type`: The appropriate memory type
- `summary`: A brief one-line summary
- `tags`: Relevant tags for categorization

Then confirm what was saved.""",
                    ),
                )
            ],
        )

    elif name == "contextfs-index":
        return GetPromptResult(
            description="Index Repository",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text="""Please index the current repository for semantic code search.

Use the `contextfs_index` tool to index all code files in this repository. This enables:
- Semantic search across the codebase
- Finding related code and patterns
- Better context for future conversations

After indexing, confirm how many files were processed.""",
                    ),
                )
            ],
        )

    elif name == "contextfs-init-repo":
        auto_index = (arguments or {}).get("auto_index", "true")
        auto_index_str = "enabled" if auto_index.lower() in ("true", "yes", "1") else "disabled"

        return GetPromptResult(
            description="Initialize Repository for ContextFS",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"""Please initialize this repository for ContextFS indexing.

Auto-indexing: {auto_index_str}

Use the `contextfs_init` tool to create the initialization marker. This:
- Creates a .contextfs/config.yaml file in the repository
- Enables automatic indexing on session start (if auto_index is true)
- Allows this repository to be indexed by the SessionStart hook

After initialization, optionally run `contextfs_index` to index the repository immediately.""",
                    ),
                )
            ],
        )

    return GetPromptResult(description="Unknown prompt", messages=[])


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="contextfs_save",
            description="Save a memory to ContextFS. Use for facts, decisions, procedures, or session summaries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Content to save",
                    },
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
                    "summary": {
                        "type": "string",
                        "description": "Brief summary",
                    },
                    "project": {
                        "type": "string",
                        "description": "Project name for grouping memories across repos (e.g., 'haven', 'my-app')",
                    },
                    "save_session": {
                        "type": "string",
                        "enum": ["current", "previous"],
                        "description": "Save session instead of memory",
                    },
                    "label": {
                        "type": "string",
                        "description": "Label for session",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="contextfs_search",
            description="Search memories using hybrid search (combines keyword + semantic). Supports cross-repo search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum results",
                        "default": 5,
                    },
                    "type": {
                        "type": "string",
                        "enum": MEMORY_TYPE_ENUM,
                        "description": "Filter by type",
                    },
                    "cross_repo": {
                        "type": "boolean",
                        "description": "Search across all repos (default: true for best results)",
                        "default": True,
                    },
                    "source_tool": {
                        "type": "string",
                        "description": "Filter by source tool (claude-code, claude-desktop, gemini, etc.)",
                    },
                    "source_repo": {
                        "type": "string",
                        "description": "Filter by source repository name",
                    },
                    "project": {
                        "type": "string",
                        "description": "Filter by project name (searches across all repos in project)",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="contextfs_list_repos",
            description="List all repositories with saved memories",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="contextfs_list_tools",
            description="List all source tools (Claude, Gemini, etc.) with saved memories",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="contextfs_list_projects",
            description="List all projects with saved memories (projects group memories across repos)",
            inputSchema={
                "type": "object",
                "properties": {},
            },
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
                    "limit": {
                        "type": "number",
                        "description": "Maximum results",
                        "default": 10,
                    },
                    "type": {
                        "type": "string",
                        "enum": MEMORY_TYPE_ENUM,
                        "description": "Filter by memory type",
                    },
                    "source_tool": {
                        "type": "string",
                        "description": "Filter by source tool (claude-desktop, claude-code, gemini, chatgpt, ollama, etc.)",
                    },
                    "project": {
                        "type": "string",
                        "description": "Filter by project name",
                    },
                },
            },
        ),
        Tool(
            name="contextfs_sessions",
            description="List recent sessions",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "number",
                        "description": "Maximum results",
                        "default": 10,
                    },
                    "label": {
                        "type": "string",
                        "description": "Filter by label",
                    },
                    "tool": {
                        "type": "string",
                        "description": "Filter by tool (claude-code, gemini, etc.)",
                    },
                },
            },
        ),
        Tool(
            name="contextfs_load_session",
            description="Load a session's messages into context",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID (can be partial)",
                    },
                    "label": {
                        "type": "string",
                        "description": "Session label",
                    },
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
                    "content": {
                        "type": "string",
                        "description": "Message content",
                    },
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
                    "max_commits": {
                        "type": "number",
                        "description": "Maximum commits to index (default: 100)",
                        "default": 100,
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
                        "description": "Path to repository to index. Can be full path or repo name from list-indexes. Defaults to current directory.",
                    },
                    "incremental": {
                        "type": "boolean",
                        "description": "Only index new/changed files (default: true)",
                        "default": True,
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Force re-index even if already indexed",
                        "default": False,
                    },
                    "mode": {
                        "type": "string",
                        "description": "Index mode: 'all' (files+commits), 'files_only', or 'commits_only'",
                        "enum": ["all", "files_only", "commits_only"],
                        "default": "all",
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
            inputSchema={
                "type": "object",
                "properties": {},
            },
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
                    "content": {
                        "type": "string",
                        "description": "New content (optional)",
                    },
                    "type": {
                        "type": "string",
                        "enum": MEMORY_TYPE_ENUM,
                        "description": "New memory type (optional)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New tags (optional)",
                    },
                    "summary": {
                        "type": "string",
                        "description": "New summary (optional)",
                    },
                    "project": {
                        "type": "string",
                        "description": "New project name (optional)",
                    },
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
            name="contextfs_update_session",
            description="Update an existing session",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID (can be partial)",
                    },
                    "label": {
                        "type": "string",
                        "description": "New label (optional)",
                    },
                    "summary": {
                        "type": "string",
                        "description": "New summary (optional)",
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="contextfs_delete_session",
            description="Delete a session and its messages",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID (can be partial)",
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="contextfs_import_conversation",
            description="Import a JSON conversation export as an episodic memory. Use this to save Claude Desktop or other AI conversations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "json_content": {
                        "type": "string",
                        "description": "JSON string containing the conversation export",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of the conversation",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization",
                    },
                    "project": {
                        "type": "string",
                        "description": "Project name for grouping",
                    },
                },
                "required": ["json_content"],
            },
        ),
        Tool(
            name="contextfs_discover_repos",
            description="Discover git repositories in a directory without indexing. Shows detected project groupings, languages, and frameworks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Root directory to scan (default: current working directory)",
                    },
                    "max_depth": {
                        "type": "number",
                        "description": "Maximum directory depth to search",
                        "default": 5,
                    },
                },
            },
        ),
        Tool(
            name="contextfs_index_directory",
            description="Recursively scan a directory for git repos and index each. Auto-detects project groupings and tags.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Root directory to scan for git repositories",
                    },
                    "max_depth": {
                        "type": "number",
                        "description": "Maximum directory depth to search",
                        "default": 5,
                    },
                    "project": {
                        "type": "string",
                        "description": "Override auto-detected project name for all repos",
                    },
                    "incremental": {
                        "type": "boolean",
                        "description": "Only index new/changed files (default: true)",
                        "default": True,
                    },
                },
                "required": ["path"],
            },
        ),
        # =========================================================================
        # Memory Lineage Tools
        # =========================================================================
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
            name="contextfs_merge",
            description="Merge multiple memories into one. Combines knowledge from multiple sources with configurable strategies. Use when consolidating related information.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of memory IDs to merge (can be partial, at least 8 chars each)",
                    },
                    "merged_content": {
                        "type": "string",
                        "description": "Optional custom content for the merged memory. If not provided, content is auto-combined.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Summary for the merged memory",
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["union", "intersection", "latest", "oldest"],
                        "description": "Tag merge strategy: union (all tags), intersection (common tags), latest (from newest), oldest (from oldest)",
                        "default": "union",
                    },
                    "type": {
                        "type": "string",
                        "enum": MEMORY_TYPE_ENUM,
                        "description": "Type for merged memory (defaults to first memory's type)",
                    },
                },
                "required": ["memory_ids"],
            },
        ),
        Tool(
            name="contextfs_split",
            description="Split a memory into multiple parts. Use when a memory contains distinct topics that should be separate.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory to split (can be partial, at least 8 chars)",
                    },
                    "parts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Content for each split part",
                    },
                    "summaries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional summaries for each part (same length as parts)",
                    },
                    "preserve_tags": {
                        "type": "boolean",
                        "description": "Whether to preserve original tags on all parts (default: true)",
                        "default": True,
                    },
                },
                "required": ["memory_id", "parts"],
            },
        ),
        Tool(
            name="contextfs_lineage",
            description="Get the lineage (history) of a memory. Shows ancestors (what it evolved from) and descendants (what evolved from it).",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory to get lineage for (can be partial, at least 8 chars)",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["ancestors", "descendants", "both"],
                        "description": "Which direction to traverse: ancestors (history), descendants (evolutions), or both",
                        "default": "both",
                    },
                },
                "required": ["memory_id"],
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
                        "default": 1.0,
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
            name="contextfs_related",
            description="Get memories related to a given memory through graph relationships.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory to find relationships for (can be partial, at least 8 chars)",
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
                            "evolved_from",
                            "merged_from",
                            "split_from",
                        ],
                        "description": "Filter by specific relation type (optional)",
                    },
                    "max_depth": {
                        "type": "number",
                        "description": "Maximum traversal depth (default: 1)",
                        "default": 1,
                    },
                },
                "required": ["memory_id"],
            },
        ),
        # =========================================================================
        # Index Management Tools
        # =========================================================================
        Tool(
            name="contextfs_cleanup_indexes",
            description="Remove stale indexes for repositories that no longer exist on disk or are not git repos.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dry_run": {
                        "type": "boolean",
                        "description": "If true, only report what would be deleted without deleting",
                        "default": False,
                    },
                },
            },
        ),
        Tool(
            name="contextfs_delete_index",
            description="Delete a specific repository index by path or namespace ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_path": {
                        "type": "string",
                        "description": "Repository path to delete index for",
                    },
                    "namespace_id": {
                        "type": "string",
                        "description": "Namespace ID to delete",
                    },
                },
            },
        ),
        Tool(
            name="contextfs_rebuild_chroma",
            description="Rebuild ChromaDB search index from SQLite data. Fast recovery after corruption without re-scanning files.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="contextfs_reindex",
            description="Reindex repository/repositories. Use 'all_repos: true' to reindex all repos in database, or provide 'repo_path' for a single repo.",
            inputSchema={
                "type": "object",
                "properties": {
                    "all_repos": {
                        "type": "boolean",
                        "description": "Reindex all repositories in the database",
                        "default": False,
                    },
                    "repo_path": {
                        "type": "string",
                        "description": "Path to specific repository to reindex (ignored if all_repos is true)",
                    },
                    "incremental": {
                        "type": "boolean",
                        "description": "Only index new/changed files (default: true)",
                        "default": True,
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["all", "files_only", "commits_only"],
                        "description": "Index mode: 'all' (files+commits), 'files_only', or 'commits_only'",
                        "default": "all",
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    global _indexing_state
    ctx = get_ctx()

    try:
        if name == "contextfs_save":
            # Check if saving session
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
                else:
                    return [TextContent(type="text", text="No active session to save.")]

            # Save memory
            content = arguments.get("content", "")
            if not content:
                return [TextContent(type="text", text="Error: content is required")]

            memory_type = MemoryType(arguments.get("type", "fact"))
            tags = arguments.get("tags", [])
            summary = arguments.get("summary")

            project = arguments.get("project")

            # Detect repo at save time (not just at init) for accurate tracking
            source_repo = detect_current_repo()

            # Trigger incremental indexing (auto-indexes if not done yet)
            index_result = None
            try:
                from pathlib import Path

                cwd = Path.cwd()
                if source_repo:
                    index_result = ctx.index_repository(repo_path=cwd, incremental=True)
            except Exception:
                pass  # Indexing is best-effort, don't fail the save

            memory = ctx.save(
                content=content,
                type=memory_type,
                tags=tags,
                summary=summary,
                source_tool=get_source_tool(),
                source_repo=source_repo,
                project=project,
            )

            # Also log to current session (best-effort, don't fail the save)
            try:
                ctx.add_message(
                    role="assistant",
                    content=f"[Memory saved] {memory_type.value}: {summary or content[:100]}",
                )
            except Exception:
                pass

            # Build response with repo info
            response = f"Memory saved successfully.\nID: {memory.id}\nType: {memory.type.value}"
            if source_repo:
                response += f"\nRepo: {source_repo}"
            if index_result and index_result.get("files_indexed", 0) > 0:
                response += f"\nIndexed: {index_result['files_indexed']} files"

            return [TextContent(type="text", text=response)]

        elif name == "contextfs_search":
            query = arguments.get("query", "")
            limit = arguments.get("limit", 5)
            type_filter = MemoryType(arguments["type"]) if arguments.get("type") else None
            # Default to cross_repo=True for better search results (matches CLI behavior)
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
                if r.memory.source_tool:
                    line += f" via {r.memory.source_tool}"
                output.append(line)
                if r.memory.summary:
                    output.append(f"  Summary: {r.memory.summary}")
                output.append(f"  {r.memory.content[:200]}...")
                if r.memory.tags:
                    output.append(f"  Tags: {', '.join(r.memory.tags)}")
                output.append("")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_list_repos":
            repos = ctx.list_repos()
            indexes = ctx.list_indexes()

            output = []

            # Repos with memories
            if repos:
                output.append("Repositories with memories:")
                for r in repos:
                    output.append(f"  • {r['source_repo']} ({r['memory_count']} memories)")
            else:
                output.append("No repositories with memories found.")

            output.append("")

            # Indexed repositories
            if indexes:
                output.append("Indexed repositories:")
                for idx in indexes:
                    repo_name = idx.repo_path.split("/")[-1] if idx.repo_path else idx.namespace_id
                    output.append(
                        f"  • {repo_name} ({idx.files_indexed} files, {idx.commits_indexed} commits, {idx.memories_created} code chunks)"
                    )
            else:
                output.append("No indexed repositories found.")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_list_tools":
            tools = ctx.list_tools()

            if not tools:
                return [TextContent(type="text", text="No source tools found.")]

            output = ["Source tools with memories:"]
            for t in tools:
                output.append(f"  • {t['source_tool']} ({t['memory_count']} memories)")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_list_projects":
            projects = ctx.list_projects()

            if not projects:
                return [TextContent(type="text", text="No projects found.")]

            output = ["Projects with memories:"]
            for p in projects:
                repos_str = ", ".join(p["repos"]) if p["repos"] else "no repos"
                output.append(f"  • {p['project']} ({p['memory_count']} memories)")
                output.append(f"    Repos: {repos_str}")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_recall":
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

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_list":
            limit = arguments.get("limit", 10)
            type_filter = MemoryType(arguments["type"]) if arguments.get("type") else None
            source_tool = arguments.get("source_tool")
            project = arguments.get("project")

            memories = ctx.list_recent(
                limit=limit,
                type=type_filter,
                source_tool=source_tool,
                project=project,
            )

            if not memories:
                filters = []
                if source_tool:
                    filters.append(f"source_tool={source_tool}")
                if project:
                    filters.append(f"project={project}")
                if type_filter:
                    filters.append(f"type={type_filter.value}")
                filter_str = f" (filters: {', '.join(filters)})" if filters else ""
                return [TextContent(type="text", text=f"No memories found{filter_str}.")]

            output = []
            for m in memories:
                # Format: [id] [type] [project?] @repo? via tool?
                line = f"[{m.id[:8]}] [{m.type.value}]"
                if m.project:
                    line += f" [{m.project}]"
                if m.source_repo:
                    line += f" @{m.source_repo}"
                if m.source_tool:
                    line += f" via {m.source_tool}"
                output.append(line)
                # Summary or content preview on next line
                if m.summary:
                    output.append(f"  {m.summary}")
                else:
                    output.append(f"  {m.content[:60]}...")
                output.append("")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_sessions":
            limit = arguments.get("limit", 10)
            label = arguments.get("label")
            tool = arguments.get("tool")

            # Search all namespaces by default to find all sessions
            sessions = ctx.list_sessions(limit=limit, label=label, tool=tool, all_namespaces=True)

            if not sessions:
                return [TextContent(type="text", text="No sessions found.")]

            output = []
            for s in sessions:
                line = f"[{s.id[:12]}] {s.tool}"
                if s.label:
                    line += f" ({s.label})"
                line += f" - {s.started_at.strftime('%Y-%m-%d %H:%M')}"
                line += f" ({len(s.messages)} msgs)"
                output.append(line)

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_load_session":
            session_id = arguments.get("session_id")
            label = arguments.get("label")
            max_messages = arguments.get("max_messages", 20)

            session = ctx.load_session(session_id=session_id, label=label)

            if not session:
                return [TextContent(type="text", text="Session not found.")]

            output = [
                f"Session: {session.id}",
                f"Tool: {session.tool}",
                f"Started: {session.started_at.isoformat()}",
            ]
            if session.label:
                output.append(f"Label: {session.label}")
            if session.summary:
                output.append(f"Summary: {session.summary}")

            output.append(f"\nMessages ({len(session.messages)}):\n")

            for msg in session.messages[-max_messages:]:
                output.append(f"[{msg.role}] {msg.content[:500]}")
                output.append("")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_message":
            role = arguments.get("role", "user")
            content = arguments.get("content", "")

            msg = ctx.add_message(role, content)

            return [
                TextContent(
                    type="text",
                    text=f"Message added to session.\nMessage ID: {msg.id}",
                )
            ]

        elif name == "contextfs_init":
            from pathlib import Path

            from contextfs.cli import create_repo_config, is_repo_initialized

            repo_path_arg = arguments.get("repo_path")
            auto_index = arguments.get("auto_index", True)
            max_commits = arguments.get("max_commits", 100)
            run_index = arguments.get("run_index", True)
            force = arguments.get("force", False)

            # Resolve repo path
            repo_path = Path(repo_path_arg).resolve() if repo_path_arg else Path.cwd()

            # Check if in a git repo
            repo_name = detect_current_repo()
            if not repo_name:
                return [TextContent(type="text", text="Error: Not in a git repository")]

            # Check if already initialized
            if is_repo_initialized(repo_path) and not force:
                return [
                    TextContent(
                        type="text",
                        text=f"Repository '{repo_name}' already initialized for ContextFS.\n"
                        f"Use force=true to reinitialize.",
                    )
                ]

            # Create the config
            config_path = create_repo_config(
                repo_path=repo_path,
                auto_index=auto_index,
                created_by=get_source_tool(),
                max_commits=int(max_commits),
            )

            output = [
                f"Repository initialized for ContextFS: {repo_name}",
                f"Config: {config_path}",
                f"Auto-index: {'enabled' if auto_index else 'disabled'}",
                f"Max commits: {max_commits}",
            ]

            # Optionally run indexing
            if run_index:
                try:
                    result = ctx.index_repository(repo_path=repo_path, incremental=True)
                    output.append("")
                    output.append(
                        f"Indexed: {result.get('files_indexed', 0)} files, {result.get('commits_indexed', 0)} commits"
                    )
                except Exception as e:
                    output.append("")
                    output.append(f"Indexing started but encountered error: {e}")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_index":
            from pathlib import Path

            incremental = arguments.get("incremental", True)
            force = arguments.get("force", False)
            mode = arguments.get("mode", "all")
            repo_path_arg = arguments.get("repo_path")

            # Resolve repo path
            if repo_path_arg:
                # Check if it's a full path
                if repo_path_arg.startswith("/"):
                    repo_path = Path(repo_path_arg)
                else:
                    # Try to find by repo name in indexed repos
                    indexes = ctx.list_indexes()
                    matching = [
                        idx
                        for idx in indexes
                        if idx.repo_path and idx.repo_path.split("/")[-1] == repo_path_arg
                    ]
                    if matching:
                        repo_path = Path(matching[0].repo_path)
                    else:
                        return [
                            TextContent(
                                type="text",
                                text=f"Repository '{repo_path_arg}' not found. Use full path or a name from list-indexes.",
                            )
                        ]

                if not repo_path.exists():
                    return [
                        TextContent(type="text", text=f"Error: Path does not exist: {repo_path}")
                    ]
                repo_name = repo_path.name
            else:
                # Default to current working directory
                repo_path = Path.cwd()
                repo_name = detect_current_repo()
                if not repo_name:
                    return [TextContent(type="text", text="Error: Not in a git repository")]

            # Check if indexing is already running
            if _indexing_state.running:
                # Verify the task is actually still alive
                task_alive = _indexing_state.task is not None and not _indexing_state.task.done()

                if task_alive and not force:
                    return [
                        TextContent(
                            type="text",
                            text=f"Indexing already in progress for '{_indexing_state.repo_name}'.\n"
                            f"Progress: {_indexing_state.current}/{_indexing_state.total} files\n"
                            f"Use force=true to cancel and restart.",
                        )
                    ]
                elif task_alive and force:
                    # Cancel the running task
                    _indexing_state.task.cancel()
                    _indexing_state = IndexingState()
                else:
                    # Task is dead but state wasn't cleaned up - reset it
                    _indexing_state = IndexingState()

            # Check if already indexed (unless force or commits_only mode)
            status = ctx.get_index_status(repo_path=repo_path)
            if status and status.indexed and not force and mode != "commits_only":
                return [
                    TextContent(
                        type="text",
                        text=f"Repository '{repo_name}' already indexed.\n"
                        f"Files: {status.files_indexed}, Commits: {status.commits_indexed}\n"
                        f"Use force=true to re-index, or mode='commits_only' to just index commits.",
                    )
                ]

            # Clear index if forcing (but not for commits_only mode which is additive)
            if force and mode != "commits_only":
                ctx.clear_index(repo_path=repo_path)

            # Reset indexing state
            _indexing_state = IndexingState(
                running=True,
                repo_name=repo_name,
            )

            # Get progress token for MCP notifications
            progress_token = None
            mcp_session = None
            try:
                req_ctx = server.request_context
                if req_ctx.meta and req_ctx.meta.progressToken:
                    progress_token = req_ctx.meta.progressToken
                    mcp_session = req_ctx.session
            except (LookupError, AttributeError):
                pass

            # Pre-initialize the embedder on main thread to avoid tokio runtime issues
            # The tokio runtime in tokenizers crashes when initialized from a thread pool
            try:
                ctx.rag._ensure_initialized()
            except Exception:
                pass  # Best effort - will retry in thread

            async def run_indexing():
                """Run indexing in background."""
                import os

                # Disable tokenizers parallelism to avoid crashes in threaded context
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

                loop = asyncio.get_running_loop()

                def on_progress(current: int, total: int, filename: str) -> None:
                    """Update state and send MCP progress notification."""
                    _indexing_state.current = current
                    _indexing_state.total = total
                    _indexing_state.current_file = filename

                    # Send MCP progress notification if available
                    if progress_token and mcp_session:
                        try:
                            coro = mcp_session.send_progress_notification(
                                progress_token=progress_token,
                                progress=float(current),
                                total=float(total),
                                message=f"Indexing: {filename}",
                            )
                            asyncio.run_coroutine_threadsafe(coro, loop)
                        except Exception:
                            pass

                try:
                    # Run blocking indexing in thread pool
                    result = await asyncio.to_thread(
                        ctx.index_repository,
                        repo_path=repo_path,
                        incremental=incremental,
                        on_progress=on_progress,
                        mode=mode,
                    )
                    _indexing_state.result = result
                    _indexing_state.running = False
                except asyncio.CancelledError:
                    _indexing_state.error = "Indexing cancelled"
                    _indexing_state.running = False
                except Exception as e:
                    import traceback

                    _indexing_state.error = f"{str(e)}\n{traceback.format_exc()}"
                    _indexing_state.running = False

            # Start background task
            _indexing_state.task = asyncio.create_task(run_indexing())

            mode_desc = {
                "all": "files and commits",
                "files_only": "files only",
                "commits_only": "commits only",
            }
            return [
                TextContent(
                    type="text",
                    text=f"Started indexing '{repo_name}' in background ({mode_desc.get(mode, mode)}).\n"
                    f"Use contextfs_index_status to check progress.",
                )
            ]

        elif name == "contextfs_index_status":
            cancel = arguments.get("cancel", False)

            if _indexing_state.running:
                # Handle cancel request
                if cancel:
                    if _indexing_state.task and not _indexing_state.task.done():
                        _indexing_state.task.cancel()
                    repo = _indexing_state.repo_name
                    progress = f"{_indexing_state.current}/{_indexing_state.total}"
                    _indexing_state = IndexingState()
                    return [
                        TextContent(
                            type="text",
                            text=f"Indexing cancelled for '{repo}'.\n"
                            f"Progress at cancellation: {progress} files",
                        )
                    ]

                # Return progress
                pct = 0
                if _indexing_state.total > 0:
                    pct = int(100 * _indexing_state.current / _indexing_state.total)
                return [
                    TextContent(
                        type="text",
                        text=f"Indexing in progress: {_indexing_state.repo_name}\n"
                        f"Progress: {_indexing_state.current}/{_indexing_state.total} files ({pct}%)\n"
                        f"Current: {_indexing_state.current_file}\n"
                        f"Use cancel=true to stop indexing.",
                    )
                ]
            elif _indexing_state.error:
                error = _indexing_state.error
                repo = _indexing_state.repo_name
                # Reset state after reporting error
                _indexing_state = IndexingState()
                return [
                    TextContent(
                        type="text",
                        text=f"Indexing failed for '{repo}'.\nError: {error}",
                    )
                ]
            elif _indexing_state.result:
                result = _indexing_state.result
                repo = _indexing_state.repo_name
                # Reset state after reporting completion
                _indexing_state = IndexingState()
                return [
                    TextContent(
                        type="text",
                        text=f"Indexing complete: {repo}\n"
                        f"Files indexed: {result.get('files_indexed', 0)}\n"
                        f"Commits indexed: {result.get('commits_indexed', 0)}\n"
                        f"Memories created: {result.get('memories_created', 0)}\n"
                        f"Skipped: {result.get('skipped', 0)}",
                    )
                ]
            else:
                # Check if there's existing index data for current repo
                from pathlib import Path

                cwd = Path.cwd()
                status = ctx.get_index_status(repo_path=cwd)
                if status and status.indexed:
                    return [
                        TextContent(
                            type="text",
                            text=f"No indexing in progress.\n"
                            f"Repository indexed: {status.files_indexed} files, {status.commits_indexed} commits, {status.memories_created} code chunks",
                        )
                    ]
                return [
                    TextContent(
                        type="text",
                        text="No indexing in progress. Use contextfs_index to start.",
                    )
                ]

        elif name == "contextfs_list_indexes":
            indexes = ctx.list_indexes()

            if not indexes:
                return [
                    TextContent(
                        type="text",
                        text="No indexed repositories found. Use contextfs_index to index the current repository.",
                    )
                ]

            # Build table output
            output = ["Full Index Status - All Repositories", ""]
            output.append(
                f"{'Namespace':<18} {'Repository':<20} {'Files':>7} {'Commits':>8} {'Memories':>9} {'Indexed At':<16}"
            )
            output.append("-" * 85)

            total_files = 0
            total_commits = 0
            total_memories = 0

            for idx in sorted(indexes, key=lambda x: x.memories_created, reverse=True):
                repo_name = idx.repo_path.split("/")[-1] if idx.repo_path else "unknown"
                indexed_at = str(idx.indexed_at)[:16] if idx.indexed_at else "N/A"
                output.append(
                    f"{idx.namespace_id[:18]:<18} {repo_name[:20]:<20} {idx.files_indexed:>7} {idx.commits_indexed:>8} {idx.memories_created:>9} {indexed_at:<16}"
                )
                total_files += idx.files_indexed
                total_commits += idx.commits_indexed
                total_memories += idx.memories_created

            output.append("-" * 85)
            output.append(
                f"{'TOTAL':<18} {f'{len(indexes)} repos':<20} {total_files:>7} {total_commits:>8} {total_memories:>9}"
            )

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_update":
            memory_id = arguments.get("id", "")
            if not memory_id:
                return [TextContent(type="text", text="Error: id is required")]

            memory_type = MemoryType(arguments["type"]) if arguments.get("type") else None

            memory = ctx.update(
                memory_id=memory_id,
                content=arguments.get("content"),
                type=memory_type,
                tags=arguments.get("tags"),
                summary=arguments.get("summary"),
                project=arguments.get("project"),
            )

            if not memory:
                return [TextContent(type="text", text=f"Memory not found: {memory_id}")]

            return [
                TextContent(
                    type="text",
                    text=f"Memory updated successfully.\nID: {memory.id}\nType: {memory.type.value}",
                )
            ]

        elif name == "contextfs_delete":
            memory_id = arguments.get("id", "")
            if not memory_id:
                return [TextContent(type="text", text="Error: id is required")]

            deleted = ctx.delete(memory_id)

            if not deleted:
                return [TextContent(type="text", text=f"Memory not found: {memory_id}")]

            return [TextContent(type="text", text=f"Memory deleted: {memory_id}")]

        elif name == "contextfs_update_session":
            session_id = arguments.get("session_id", "")
            if not session_id:
                return [TextContent(type="text", text="Error: session_id is required")]

            session = ctx.update_session(
                session_id=session_id,
                label=arguments.get("label"),
                summary=arguments.get("summary"),
            )

            if not session:
                return [TextContent(type="text", text=f"Session not found: {session_id}")]

            return [
                TextContent(
                    type="text",
                    text=f"Session updated successfully.\nID: {session.id}\nLabel: {session.label or 'none'}",
                )
            ]

        elif name == "contextfs_delete_session":
            session_id = arguments.get("session_id", "")
            if not session_id:
                return [TextContent(type="text", text="Error: session_id is required")]

            deleted = ctx.delete_session(session_id)

            if not deleted:
                return [TextContent(type="text", text=f"Session not found: {session_id}")]

            return [TextContent(type="text", text=f"Session deleted: {session_id}")]

        elif name == "contextfs_import_conversation":
            import json

            json_content = arguments.get("json_content", "")
            if not json_content:
                return [TextContent(type="text", text="Error: json_content is required")]

            summary = arguments.get("summary")
            tags = arguments.get("tags", [])
            project = arguments.get("project")

            # Try to parse and format the JSON
            try:
                conversation_data = json.loads(json_content)
            except json.JSONDecodeError as e:
                return [TextContent(type="text", text=f"Error parsing JSON: {e}")]

            # Format the conversation for storage
            formatted_content = []

            # Handle different JSON formats
            if isinstance(conversation_data, list):
                # Array of messages format
                for msg in conversation_data:
                    if isinstance(msg, dict):
                        role = msg.get("role", msg.get("sender", "unknown"))
                        content = msg.get("content", msg.get("text", msg.get("message", "")))
                        if isinstance(content, list):
                            # Handle content blocks (Claude format)
                            content = " ".join(
                                c.get("text", "") for c in content if isinstance(c, dict)
                            )
                        formatted_content.append(f"[{role}]: {content}")
            elif isinstance(conversation_data, dict):
                # Object format - look for messages array
                messages = conversation_data.get(
                    "messages", conversation_data.get("conversation", [])
                )
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get("role", msg.get("sender", "unknown"))
                        content = msg.get("content", msg.get("text", msg.get("message", "")))
                        if isinstance(content, list):
                            content = " ".join(
                                c.get("text", "") for c in content if isinstance(c, dict)
                            )
                        formatted_content.append(f"[{role}]: {content}")

                # Also capture any metadata
                if "title" in conversation_data and not summary:
                    summary = conversation_data["title"]
                if "name" in conversation_data and not summary:
                    summary = conversation_data["name"]

            if not formatted_content:
                # Just store raw JSON if we couldn't parse it
                formatted_content = [json_content[:5000]]

            content = "\n\n".join(formatted_content)

            # Auto-generate summary if not provided
            if not summary:
                # Take first 100 chars of content as summary
                summary = content[:100] + "..." if len(content) > 100 else content

            # Add conversation tag
            if "conversation" not in tags:
                tags = ["conversation", "import"] + list(tags)

            source_repo = detect_current_repo()

            memory = ctx.save(
                content=content,
                type=MemoryType.EPISODIC,
                tags=tags,
                summary=summary,
                source_tool=get_source_tool(),
                source_repo=source_repo,
                project=project,
            )

            msg_count = len(formatted_content) if formatted_content else 0
            return [
                TextContent(
                    type="text",
                    text=f"Conversation imported successfully.\nID: {memory.id}\nMessages: {msg_count}\nSummary: {summary}",
                )
            ]

        elif name == "contextfs_discover_repos":
            from pathlib import Path

            path_str = arguments.get("path")
            max_depth = arguments.get("max_depth", 5)

            # Default to current working directory
            target_path = Path(path_str).resolve() if path_str else Path.cwd()

            if not target_path.exists():
                return [TextContent(type="text", text=f"Error: Path does not exist: {target_path}")]

            repos = ctx.discover_repos(target_path, max_depth=max_depth)

            if not repos:
                return [
                    TextContent(type="text", text=f"No git repositories found in {target_path}")
                ]

            output = [f"Found {len(repos)} repositories in {target_path}:\n"]

            for repo in repos:
                line = f"• {repo['name']}"
                if repo["project"]:
                    line += f" (project: {repo['project']})"
                output.append(line)

                # Show detected tags
                lang_tags = [t for t in repo["suggested_tags"] if t.startswith("lang:")]
                fw_tags = [t for t in repo["suggested_tags"] if t.startswith("framework:")]
                other_tags = [
                    t
                    for t in repo["suggested_tags"]
                    if not t.startswith("lang:") and not t.startswith("framework:")
                ]

                if lang_tags or fw_tags:
                    tags_str = ", ".join([t.split(":")[1] for t in lang_tags + fw_tags])
                    output.append(f"  Languages/Frameworks: {tags_str}")
                if other_tags:
                    output.append(f"  Tags: {', '.join(other_tags)}")
                output.append(f"  Path: {repo['relative_path']}")
                output.append("")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_index_directory":
            from pathlib import Path

            path_str = arguments.get("path")
            max_depth = arguments.get("max_depth", 5)
            project_override = arguments.get("project")
            incremental = arguments.get("incremental", True)

            if not path_str:
                return [TextContent(type="text", text="Error: path is required")]

            target_path = Path(path_str).resolve()

            if not target_path.exists():
                return [TextContent(type="text", text=f"Error: Path does not exist: {target_path}")]

            # First discover repos
            repos = ctx.discover_repos(target_path, max_depth=max_depth)

            if not repos:
                return [
                    TextContent(type="text", text=f"No git repositories found in {target_path}")
                ]

            # Run indexing
            result = ctx.index_directory(
                root_dir=target_path,
                max_depth=max_depth,
                incremental=incremental,
                project_override=project_override,
            )

            # Build response
            output = [
                f"Directory indexing complete for {target_path}:\n",
                f"Repositories found: {result['repos_found']}",
                f"Repositories indexed: {result['repos_indexed']}",
                f"Total files: {result['total_files']}",
                f"Total commits: {result.get('total_commits', 0)}",
                f"Total memories: {result['total_memories']}",
                "",
            ]

            # Per-repo details
            if result["repos"]:
                output.append("Per-repository breakdown:")
                for repo in result["repos"]:
                    status = "Error" if "error" in repo else "OK"
                    line = f"  • {repo['name']}"
                    if repo.get("project"):
                        line += f" [{repo['project']}]"
                    if "error" in repo:
                        output.append(f"{line} - {status}: {repo['error']}")
                    else:
                        output.append(
                            f"{line} - {repo.get('files_indexed', 0)} files, "
                            f"{repo.get('memories_created', 0)} memories"
                        )

            return [TextContent(type="text", text="\n".join(output))]

        # =====================================================================
        # Memory Lineage Tool Handlers
        # =====================================================================
        elif name == "contextfs_evolve":
            memory_id = arguments.get("memory_id", "")
            new_content = arguments.get("new_content", "")

            if not memory_id:
                return [TextContent(type="text", text="Error: memory_id is required")]
            if not new_content:
                return [TextContent(type="text", text="Error: new_content is required")]

            summary = arguments.get("summary")
            preserve_tags = arguments.get("preserve_tags", True)
            additional_tags = arguments.get("additional_tags")

            new_memory = ctx.evolve(
                memory_id=memory_id,
                new_content=new_content,
                summary=summary,
                preserve_tags=preserve_tags,
                additional_tags=additional_tags,
            )

            if not new_memory:
                return [TextContent(type="text", text=f"Memory not found: {memory_id}")]

            output = [
                "Memory evolved successfully.",
                f"Original: {memory_id[:12]}...",
                f"New ID: {new_memory.id}",
                f"Type: {new_memory.type.value}",
            ]
            if new_memory.summary:
                output.append(f"Summary: {new_memory.summary}")
            if new_memory.tags:
                output.append(f"Tags: {', '.join(new_memory.tags)}")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_merge":
            memory_ids = arguments.get("memory_ids", [])

            if not memory_ids or len(memory_ids) < 2:
                return [TextContent(type="text", text="Error: At least 2 memory_ids are required")]

            merged_content = arguments.get("merged_content")
            summary = arguments.get("summary")
            strategy = arguments.get("strategy", "union")
            memory_type = MemoryType(arguments["type"]) if arguments.get("type") else None

            merged_memory = ctx.merge(
                memory_ids=memory_ids,
                merged_content=merged_content,
                summary=summary,
                strategy=strategy,
                memory_type=memory_type,
            )

            if not merged_memory:
                return [
                    TextContent(
                        type="text", text="Error: Failed to merge memories. Some IDs may not exist."
                    )
                ]

            output = [
                "Memories merged successfully.",
                f"Merged {len(memory_ids)} memories:",
            ]
            for mid in memory_ids:
                output.append(f"  • {mid[:12]}...")
            output.append(f"New ID: {merged_memory.id}")
            output.append(f"Type: {merged_memory.type.value}")
            output.append(f"Strategy: {strategy}")
            if merged_memory.summary:
                output.append(f"Summary: {merged_memory.summary}")
            if merged_memory.tags:
                output.append(f"Tags: {', '.join(merged_memory.tags)}")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_split":
            memory_id = arguments.get("memory_id", "")
            parts = arguments.get("parts", [])

            if not memory_id:
                return [TextContent(type="text", text="Error: memory_id is required")]
            if not parts or len(parts) < 2:
                return [TextContent(type="text", text="Error: At least 2 parts are required")]

            summaries = arguments.get("summaries")
            preserve_tags = arguments.get("preserve_tags", True)

            split_memories = ctx.split(
                memory_id=memory_id,
                parts=parts,
                summaries=summaries,
                preserve_tags=preserve_tags,
            )

            if not split_memories:
                return [TextContent(type="text", text=f"Memory not found: {memory_id}")]

            output = [
                "Memory split successfully.",
                f"Original: {memory_id[:12]}...",
                f"Created {len(split_memories)} new memories:",
            ]
            for i, mem in enumerate(split_memories):
                line = f"  {i + 1}. {mem.id}"
                if mem.summary:
                    line += f" - {mem.summary}"
                output.append(line)

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_lineage":
            memory_id = arguments.get("memory_id", "")

            if not memory_id:
                return [TextContent(type="text", text="Error: memory_id is required")]

            direction = arguments.get("direction", "both")

            lineage = ctx.get_lineage(memory_id=memory_id, direction=direction)

            if not lineage:
                return [
                    TextContent(type="text", text=f"Memory not found or no lineage: {memory_id}")
                ]

            output = [f"Lineage for {memory_id[:12]}..."]

            # Show ancestors (history)
            ancestors = lineage.get("ancestors", [])
            if ancestors:
                output.append(f"\nAncestors ({len(ancestors)}):")
                for anc in ancestors:
                    rel = anc.get("relation", "unknown")
                    aid = anc.get("id", "")[:12]
                    depth = anc.get("depth", 1)
                    output.append(f"  {'  ' * (depth - 1)}↑ [{rel}] {aid}...")
            elif direction in ("ancestors", "both"):
                output.append("\nNo ancestors (this is the original).")

            # Show descendants (evolutions)
            descendants = lineage.get("descendants", [])
            if descendants:
                output.append(f"\nDescendants ({len(descendants)}):")
                for desc in descendants:
                    rel = desc.get("relation", "unknown")
                    did = desc.get("id", "")[:12]
                    depth = desc.get("depth", 1)
                    output.append(f"  {'  ' * (depth - 1)}↓ [{rel}] {did}...")
            elif direction in ("descendants", "both"):
                output.append("\nNo descendants (not evolved yet).")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_link":
            from_id = arguments.get("from_id", "")
            to_id = arguments.get("to_id", "")
            relation = arguments.get("relation", "")

            if not from_id:
                return [TextContent(type="text", text="Error: from_id is required")]
            if not to_id:
                return [TextContent(type="text", text="Error: to_id is required")]
            if not relation:
                return [TextContent(type="text", text="Error: relation is required")]

            weight = arguments.get("weight", 1.0)
            bidirectional = arguments.get("bidirectional", False)

            success = ctx.link(
                from_memory_id=from_id,
                to_memory_id=to_id,
                relation=relation,
                weight=weight,
                bidirectional=bidirectional,
            )

            if not success:
                return [
                    TextContent(
                        type="text", text="Error: Failed to create link. Memories may not exist."
                    )
                ]

            output = [
                "Link created successfully.",
                f"From: {from_id[:12]}...",
                f"To: {to_id[:12]}...",
                f"Relation: {relation}",
                f"Weight: {weight}",
            ]
            if bidirectional:
                output.append("Bidirectional: Yes")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_related":
            memory_id = arguments.get("memory_id", "")

            if not memory_id:
                return [TextContent(type="text", text="Error: memory_id is required")]

            relation = arguments.get("relation")
            max_depth = arguments.get("max_depth", 1)

            related = ctx.get_related(
                memory_id=memory_id,
                relation=relation,
                max_depth=max_depth,
            )

            if not related:
                msg = f"No related memories found for {memory_id[:12]}..."
                if relation:
                    msg += f" with relation '{relation}'"
                return [TextContent(type="text", text=msg)]

            output = [f"Related memories for {memory_id[:12]}..."]
            if relation:
                output[0] += f" (relation: {relation})"
            output.append("")

            for item in related:
                rel = item.get("relation", "unknown")
                rid = item.get("id", "")[:12]
                depth = item.get("depth", 1)
                direction = item.get("direction", "outgoing")
                arrow = "→" if direction == "outgoing" else "←"

                line = f"  {arrow} [{rel}] {rid}..."
                if depth > 1:
                    line += f" (depth: {depth})"
                output.append(line)

                # Show preview of related memory content if available
                if item.get("summary"):
                    output.append(f"    {item['summary']}")
                elif item.get("content"):
                    content_preview = (
                        item["content"][:60] + "..."
                        if len(item.get("content", "")) > 60
                        else item.get("content", "")
                    )
                    output.append(f"    {content_preview}")

            return [TextContent(type="text", text="\n".join(output))]

        # =====================================================================
        # Index Management Tool Handlers
        # =====================================================================
        elif name == "contextfs_cleanup_indexes":
            dry_run = arguments.get("dry_run", False)

            result = ctx.cleanup_indexes(dry_run=dry_run)

            if not result["removed"]:
                return [
                    TextContent(
                        type="text",
                        text="No stale indexes found. All indexes are valid.",
                    )
                ]

            reason_labels = {
                "no_path": "No path stored",
                "path_missing": "Path missing",
                "not_git_repo": "Not a git repo",
            }

            output = []
            if dry_run:
                output.append(f"Found {len(result['removed'])} stale index(es) (dry run):\n")
            else:
                output.append(f"Removed {len(result['removed'])} stale index(es):\n")

            for idx in result["removed"]:
                repo_name = (
                    idx["repo_path"].split("/")[-1] if idx["repo_path"] else idx["namespace_id"]
                )
                reason = reason_labels.get(idx.get("reason", ""), idx.get("reason", "unknown"))
                output.append(f"  • {repo_name}: {reason}")
                output.append(f"    Path: {idx['repo_path'] or 'none'}")
                output.append(
                    f"    Files: {idx['files_indexed']}, Commits: {idx['commits_indexed']}"
                )

            output.append(f"\nKept {len(result['kept'])} valid index(es)")

            return [TextContent(type="text", text="\n".join(output))]

        elif name == "contextfs_delete_index":
            repo_path = arguments.get("repo_path")
            namespace_id = arguments.get("namespace_id")

            if not repo_path and not namespace_id:
                return [
                    TextContent(
                        type="text",
                        text="Error: Provide either repo_path or namespace_id",
                    )
                ]

            deleted = ctx.delete_index(namespace_id=namespace_id, repo_path=repo_path)

            if deleted:
                target = repo_path or namespace_id
                return [
                    TextContent(
                        type="text",
                        text=f"Successfully deleted index for: {target}",
                    )
                ]
            else:
                return [
                    TextContent(
                        type="text",
                        text=f"Index not found for: {repo_path or namespace_id}",
                    )
                ]

        elif name == "contextfs_rebuild_chroma":
            result = ctx.rebuild_chromadb()

            if result.get("success"):
                return [
                    TextContent(
                        type="text",
                        text=f"ChromaDB rebuilt successfully!\nMemories rebuilt: {result.get('memories_rebuilt', 0)}",
                    )
                ]
            else:
                return [
                    TextContent(
                        type="text",
                        text=f"Failed to rebuild ChromaDB: {result.get('error', 'Unknown error')}",
                    )
                ]

        elif name == "contextfs_reindex":
            all_repos = arguments.get("all_repos", False)
            repo_path = arguments.get("repo_path")
            incremental = arguments.get("incremental", True)
            mode = arguments.get("mode", "all")

            if all_repos:
                # Reindex all repos in database
                result = ctx.reindex_all_repos(
                    incremental=incremental,
                    mode=mode,
                )

                output = [
                    f"Repos found: {result['repos_found']}",
                    f"Repos indexed: {result['repos_indexed']}",
                    f"Repos failed: {result['repos_failed']}",
                    f"Total files: {result['total_files']}",
                    f"Total memories: {result['total_memories']}",
                ]

                if result["errors"]:
                    output.append("\nErrors:")
                    for err in result["errors"]:
                        output.append(f"  - {err}")

                return [TextContent(type="text", text="\n".join(output))]
            else:
                # Reindex single repo
                from pathlib import Path

                path = Path(repo_path) if repo_path else None
                result = ctx.index_repository(
                    repo_path=path,
                    incremental=incremental,
                    mode=mode,
                )

                return [
                    TextContent(
                        type="text",
                        text=f"Repository reindexed!\nFiles indexed: {result.get('files_indexed', 0)}\nMemories created: {result.get('memories_created', 0)}",
                    )
                ]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def run_server():
    """Run the MCP server."""
    global _ctx, _session_started

    # Set up signal handlers for clean shutdown
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def signal_handler():
        shutdown_event.set()

    # Handle SIGINT and SIGTERM
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except (NotImplementedError, OSError):
            # Windows doesn't support add_signal_handler
            pass

    try:
        async with stdio_server() as (read_stream, write_stream):
            # Run server until shutdown signal
            server_task = asyncio.create_task(
                server.run(read_stream, write_stream, server.create_initialization_options())
            )
            shutdown_task = asyncio.create_task(shutdown_event.wait())

            done, pending = await asyncio.wait(
                [server_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    except asyncio.CancelledError:
        # Normal shutdown - don't treat as error
        pass
    except Exception:
        # Suppress other exceptions during shutdown
        pass
    finally:
        # Clean shutdown - close ContextFS if open
        if _ctx is not None:
            try:
                _ctx.close()
            except Exception:
                pass
        _ctx = None
        _session_started = False


def main():
    """Entry point for MCP server."""
    try:
        asyncio.run(run_server())
    except (KeyboardInterrupt, SystemExit, BrokenPipeError):
        pass
    except Exception:
        # Exit cleanly without error
        pass
    # Always exit with success code to prevent "MCP server failed" messages
    # This is normal shutdown - stdin closing is how Claude Code signals exit
    os._exit(0)


if __name__ == "__main__":
    main()
