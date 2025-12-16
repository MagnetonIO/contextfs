"""
MCP Server for ContextFS.

Provides memory operations via Model Context Protocol.
Works with Claude Desktop, Claude Code, and any MCP client.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import GetPromptResult, Prompt, PromptArgument, PromptMessage, TextContent, Tool

from contextfs.core import ContextFS
from contextfs.schemas import MemoryType

# Global ContextFS instance
_ctx: ContextFS | None = None

# Source tool name (detected from environment or set by client)
_source_tool: str = "claude-desktop"  # Default for Claude Desktop MCP

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

    # Auto-start session for Claude Desktop
    if not _session_started and get_source_tool() == "claude-desktop":
        _ctx.start_session(tool="claude-desktop")
        _session_started = True

    return _ctx


def get_source_tool() -> str:
    """Get source tool name."""
    import os

    # Allow override via environment
    return os.environ.get("CONTEXTFS_SOURCE_TOOL", _source_tool)


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
                        "enum": [
                            "fact",
                            "decision",
                            "procedural",
                            "episodic",
                            "user",
                            "code",
                            "error",
                        ],
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
            description="Search memories using semantic similarity. Supports cross-repo search.",
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
                        "enum": [
                            "fact",
                            "decision",
                            "procedural",
                            "episodic",
                            "user",
                            "code",
                            "error",
                        ],
                        "description": "Filter by type",
                    },
                    "cross_repo": {
                        "type": "boolean",
                        "description": "Search across all repos (default: current repo only)",
                        "default": False,
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
                        "enum": [
                            "fact",
                            "decision",
                            "procedural",
                            "episodic",
                            "user",
                            "code",
                            "error",
                        ],
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
            name="contextfs_index",
            description="Start indexing the current repository's codebase in background. Use contextfs_index_status to check progress.",
            inputSchema={
                "type": "object",
                "properties": {
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
            cross_repo = arguments.get("cross_repo", False)
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

            if not repos:
                return [TextContent(type="text", text="No repositories found.")]

            output = ["Repositories with memories:"]
            for r in repos:
                output.append(f"  • {r['source_repo']} ({r['memory_count']} memories)")

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

        elif name == "contextfs_index":
            from pathlib import Path

            incremental = arguments.get("incremental", True)
            force = arguments.get("force", False)

            # Get current working directory
            cwd = Path.cwd()
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

            # Check if already indexed (unless force)
            status = ctx.get_index_status()
            if status and status.get("indexed") and not force:
                return [
                    TextContent(
                        type="text",
                        text=f"Repository '{repo_name}' already indexed.\n"
                        f"Files: {status.get('total_files', 0)}\n"
                        f"Use force=true to re-index.",
                    )
                ]

            # Clear index if forcing
            if force:
                ctx.clear_index()

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

            async def run_indexing():
                """Run indexing in background."""
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
                        repo_path=cwd,
                        incremental=incremental,
                        on_progress=on_progress,
                    )
                    _indexing_state.result = result
                    _indexing_state.running = False
                except Exception as e:
                    _indexing_state.error = str(e)
                    _indexing_state.running = False

            # Start background task
            _indexing_state.task = asyncio.create_task(run_indexing())

            return [
                TextContent(
                    type="text",
                    text=f"Started indexing '{repo_name}' in background.\n"
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
                        f"Chunks created: {result.get('chunks_created', 0)}\n"
                        f"Skipped: {result.get('files_skipped', 0)}",
                    )
                ]
            else:
                # Check if there's existing index data
                status = ctx.get_index_status()
                if status and status.get("indexed"):
                    return [
                        TextContent(
                            type="text",
                            text=f"No indexing in progress.\n"
                            f"Repository indexed: {status.get('total_files', 0)} files",
                        )
                    ]
                return [
                    TextContent(
                        type="text",
                        text="No indexing in progress. Use contextfs_index to start.",
                    )
                ]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def run_server():
    """Run the MCP server."""
    global _ctx, _session_started
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        # Auto-save session on server shutdown
        if _ctx is not None and _session_started:
            try:
                _ctx.end_session(summary="Auto-saved on MCP server shutdown")
            except Exception:
                pass  # Don't fail on cleanup errors


def main():
    """Entry point for MCP server."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
