"""
CLI for ContextFS.

Provides command-line access to memory operations.
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from contextfs.core import ContextFS
from contextfs.schemas import MemoryType

app = typer.Typer(
    name="contextfs",
    help="ContextFS - Universal AI Memory Layer",
    no_args_is_help=True,
)
console = Console()


def get_ctx() -> ContextFS:
    """Get ContextFS instance."""
    return ContextFS(auto_load=True)


@app.command()
def save(
    content: str = typer.Argument(..., help="Content to save"),
    type: str = typer.Option("fact", "--type", "-t", help="Memory type"),
    tags: str | None = typer.Option(None, "--tags", help="Comma-separated tags"),
    summary: str | None = typer.Option(None, "--summary", "-s", help="Brief summary"),
):
    """Save a memory."""
    ctx = get_ctx()

    tag_list = [t.strip() for t in tags.split(",")] if tags else []

    try:
        memory_type = MemoryType(type)
    except ValueError:
        console.print(f"[red]Invalid type: {type}[/red]")
        raise typer.Exit(1)

    memory = ctx.save(
        content=content,
        type=memory_type,
        tags=tag_list,
        summary=summary,
    )

    console.print("[green]Memory saved[/green]")
    console.print(f"ID: {memory.id}")
    console.print(f"Type: {memory.type.value}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results"),
    type: str | None = typer.Option(None, "--type", "-t", help="Filter by type"),
):
    """Search memories."""
    ctx = get_ctx()

    type_filter = MemoryType(type) if type else None
    results = ctx.search(query, limit=limit, type=type_filter)

    if not results:
        console.print("[yellow]No memories found[/yellow]")
        return

    table = Table(title="Search Results")
    table.add_column("ID", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Type", style="magenta")
    table.add_column("Content")
    table.add_column("Tags", style="blue")

    for r in results:
        table.add_row(
            r.memory.id[:8],
            f"{r.score:.2f}",
            r.memory.type.value,
            r.memory.content[:60] + "..." if len(r.memory.content) > 60 else r.memory.content,
            ", ".join(r.memory.tags) if r.memory.tags else "",
        )

    console.print(table)


@app.command()
def recall(
    memory_id: str = typer.Argument(..., help="Memory ID (can be partial)"),
):
    """Recall a specific memory."""
    ctx = get_ctx()
    memory = ctx.recall(memory_id)

    if not memory:
        console.print(f"[red]Memory not found: {memory_id}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]ID:[/cyan] {memory.id}")
    console.print(f"[cyan]Type:[/cyan] {memory.type.value}")
    console.print(f"[cyan]Created:[/cyan] {memory.created_at}")
    if memory.summary:
        console.print(f"[cyan]Summary:[/cyan] {memory.summary}")
    if memory.tags:
        console.print(f"[cyan]Tags:[/cyan] {', '.join(memory.tags)}")
    console.print(f"\n[cyan]Content:[/cyan]\n{memory.content}")


@app.command("list")
def list_memories(
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results"),
    type: str | None = typer.Option(None, "--type", "-t", help="Filter by type"),
):
    """List recent memories."""
    ctx = get_ctx()

    type_filter = MemoryType(type) if type else None
    memories = ctx.list_recent(limit=limit, type=type_filter)

    if not memories:
        console.print("[yellow]No memories found[/yellow]")
        return

    table = Table(title="Recent Memories")
    table.add_column("ID", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Created", style="green")
    table.add_column("Content/Summary")

    for m in memories:
        content = m.summary or m.content[:50] + "..."
        table.add_row(
            m.id[:8],
            m.type.value,
            m.created_at.strftime("%Y-%m-%d %H:%M"),
            content,
        )

    console.print(table)


@app.command()
def delete(
    memory_id: str = typer.Argument(..., help="Memory ID to delete"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Delete a memory."""
    ctx = get_ctx()

    memory = ctx.recall(memory_id)
    if not memory:
        console.print(f"[red]Memory not found: {memory_id}[/red]")
        raise typer.Exit(1)

    if not confirm:
        console.print(f"About to delete: {memory.content[:100]}...")
        if not typer.confirm("Are you sure?"):
            raise typer.Abort()

    if ctx.delete(memory.id):
        console.print(f"[green]Memory deleted: {memory.id}[/green]")
    else:
        console.print("[red]Failed to delete memory[/red]")


@app.command()
def sessions(
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results"),
    tool: str | None = typer.Option(None, "--tool", help="Filter by tool"),
    label: str | None = typer.Option(None, "--label", help="Filter by label"),
):
    """List recent sessions."""
    ctx = get_ctx()

    session_list = ctx.list_sessions(limit=limit, tool=tool, label=label)

    if not session_list:
        console.print("[yellow]No sessions found[/yellow]")
        return

    table = Table(title="Recent Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Tool", style="magenta")
    table.add_column("Label", style="blue")
    table.add_column("Started", style="green")
    table.add_column("Messages")

    for s in session_list:
        table.add_row(
            s.id[:12],
            s.tool,
            s.label or "",
            s.started_at.strftime("%Y-%m-%d %H:%M"),
            str(len(s.messages)),
        )

    console.print(table)


@app.command("save-session")
def save_session(
    label: str = typer.Option(None, "--label", "-l", help="Session label"),
    transcript: Path | None = typer.Option(None, "--transcript", "-t", help="Path to transcript JSONL file"),
):
    """Save the current session to memory (for use with hooks)."""
    import json
    import sys

    ctx = get_ctx()

    # Try to read hook input from stdin for transcript path
    transcript_path = transcript
    if not transcript_path and not sys.stdin.isatty():
        try:
            hook_input = json.load(sys.stdin)
            if "transcript_path" in hook_input:
                transcript_path = Path(hook_input["transcript_path"]).expanduser()
        except Exception:
            pass

    # Get or create session
    session = ctx.get_current_session()
    if not session:
        session = ctx.start_session(tool="claude-code", label=label)
    elif label:
        session.label = label

    # If we have a transcript path, read and save messages
    if transcript_path and transcript_path.exists():
        try:
            with open(transcript_path) as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get("type") == "human":
                            ctx.log_message("user", entry.get("message", {}).get("content", ""))
                        elif entry.get("type") == "assistant":
                            content = entry.get("message", {}).get("content", "")
                            if isinstance(content, list):
                                # Handle content blocks
                                text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
                                content = "\n".join(text_parts)
                            ctx.log_message("assistant", content)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read transcript: {e}[/yellow]")

    # End session to save it
    ctx.end_session(summary=f"Auto-saved session: {label or 'unnamed'}")

    console.print(f"[green]Session saved[/green]")
    console.print(f"ID: {session.id}")
    if label:
        console.print(f"Label: {label}")


@app.command()
def status():
    """Show ContextFS status."""
    ctx = get_ctx()

    console.print("[bold]ContextFS Status[/bold]\n")
    console.print(f"Data directory: {ctx.data_dir}")
    console.print(f"Namespace: {ctx.namespace_id}")

    # Count memories
    memories = ctx.list_recent(limit=1000)
    console.print(f"Total memories: {len(memories)}")

    # Count by type
    type_counts = {}
    for m in memories:
        type_counts[m.type.value] = type_counts.get(m.type.value, 0) + 1

    if type_counts:
        console.print("\nMemories by type:")
        for t, c in sorted(type_counts.items()):
            console.print(f"  {t}: {c}")

    # RAG stats
    try:
        rag_stats = ctx.rag.get_stats()
        console.print(f"\nVector store: {rag_stats['total_memories']} embeddings")
        console.print(f"Embedding model: {rag_stats['embedding_model']}")
    except Exception:
        console.print("\n[yellow]Vector store not initialized[/yellow]")

    # Current session
    session = ctx.get_current_session()
    if session:
        console.print(f"\nActive session: {session.id[:12]}")
        console.print(f"  Messages: {len(session.messages)}")


def find_git_root(start_path: Path) -> Path | None:
    """Find the git root directory from start_path."""
    current = start_path.resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    return None


@app.command()
def index(
    path: Path | None = typer.Argument(None, help="Repository path (auto-detects git root)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-index even if already indexed"),
    incremental: bool = typer.Option(True, "--incremental/--full", help="Only index new/changed files"),
):
    """Index a repository's codebase for semantic search."""
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

    # Determine repo path
    start_path = path or Path.cwd()
    repo_path = find_git_root(start_path)

    if not repo_path:
        # Not in a git repo, use the provided path or cwd
        repo_path = start_path.resolve()
        console.print(f"[yellow]Not a git repository, indexing: {repo_path}[/yellow]")
    else:
        console.print(f"[cyan]Found git repository: {repo_path}[/cyan]")

    if not repo_path.exists():
        console.print(f"[red]Path does not exist: {repo_path}[/red]")
        raise typer.Exit(1)

    ctx = get_ctx()

    # Check if already indexed
    status = ctx.get_index_status()
    if status and status.indexed and not force:
        console.print(f"\n[yellow]Repository already indexed:[/yellow]")
        console.print(f"  Files: {status.files_indexed}")
        console.print(f"  Memories: {status.memories_created}")
        console.print(f"  Indexed at: {status.indexed_at}")
        console.print(f"\n[dim]Use --force to re-index[/dim]")
        return

    if force:
        console.print("[yellow]Force re-indexing...[/yellow]")
        ctx.clear_index()

    # Index with progress
    console.print(f"\n[bold]Indexing {repo_path.name}...[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Discovering files...", total=None)

        def on_progress(current: int, total: int, filename: str):
            progress.update(task, total=total, completed=current, description=f"[cyan]{filename[:50]}[/cyan]")

        result = ctx.index_repository(
            repo_path=repo_path,
            on_progress=on_progress,
            incremental=incremental,
        )

    # Display results
    console.print(f"\n[green]✅ Indexing complete![/green]\n")

    table = Table(title="Indexing Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")

    table.add_row("Files discovered", str(result.get("files_discovered", 0)))
    table.add_row("Files indexed", str(result.get("files_indexed", 0)))
    table.add_row("Commits indexed", str(result.get("commits_indexed", 0)))
    table.add_row("Memories created", str(result.get("memories_created", 0)))
    table.add_row("Skipped (unchanged)", str(result.get("skipped", 0)))

    console.print(table)

    if result.get("errors"):
        console.print(f"\n[yellow]Warnings: {len(result['errors'])} files had errors[/yellow]")


@app.command()
def init(
    path: Path | None = typer.Argument(None, help="Directory to initialize"),
):
    """Initialize ContextFS in a directory."""
    target = path or Path.cwd()
    ctx_dir = target / ".contextfs"

    if ctx_dir.exists():
        console.print(f"[yellow]ContextFS already initialized at {ctx_dir}[/yellow]")
        return

    ctx_dir.mkdir(parents=True, exist_ok=True)

    # Add to .gitignore
    gitignore = target / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text()
        if ".contextfs/" not in content:
            with gitignore.open("a") as f:
                f.write("\n# ContextFS\n.contextfs/\n")

    console.print(f"[green]Initialized ContextFS at {ctx_dir}[/green]")


@app.command()
def serve():
    """Start the MCP server."""
    from contextfs.mcp_server import main as mcp_main

    console.print("[green]Starting ContextFS MCP server...[/green]")
    mcp_main()


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
):
    """Start the web UI server."""
    import uvicorn

    from contextfs.web.server import create_app

    console.print(f"[green]Starting ContextFS Web UI at http://{host}:{port}[/green]")
    app = create_app()
    uvicorn.run(app, host=host, port=port)


@app.command("install-claude-desktop")
def install_claude_desktop(
    uninstall: bool = typer.Option(False, "--uninstall", help="Remove from Claude Desktop"),
):
    """Install ContextFS MCP server for Claude Desktop."""
    import json
    import os
    import platform
    import shutil
    import sys

    def get_claude_desktop_config_path() -> Path:
        system = platform.system()
        if system == "Darwin":
            return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
        elif system == "Windows":
            return Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json"
        else:
            return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"

    def find_contextfs_mcp_path() -> str | None:
        path = shutil.which("contextfs-mcp")
        if path:
            if platform.system() != "Windows":
                path = os.path.realpath(path)
            return path
        return None

    config_path = get_claude_desktop_config_path()

    if uninstall:
        if not config_path.exists():
            console.print("[yellow]Claude Desktop config not found.[/yellow]")
            return
        with open(config_path) as f:
            config = json.load(f)
        if "mcpServers" in config and "contextfs" in config["mcpServers"]:
            del config["mcpServers"]["contextfs"]
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            console.print("[green]✅ ContextFS removed from Claude Desktop config.[/green]")
        else:
            console.print("[yellow]ContextFS not found in config.[/yellow]")
        return

    # Install
    console.print("[bold]Installing ContextFS MCP for Claude Desktop...[/bold]\n")

    contextfs_path = find_contextfs_mcp_path()
    if contextfs_path:
        console.print(f"Found contextfs-mcp: [cyan]{contextfs_path}[/cyan]")
    else:
        contextfs_path = sys.executable
        console.print(f"Using Python fallback: [cyan]{contextfs_path}[/cyan]")

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}
        config_path.parent.mkdir(parents=True, exist_ok=True)

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Set up MCP config
    if find_contextfs_mcp_path():
        config["mcpServers"]["contextfs"] = {
            "command": contextfs_path,
            "env": {"CONTEXTFS_SOURCE_TOOL": "claude-desktop"}
        }
    else:
        config["mcpServers"]["contextfs"] = {
            "command": sys.executable,
            "args": ["-m", "contextfs.mcp_server"],
            "env": {"CONTEXTFS_SOURCE_TOOL": "claude-desktop"}
        }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    console.print("\n[green]✅ ContextFS MCP server installed![/green]")
    console.print(f"\nConfig: [dim]{config_path}[/dim]")
    console.print("\n[yellow]⚠️  Restart Claude Desktop to activate.[/yellow]")

    console.print("\n[bold]Available MCP tools:[/bold]")
    tools = [
        ("contextfs_save", "Save memories (with project grouping)"),
        ("contextfs_search", "Search (cross-repo, by project/tool)"),
        ("contextfs_list", "List recent memories"),
        ("contextfs_list_repos", "List repositories"),
        ("contextfs_list_projects", "List projects"),
        ("contextfs_list_tools", "List source tools"),
        ("contextfs_recall", "Recall by ID"),
    ]
    for name, desc in tools:
        console.print(f"  • [cyan]{name}[/cyan] - {desc}")


if __name__ == "__main__":
    app()
