"""
CLI for ContextFS.

Provides command-line access to memory operations.
"""

from importlib.metadata import version as get_version
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from contextfs.core import ContextFS
from contextfs.schemas import MemoryType


def version_callback(value: bool):
    if value:
        print(f"contextfs {get_version('contextfs')}")
        raise typer.Exit()


app = typer.Typer(
    name="contextfs",
    help="ContextFS - Universal AI Memory Layer",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    pass


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
        source_tool="contextfs-cli",
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
def prune(
    days: int | None = typer.Option(None, "--days", "-d", help="Delete memories older than N days"),
    type: str | None = typer.Option(None, "--type", "-t", help="Only delete memories of this type"),
    repo: str | None = typer.Option(
        None, "--repo", "-r", help="Only delete memories from this repo"
    ),
    auto_indexed: bool = typer.Option(
        False, "--auto-indexed", help="Only delete auto-indexed memories"
    ),
    all_memories: bool = typer.Option(
        False, "--all", help="Delete ALL memories (use with caution)"
    ),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be deleted without deleting"
    ),
):
    """Prune memories based on criteria.

    Examples:
        contextfs prune --days 30                    # Delete memories older than 30 days
        contextfs prune --repo myproject --all      # Delete all memories from a repo
        contextfs prune --auto-indexed --days 7     # Delete week-old auto-indexed memories
        contextfs prune --all --yes                 # Delete ALL memories (dangerous!)
    """
    from datetime import datetime, timedelta

    # Require at least one filter unless --all is specified
    if not all_memories and not days and not repo and not type and not auto_indexed:
        console.print(
            "[red]Error: Specify at least one filter (--days, --repo, --type, --auto-indexed) or use --all[/red]"
        )
        raise typer.Exit(1)

    ctx = get_ctx()
    cutoff = datetime.now() - timedelta(days=days) if days else None

    # Get all memories
    memories = ctx.list_recent(limit=10000)

    # Filter by criteria
    to_delete = []
    for m in memories:
        # If --all without other filters, include everything
        if all_memories and not days and not repo and not type and not auto_indexed:
            to_delete.append(m)
            continue

        # Check age
        if cutoff and m.created_at >= cutoff:
            continue

        # Check repo filter
        if repo and m.source_repo != repo:
            continue

        # Check type filter
        if type and m.type.value != type:
            continue

        # Check auto-indexed filter
        if auto_indexed and not m.metadata.get("auto_indexed"):
            continue

        # If we got here with filters, add it
        if days or repo or type or auto_indexed:
            to_delete.append(m)

    if not to_delete:
        console.print("[yellow]No memories match the criteria[/yellow]")
        return

    # Build filter description
    filters = []
    if days:
        filters.append(f"older than {days} days")
    if repo:
        filters.append(f"from repo '{repo}'")
    if type:
        filters.append(f"type={type}")
    if auto_indexed:
        filters.append("auto-indexed")
    if all_memories and not filters:
        filters.append("ALL MEMORIES")
    filter_desc = ", ".join(filters) if filters else "all"

    # Show what would be deleted
    console.print(f"\n[bold]Found {len(to_delete)} memories to delete ({filter_desc}):[/bold]")

    table = Table()
    table.add_column("ID", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Repo", style="blue")
    table.add_column("Created", style="green")
    table.add_column("Content/Summary")

    for m in to_delete[:20]:  # Show first 20
        content = m.summary or m.content[:50] + "..."
        table.add_row(
            m.id[:8],
            m.type.value,
            m.source_repo or "-",
            m.created_at.strftime("%Y-%m-%d"),
            content,
        )

    console.print(table)

    if len(to_delete) > 20:
        console.print(f"[dim]... and {len(to_delete) - 20} more[/dim]")

    if dry_run:
        console.print("\n[yellow]Dry run - no memories deleted[/yellow]")
        return

    if not confirm:
        warning = ""
        if all_memories and not days and not repo:
            warning = "[bold red]WARNING: This will delete ALL memories![/bold red]\n"
        console.print(warning)
        if not typer.confirm(f"Delete {len(to_delete)} memories?"):
            raise typer.Abort()

    # Delete memories
    deleted = 0
    for m in to_delete:
        if ctx.delete(m.id):
            deleted += 1

    console.print(f"\n[green]Deleted {deleted} memories[/green]")


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
    transcript: Path | None = typer.Option(
        None, "--transcript", "-t", help="Path to transcript JSONL file"
    ),
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
                                text_parts = [
                                    c.get("text", "") for c in content if c.get("type") == "text"
                                ]
                                content = "\n".join(text_parts)
                            ctx.log_message("assistant", content)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read transcript: {e}[/yellow]")

    # Set summary and end session to save it
    session.summary = f"Auto-saved session: {label or 'unnamed'}"
    ctx.end_session(generate_summary=False)

    console.print("[green]Session saved[/green]")
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
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-index even if already indexed"
    ),
    incremental: bool = typer.Option(
        True, "--incremental/--full", help="Only index new/changed files"
    ),
    mode: str = typer.Option(
        "all",
        "--mode",
        "-m",
        help="Index mode: 'all' (files+commits), 'files_only', or 'commits_only'",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Quiet mode for hooks (minimal output)"
    ),
    background: bool = typer.Option(
        False, "--background", "-b", help="Run indexing in background (for hooks)"
    ),
    reset_chroma: bool = typer.Option(
        False, "--reset-chroma", help="Reset ChromaDB before indexing (use if corrupted)"
    ),
    allow_dir: bool = typer.Option(
        False,
        "--allow-dir",
        help="Allow indexing non-git directories (use index-dir for multiple repos)",
    ),
):
    """Index a repository's codebase for semantic search.

    By default, only indexes git repositories. Use --allow-dir to index
    non-git directories, or use 'index-dir' command to scan for multiple repos.
    """
    import subprocess
    import sys

    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

    # Determine repo path
    start_path = path or Path.cwd()
    repo_path = find_git_root(start_path)

    # Background mode: spawn subprocess and return immediately
    if background:
        cmd = [sys.executable, "-m", "contextfs.cli", "index", "--quiet"]
        if force:
            cmd.append("--force")
        if not incremental:
            cmd.append("--full")
        if mode != "all":
            cmd.extend(["--mode", mode])
        if allow_dir:
            cmd.append("--allow-dir")
        if repo_path:
            cmd.append(str(repo_path))

        # Start detached subprocess
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        if not quiet:
            console.print("[cyan]Indexing started in background[/cyan]")
        return

    if not repo_path:
        # Not in a git repo
        if not allow_dir:
            console.print(f"[red]Error: Not a git repository: {start_path.resolve()}[/red]")
            console.print("[yellow]Use --allow-dir to index non-git directories,[/yellow]")
            console.print(
                "[yellow]or use 'contextfs index-dir' to scan for multiple repos.[/yellow]"
            )
            raise typer.Exit(1)
        repo_path = start_path.resolve()
        if not quiet:
            console.print(f"[yellow]Indexing non-git directory: {repo_path}[/yellow]")
    else:
        if not quiet:
            console.print(f"[cyan]Found git repository: {repo_path}[/cyan]")

    if not repo_path.exists():
        if not quiet:
            console.print(f"[red]Path does not exist: {repo_path}[/red]")
        raise typer.Exit(1)

    ctx = get_ctx()

    # Validate mode parameter
    valid_modes = ["all", "files_only", "commits_only"]
    if mode not in valid_modes:
        console.print(f"[red]Invalid mode: {mode}. Must be one of: {', '.join(valid_modes)}[/red]")
        raise typer.Exit(1)

    # Reset ChromaDB if requested (fixes corruption issues)
    if reset_chroma:
        if not quiet:
            console.print("[yellow]Resetting ChromaDB (fixing corruption)...[/yellow]")
        if ctx.reset_chromadb():
            if not quiet:
                console.print("[green]ChromaDB reset successfully[/green]")
            force = True  # Force full re-index after reset
            incremental = False
        else:
            if not quiet:
                console.print("[red]Failed to reset ChromaDB[/red]")
            raise typer.Exit(1)

    # Check if already indexed
    status = ctx.get_index_status()
    already_indexed = status and status.indexed

    if force:
        if not quiet:
            console.print("[yellow]Force re-indexing (full)...[/yellow]")
        ctx.clear_index()
        incremental = False  # Force means full re-index
    elif already_indexed:
        if not quiet:
            console.print(
                f"[cyan]Running incremental index (previously indexed {status.files_indexed} files)[/cyan]"
            )
        # Continue with incremental=True (default)

    # Index with progress
    if not quiet:
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
                progress.update(
                    task,
                    total=total,
                    completed=current,
                    description=f"[cyan]{filename[:50]}[/cyan]",
                )

            result = ctx.index_repository(
                repo_path=repo_path,
                on_progress=on_progress,
                incremental=incremental,
                mode=mode,
            )
    else:
        # Quiet mode - no progress output
        result = ctx.index_repository(
            repo_path=repo_path,
            incremental=incremental,
            mode=mode,
        )

    # Display results
    if not quiet:
        console.print("\n[green]✅ Indexing complete![/green]\n")

        table = Table(title="Indexing Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green", justify="right")

        table.add_row("Mode", result.get("mode", mode))
        table.add_row("Files discovered", str(result.get("files_discovered", 0)))
        table.add_row("Files indexed", str(result.get("files_indexed", 0)))
        table.add_row("Commits indexed", str(result.get("commits_indexed", 0)))
        table.add_row("Memories created", str(result.get("memories_created", 0)))
        table.add_row("Skipped (unchanged)", str(result.get("skipped", 0)))

        console.print(table)

        if result.get("errors"):
            console.print(f"\n[yellow]Warnings: {len(result['errors'])} files had errors[/yellow]")


@app.command("index-dir")
def index_directory(
    path: Path = typer.Argument(..., help="Root directory to scan for git repositories"),
    max_depth: int = typer.Option(5, "--depth", "-d", help="Maximum directory depth to search"),
    project: str | None = typer.Option(
        None, "--project", "-p", help="Override project name for all repos"
    ),
    incremental: bool = typer.Option(
        True, "--incremental/--full", help="Only index new/changed files"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Discover repos without indexing"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
):
    """Recursively scan a directory for git repos and index each.

    Automatically detects:
    - Project groupings from directory structure
    - Programming languages and frameworks
    - CI/CD configurations and project types

    Examples:
        contextfs index-dir ~/Development
        contextfs index-dir ~/work/haven --project haven
        contextfs index-dir . --dry-run  # Preview without indexing
    """

    if not path.exists():
        console.print(f"[red]Path does not exist: {path}[/red]")
        raise typer.Exit(1)

    path = path.resolve()
    ctx = get_ctx()

    if dry_run:
        # Discovery mode only
        console.print(f"\n[bold]Discovering git repositories in {path}...[/bold]\n")

        repos = ctx.discover_repos(path, max_depth=max_depth)

        if not repos:
            console.print("[yellow]No git repositories found[/yellow]")
            return

        from rich.table import Table

        table = Table(title=f"Found {len(repos)} repositories")
        table.add_column("Repository", style="cyan")
        table.add_column("Project", style="magenta")
        table.add_column("Tags", style="blue")
        table.add_column("Path", style="dim")

        for repo in repos:
            table.add_row(
                repo["name"],
                repo["project"] or "-",
                ", ".join(repo["suggested_tags"][:4]) or "-",
                repo["relative_path"],
            )

        console.print(table)
        console.print("\n[dim]Run without --dry-run to index these repositories[/dim]")
        return

    # Full indexing mode
    if not quiet:
        console.print(f"\n[bold]Scanning {path} for git repositories...[/bold]\n")

    current_repo = {"name": "", "project": ""}

    def on_repo_start(repo_name: str, project_name: str | None) -> None:
        current_repo["name"] = repo_name
        current_repo["project"] = project_name or ""
        if not quiet:
            proj_str = f" (project: {project_name})" if project_name else ""
            console.print(f"\n[cyan]Indexing {repo_name}{proj_str}...[/cyan]")

    def on_repo_complete(repo_name: str, stats: dict) -> None:
        if not quiet and "error" not in stats:
            console.print(
                f"  [green]✓[/green] {stats['files_indexed']} files, "
                f"{stats.get('commits_indexed', 0)} commits, "
                f"{stats['memories_created']} memories"
            )
        elif not quiet and "error" in stats:
            console.print(f"  [red]✗ Error: {stats['error']}[/red]")

    result = ctx.index_directory(
        root_dir=path,
        max_depth=max_depth,
        on_repo_start=on_repo_start,
        on_repo_complete=on_repo_complete,
        incremental=incremental,
        project_override=project,
    )

    # Summary
    if not quiet:
        console.print("\n[green]✅ Directory indexing complete![/green]\n")

        from rich.table import Table

        table = Table(title="Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green", justify="right")

        table.add_row("Repositories found", str(result["repos_found"]))
        table.add_row("Repositories indexed", str(result["repos_indexed"]))
        table.add_row("Total files", str(result["total_files"]))
        table.add_row("Total commits", str(result.get("total_commits", 0)))
        table.add_row("Total memories", str(result["total_memories"]))

        console.print(table)

        # Show per-repo breakdown
        if result["repos"]:
            console.print("\n[bold]Per-repository breakdown:[/bold]")
            repo_table = Table()
            repo_table.add_column("Repository", style="cyan")
            repo_table.add_column("Project", style="magenta")
            repo_table.add_column("Files", justify="right")
            repo_table.add_column("Memories", justify="right")
            repo_table.add_column("Status", style="green")

            for repo in result["repos"]:
                status = "[red]Error[/red]" if "error" in repo else "[green]✓[/green]"
                repo_table.add_row(
                    repo["name"],
                    repo.get("project") or "-",
                    str(repo.get("files_indexed", 0)),
                    str(repo.get("memories_created", 0)),
                    status,
                )

            console.print(repo_table)


@app.command("discover")
def discover_repos(
    path: Path = typer.Argument(None, help="Root directory to scan (default: current directory)"),
    max_depth: int = typer.Option(5, "--depth", "-d", help="Maximum directory depth"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Discover git repositories without indexing.

    Useful for previewing what would be indexed.
    """
    import json as json_mod

    target = (path or Path.cwd()).resolve()

    if not target.exists():
        console.print(f"[red]Path does not exist: {target}[/red]")
        raise typer.Exit(1)

    ctx = get_ctx()
    repos = ctx.discover_repos(target, max_depth=max_depth)

    if json_output:
        # Convert Path objects to strings for JSON
        output = []
        for repo in repos:
            output.append(
                {
                    "name": repo["name"],
                    "path": str(repo["path"]),
                    "project": repo["project"],
                    "tags": repo["suggested_tags"],
                    "remote_url": repo.get("remote_url"),
                }
            )
        console.print(json_mod.dumps(output, indent=2))
        return

    if not repos:
        console.print("[yellow]No git repositories found[/yellow]")
        return

    from rich.table import Table

    table = Table(title=f"Found {len(repos)} repositories in {target}")
    table.add_column("Repository", style="cyan")
    table.add_column("Project", style="magenta")
    table.add_column("Languages/Frameworks", style="blue")
    table.add_column("Path", style="dim")

    for repo in repos:
        # Separate language and framework tags
        lang_tags = [t for t in repo["suggested_tags"] if t.startswith("lang:")]
        fw_tags = [t for t in repo["suggested_tags"] if t.startswith("framework:")]
        tags_str = ", ".join([t.split(":")[1] for t in lang_tags + fw_tags][:3])

        table.add_row(
            repo["name"],
            repo["project"] or "-",
            tags_str or "-",
            repo["relative_path"],
        )

    console.print(table)


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

    # MCP uses stdout for JSON-RPC - no printing allowed
    mcp_main()


@app.command("reset-chroma")
def reset_chroma(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Reset the ChromaDB database.

    Use this when ChromaDB becomes corrupted (e.g., after version upgrades).
    This will delete all vector embeddings but SQLite memories remain intact.

    After reset, run 'contextfs rebuild-chroma' to restore search from SQLite.

    Common symptoms of corruption:
    - "Error in compaction" errors during indexing
    - "mismatched types" errors
    - Indexing shows 0 files/memories created despite discovering files
    """
    if not confirm:
        console.print("[yellow]This will delete the ChromaDB vector database.[/yellow]")
        console.print("Your memories in SQLite will be preserved.")
        console.print("After reset, run 'contextfs rebuild-chroma' to restore search.\n")

        if not typer.confirm("Proceed with reset?"):
            console.print("[dim]Cancelled[/dim]")
            raise typer.Exit(0)

    ctx = get_ctx()

    console.print("[cyan]Resetting ChromaDB...[/cyan]")

    if ctx.reset_chromadb():
        console.print("[green]✅ ChromaDB reset successfully![/green]")
        console.print("\nNext steps:")
        console.print("  1. Run [cyan]contextfs rebuild-chroma[/cyan] to restore search (fast)")
        console.print("  2. Or run [cyan]contextfs index -f[/cyan] to fully re-index from files")
    else:
        console.print("[red]❌ Failed to reset ChromaDB[/red]")
        raise typer.Exit(1)


@app.command("rebuild-chroma")
def rebuild_chroma(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Rebuild ChromaDB search index from SQLite data.

    Use this to restore search capability after ChromaDB corruption.
    This is MUCH faster than 'index -f' because it rebuilds from SQLite
    rather than re-scanning all source files.

    This command:
    - Preserves all your memories (they're safe in SQLite)
    - Restores semantic search functionality
    - Does NOT re-index source files (use 'index' for that)

    Example:
        contextfs rebuild-chroma        # Interactive confirmation
        contextfs rebuild-chroma -y     # Skip confirmation
    """
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

    if not confirm:
        console.print("[cyan]This will rebuild ChromaDB from your SQLite memories.[/cyan]")
        console.print("This is safe - your memories will not be affected.\n")

        if not typer.confirm("Proceed with rebuild?"):
            console.print("[dim]Cancelled[/dim]")
            raise typer.Exit(0)

    ctx = get_ctx()

    console.print("\n[bold]Rebuilding ChromaDB from SQLite...[/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Rebuilding...", total=None)

        def on_progress(current: int, total: int):
            progress.update(
                task,
                total=total,
                completed=current,
                description=f"[cyan]Rebuilding memories ({current}/{total})[/cyan]",
            )

        result = ctx.rebuild_chromadb(on_progress=on_progress)

    if result.get("success"):
        console.print("\n[green]✅ ChromaDB rebuilt successfully![/green]\n")

        table = Table(title="Rebuild Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green", justify="right")

        table.add_row("Total memories", str(result.get("total", 0)))
        table.add_row("Rebuilt", str(result.get("rebuilt", 0)))
        if result.get("errors", 0) > 0:
            table.add_row("Errors", str(result.get("errors", 0)))

        console.print(table)
    else:
        console.print(f"[red]❌ Rebuild failed: {result.get('error', 'Unknown error')}[/red]")
        raise typer.Exit(1)


@app.command("reindex-all")
def reindex_all(
    incremental: bool = typer.Option(
        True, "--incremental/--full", help="Only index new/changed files"
    ),
    mode: str = typer.Option(
        "all", "--mode", "-m", help="Index mode: 'all', 'files_only', or 'commits_only'"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
):
    """Reindex all previously indexed repositories.

    Uses stored repo paths from index_status table to reindex.
    Useful for rebuilding indexes after ChromaDB corruption or upgrades.

    Examples:
        contextfs reindex-all                    # Incremental reindex all repos
        contextfs reindex-all --full             # Full reindex all repos
        contextfs reindex-all --mode files_only  # Only reindex files
    """

    ctx = get_ctx()

    # Progress callback for CLI output
    current_repo = {"name": "", "done": False}

    def on_progress(repo_name: str, current: int, total: int):
        if not quiet:
            if current_repo["name"] and not current_repo["done"]:
                # Previous repo finished without error
                pass
            current_repo["name"] = repo_name
            current_repo["done"] = False
            console.print(f"  [cyan]Indexing {repo_name}...[/cyan]", end=" ")

    result = ctx.reindex_all_repos(
        incremental=incremental,
        mode=mode,
        on_progress=on_progress if not quiet else None,
    )

    if result["repos_found"] == 0:
        console.print("[yellow]No indexed repositories found in database[/yellow]")
        return

    if not quiet:
        # Print final status for last repo
        if current_repo["name"]:
            console.print("[green]✓[/green]")

        if result["errors"]:
            console.print("\n[yellow]Errors:[/yellow]")
            for err in result["errors"]:
                console.print(f"  [yellow]⚠ {err}[/yellow]")

        console.print("\n[green]✅ Reindexing complete![/green]")
        console.print(f"  Successful: {result['repos_indexed']}")
        console.print(f"  Failed: {result['repos_failed']}")
        console.print(f"  Total files: {result['total_files']}")
        console.print(f"  Total memories: {result['total_memories']}")


@app.command("install-hooks")
def install_hooks(
    repo_path: str = typer.Argument(None, help="Repository path (default: current directory)"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing hooks"),
):
    """Install git hooks for automatic indexing.

    Installs post-commit and post-merge hooks that automatically
    run incremental indexing after commits and pulls.

    Examples:
        contextfs install-hooks              # Install to current repo
        contextfs install-hooks /path/to/repo
        contextfs install-hooks --force      # Overwrite existing hooks
    """
    import shutil

    # Determine target repo
    target = Path(repo_path).resolve() if repo_path else Path.cwd()

    # Verify it's a git repo
    git_dir = target / ".git"
    if not git_dir.exists():
        console.print(f"[red]Error: {target} is not a git repository[/red]")
        raise typer.Exit(1)

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)

    # Find source hooks (bundled with package)
    import contextfs

    pkg_dir = Path(contextfs.__file__).parent.parent.parent
    source_hooks_dir = pkg_dir / "hooks"

    # If not found in package, create hooks inline
    hooks = {
        "post-commit": """#!/bin/bash
# ContextFS Post-Commit Hook - Auto-index on commit
set -e
if command -v contextfs &> /dev/null; then
    CONTEXTFS="contextfs"
elif [ -f "$HOME/.local/bin/contextfs" ]; then
    CONTEXTFS="$HOME/.local/bin/contextfs"
else
    exit 0
fi
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
(cd "$REPO_ROOT" && $CONTEXTFS index --incremental --mode files_only --quiet 2>/dev/null &) &
exit 0
""",
        "post-merge": """#!/bin/bash
# ContextFS Post-Merge Hook - Auto-index on pull/merge
set -e
if command -v contextfs &> /dev/null; then
    CONTEXTFS="contextfs"
elif [ -f "$HOME/.local/bin/contextfs" ]; then
    CONTEXTFS="$HOME/.local/bin/contextfs"
else
    exit 0
fi
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
(cd "$REPO_ROOT" && $CONTEXTFS index --incremental --quiet 2>/dev/null &) &
exit 0
""",
    }

    console.print(f"Installing ContextFS git hooks to: [cyan]{target}[/cyan]\n")

    for hook_name, hook_content in hooks.items():
        hook_path = hooks_dir / hook_name
        source_path = source_hooks_dir / hook_name if source_hooks_dir.exists() else None

        # Check if hook exists
        if hook_path.exists() and not force:
            console.print(f"  [yellow]{hook_name}:[/yellow] exists (use --force to overwrite)")
            continue

        # Backup existing hook
        if hook_path.exists():
            backup_path = hooks_dir / f"{hook_name}.bak"
            shutil.copy(hook_path, backup_path)
            console.print(f"  [dim]{hook_name}: backed up to {hook_name}.bak[/dim]")

        # Write hook (prefer source file if available)
        if source_path and source_path.exists():
            shutil.copy(source_path, hook_path)
        else:
            hook_path.write_text(hook_content)

        # Make executable
        hook_path.chmod(0o755)
        console.print(f"  [green]{hook_name}:[/green] installed")

    console.print("\n[green]Done![/green] ContextFS will auto-index on:")
    console.print("  - git commit (indexes changed files)")
    console.print("  - git pull/merge (indexes new files and commits)")


@app.command("list-indexes")
def list_indexes():
    """Show full index status for all repositories.

    Displays a table with all indexed repositories, including:
    - Namespace ID
    - Repository name
    - Files indexed
    - Commits indexed
    - Memories created
    - Last indexed timestamp
    """
    from rich.table import Table

    ctx = get_ctx()
    indexes = ctx.list_indexes()

    if not indexes:
        console.print("[yellow]No indexed repositories found.[/yellow]")
        console.print("Run 'contextfs index' to index the current repository.")
        return

    table = Table(title="Full Index Status - All Repositories")
    table.add_column("Namespace", style="cyan")
    table.add_column("Repository", style="white")
    table.add_column("Files", justify="right", style="green")
    table.add_column("Commits", justify="right", style="green")
    table.add_column("Memories", justify="right", style="green")
    table.add_column("Indexed At", style="dim")

    total_files = 0
    total_commits = 0
    total_memories = 0

    for idx in sorted(indexes, key=lambda x: x.memories_created, reverse=True):
        # Shorten repo path to just repo name
        repo_name = idx.repo_path.split("/")[-1] if idx.repo_path else "unknown"
        # Format datetime
        indexed_at = str(idx.indexed_at)[:16] if idx.indexed_at else "N/A"

        table.add_row(
            idx.namespace_id[:16],
            repo_name,
            str(idx.files_indexed),
            str(idx.commits_indexed),
            str(idx.memories_created),
            indexed_at,
        )

        total_files += idx.files_indexed
        total_commits += idx.commits_indexed
        total_memories += idx.memories_created

    # Add totals row
    table.add_section()
    table.add_row(
        "TOTAL",
        f"{len(indexes)} repos",
        str(total_files),
        str(total_commits),
        str(total_memories),
        "",
    )

    console.print(table)


@app.command("cleanup-indexes")
def cleanup_indexes(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be removed without removing"
    ),
    include_non_git: bool = typer.Option(
        True, "--include-non-git/--git-only", help="Also remove non-git directories"
    ),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Remove stale indexes for repositories that no longer exist.

    Cleans up indexes for:
    - Repositories that have been deleted or moved
    - Paths that no longer exist on disk
    - Directories that are no longer git repositories (if --include-non-git)

    Examples:
        contextfs cleanup-indexes --dry-run     # Preview what would be removed
        contextfs cleanup-indexes -y            # Remove without confirmation
        contextfs cleanup-indexes --git-only    # Only keep git repositories
    """
    ctx = get_ctx()

    result = ctx.cleanup_indexes(dry_run=True)

    if not result["removed"]:
        console.print("[green]No stale indexes found. All indexes are valid.[/green]")
        return

    # Show what would be removed
    console.print(f"\n[bold]Found {len(result['removed'])} stale index(es):[/bold]\n")

    table = Table()
    table.add_column("Repository", style="cyan")
    table.add_column("Path", style="dim")
    table.add_column("Files", justify="right")
    table.add_column("Commits", justify="right")
    table.add_column("Reason", style="yellow")

    reason_labels = {
        "no_path": "No path stored",
        "path_missing": "Path missing",
        "not_git_repo": "Not a git repo",
    }

    for idx in result["removed"]:
        repo_name = idx["repo_path"].split("/")[-1] if idx["repo_path"] else idx["namespace_id"]
        reason = reason_labels.get(idx.get("reason", ""), idx.get("reason", "unknown"))
        table.add_row(
            repo_name,
            idx["repo_path"] or "-",
            str(idx["files_indexed"]),
            str(idx["commits_indexed"]),
            reason,
        )

    console.print(table)

    if dry_run:
        console.print("\n[yellow]Dry run - no indexes removed[/yellow]")
        return

    if not confirm:
        console.print()
        if not typer.confirm(f"Remove {len(result['removed'])} stale index(es)?"):
            console.print("[dim]Cancelled[/dim]")
            raise typer.Exit(0)

    # Actually delete
    result = ctx.cleanup_indexes(dry_run=False)

    console.print(f"\n[green]✅ Removed {len(result['removed'])} stale index(es)[/green]")
    console.print(f"[green]   Kept {len(result['kept'])} valid index(es)[/green]")


@app.command("delete-index")
def delete_index_cmd(
    path: str = typer.Argument(None, help="Repository path to delete index for"),
    namespace_id: str = typer.Option(None, "--id", help="Namespace ID to delete"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Delete a specific repository index.

    Delete by path or namespace ID.

    Examples:
        contextfs delete-index /path/to/repo
        contextfs delete-index --id repo-abc123
    """
    if not path and not namespace_id:
        console.print("[red]Error: Provide either a path or --id[/red]")
        raise typer.Exit(1)

    ctx = get_ctx()

    # Find the index to show details
    indexes = ctx.list_indexes()
    target_idx = None

    for idx in indexes:
        if (
            namespace_id
            and idx.namespace_id == namespace_id
            or path
            and idx.repo_path == str(Path(path).resolve())
        ):
            target_idx = idx
            break

    if not target_idx:
        console.print(f"[red]Index not found for: {path or namespace_id}[/red]")
        raise typer.Exit(1)

    repo_name = (
        target_idx.repo_path.split("/")[-1] if target_idx.repo_path else target_idx.namespace_id
    )

    if not confirm:
        console.print(f"\nAbout to delete index for: [cyan]{repo_name}[/cyan]")
        console.print(f"  Files: {target_idx.files_indexed}")
        console.print(f"  Commits: {target_idx.commits_indexed}")
        console.print(f"  Memories: {target_idx.memories_created}")
        console.print()

        if not typer.confirm("Delete this index?"):
            console.print("[dim]Cancelled[/dim]")
            raise typer.Exit(0)

    if ctx.delete_index(namespace_id=namespace_id, repo_path=path):
        console.print(f"[green]✅ Deleted index for {repo_name}[/green]")
    else:
        console.print("[red]Failed to delete index[/red]")
        raise typer.Exit(1)


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
            return (
                Path.home()
                / "Library"
                / "Application Support"
                / "Claude"
                / "claude_desktop_config.json"
            )
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
            "env": {"CONTEXTFS_SOURCE_TOOL": "claude-desktop"},
        }
    else:
        config["mcpServers"]["contextfs"] = {
            "command": sys.executable,
            "args": ["-m", "contextfs.mcp_server"],
            "env": {"CONTEXTFS_SOURCE_TOOL": "claude-desktop"},
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
        ("contextfs_evolve", "Evolve a memory with history tracking"),
        ("contextfs_merge", "Merge multiple memories"),
        ("contextfs_split", "Split a memory into parts"),
        ("contextfs_lineage", "Get memory lineage (history)"),
        ("contextfs_link", "Create relationships between memories"),
        ("contextfs_related", "Find related memories"),
    ]
    for name, desc in tools:
        console.print(f"  • [cyan]{name}[/cyan] - {desc}")


# =============================================================================
# Memory Lineage Commands
# =============================================================================


@app.command()
def evolve(
    memory_id: str = typer.Argument(..., help="Memory ID to evolve (can be partial)"),
    content: str = typer.Argument(..., help="New content for the evolved memory"),
    summary: str | None = typer.Option(None, "--summary", "-s", help="Summary for new version"),
    no_preserve_tags: bool = typer.Option(False, "--no-tags", help="Don't preserve original tags"),
    tags: str | None = typer.Option(None, "--tags", "-t", help="Additional comma-separated tags"),
):
    """Evolve a memory to a new version with history tracking.

    Creates a new memory linked to the original, preserving the history
    of how knowledge evolved over time.

    Examples:
        contextfs evolve abc123 "Updated API endpoint is /v2/users"
        contextfs evolve abc123 "New content" --summary "Updated for v2"
        contextfs evolve abc123 "Content" --tags "v2,api-change"
    """
    ctx = get_ctx()

    additional_tags = [t.strip() for t in tags.split(",")] if tags else None

    new_memory = ctx.evolve(
        memory_id=memory_id,
        new_content=content,
        summary=summary,
        preserve_tags=not no_preserve_tags,
        additional_tags=additional_tags,
    )

    if not new_memory:
        console.print(f"[red]Memory not found: {memory_id}[/red]")
        raise typer.Exit(1)

    console.print("[green]Memory evolved successfully[/green]")
    console.print(f"Original: {memory_id[:12]}...")
    console.print(f"New ID: {new_memory.id}")
    console.print(f"Type: {new_memory.type.value}")
    if new_memory.summary:
        console.print(f"Summary: {new_memory.summary}")
    if new_memory.tags:
        console.print(f"Tags: {', '.join(new_memory.tags)}")


@app.command()
def merge(
    memory_ids: list[str] = typer.Argument(..., help="Memory IDs to merge (at least 2)"),
    content: str | None = typer.Option(
        None, "--content", "-c", help="Custom merged content (auto-combined if not provided)"
    ),
    summary: str | None = typer.Option(None, "--summary", "-s", help="Summary for merged memory"),
    strategy: str = typer.Option(
        "union",
        "--strategy",
        help="Tag merge strategy: union, intersection, latest, oldest",
    ),
    type: str | None = typer.Option(None, "--type", "-t", help="Memory type for result"),
):
    """Merge multiple memories into one.

    Combines knowledge from multiple memories with configurable tag strategies.
    The original memories remain unchanged; a new merged memory is created.

    Strategies:
        union: All tags from all memories (default)
        intersection: Only common tags
        latest: Tags from the newest memory
        oldest: Tags from the oldest memory

    Examples:
        contextfs merge abc123 def456
        contextfs merge abc123 def456 ghi789 --strategy intersection
        contextfs merge abc123 def456 --content "Combined knowledge"
    """
    if len(memory_ids) < 2:
        console.print("[red]At least 2 memory IDs are required[/red]")
        raise typer.Exit(1)

    ctx = get_ctx()

    memory_type = MemoryType(type) if type else None

    merged_memory = ctx.merge(
        memory_ids=memory_ids,
        merged_content=content,
        summary=summary,
        strategy=strategy,
        memory_type=memory_type,
    )

    if not merged_memory:
        console.print("[red]Failed to merge memories. Some IDs may not exist.[/red]")
        raise typer.Exit(1)

    console.print("[green]Memories merged successfully[/green]")
    console.print(f"Merged {len(memory_ids)} memories:")
    for mid in memory_ids:
        console.print(f"  • {mid[:12]}...")
    console.print(f"New ID: {merged_memory.id}")
    console.print(f"Type: {merged_memory.type.value}")
    console.print(f"Strategy: {strategy}")
    if merged_memory.summary:
        console.print(f"Summary: {merged_memory.summary}")
    if merged_memory.tags:
        console.print(f"Tags: {', '.join(merged_memory.tags)}")


@app.command()
def split(
    memory_id: str = typer.Argument(..., help="Memory ID to split"),
    parts: list[str] = typer.Argument(..., help="Content for each split part"),
    summaries: str | None = typer.Option(
        None, "--summaries", help="Pipe-separated summaries for each part"
    ),
    no_preserve_tags: bool = typer.Option(
        False, "--no-tags", help="Don't preserve original tags on parts"
    ),
):
    """Split a memory into multiple parts.

    Use when a memory contains distinct topics that should be separate.
    Each part creates a new memory linked back to the original.

    Examples:
        contextfs split abc123 "Part 1 content" "Part 2 content"
        contextfs split abc123 "Part 1" "Part 2" --summaries "Summary 1|Summary 2"
    """
    if len(parts) < 2:
        console.print("[red]At least 2 parts are required[/red]")
        raise typer.Exit(1)

    ctx = get_ctx()

    summary_list = [s.strip() for s in summaries.split("|")] if summaries else None

    split_memories = ctx.split(
        memory_id=memory_id,
        parts=parts,
        summaries=summary_list,
        preserve_tags=not no_preserve_tags,
    )

    if not split_memories:
        console.print(f"[red]Memory not found: {memory_id}[/red]")
        raise typer.Exit(1)

    console.print("[green]Memory split successfully[/green]")
    console.print(f"Original: {memory_id[:12]}...")
    console.print(f"Created {len(split_memories)} new memories:")

    table = Table()
    table.add_column("#", style="cyan")
    table.add_column("ID", style="green")
    table.add_column("Summary/Content")

    for i, mem in enumerate(split_memories):
        preview = mem.summary or (
            mem.content[:50] + "..." if len(mem.content) > 50 else mem.content
        )
        table.add_row(str(i + 1), mem.id[:12], preview)

    console.print(table)


@app.command()
def lineage(
    memory_id: str = typer.Argument(..., help="Memory ID to get lineage for"),
    direction: str = typer.Option(
        "both",
        "--direction",
        "-d",
        help="Direction: ancestors, descendants, or both",
    ),
):
    """Show the lineage (history) of a memory.

    Displays:
        - Ancestors: What this memory evolved from (history)
        - Descendants: What evolved from this memory (future versions)

    Examples:
        contextfs lineage abc123
        contextfs lineage abc123 --direction ancestors
    """
    ctx = get_ctx()

    result = ctx.get_lineage(memory_id=memory_id, direction=direction)

    if not result:
        console.print(f"[red]Memory not found or no lineage: {memory_id}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Lineage for {memory_id[:12]}...[/bold]\n")

    # Show ancestors
    ancestors = result.get("ancestors", [])
    if ancestors:
        console.print(f"[cyan]Ancestors ({len(ancestors)}):[/cyan]")
        for anc in ancestors:
            rel = anc.get("relation", "unknown")
            aid = anc.get("id", "")[:12]
            depth = anc.get("depth", 1)
            indent = "  " * depth
            console.print(f"{indent}↑ [{rel}] {aid}...")
    elif direction in ("ancestors", "both"):
        console.print("[dim]No ancestors (this is the original)[/dim]")

    console.print()

    # Show descendants
    descendants = result.get("descendants", [])
    if descendants:
        console.print(f"[cyan]Descendants ({len(descendants)}):[/cyan]")
        for desc in descendants:
            rel = desc.get("relation", "unknown")
            did = desc.get("id", "")[:12]
            depth = desc.get("depth", 1)
            indent = "  " * depth
            console.print(f"{indent}↓ [{rel}] {did}...")
    elif direction in ("descendants", "both"):
        console.print("[dim]No descendants (not evolved yet)[/dim]")


@app.command()
def link(
    from_id: str = typer.Argument(..., help="Source memory ID"),
    to_id: str = typer.Argument(..., help="Target memory ID"),
    relation: str = typer.Argument(
        ...,
        help="Relation type: references, depends_on, contradicts, supports, supersedes, related_to, derived_from, example_of, part_of, implements",
    ),
    weight: float = typer.Option(1.0, "--weight", "-w", help="Relationship strength (0.0-1.0)"),
    bidirectional: bool = typer.Option(
        False, "--bidirectional", "-b", help="Create link in both directions"
    ),
):
    """Create a relationship between two memories.

    Relationships help connect related knowledge across your memory graph.

    Relation types:
        references: One memory references another
        depends_on: One memory depends on another
        contradicts: Memories contain conflicting info
        supports: One memory supports/validates another
        supersedes: One memory replaces/updates another
        related_to: General relationship
        derived_from: One is derived from another
        example_of: One is an example of another
        part_of: One is part of another
        implements: One implements another

    Examples:
        contextfs link abc123 def456 references
        contextfs link abc123 def456 depends_on --bidirectional
        contextfs link abc123 def456 contradicts --weight 0.8
    """
    ctx = get_ctx()

    success = ctx.link(
        from_memory_id=from_id,
        to_memory_id=to_id,
        relation=relation,
        weight=weight,
        bidirectional=bidirectional,
    )

    if not success:
        console.print("[red]Failed to create link. Memories may not exist.[/red]")
        raise typer.Exit(1)

    console.print("[green]Link created successfully[/green]")
    console.print(f"From: {from_id[:12]}...")
    console.print(f"To: {to_id[:12]}...")
    console.print(f"Relation: {relation}")
    console.print(f"Weight: {weight}")
    if bidirectional:
        console.print("Bidirectional: Yes")


@app.command()
def related(
    memory_id: str = typer.Argument(..., help="Memory ID to find relationships for"),
    relation: str | None = typer.Option(None, "--relation", "-r", help="Filter by relation type"),
    max_depth: int = typer.Option(1, "--depth", "-d", help="Maximum traversal depth"),
):
    """Find memories related to a given memory.

    Shows all memories connected through graph relationships.

    Examples:
        contextfs related abc123
        contextfs related abc123 --relation references
        contextfs related abc123 --depth 2
    """
    ctx = get_ctx()

    results = ctx.get_related(
        memory_id=memory_id,
        relation=relation,
        max_depth=max_depth,
    )

    if not results:
        msg = f"No related memories found for {memory_id[:12]}..."
        if relation:
            msg += f" with relation '{relation}'"
        console.print(f"[yellow]{msg}[/yellow]")
        return

    console.print(f"[bold]Related memories for {memory_id[:12]}...[/bold]")
    if relation:
        console.print(f"[dim]Filtered by relation: {relation}[/dim]")
    console.print()

    table = Table()
    table.add_column("Direction", style="cyan")
    table.add_column("Relation", style="magenta")
    table.add_column("ID", style="green")
    table.add_column("Summary/Content")
    table.add_column("Depth", style="dim")

    for item in results:
        rel = item.get("relation", "unknown")
        rid = item.get("id", "")[:12]
        depth = item.get("depth", 1)
        direction = item.get("direction", "outgoing")
        arrow = "→" if direction == "outgoing" else "←"

        preview = item.get("summary", "")
        if not preview and item.get("content"):
            preview = (
                item["content"][:40] + "..."
                if len(item.get("content", "")) > 40
                else item.get("content", "")
            )

        table.add_row(arrow, rel, rid, preview, str(depth) if depth > 1 else "")

    console.print(table)


@app.command("graph-status")
def graph_status():
    """Show graph backend status and statistics."""
    ctx = get_ctx()

    console.print("[bold]Graph Backend Status[/bold]\n")

    if ctx.has_graph():
        console.print("[green]Graph backend: Active[/green]")

        # Try to get some stats
        try:
            from contextfs.config import get_config

            config = get_config()
            console.print(f"Backend type: {config.backend.value}")

            if "falkordb" in config.backend.value:
                console.print(f"FalkorDB host: {config.falkordb_host}:{config.falkordb_port}")
                console.print(f"Graph name: {config.falkordb_graph_name}")
        except Exception:
            pass

        console.print("\nLineage settings:")
        try:
            from contextfs.config import get_config

            config = get_config()
            console.print(f"  Auto-track: {config.lineage_auto_track}")
            console.print(f"  Merge strategy: {config.lineage_merge_strategy.value}")
            console.print(f"  Preserve tags: {config.lineage_preserve_tags}")
        except Exception:
            console.print("  [dim]Could not load settings[/dim]")
    else:
        console.print("[yellow]Graph backend: Not active[/yellow]")
        console.print("\nTo enable graph features:")
        console.print("  1. Set CONTEXTFS_BACKEND=sqlite+falkordb or postgres+falkordb")
        console.print("  2. Start FalkorDB: docker-compose up -d falkordb")
        console.print("  3. Restart contextfs")


# Register sync subcommand group (Click-based)
# We need to add the Click group directly to Typer's underlying Click app
def _register_sync_commands():
    try:
        from contextfs.sync.cli import get_sync_commands

        sync_click_group = get_sync_commands()

        # Get the underlying Click app and add the sync group
        click_app = typer.main.get_command(app)
        click_app.add_command(sync_click_group, name="sync")
    except ImportError:
        pass  # Sync module not available


_register_sync_commands()


if __name__ == "__main__":
    app()
