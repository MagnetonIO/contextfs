"""Index-related CLI commands."""

from pathlib import Path

import typer
from rich.table import Table

from .utils import (
    console,
    create_repo_config,
    find_git_root,
    get_ctx,
    is_repo_initialized,
)

index_app = typer.Typer(help="Index commands", no_args_is_help=True)


@index_app.command()
def init(
    path: Path | None = typer.Argument(None, help="Repository path (default: current directory)"),
    no_index: bool = typer.Option(False, "--no-index", help="Don't run index after init"),
    auto_index: bool = typer.Option(
        True, "--auto-index/--no-auto-index", help="Enable auto-indexing"
    ),
    max_commits: int = typer.Option(100, "--max-commits", help="Maximum commits to index"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Reinitialize even if already initialized"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
):
    """Initialize a repository for ContextFS indexing.

    Creates a .contextfs/config.yaml marker file that enables auto-indexing
    for this repository. The SessionStart hook will only index repositories
    that have been initialized.

    Examples:
        contextfs init                    # Initialize current repo
        contextfs init /path/to/repo      # Initialize specific repo
        contextfs init --no-index         # Initialize without indexing
        contextfs init --no-auto-index    # Initialize but disable auto-index
    """
    # Determine repo path
    start_path = path or Path.cwd()
    repo_path = find_git_root(start_path)

    if not repo_path:
        if not quiet:
            console.print(f"[red]Error: Not a git repository: {start_path.resolve()}[/red]")
            console.print("[yellow]contextfs init requires a git repository.[/yellow]")
        raise typer.Exit(1)

    # Check if already initialized
    if is_repo_initialized(repo_path) and not force:
        if not quiet:
            console.print(f"[yellow]Repository already initialized: {repo_path}[/yellow]")
            console.print("[dim]Use --force to reinitialize.[/dim]")
        raise typer.Exit(0)

    # Create config
    config_path = create_repo_config(
        repo_path=repo_path,
        auto_index=auto_index,
        created_by="cli",
        max_commits=max_commits,
    )

    if not quiet:
        console.print(f"[green]Initialized ContextFS for: {repo_path.name}[/green]")
        console.print(f"   Config: {config_path}")
        if auto_index:
            console.print("   Auto-index: [green]enabled[/green]")
        else:
            console.print("   Auto-index: [yellow]disabled[/yellow]")

    # Run index unless --no-index
    if not no_index:
        if not quiet:
            console.print()
        ctx = get_ctx()
        result = ctx.index_repository(repo_path=repo_path, incremental=True)
        if not quiet:
            console.print(
                f"[green]Indexed {result.get('files_indexed', 0)} files, {result.get('commits_indexed', 0)} commits[/green]"
            )


@index_app.command()
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
    allow_dir: bool = typer.Option(
        False,
        "--allow-dir",
        help="Allow indexing non-git directories (use index-dir for multiple repos)",
    ),
    require_init: bool = typer.Option(
        False,
        "--require-init",
        help="Only index if repo has been initialized with 'contextfs init'",
    ),
):
    """Index a repository's codebase for semantic search.

    By default, only indexes git repositories. Use --allow-dir to index
    non-git directories, or use 'index-dir' command to scan for multiple repos.

    Use --require-init for hooks to only index repos that have been explicitly
    initialized with 'contextfs init'.
    """
    import subprocess
    import sys

    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

    # Determine repo path
    start_path = path or Path.cwd()
    repo_path = find_git_root(start_path)

    # Check if --require-init is set and repo is not initialized
    if require_init and repo_path and not is_repo_initialized(repo_path):
        # Silently exit - repo not initialized for contextfs
        if not quiet:
            console.print(
                f"[yellow]Skipping: {repo_path.name} not initialized for ContextFS[/yellow]"
            )
            console.print("[dim]Run 'contextfs init' to enable indexing for this repo.[/dim]")
        return

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
        if require_init:
            cmd.append("--require-init")
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
            if not quiet:
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
        console.print("\n[green]Indexing complete![/green]\n")

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


@index_app.command("index-dir")
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
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON (with --dry-run)"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
    require_init: bool = typer.Option(
        False,
        "--require-init",
        help="Only index repos initialized with 'contextfs init'",
    ),
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
        repos = ctx.discover_repos(path, max_depth=max_depth)

        # Filter to initialized repos if --require-init
        if require_init:
            repos = [r for r in repos if is_repo_initialized(Path(r["path"]))]

        if json_output:
            import json as json_mod

            output = [
                {
                    "name": repo["name"],
                    "path": str(repo["path"]),
                    "project": repo["project"],
                    "tags": repo["suggested_tags"],
                    "remote_url": repo.get("remote_url"),
                }
                for repo in repos
            ]
            console.print(json_mod.dumps(output, indent=2))
            return

        console.print(f"\n[bold]Discovering git repositories in {path}...[/bold]\n")

        if not repos:
            console.print("[yellow]No git repositories found[/yellow]")
            return

        table = Table(title=f"Found {len(repos)} repositories")
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
                f"  [green][/green] {stats['files_indexed']} files, "
                f"{stats.get('commits_indexed', 0)} commits, "
                f"{stats['memories_created']} memories"
            )
        elif not quiet and "error" in stats:
            console.print(f"  [red] Error: {stats['error']}[/red]")

    # Create filter for --require-init
    def require_init_filter(repo_path: Path) -> bool:
        return is_repo_initialized(repo_path)

    repo_filter_fn = require_init_filter if require_init else None

    result = ctx.index_directory(
        root_dir=path,
        max_depth=max_depth,
        on_repo_start=on_repo_start,
        on_repo_complete=on_repo_complete,
        incremental=incremental,
        project_override=project,
        repo_filter=repo_filter_fn,
    )

    # Summary
    if not quiet:
        console.print("\n[green]Directory indexing complete![/green]\n")

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
                status = "[red]Error[/red]" if "error" in repo else "[green][/green]"
                repo_table.add_row(
                    repo["name"],
                    repo.get("project") or "-",
                    str(repo.get("files_indexed", 0)),
                    str(repo.get("memories_created", 0)),
                    status,
                )

            console.print(repo_table)


@index_app.command("reindex-all")
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
            console.print("[green][/green]")

        if result["errors"]:
            console.print("\n[yellow]Errors:[/yellow]")
            for err in result["errors"]:
                console.print(f"  [yellow] {err}[/yellow]")

        console.print("\n[green]Reindexing complete![/green]")
        console.print(f"  Successful: {result['repos_indexed']}")
        console.print(f"  Failed: {result['repos_failed']}")
        console.print(f"  Total files: {result['total_files']}")
        console.print(f"  Total memories: {result['total_memories']}")


@index_app.command("list-indexes")
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


@index_app.command("cleanup-indexes")
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

    console.print(f"\n[green]Removed {len(result['removed'])} stale index(es)[/green]")
    console.print(f"[green]   Kept {len(result['kept'])} valid index(es)[/green]")


@index_app.command("delete-index")
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
        console.print(f"[green]Deleted index for {repo_name}[/green]")
    else:
        console.print("[red]Failed to delete index[/red]")
        raise typer.Exit(1)
