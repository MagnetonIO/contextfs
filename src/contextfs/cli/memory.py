"""Memory-related CLI commands."""

import json
import re
import select
import sys
from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from contextfs.schemas import MemoryType

from .utils import console, get_ctx

memory_app = typer.Typer(help="Memory commands", no_args_is_help=True)


@memory_app.command()
def save(
    content: str = typer.Argument(..., help="Content to save"),
    type: str = typer.Option("fact", "--type", "-t", help="Memory type"),
    tags: str | None = typer.Option(None, "--tags", help="Comma-separated tags"),
    summary: str | None = typer.Option(None, "--summary", "-s", help="Brief summary"),
    structured: str | None = typer.Option(
        None,
        "--structured",
        help="JSON structured data validated against type's schema",
    ),
):
    """Save a memory.

    Use --structured to add typed structured data. For example:
        contextfs save "Auth decision" -t decision --structured '{"decision": "Use JWT", "rationale": "Stateless auth"}'

    To see available schemas for each type, use: contextfs type-schema <type>
    """
    import json

    ctx = get_ctx()

    tag_list = [t.strip() for t in tags.split(",")] if tags else []

    try:
        memory_type = MemoryType(type)
    except ValueError:
        console.print(f"[red]Invalid type: {type}[/red]")
        raise typer.Exit(1)

    structured_data = None
    if structured:
        try:
            structured_data = json.loads(structured)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON in --structured: {e}[/red]")
            raise typer.Exit(1)

    try:
        memory = ctx.save(
            content=content,
            type=memory_type,
            tags=tag_list,
            summary=summary,
            source_tool="contextfs-cli",
            structured_data=structured_data,
        )
    except ValueError as e:
        console.print(f"[red]Schema validation error: {e}[/red]")
        raise typer.Exit(1)

    console.print("[green]Memory saved[/green]")
    console.print(f"ID: {memory.id}")
    console.print(f"Type: {memory.type.value}")
    if memory.structured_data:
        console.print(f"Structured: {len(memory.structured_data)} fields")


@memory_app.command("type-schema")
def type_schema(
    memory_type: str = typer.Argument(..., help="Memory type to show schema for"),
):
    """Show the JSON schema for a memory type's structured_data."""
    import json

    from contextfs.schemas import TYPE_SCHEMAS, get_type_schema

    schema = get_type_schema(memory_type)
    if not schema:
        types_with_schemas = list(TYPE_SCHEMAS.keys())
        console.print(f"[yellow]No schema defined for type '{memory_type}'[/yellow]")
        console.print(f"\nTypes with schemas: {', '.join(types_with_schemas)}")
        return

    console.print(f"[cyan]JSON Schema for type '{memory_type}':[/cyan]\n")
    console.print(json.dumps(schema, indent=2))

    required = schema.get("required", [])
    if required:
        console.print(f"\n[green]Required fields:[/green] {', '.join(required)}")
    else:
        console.print("\n[green]No required fields[/green]")


@memory_app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results"),
    type: str | None = typer.Option(None, "--type", "-t", help="Filter by type"),
    namespace: str | None = typer.Option(
        None, "--namespace", "-ns", help="Filter to specific namespace"
    ),
    mode: str = typer.Option(
        "hybrid",
        "--mode",
        "-m",
        help="Search mode: hybrid, semantic, keyword, smart",
    ),
):
    """Search memories across all repos/namespaces."""
    ctx = get_ctx()

    type_filter = MemoryType(type) if type else None
    results = ctx.search(
        query,
        limit=limit,
        type=type_filter,
        namespace_id=namespace,
        cross_repo=(namespace is None),
        mode=mode,
    )

    if not results:
        console.print("[yellow]No memories found[/yellow]")
        return

    table = Table(title="Search Results")
    table.add_column("ID", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Type", style="magenta")
    table.add_column("Content")
    table.add_column("Tags", style="blue")
    if mode == "hybrid":
        table.add_column("Source", style="dim")

    for r in results:
        row = [
            r.memory.id[:8],
            f"{r.score:.2f}",
            r.memory.type.value,
            r.memory.content[:60] + "..." if len(r.memory.content) > 60 else r.memory.content,
            ", ".join(r.memory.tags) if r.memory.tags else "",
        ]
        if mode == "hybrid":
            row.append(getattr(r, "source", "") or "")
        table.add_row(*row)

    console.print(table)


@memory_app.command()
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

    if memory.structured_data:
        import json

        console.print("\n[cyan]Structured Data:[/cyan]")
        console.print(json.dumps(memory.structured_data, indent=2))


@memory_app.command("list")
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


@memory_app.command()
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


@memory_app.command()
def evolve(
    memory_id: str = typer.Argument(..., help="Memory ID to evolve (can be partial)"),
    content: str = typer.Argument(..., help="New content for the evolved memory"),
    summary: str | None = typer.Option(None, "--summary", "-s", help="Summary for new version"),
    no_preserve_tags: bool = typer.Option(False, "--no-tags", help="Don't preserve original tags"),
    tags: str | None = typer.Option(None, "--tags", "-t", help="Additional comma-separated tags"),
):
    """Evolve a memory to a new version with history tracking."""
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


@memory_app.command()
def merge(
    memory_ids: list[str] = typer.Argument(..., help="Memory IDs to merge (at least 2)"),
    content: str | None = typer.Option(None, "--content", "-c", help="Custom merged content"),
    summary: str | None = typer.Option(None, "--summary", "-s", help="Summary for merged memory"),
    strategy: str = typer.Option(
        "union",
        "--strategy",
        help="Tag merge strategy: union, intersection, latest, oldest",
    ),
    type: str | None = typer.Option(None, "--type", "-t", help="Memory type for result"),
):
    """Merge multiple memories into one."""
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
        console.print(f"  - {mid[:12]}...")
    console.print(f"New ID: {merged_memory.id}")
    console.print(f"Type: {merged_memory.type.value}")
    if merged_memory.tags:
        console.print(f"Tags: {', '.join(merged_memory.tags)}")


@memory_app.command()
def split(
    memory_id: str = typer.Argument(..., help="Memory ID to split"),
    parts: list[str] = typer.Argument(..., help="Content for each split part"),
    summaries: str | None = typer.Option(
        None, "--summaries", help="Pipe-separated summaries for each part"
    ),
    no_preserve_tags: bool = typer.Option(False, "--no-tags", help="Don't preserve original tags"),
):
    """Split a memory into multiple parts."""
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


@memory_app.command()
def lineage(
    memory_id: str = typer.Argument(..., help="Memory ID to get lineage for"),
    direction: str = typer.Option(
        "both",
        "--direction",
        "-d",
        help="Direction: ancestors, descendants, or both",
    ),
):
    """Show the lineage (history) of a memory."""
    ctx = get_ctx()

    result = ctx.get_lineage(memory_id=memory_id, direction=direction)

    if not result:
        console.print(f"[red]Memory not found or no lineage: {memory_id}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Lineage for {memory_id[:12]}...[/bold]\n")

    ancestors = result.get("ancestors", [])
    if ancestors:
        console.print(f"[cyan]Ancestors ({len(ancestors)}):[/cyan]")
        for anc in ancestors:
            rel = anc.get("relation", "unknown")
            aid = anc.get("id", "")[:12]
            depth = anc.get("depth", 1)
            indent = "  " * depth
            console.print(f"{indent}^ [{rel}] {aid}...")
    elif direction in ("ancestors", "both"):
        console.print("[dim]No ancestors (this is the original)[/dim]")

    console.print()

    descendants = result.get("descendants", [])
    if descendants:
        console.print(f"[cyan]Descendants ({len(descendants)}):[/cyan]")
        for desc in descendants:
            rel = desc.get("relation", "unknown")
            did = desc.get("id", "")[:12]
            depth = desc.get("depth", 1)
            indent = "  " * depth
            console.print(f"{indent}v [{rel}] {did}...")
    elif direction in ("descendants", "both"):
        console.print("[dim]No descendants (not evolved yet)[/dim]")


@memory_app.command()
def link(
    from_id: str = typer.Argument(..., help="Source memory ID"),
    to_id: str = typer.Argument(..., help="Target memory ID"),
    relation: str = typer.Argument(
        ...,
        help="Relation: references, depends_on, contradicts, supports, supersedes, related_to, etc.",
    ),
    weight: float = typer.Option(1.0, "--weight", "-w", help="Relationship strength (0.0-1.0)"),
    bidirectional: bool = typer.Option(
        False, "--bidirectional", "-b", help="Create link in both directions"
    ),
):
    """Create a relationship between two memories."""
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
    if bidirectional:
        console.print("Bidirectional: Yes")


@memory_app.command()
def related(
    memory_id: str = typer.Argument(..., help="Memory ID to find relationships for"),
    relation: str | None = typer.Option(None, "--relation", "-r", help="Filter by relation type"),
    max_depth: int = typer.Option(1, "--depth", "-d", help="Maximum traversal depth"),
):
    """Find memories related to a given memory."""
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

    for item in results:
        rel = item.get("relation", "unknown")
        rid = item.get("id", "")[:12]
        direction = item.get("direction", "outgoing")
        arrow = "->" if direction == "outgoing" else "<-"

        preview = item.get("summary", "")
        if not preview and item.get("content"):
            preview = (
                item["content"][:40] + "..."
                if len(item.get("content", "")) > 40
                else item.get("content", "")
            )

        table.add_row(arrow, rel, rid, preview)

    console.print(table)


@memory_app.command()
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

    if not all_memories and not days and not repo and not type and not auto_indexed:
        console.print(
            "[red]Error: Specify at least one filter (--days, --repo, --type, --auto-indexed) or use --all[/red]"
        )
        raise typer.Exit(1)

    ctx = get_ctx()
    cutoff = datetime.now() - timedelta(days=days) if days else None

    memories = ctx.list_recent(limit=10000)

    to_delete = []
    for m in memories:
        if all_memories and not days and not repo and not type and not auto_indexed:
            to_delete.append(m)
            continue

        if cutoff and m.created_at >= cutoff:
            continue
        if repo and m.source_repo != repo:
            continue
        if type and m.type.value != type:
            continue
        if auto_indexed and not m.metadata.get("auto_indexed"):
            continue

        if days or repo or type or auto_indexed:
            to_delete.append(m)

    if not to_delete:
        console.print("[yellow]No memories match the criteria[/yellow]")
        return

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

    console.print(f"\n[bold]Found {len(to_delete)} memories to delete ({filter_desc}):[/bold]")

    table = Table()
    table.add_column("ID", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Repo", style="blue")
    table.add_column("Created", style="green")
    table.add_column("Content/Summary")

    for m in to_delete[:20]:
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

    deleted = 0
    for m in to_delete:
        if ctx.delete(m.id):
            deleted += 1

    console.print(f"\n[green]Deleted {deleted} memories[/green]")


@memory_app.command()
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


@memory_app.command("save-session")
def save_session(
    label: str = typer.Option(None, "--label", "-l", help="Session label"),
    transcript: Path | None = typer.Option(
        None, "--transcript", "-t", help="Path to transcript JSONL file"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
):
    """Save the current session to memory (for use with hooks)."""
    import json
    import sys

    ctx = get_ctx()

    transcript_path = transcript
    # Check if there's actually data to read from stdin (non-blocking)
    if (
        not transcript_path
        and not sys.stdin.isatty()
        and select.select([sys.stdin], [], [], 0.1)[0]
    ):
        try:
            hook_input = json.load(sys.stdin)
            if "transcript_path" in hook_input:
                transcript_path = Path(hook_input["transcript_path"]).expanduser()
        except Exception:
            pass

    session = ctx.get_current_session()
    if not session:
        session = ctx.start_session(tool="claude-code", label=label)
    elif label:
        session.label = label

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
                                text_parts = [
                                    c.get("text", "") for c in content if c.get("type") == "text"
                                ]
                                content = "\n".join(text_parts)
                            ctx.log_message("assistant", content)
        except Exception as e:
            if not quiet:
                console.print(f"[yellow]Warning: Could not read transcript: {e}[/yellow]")

    session.summary = f"Auto-saved session: {label or 'unnamed'}"
    ctx.end_session(generate_summary=False)

    if not quiet:
        console.print("[green]Session saved[/green]")
        console.print(f"ID: {session.id}")
        if label:
            console.print(f"Label: {label}")


@memory_app.command()
def status():
    """Show ContextFS status."""
    ctx = get_ctx()

    console.print("[bold]ContextFS Status[/bold]\n")
    console.print(f"Data directory: {ctx.data_dir}")
    console.print(f"Namespace: {ctx.namespace_id}")

    memories = ctx.list_recent(limit=1000)
    console.print(f"Total memories: {len(memories)}")

    type_counts = {}
    for m in memories:
        type_counts[m.type.value] = type_counts.get(m.type.value, 0) + 1

    if type_counts:
        console.print("\nMemories by type:")
        for t, c in sorted(type_counts.items()):
            console.print(f"  {t}: {c}")

    try:
        rag_stats = ctx.rag.get_stats()
        console.print(f"\nVector store: {rag_stats['total_memories']} embeddings")
        console.print(f"Embedding model: {rag_stats['embedding_model']}")
    except Exception:
        console.print("\n[yellow]Vector store not initialized[/yellow]")

    session = ctx.get_current_session()
    if session:
        console.print(f"\nActive session: {session.id[:12]}")
        console.print(f"  Messages: {len(session.messages)}")


@memory_app.command("graph-status")
def graph_status():
    """Show graph backend status and statistics."""
    ctx = get_ctx()

    console.print("[bold]Graph Backend Status[/bold]\n")

    if ctx.has_graph():
        console.print("[green]Graph backend: Active[/green]")

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


@memory_app.command("auto-recall")
def auto_recall(
    limit: int = typer.Option(5, "--limit", "-l", help="Max memories per type"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output for hooks"),
):
    """Auto-recall relevant context for current repo (for SessionStart hooks).

    Searches for recent decisions, procedures, and errors for the current
    repository and outputs them in a format Claude can use.

    Example hook usage:
        "command": "uvx contextfs memory auto-recall --quiet"
    """
    import subprocess

    ctx = get_ctx()

    # Detect current repo name
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            repo_path = Path(result.stdout.strip())
            repo_name = repo_path.name
        else:
            repo_name = Path.cwd().name
    except Exception:
        repo_name = Path.cwd().name

    # Search for relevant context by type
    memory_types = [
        ("decision", "Recent Decisions"),
        ("procedural", "Procedures"),
        ("error", "Known Errors & Solutions"),
    ]

    output_lines = []
    output_lines.append(f"# ContextFS Auto-Recall: {repo_name}")
    output_lines.append("")

    has_content = False
    for mem_type, header in memory_types:
        memories = ctx.search(
            query=repo_name,
            limit=limit,
            type=MemoryType(mem_type),
        )

        if memories:
            has_content = True
            output_lines.append(f"## {header}")
            for result in memories:
                mem = result.memory
                summary = mem.summary or mem.content[:100]
                output_lines.append(f"- [{mem.id[:8]}] {summary}")
            output_lines.append("")

    if not has_content:
        output_lines.append("No relevant memories found for this repo.")
        output_lines.append("Use contextfs_save to store decisions, procedures, and errors.")

    output_lines.append("")
    output_lines.append("---")
    output_lines.append("Memory operations: search → save → evolve/link → delete")

    # Output
    if quiet:
        # Minimal output for hooks
        print("\n".join(output_lines))
    else:
        for line in output_lines:
            console.print(line)


# =============================================================================
# Extract commands (moved from separate extract module)
# =============================================================================

# Patterns that indicate learnings/decisions
LEARNING_PATTERNS = [
    # Decision patterns
    (r"(?i)decided to|chose to|going with|we('ll)? use|let's use|better to", "decision"),
    (r"(?i)the reason|because|rationale|this approach|tradeoff", "decision"),
    # Error patterns
    (r"(?i)error:|exception:|failed:|traceback|bug|issue:", "error"),
    (r"(?i)fixed by|resolved by|solution:|the fix", "error"),
    # Fact patterns
    (r"(?i)i learned|discovered|found out|turns out|important:", "fact"),
    (r"(?i)note:|remember:|key point|takeaway", "fact"),
    # Procedure patterns
    (r"(?i)steps?:|workflow:|process:|to do this", "procedural"),
    (r"(?i)first,.*then,|step \d", "procedural"),
]


def _extract_from_text(text: str) -> list[dict[str, Any]]:
    """Extract potential memories from text using pattern matching."""
    extractions = []

    # Split into sentences/paragraphs
    chunks = re.split(r"\n\n+|(?<=[.!?])\s+", text)

    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) < 100:  # Skip short chunks (avoid noisy extractions)
            continue

        for pattern, mem_type in LEARNING_PATTERNS:
            if re.search(pattern, chunk):
                extractions.append(
                    {
                        "type": mem_type,
                        "content": chunk,  # No limit
                        "pattern": pattern,
                    }
                )
                break  # Only match first pattern per chunk

    return extractions


def _parse_transcript(transcript_path: Path) -> list[dict]:
    """Parse a Claude Code transcript file."""
    messages = []

    with open(transcript_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                msg_type = entry.get("type")

                if msg_type == "assistant":
                    content = entry.get("message", {}).get("content", "")
                    if isinstance(content, list):
                        # Handle structured content
                        text_parts = []
                        for part in content:
                            if part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                        content = "\n".join(text_parts)
                    messages.append({"role": "assistant", "content": content})

                elif msg_type == "human":
                    content = entry.get("message", {}).get("content", "")
                    messages.append({"role": "user", "content": content})

            except json.JSONDecodeError:
                continue

    return messages


@memory_app.command("extract-transcript")
def extract_transcript(
    transcript: Path = typer.Argument(
        None, help="Path to transcript JSONL file (or reads from stdin)"
    ),
    save: bool = typer.Option(False, "--save", "-s", help="Save extracted memories"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
    agent_role: str = typer.Option(None, "--agent-role", "-a", help="Agent role tag"),
    project: str = typer.Option(None, "--project", "-p", help="Project name for grouping"),
):
    """Extract memories from a Claude Code transcript.

    Analyzes the conversation and extracts:
    - Decisions made (with rationale)
    - Errors encountered and solutions
    - Facts learned
    - Procedures discovered

    Example (Stop hook):
        contextfs memory extract-transcript --save --quiet

    Example (manual):
        contextfs memory extract-transcript /path/to/transcript.jsonl
    """
    transcript_path = transcript

    # Try to get transcript from stdin (hook input)
    if (
        not transcript_path
        and not sys.stdin.isatty()
        and select.select([sys.stdin], [], [], 0.1)[0]
    ):
        try:
            hook_input = json.load(sys.stdin)
            if "transcript_path" in hook_input:
                transcript_path = Path(hook_input["transcript_path"]).expanduser()
        except Exception:
            pass

    if not transcript_path:
        if not quiet:
            console.print("[yellow]No transcript provided[/yellow]")
        return

    if not transcript_path.exists():
        if not quiet:
            console.print(f"[red]Transcript not found: {transcript_path}[/red]")
        raise typer.Exit(1)

    # Parse transcript
    messages = _parse_transcript(transcript_path)
    if not messages:
        if not quiet:
            console.print("[yellow]No messages found in transcript[/yellow]")
        return

    # Extract from assistant messages only
    assistant_text = "\n\n".join(m["content"] for m in messages if m["role"] == "assistant")

    extractions = _extract_from_text(assistant_text)

    if not extractions:
        if not quiet:
            console.print("[dim]No extractable memories found[/dim]")
        return

    # Deduplicate similar extractions
    seen_content = set()
    unique_extractions = []
    for ext in extractions:
        content_hash = ext["content"][:100]
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_extractions.append(ext)

    if not quiet:
        console.print(f"[cyan]Found {len(unique_extractions)} potential memories[/cyan]")

    # Save if requested
    if save:
        ctx = get_ctx()
        saved_count = 0

        tags = ["auto-extracted"]
        if agent_role:
            tags.append(f"{agent_role}-agent")

        for ext in unique_extractions[:10]:  # Limit to 10 per session
            try:
                memory_type = MemoryType(ext["type"])

                # Build structured data for types that require it
                structured_data = None
                if memory_type == MemoryType.DECISION:
                    structured_data = {
                        "decision": ext["content"][:200],
                        "rationale": "Auto-extracted from conversation",
                        "alternatives": [],
                    }
                elif memory_type == MemoryType.ERROR:
                    structured_data = {
                        "error_type": "auto-extracted",
                        "message": ext["content"][:200],
                        "resolution": "See full content",
                    }
                elif memory_type == MemoryType.PROCEDURAL:
                    structured_data = {
                        "title": "Auto-extracted procedure",
                        "steps": [ext["content"][:500]],
                    }

                ctx.save(
                    content=ext["content"],
                    type=memory_type,
                    tags=tags,
                    summary=f"Auto-extracted {ext['type']}: {ext['content'][:50]}...",
                    source_tool=f"contextfs-extract{f'-{agent_role}' if agent_role else ''}",
                    structured_data=structured_data,
                    project=project,
                )
                saved_count += 1

            except Exception as e:
                if not quiet:
                    console.print(f"[yellow]Failed to save: {e}[/yellow]")

        if not quiet:
            console.print(f"[green]Saved {saved_count} memories[/green]")
        elif saved_count > 0:
            print(f"Auto-extracted {saved_count} memories")

    else:
        # Just display extractions
        for ext in unique_extractions:
            console.print(f"\n[cyan]{ext['type'].upper()}[/cyan]")
            console.print(
                ext["content"][:200] + "..." if len(ext["content"]) > 200 else ext["content"]
            )


@memory_app.command("extract-patterns")
def show_patterns():
    """Show the patterns used for memory extraction."""
    console.print("[bold]Memory Extraction Patterns[/bold]\n")

    for pattern, mem_type in LEARNING_PATTERNS:
        console.print(f"[cyan]{mem_type}[/cyan]: {pattern}")


@memory_app.command("extract-test")
def test_extraction(
    text: str = typer.Argument(..., help="Text to test extraction on"),
):
    """Test memory extraction on sample text."""
    extractions = _extract_from_text(text)

    if not extractions:
        console.print("[yellow]No patterns matched[/yellow]")
        return

    for ext in extractions:
        console.print(f"\n[cyan]{ext['type']}[/cyan] (pattern: {ext['pattern']})")
        console.print(ext["content"])
