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


if __name__ == "__main__":
    app()
