"""CLI for ContextFS."""

import typer

from .cloud import cloud_app
from .index import index_app
from .memory import memory_app
from .server import server_app
from .sync import sync_app
from .utils import console, get_ctx

app = typer.Typer(
    name="contextfs",
    help="ContextFS - Semantic memory for AI agents",
    no_args_is_help=True,
)

# Register top-level commands from memory_app
for command in memory_app.registered_commands:
    app.command(command.name)(command.callback)

# Register top-level commands from index_app
for command in index_app.registered_commands:
    app.command(command.name)(command.callback)

# Register top-level commands from server_app
for command in server_app.registered_commands:
    app.command(command.name)(command.callback)

# Register subcommand groups
app.add_typer(sync_app, name="sync")
app.add_typer(cloud_app, name="cloud")

__all__ = ["app", "console", "get_ctx"]


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
