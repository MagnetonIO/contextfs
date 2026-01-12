"""Sync-related CLI commands."""

import typer

from .utils import console

sync_app = typer.Typer(
    name="sync",
    help="Sync commands for multi-device memory synchronization.",
    no_args_is_help=True,
)

try:
    from contextfs.sync.cli import sync_cli as _sync_click_group

    @sync_app.command()
    def register(
        server: str = typer.Option(
            "http://localhost:8766", "-s", "--server", help="Sync server URL"
        ),
        name: str = typer.Option(None, "-n", "--name", help="Device name (defaults to hostname)"),
    ):
        """Register this device with the sync server."""
        _sync_click_group(
            ["register", "-s", server] + (["-n", name] if name else []), standalone_mode=False
        )

    @sync_app.command()
    def push(
        server: str = typer.Option(
            "http://localhost:8766", "-s", "--server", help="Sync server URL"
        ),
        namespace: list[str] = typer.Option([], "-n", "--namespace", help="Namespace ID to sync"),
        push_all: bool = typer.Option(False, "--all", help="Push all memories"),
    ):
        """Push local changes to the sync server."""
        args = ["push", "-s", server]
        for ns in namespace:
            args.extend(["-n", ns])
        if push_all:
            args.append("--all")
        _sync_click_group(args, standalone_mode=False)

    @sync_app.command()
    def pull(
        server: str = typer.Option(
            "http://localhost:8766", "-s", "--server", help="Sync server URL"
        ),
        namespace: list[str] = typer.Option([], "-n", "--namespace", help="Namespace ID to sync"),
        since: str = typer.Option(None, help="Pull changes after this ISO timestamp"),
        pull_all: bool = typer.Option(False, "--all", help="Pull all memories from server"),
    ):
        """Pull changes from the sync server."""
        args = ["pull", "-s", server]
        for ns in namespace:
            args.extend(["-n", ns])
        if since:
            args.extend(["--since", since])
        if pull_all:
            args.append("--all")
        _sync_click_group(args, standalone_mode=False)

    @sync_app.command(name="all")
    def sync_all(
        server: str = typer.Option(
            "http://localhost:8766", "-s", "--server", help="Sync server URL"
        ),
        namespace: list[str] = typer.Option([], "-n", "--namespace", help="Namespace ID to sync"),
    ):
        """Full bidirectional sync (push + pull)."""
        args = ["all", "-s", server]
        for ns in namespace:
            args.extend(["-n", ns])
        _sync_click_group(args, standalone_mode=False)

    @sync_app.command()
    def diff(
        server: str = typer.Option(
            "http://localhost:8766", "-s", "--server", help="Sync server URL"
        ),
        namespace: list[str] = typer.Option([], "-n", "--namespace", help="Namespace ID to sync"),
    ):
        """Content-addressed sync (idempotent, Merkle-style)."""
        args = ["diff", "-s", server]
        for ns in namespace:
            args.extend(["-n", ns])
        _sync_click_group(args, standalone_mode=False)

    @sync_app.command()
    def status(
        server: str = typer.Option(
            "http://localhost:8766", "-s", "--server", help="Sync server URL"
        ),
    ):
        """Get sync status from server."""
        _sync_click_group(["status", "-s", server], standalone_mode=False)

    @sync_app.command()
    def daemon(
        server: str = typer.Option(
            "http://localhost:8766", "-s", "--server", help="Sync server URL"
        ),
        interval: int = typer.Option(300, "-i", "--interval", help="Sync interval in seconds"),
        namespace: list[str] = typer.Option([], "-n", "--namespace", help="Namespace ID to sync"),
    ):
        """Run sync daemon in background."""
        args = ["daemon", "-s", server, "-i", str(interval)]
        for ns in namespace:
            args.extend(["-n", ns])
        _sync_click_group(args, standalone_mode=False)

    SYNC_AVAILABLE = True
except ImportError:
    SYNC_AVAILABLE = False

    @sync_app.callback(invoke_without_command=True)
    def sync_unavailable():
        """Sync module not available."""
        console.print("[yellow]Sync module not available. Install sync dependencies.[/yellow]")
        raise typer.Exit(1)
