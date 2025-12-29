"""CLI commands for sync operations.

Provides command-line interface for sync operations:
- contextfs sync register
- contextfs sync push
- contextfs sync pull
- contextfs sync all
- contextfs sync status
- contextfs sync daemon
"""

from __future__ import annotations

import asyncio
import logging
import platform
import signal
import socket
import sys
from datetime import datetime
from typing import Any

import click

logger = logging.getLogger(__name__)

# Default sync server URL
DEFAULT_SERVER_URL = "http://localhost:8766"


def run_async(coro: Any) -> Any:
    """Run async function in sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@click.group(name="sync")
def sync_cli():
    """Sync commands for multi-device memory synchronization."""
    pass


@sync_cli.command(name="register")
@click.option(
    "--server",
    "-s",
    default=DEFAULT_SERVER_URL,
    help="Sync server URL",
    envvar="CONTEXTFS_SYNC_SERVER",
)
@click.option(
    "--name",
    "-n",
    default=None,
    help="Device name (defaults to hostname)",
)
def register(server: str, name: str | None):
    """Register this device with the sync server."""
    from contextfs.sync import SyncClient

    async def _register():
        async with SyncClient(server) as client:
            info = await client.register_device(
                device_name=name or socket.gethostname(),
                device_platform=platform.system().lower(),
            )
            return info

    try:
        info = run_async(_register())
        click.echo("Device registered successfully!")
        click.echo(f"  Device ID: {info.device_id}")
        click.echo(f"  Name: {info.device_name}")
        click.echo(f"  Platform: {info.platform}")
        click.echo(f"  Registered: {info.registered_at}")
    except Exception as e:
        click.echo(f"Failed to register device: {e}", err=True)
        sys.exit(1)


@sync_cli.command(name="push")
@click.option(
    "--server",
    "-s",
    default=DEFAULT_SERVER_URL,
    help="Sync server URL",
    envvar="CONTEXTFS_SYNC_SERVER",
)
@click.option(
    "--namespace",
    "-n",
    multiple=True,
    help="Namespace ID to sync (can specify multiple)",
)
@click.option(
    "--all",
    "push_all",
    is_flag=True,
    default=False,
    help="Push all memories, not just changes since last sync",
)
def push(server: str, namespace: tuple[str, ...], push_all: bool):
    """Push local changes to the sync server."""
    from contextfs.sync import SyncClient

    namespace_ids = list(namespace) if namespace else None

    async def _push():
        async with SyncClient(server) as client:
            return await client.push(namespace_ids=namespace_ids, push_all=push_all)

    try:
        result = run_async(_push())
        click.echo("Push complete:")
        click.echo(f"  Accepted: {result.accepted}")
        click.echo(f"  Rejected: {result.rejected}")
        if result.conflicts:
            click.echo(f"  Conflicts: {len(result.conflicts)}")
            for conflict in result.conflicts[:5]:  # Show first 5
                click.echo(f"    - {conflict.entity_type}: {conflict.entity_id}")
        click.echo(f"  Server time: {result.server_timestamp}")
    except Exception as e:
        click.echo(f"Push failed: {e}", err=True)
        sys.exit(1)


@sync_cli.command(name="pull")
@click.option(
    "--server",
    "-s",
    default=DEFAULT_SERVER_URL,
    help="Sync server URL",
    envvar="CONTEXTFS_SYNC_SERVER",
)
@click.option(
    "--namespace",
    "-n",
    multiple=True,
    help="Namespace ID to sync (can specify multiple)",
)
@click.option(
    "--since",
    default=None,
    help="Pull changes after this ISO timestamp",
)
@click.option(
    "--all",
    "pull_all",
    is_flag=True,
    default=False,
    help="Pull all memories from server, ignoring last sync time",
)
def pull(server: str, namespace: tuple[str, ...], since: str | None, pull_all: bool):
    """Pull changes from the sync server."""
    from contextfs.sync import SyncClient

    namespace_ids = list(namespace) if namespace else None
    since_dt = datetime.fromisoformat(since) if since else None

    async def _pull():
        async with SyncClient(server) as client:
            total_memories = 0
            total_sessions = 0
            total_edges = 0
            offset = 0

            # If --all, use epoch time to get all memories
            # since=None would use _last_sync from state, so we need explicit epoch time
            from datetime import timezone

            epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
            since_value = since_dt if since_dt else (epoch if pull_all else None)

            while True:
                # Only update sync state on final page
                is_final_page = True  # Will be updated below if has_more

                result = await client.pull(
                    since=since_value,
                    namespace_ids=namespace_ids,
                    offset=offset if pull_all else 0,
                    update_sync_state=False,  # Don't update until we know it's the final page
                )
                total_memories += len(result.memories)
                total_sessions += len(result.sessions)
                total_edges += len(result.edges)

                # Check if we need to continue pagination
                if result.has_more and pull_all and result.next_offset > 0:
                    is_final_page = False
                    offset = result.next_offset
                else:
                    is_final_page = True

                if is_final_page:
                    # Update sync state now that we're done
                    client._last_sync = result.server_timestamp
                    client._save_sync_state()
                    break

            # Return final result with totals
            return result, total_memories, total_sessions, total_edges

    try:
        result, total_mem, total_sess, total_edges = run_async(_pull())
        click.echo("Pull complete:")
        click.echo(f"  Memories: {total_mem}")
        click.echo(f"  Sessions: {total_sess}")
        click.echo(f"  Edges: {total_edges}")
        if result.has_more and not pull_all:
            click.echo("  (More available, use --all to pull everything)")
        click.echo(f"  Server time: {result.server_timestamp}")
    except Exception as e:
        click.echo(f"Pull failed: {e}", err=True)
        sys.exit(1)


@sync_cli.command(name="diff")
@click.option(
    "--server",
    "-s",
    default=DEFAULT_SERVER_URL,
    help="Sync server URL",
    envvar="CONTEXTFS_SYNC_SERVER",
)
@click.option(
    "--namespace",
    "-n",
    multiple=True,
    help="Namespace ID to sync (can specify multiple)",
)
def diff_sync(server: str, namespace: tuple[str, ...]):
    """Content-addressed sync (idempotent, Merkle-style).

    Compares local content hashes with server state.
    Always produces correct result regardless of sync state.
    """
    from contextfs.sync import SyncClient

    namespace_ids = list(namespace) if namespace else None

    async def _diff():
        async with SyncClient(server) as client:
            return await client.pull_diff(namespace_ids=namespace_ids)

    try:
        result = run_async(_diff())
        click.echo("Diff sync complete:")
        click.echo(
            f"  To pull: {len(result.missing_memories)} memories, {len(result.missing_sessions)} sessions"
        )
        click.echo(
            f"  Server needs: {len(result.server_missing_memory_ids)} memories, {len(result.server_missing_session_ids)} sessions"
        )
        click.echo(f"  Deleted: {result.total_deleted}")
        click.echo(f"  Updated: {result.total_updated}")
        click.echo(f"  Server time: {result.server_timestamp}")
    except Exception as e:
        click.echo(f"Diff sync failed: {e}", err=True)
        sys.exit(1)


@sync_cli.command(name="all")
@click.option(
    "--server",
    "-s",
    default=DEFAULT_SERVER_URL,
    help="Sync server URL",
    envvar="CONTEXTFS_SYNC_SERVER",
)
@click.option(
    "--namespace",
    "-n",
    multiple=True,
    help="Namespace ID to sync (can specify multiple)",
)
def sync_all(server: str, namespace: tuple[str, ...]):
    """Full bidirectional sync (push + pull)."""
    from contextfs.sync import SyncClient

    namespace_ids = list(namespace) if namespace else None

    async def _sync():
        async with SyncClient(server) as client:
            return await client.sync_all(namespace_ids=namespace_ids)

    try:
        result = run_async(_sync())
        click.echo(f"Sync complete in {result.duration_ms:.0f}ms:")
        click.echo(
            f"  Pushed: {result.pushed.accepted} accepted, {result.pushed.rejected} rejected"
        )
        if result.pushed.conflicts:
            click.echo(f"    Conflicts: {len(result.pushed.conflicts)}")
        click.echo(
            f"  Pulled: {len(result.pulled.memories)} memories, {len(result.pulled.sessions)} sessions"
        )
        if result.errors:
            click.echo("  Errors:")
            for error in result.errors:
                click.echo(f"    - {error}")
    except Exception as e:
        click.echo(f"Sync failed: {e}", err=True)
        sys.exit(1)


@sync_cli.command(name="status")
@click.option(
    "--server",
    "-s",
    default=DEFAULT_SERVER_URL,
    help="Sync server URL",
    envvar="CONTEXTFS_SYNC_SERVER",
)
def status(server: str):
    """Get sync status from server."""
    from contextfs.sync import SyncClient

    async def _status():
        async with SyncClient(server) as client:
            return await client.status()

    try:
        result = run_async(_status())
        click.echo("Sync Status:")
        click.echo(f"  Device ID: {result.device_id}")
        click.echo(f"  Last sync: {result.last_sync_at or 'never'}")
        click.echo(f"  Pending pulls: {result.pending_pull_count}")
        click.echo(f"  Server time: {result.server_timestamp}")
    except Exception as e:
        click.echo(f"Failed to get status: {e}", err=True)
        sys.exit(1)


@sync_cli.command(name="daemon")
@click.option(
    "--server",
    "-s",
    default=DEFAULT_SERVER_URL,
    help="Sync server URL",
    envvar="CONTEXTFS_SYNC_SERVER",
)
@click.option(
    "--interval",
    "-i",
    default=300,
    help="Sync interval in seconds (default: 300)",
)
@click.option(
    "--namespace",
    "-n",
    multiple=True,
    help="Namespace ID to sync (can specify multiple)",
)
def daemon(server: str, interval: int, namespace: tuple[str, ...]):
    """Run sync daemon in background."""
    from contextfs.sync import SyncClient

    namespace_ids = list(namespace) if namespace else None
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        click.echo("\nShutting down sync daemon...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    click.echo(f"Starting sync daemon (interval: {interval}s)")
    click.echo(f"  Server: {server}")
    if namespace_ids:
        click.echo(f"  Namespaces: {', '.join(namespace_ids)}")
    click.echo("Press Ctrl+C to stop")

    async def _daemon():
        async with SyncClient(server) as client:
            while running:
                try:
                    result = await client.sync_all(namespace_ids=namespace_ids)
                    now = datetime.now().strftime("%H:%M:%S")
                    click.echo(
                        f"[{now}] Synced: +{result.pushed.accepted} ↓{len(result.pulled.memories)}"
                    )
                except Exception as e:
                    now = datetime.now().strftime("%H:%M:%S")
                    click.echo(f"[{now}] Sync error: {e}", err=True)

                # Wait for next interval
                for _ in range(interval):
                    if not running:
                        break
                    await asyncio.sleep(1)

    try:
        run_async(_daemon())
        click.echo("Sync daemon stopped")
    except Exception as e:
        click.echo(f"Daemon error: {e}", err=True)
        sys.exit(1)


# For integration with main CLI
def get_sync_commands() -> click.Group:
    """Get sync command group for integration with main CLI."""
    return sync_cli


if __name__ == "__main__":
    sync_cli()
