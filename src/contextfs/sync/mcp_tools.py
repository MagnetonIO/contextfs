"""MCP tool handlers for sync operations.

Provides MCP-compatible tools for sync operations that can be
called from Claude Code or other MCP clients.
"""

from __future__ import annotations

import logging
import platform
import socket
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Default sync server URL
DEFAULT_SERVER_URL = "http://localhost:8766"


async def contextfs_sync_register(
    server_url: str = DEFAULT_SERVER_URL,
    device_name: str | None = None,
) -> dict[str, Any]:
    """
    Register this device with a sync server.

    Args:
        server_url: URL of the sync server
        device_name: Human-readable device name (defaults to hostname)

    Returns:
        Dict with registration result
    """
    from contextfs.sync import SyncClient

    async with SyncClient(server_url) as client:
        try:
            info = await client.register_device(
                device_name=device_name or socket.gethostname(),
                device_platform=platform.system().lower(),
            )
            return {
                "success": True,
                "device_id": info.device_id,
                "device_name": info.device_name,
                "platform": info.platform,
                "registered_at": info.registered_at.isoformat(),
                "message": f"Device '{info.device_name}' registered successfully",
            }
        except Exception as e:
            logger.error(f"Failed to register device: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to register device: {e}",
            }


async def contextfs_sync_push(
    server_url: str = DEFAULT_SERVER_URL,
    namespace_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Push local changes to sync server.

    Args:
        server_url: URL of the sync server
        namespace_ids: Optional list of namespace IDs to sync

    Returns:
        Dict with push result
    """
    from contextfs.sync import SyncClient

    async with SyncClient(server_url) as client:
        try:
            result = await client.push(namespace_ids=namespace_ids)
            return {
                "success": result.success,
                "status": result.status.value,
                "accepted": result.accepted,
                "rejected": result.rejected,
                "conflicts": len(result.conflicts),
                "server_timestamp": result.server_timestamp.isoformat(),
                "message": (
                    f"Pushed {result.accepted} items"
                    + (f", {result.rejected} rejected" if result.rejected else "")
                    + (f", {len(result.conflicts)} conflicts" if result.conflicts else "")
                ),
            }
        except Exception as e:
            logger.error(f"Push failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Push failed: {e}",
            }


async def contextfs_sync_pull(
    server_url: str = DEFAULT_SERVER_URL,
    namespace_ids: list[str] | None = None,
    since: str | None = None,
) -> dict[str, Any]:
    """
    Pull changes from sync server.

    Args:
        server_url: URL of the sync server
        namespace_ids: Optional list of namespace IDs to sync
        since: ISO timestamp to pull changes after

    Returns:
        Dict with pull result
    """
    from contextfs.sync import SyncClient

    since_dt = datetime.fromisoformat(since) if since else None

    async with SyncClient(server_url) as client:
        try:
            result = await client.pull(
                since=since_dt,
                namespace_ids=namespace_ids,
            )
            return {
                "success": result.success,
                "memories_count": len(result.memories),
                "sessions_count": len(result.sessions),
                "edges_count": len(result.edges),
                "has_more": result.has_more,
                "server_timestamp": result.server_timestamp.isoformat(),
                "message": (
                    f"Pulled {len(result.memories)} memories, {len(result.sessions)} sessions"
                ),
            }
        except Exception as e:
            logger.error(f"Pull failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Pull failed: {e}",
            }


async def contextfs_sync_all(
    server_url: str = DEFAULT_SERVER_URL,
    namespace_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Full bidirectional sync with server.

    Args:
        server_url: URL of the sync server
        namespace_ids: Optional list of namespace IDs to sync

    Returns:
        Dict with sync result
    """
    from contextfs.sync import SyncClient

    async with SyncClient(server_url) as client:
        try:
            result = await client.sync_all(namespace_ids=namespace_ids)
            return {
                "success": result.success,
                "pushed": {
                    "accepted": result.pushed.accepted,
                    "rejected": result.pushed.rejected,
                    "conflicts": len(result.pushed.conflicts),
                },
                "pulled": {
                    "memories": len(result.pulled.memories),
                    "sessions": len(result.pulled.sessions),
                },
                "duration_ms": result.duration_ms,
                "errors": result.errors,
                "message": (
                    f"Synced in {result.duration_ms:.0f}ms: "
                    f"pushed {result.pushed.accepted}, "
                    f"pulled {len(result.pulled.memories)} memories"
                ),
            }
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Sync failed: {e}",
            }


async def contextfs_sync_diff(
    server_url: str = DEFAULT_SERVER_URL,
    namespace_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Content-addressed sync (Merkle-style, idempotent).

    Compares local content hashes with server state.
    Run it 100 times, always correct result.

    Args:
        server_url: URL of the sync server
        namespace_ids: Optional list of namespace IDs to sync

    Returns:
        Dict with diff sync result including what server is missing
    """
    from contextfs.sync import SyncClient

    async with SyncClient(server_url) as client:
        try:
            result = await client.pull_diff(namespace_ids=namespace_ids)
            return {
                "success": result.success,
                "missing_memories": len(result.missing_memories),
                "missing_sessions": len(result.missing_sessions),
                "missing_edges": len(result.missing_edges),
                "deleted": result.total_deleted,
                "updated": result.total_updated,
                "server_missing_memories": len(result.server_missing_memory_ids),
                "server_missing_sessions": len(result.server_missing_session_ids),
                "server_missing_edges": len(result.server_missing_edge_ids),
                "server_timestamp": result.server_timestamp.isoformat(),
                "message": (
                    f"Diff: {len(result.missing_memories)} to pull, "
                    f"{result.total_server_missing} server needs, "
                    f"{result.total_updated} updated, {result.total_deleted} deleted"
                ),
            }
        except Exception as e:
            logger.error(f"Diff sync failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Diff sync failed: {e}",
            }


async def contextfs_sync_status(
    server_url: str = DEFAULT_SERVER_URL,
) -> dict[str, Any]:
    """
    Get sync status from server.

    Args:
        server_url: URL of the sync server

    Returns:
        Dict with sync status
    """
    from contextfs.sync import SyncClient

    async with SyncClient(server_url) as client:
        try:
            status = await client.status()
            return {
                "success": True,
                "device_id": status.device_id,
                "last_sync_at": (status.last_sync_at.isoformat() if status.last_sync_at else None),
                "pending_push_count": status.pending_push_count,
                "pending_pull_count": status.pending_pull_count,
                "server_timestamp": status.server_timestamp.isoformat(),
                "message": (
                    f"Last sync: {status.last_sync_at.isoformat() if status.last_sync_at else 'never'}, "
                    f"{status.pending_pull_count} pending pulls"
                ),
            }
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get status: {e}",
            }


# MCP Tool definitions for registration
MCP_TOOLS = [
    {
        "name": "contextfs_sync_register",
        "description": "Register this device with a sync server for multi-device memory synchronization",
        "parameters": {
            "type": "object",
            "properties": {
                "server_url": {
                    "type": "string",
                    "description": "URL of the sync server",
                    "default": DEFAULT_SERVER_URL,
                },
                "device_name": {
                    "type": "string",
                    "description": "Human-readable device name (defaults to hostname)",
                },
            },
        },
        "handler": contextfs_sync_register,
    },
    {
        "name": "contextfs_sync_push",
        "description": "Push local memory changes to the sync server",
        "parameters": {
            "type": "object",
            "properties": {
                "server_url": {
                    "type": "string",
                    "description": "URL of the sync server",
                    "default": DEFAULT_SERVER_URL,
                },
                "namespace_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of namespace IDs to sync",
                },
            },
        },
        "handler": contextfs_sync_push,
    },
    {
        "name": "contextfs_sync_pull",
        "description": "Pull memory changes from the sync server",
        "parameters": {
            "type": "object",
            "properties": {
                "server_url": {
                    "type": "string",
                    "description": "URL of the sync server",
                    "default": DEFAULT_SERVER_URL,
                },
                "namespace_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of namespace IDs to sync",
                },
                "since": {
                    "type": "string",
                    "description": "ISO timestamp to pull changes after",
                },
            },
        },
        "handler": contextfs_sync_pull,
    },
    {
        "name": "contextfs_sync_all",
        "description": "Full bidirectional sync with the server (push + pull)",
        "parameters": {
            "type": "object",
            "properties": {
                "server_url": {
                    "type": "string",
                    "description": "URL of the sync server",
                    "default": DEFAULT_SERVER_URL,
                },
                "namespace_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of namespace IDs to sync",
                },
            },
        },
        "handler": contextfs_sync_all,
    },
    {
        "name": "contextfs_sync_diff",
        "description": "Content-addressed sync (idempotent, Merkle-style). Compares content hashes instead of timestamps.",
        "parameters": {
            "type": "object",
            "properties": {
                "server_url": {
                    "type": "string",
                    "description": "URL of the sync server",
                    "default": DEFAULT_SERVER_URL,
                },
                "namespace_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of namespace IDs to sync",
                },
            },
        },
        "handler": contextfs_sync_diff,
    },
    {
        "name": "contextfs_sync_status",
        "description": "Get sync status (last sync time, pending changes)",
        "parameters": {
            "type": "object",
            "properties": {
                "server_url": {
                    "type": "string",
                    "description": "URL of the sync server",
                    "default": DEFAULT_SERVER_URL,
                },
            },
        },
        "handler": contextfs_sync_status,
    },
]
