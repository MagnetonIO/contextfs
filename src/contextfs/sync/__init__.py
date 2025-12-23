"""
ContextFS Sync Module - Multi-device synchronization.

Provides:
- SyncClient: Client-side sync with remote server
- PostgresSync: Direct PostgreSQL sync (legacy)
- VectorClock: Conflict resolution via vector clocks
- Protocol models: Pydantic models for sync operations
- Path normalization: Cross-machine path handling
"""

from contextfs.sync.client import SyncClient
from contextfs.sync.path_resolver import PathResolver, PortablePath, RepoRegistry
from contextfs.sync.postgres import PostgresSync
from contextfs.sync.protocol import (
    BackendType,
    ConflictResolution,
    SyncableBackend,
    SyncableEntity,
    SyncedEdge,
    SyncedMemory,
    SyncedSession,
    SyncPullRequest,
    SyncPullResponse,
    SyncPushRequest,
    SyncPushResponse,
    SyncResult,
    SyncStatus,
)
from contextfs.sync.vector_clock import DeviceTracker, VectorClock

__all__ = [
    # Client
    "SyncClient",
    # Legacy
    "PostgresSync",
    # Vector clock
    "VectorClock",
    "DeviceTracker",
    # Protocol
    "SyncableBackend",
    "SyncableEntity",
    "SyncedMemory",
    "SyncedSession",
    "SyncedEdge",
    "SyncPushRequest",
    "SyncPushResponse",
    "SyncPullRequest",
    "SyncPullResponse",
    "SyncResult",
    "SyncStatus",
    "BackendType",
    "ConflictResolution",
    # Path handling
    "PathResolver",
    "PortablePath",
    "RepoRegistry",
]
