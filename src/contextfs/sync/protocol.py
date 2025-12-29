"""Shared sync protocol models for client and server.

This module defines the protocol interfaces and Pydantic models
used for synchronization between client devices and the sync server.

Key concepts:
- SyncableEntity: Base for all entities that can be synced
- SyncableBackend: Protocol for database-agnostic sync adapters
- Request/Response models for push/pull operations
"""

from __future__ import annotations

import hashlib
from abc import abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


# =============================================================================
# Enums
# =============================================================================


class BackendType(str, Enum):
    """Type of storage backend for sync routing."""

    RELATIONAL = "relational"  # SQLite ↔ PostgreSQL
    VECTOR = "vector"  # ChromaDB ↔ pgvector
    GRAPH = "graph"  # FalkorDB ↔ Apache AGE


class ConflictResolution(str, Enum):
    """Conflict resolution strategies."""

    LAST_WRITE_WINS = "last_write_wins"
    LOCAL_WINS = "local_wins"
    REMOTE_WINS = "remote_wins"
    MANUAL = "manual"  # Return conflict for manual resolution


class SyncStatus(str, Enum):
    """Status of a sync operation."""

    SUCCESS = "success"
    PARTIAL = "partial"  # Some items synced, some failed
    CONFLICT = "conflict"
    FAILED = "failed"


# =============================================================================
# Base Syncable Entity
# =============================================================================


class SyncableEntity(BaseModel):
    """Base for all syncable entities.

    All entities that participate in sync must include these fields
    for conflict detection and soft delete support.
    """

    id: str
    vector_clock: dict[str, int] = Field(default_factory=dict)
    content_hash: str | None = None
    deleted_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_modified_by: str | None = None  # device_id that made last change

    def compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content (truncated to 16 chars)."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def is_deleted(self) -> bool:
        """Check if entity is soft-deleted."""
        return self.deleted_at is not None

    model_config = {"extra": "allow"}


# =============================================================================
# Synced Memory
# =============================================================================


class SyncedMemory(SyncableEntity):
    """Memory with sync metadata.

    Uses portable paths (repo_url + relative_path) instead of
    absolute paths for cross-machine sync compatibility.
    """

    # Core content
    content: str
    type: str = "fact"
    tags: list[str] = Field(default_factory=list)
    summary: str | None = None

    # Namespace
    namespace_id: str = "global"

    # Portable source reference (synced)
    repo_url: str | None = None  # git@github.com:user/repo.git
    repo_name: str | None = None  # Human-readable repo name
    relative_path: str | None = None  # Path from repo root

    # Legacy fields (for backwards compatibility)
    source_file: str | None = None
    source_repo: str | None = None
    source_tool: str | None = None

    # Context
    project: str | None = None
    session_id: str | None = None

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Embedding (synced for vector backend)
    embedding: list[float] | None = None


# =============================================================================
# Synced Session
# =============================================================================


class SyncedSession(SyncableEntity):
    """Session with sync metadata."""

    label: str | None = None
    namespace_id: str = "global"
    tool: str = "contextfs"

    # Portable repo reference
    repo_url: str | None = None
    repo_name: str | None = None

    # Legacy field
    repo_path: str | None = None

    branch: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Synced Edge (for graph backend)
# =============================================================================


class SyncedEdge(SyncableEntity):
    """Memory edge/relationship with sync metadata."""

    from_id: str
    to_id: str
    relation: str  # EdgeRelation enum value
    weight: float = 1.0
    created_by: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Device Registration
# =============================================================================


class DeviceRegistration(BaseModel):
    """Register a new device for sync."""

    device_id: str
    device_name: str
    platform: str  # darwin, linux, windows
    client_version: str


class DeviceInfo(BaseModel):
    """Information about a registered device."""

    device_id: str
    device_name: str
    platform: str
    client_version: str
    registered_at: datetime
    last_sync_at: datetime | None = None
    sync_cursor: datetime | None = None


# =============================================================================
# Sync Request/Response Models
# =============================================================================


class SyncPushRequest(BaseModel):
    """Request to push changes to server."""

    device_id: str
    memories: list[SyncedMemory] = Field(default_factory=list)
    sessions: list[SyncedSession] = Field(default_factory=list)
    edges: list[SyncedEdge] = Field(default_factory=list)
    last_sync_timestamp: datetime | None = None

    # Optional: specify backend type for routing
    backend_type: BackendType | None = None


class ConflictInfo(BaseModel):
    """Information about a sync conflict."""

    entity_id: str
    entity_type: str  # "memory", "session", "edge"
    client_clock: dict[str, int]
    server_clock: dict[str, int]
    client_content: str | None = None
    server_content: str | None = None
    client_updated_at: datetime
    server_updated_at: datetime


class SyncPushResponse(BaseModel):
    """Response from push operation."""

    success: bool
    status: SyncStatus = SyncStatus.SUCCESS
    accepted: int = 0
    rejected: int = 0
    conflicts: list[ConflictInfo] = Field(default_factory=list)
    server_timestamp: datetime = Field(default_factory=utc_now)
    message: str | None = None


class SyncPullRequest(BaseModel):
    """Request to pull changes from server."""

    device_id: str
    since_timestamp: datetime | None = None
    namespace_ids: list[str] | None = None  # Optional filter
    backend_type: BackendType | None = None
    limit: int = 1000  # Max items per request
    offset: int = 0  # Offset for pagination


class SyncPullResponse(BaseModel):
    """Response with changes from server."""

    success: bool
    memories: list[SyncedMemory] = Field(default_factory=list)
    sessions: list[SyncedSession] = Field(default_factory=list)
    edges: list[SyncedEdge] = Field(default_factory=list)
    server_timestamp: datetime = Field(default_factory=utc_now)
    has_more: bool = False  # True if more pages available
    next_cursor: datetime | None = None  # Cursor for next page (deprecated, use next_offset)
    next_offset: int = 0  # Offset for next page


class SyncStatusRequest(BaseModel):
    """Request for sync status."""

    device_id: str


class SyncStatusResponse(BaseModel):
    """Response with sync status."""

    device_id: str
    last_sync_at: datetime | None = None
    pending_push_count: int = 0
    pending_pull_count: int = 0
    server_timestamp: datetime = Field(default_factory=utc_now)


# =============================================================================
# Content-Addressed Sync (Merkle-style)
# =============================================================================


class EntityManifestEntry(BaseModel):
    """Single entry in a sync manifest."""

    id: str
    content_hash: str | None = None
    updated_at: datetime | None = None


class SyncManifestRequest(BaseModel):
    """Request containing client's manifest of entities.

    Client sends list of {id, content_hash} for all entities.
    Server compares and returns what differs.
    """

    device_id: str
    memories: list[EntityManifestEntry] = Field(default_factory=list)
    sessions: list[EntityManifestEntry] = Field(default_factory=list)
    edges: list[EntityManifestEntry] = Field(default_factory=list)
    namespace_ids: list[str] | None = None  # Optional filter


class SyncDiffResponse(BaseModel):
    """Response with diff between client and server state.

    Content-addressed sync response - tells client exactly what
    they're missing, what's been updated, and what's been deleted.
    Also tells client what server is missing (for content-addressed push).
    """

    success: bool = True

    # Memories client is missing or has outdated (for pull)
    missing_memories: list[SyncedMemory] = Field(default_factory=list)
    # Sessions client is missing or has outdated (for pull)
    missing_sessions: list[SyncedSession] = Field(default_factory=list)
    # Edges client is missing or has outdated (for pull)
    missing_edges: list[SyncedEdge] = Field(default_factory=list)

    # IDs of entities that were deleted on server
    deleted_memory_ids: list[str] = Field(default_factory=list)
    deleted_session_ids: list[str] = Field(default_factory=list)
    deleted_edge_ids: list[str] = Field(default_factory=list)

    # IDs of entities that server is missing (for content-addressed push)
    server_missing_memory_ids: list[str] = Field(default_factory=list)
    server_missing_session_ids: list[str] = Field(default_factory=list)
    server_missing_edge_ids: list[str] = Field(default_factory=list)

    # Summary stats
    total_missing: int = 0
    total_updated: int = 0
    total_deleted: int = 0
    total_server_missing: int = 0  # What server needs from client

    server_timestamp: datetime = Field(default_factory=utc_now)
    has_more: bool = False  # True if more pages available
    next_offset: int = 0  # Offset for next page


# =============================================================================
# Sync Result (for client-side operations)
# =============================================================================


class SyncResult(BaseModel):
    """Result of a full sync operation."""

    pushed: SyncPushResponse
    pulled: SyncPullResponse
    duration_ms: float = 0
    errors: list[str] = Field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if sync was fully successful."""
        return self.pushed.success and self.pulled.success and len(self.pushed.conflicts) == 0


# =============================================================================
# Syncable Backend Protocol
# =============================================================================


@runtime_checkable
class SyncableBackend(Protocol):
    """Protocol for syncable storage backends.

    Implementations provide database-agnostic sync operations
    for different storage types (relational, vector, graph).
    """

    @abstractmethod
    async def get_changes_since(
        self,
        since: datetime | None,
        device_id: str,
        namespace_ids: list[str] | None = None,
        limit: int = 1000,
    ) -> list[SyncableEntity]:
        """
        Get all changes since timestamp.

        Args:
            since: Timestamp to get changes after (None for all)
            device_id: Device requesting changes
            namespace_ids: Optional namespace filter
            limit: Maximum items to return

        Returns:
            List of changed entities
        """
        ...

    @abstractmethod
    async def apply_changes(
        self,
        changes: list[SyncableEntity],
        device_id: str,
        conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITE_WINS,
    ) -> SyncPushResponse:
        """
        Apply changes from remote, handling conflicts via vector clocks.

        Args:
            changes: List of entities to apply
            device_id: Device sending changes
            conflict_resolution: Strategy for handling conflicts

        Returns:
            Push response with accepted/rejected/conflict counts
        """
        ...

    @abstractmethod
    def get_backend_type(self) -> BackendType:
        """Return backend type for routing."""
        ...

    @abstractmethod
    async def get_pending_count(
        self,
        device_id: str,
        since: datetime | None = None,
    ) -> int:
        """Get count of pending changes for a device."""
        ...
