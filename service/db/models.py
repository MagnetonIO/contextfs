"""SQLAlchemy 2.0 models for sync service.

Uses modern SQLAlchemy patterns:
- DeclarativeBase with Mapped types
- mapped_column for column definitions
- JSONB for vector clocks and metadata
- pgvector for embeddings
"""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, Index, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    # Fallback for environments without pgvector
    Vector = None


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class SyncedMemoryModel(Base):
    """Memory table with sync support.

    Includes all sync-related fields:
    - vector_clock: For conflict detection
    - content_hash: For deduplication
    - deleted_at: For soft deletes
    - last_modified_by: Device tracking
    """

    __tablename__ = "memories"

    # Primary key
    id: Mapped[str] = mapped_column(Text, primary_key=True)

    # Core content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False, default="fact")
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    summary: Mapped[str | None] = mapped_column(Text)

    # Namespace
    namespace_id: Mapped[str] = mapped_column(Text, nullable=False, default="global")

    # Portable source reference (for cross-machine sync)
    repo_url: Mapped[str | None] = mapped_column(Text)
    repo_name: Mapped[str | None] = mapped_column(Text)
    relative_path: Mapped[str | None] = mapped_column(Text)

    # Legacy source fields (for backwards compatibility)
    source_file: Mapped[str | None] = mapped_column(Text)
    source_repo: Mapped[str | None] = mapped_column(Text)
    source_tool: Mapped[str | None] = mapped_column(Text)

    # Context
    project: Mapped[str | None] = mapped_column(Text)
    session_id: Mapped[str | None] = mapped_column(Text)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Sync fields - CRITICAL
    vector_clock: Mapped[dict[str, int]] = mapped_column(JSONB, default=dict)
    content_hash: Mapped[str | None] = mapped_column(Text)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_modified_by: Mapped[str | None] = mapped_column(Text)  # device_id

    # Extra metadata (note: can't use 'metadata' as it's reserved by SQLAlchemy)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)

    # Embedding - using dynamic column definition for pgvector
    # Note: embedding column is added via migration if pgvector is available

    __table_args__ = (
        Index("idx_memories_namespace", "namespace_id"),
        Index("idx_memories_type", "type"),
        Index("idx_memories_updated", "updated_at"),
        Index("idx_memories_deleted", "deleted_at"),
        Index("idx_memories_content_hash", "content_hash"),
        Index("idx_memories_repo_url", "repo_url"),
    )


# Add embedding column if pgvector is available
if Vector is not None:
    SyncedMemoryModel.embedding = mapped_column(Vector(384), nullable=True)


class SyncedSessionModel(Base):
    """Session table with sync support."""

    __tablename__ = "sessions"

    # Primary key
    id: Mapped[str] = mapped_column(Text, primary_key=True)

    # Core fields
    label: Mapped[str | None] = mapped_column(Text)
    namespace_id: Mapped[str] = mapped_column(Text, nullable=False, default="global")
    tool: Mapped[str] = mapped_column(Text, nullable=False, default="contextfs")

    # Portable repo reference
    repo_url: Mapped[str | None] = mapped_column(Text)
    repo_name: Mapped[str | None] = mapped_column(Text)

    # Legacy field
    repo_path: Mapped[str | None] = mapped_column(Text)

    branch: Mapped[str | None] = mapped_column(Text)

    # Session timing
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    summary: Mapped[str | None] = mapped_column(Text)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Sync fields
    vector_clock: Mapped[dict[str, int]] = mapped_column(JSONB, default=dict)
    content_hash: Mapped[str | None] = mapped_column(Text)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_modified_by: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        Index("idx_sessions_namespace", "namespace_id"),
        Index("idx_sessions_updated", "updated_at"),
        Index("idx_sessions_deleted", "deleted_at"),
    )


class SyncedEdgeModel(Base):
    """Memory edge/relationship table with sync support."""

    __tablename__ = "memory_edges"

    # Composite primary key
    from_id: Mapped[str] = mapped_column(Text, primary_key=True)
    to_id: Mapped[str] = mapped_column(Text, primary_key=True)
    relation: Mapped[str] = mapped_column(Text, primary_key=True)

    # Edge attributes
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    created_by: Mapped[str | None] = mapped_column(Text)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Sync fields
    vector_clock: Mapped[dict[str, int]] = mapped_column(JSONB, default=dict)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_modified_by: Mapped[str | None] = mapped_column(Text)

    # Composite ID for SyncableEntity compatibility
    @property
    def id(self) -> str:
        """Generate composite ID for sync."""
        return f"{self.from_id}:{self.relation}:{self.to_id}"

    __table_args__ = (
        Index("idx_edges_from", "from_id"),
        Index("idx_edges_to", "to_id"),
        Index("idx_edges_updated", "updated_at"),
    )


class Device(Base):
    """Registered sync devices."""

    __tablename__ = "devices"

    device_id: Mapped[str] = mapped_column(Text, primary_key=True)
    device_name: Mapped[str] = mapped_column(Text, nullable=False)
    platform: Mapped[str] = mapped_column(Text, nullable=False)
    client_version: Mapped[str] = mapped_column(Text, nullable=False)

    registered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    sync_cursor: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Device metadata (note: can't use 'metadata' as it's reserved by SQLAlchemy)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, default=dict)


class SyncState(Base):
    """Track sync state per device."""

    __tablename__ = "sync_state"

    device_id: Mapped[str] = mapped_column(Text, primary_key=True)
    last_push_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_pull_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    push_cursor: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    pull_cursor: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Stats
    total_pushed: Mapped[int] = mapped_column(default=0)
    total_pulled: Mapped[int] = mapped_column(default=0)
    total_conflicts: Mapped[int] = mapped_column(default=0)
