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

    # Owner (for multi-tenant isolation)
    user_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)

    # Team sharing (for Team tier)
    owner_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    team_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    visibility: Mapped[str] = mapped_column(
        Text, default="private"
    )  # private, team_read, team_write

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

    # Owner (for multi-tenant isolation)
    user_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)

    # Team sharing (for Team tier)
    owner_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    team_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    visibility: Mapped[str] = mapped_column(
        Text, default="private"
    )  # private, team_read, team_write

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
    user_id: Mapped[str | None] = mapped_column(
        Text, nullable=True, index=True
    )  # Links device to user
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


# =============================================================================
# Auth Models (users, api_keys, subscriptions, usage)
# =============================================================================


class UserModel(Base):
    """User accounts."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    email: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    name: Mapped[str | None] = mapped_column(Text)
    provider: Mapped[str] = mapped_column(Text, nullable=False, default="api_key")
    provider_id: Mapped[str | None] = mapped_column(Text)
    password_hash: Mapped[str | None] = mapped_column(Text)
    email_verified: Mapped[bool] = mapped_column(default=False)
    is_admin: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class APIKeyModel(Base):
    """API keys for authentication."""

    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    key_hash: Mapped[str] = mapped_column(Text, nullable=False)
    key_prefix: Mapped[str] = mapped_column(Text, nullable=False)
    encryption_salt: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SubscriptionModel(Base):
    """User subscriptions."""

    __tablename__ = "subscriptions"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    tier: Mapped[str] = mapped_column(Text, default="free")  # free, pro, team, enterprise
    stripe_customer_id: Mapped[str | None] = mapped_column(Text)
    stripe_subscription_id: Mapped[str | None] = mapped_column(Text)
    device_limit: Mapped[int] = mapped_column(default=2)  # Free: 2, Pro: 5, Team: 10
    memory_limit: Mapped[int] = mapped_column(default=5000)  # Free: 5K, Pro: 50K, Team: unlimited
    status: Mapped[str] = mapped_column(Text, default="active")
    current_period_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Team tier fields
    team_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    seats_included: Mapped[int] = mapped_column(default=1)  # Team tier includes 5 seats
    seats_used: Mapped[int] = mapped_column(default=1)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class UsageModel(Base):
    """User usage tracking."""

    __tablename__ = "usage"

    user_id: Mapped[str] = mapped_column(Text, primary_key=True)
    device_count: Mapped[int] = mapped_column(default=0)
    memory_count: Mapped[int] = mapped_column(default=0)
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class PasswordResetToken(Base):
    """Password reset tokens for email-based login."""

    __tablename__ = "password_reset_tokens"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# =============================================================================
# Team Models (for Team tier collaboration)
# =============================================================================


class TeamModel(Base):
    """Teams for shared memory collaboration."""

    __tablename__ = "teams"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    owner_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class TeamMemberModel(Base):
    """Team membership with roles."""

    __tablename__ = "team_members"

    team_id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(Text, primary_key=True, index=True)
    role: Mapped[str] = mapped_column(
        Text, nullable=False, default="member"
    )  # owner, admin, member
    invited_by: Mapped[str | None] = mapped_column(Text)
    joined_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class TeamInvitationModel(Base):
    """Pending team invitations."""

    __tablename__ = "team_invitations"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    team_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    email: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    role: Mapped[str] = mapped_column(Text, nullable=False, default="member")
    invited_by: Mapped[str] = mapped_column(Text, nullable=False)
    token_hash: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    accepted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
