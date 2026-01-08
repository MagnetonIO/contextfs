"""
Versioned memory types for the ContextFS formal type system.

Implements from Definition 5.1 Type Grammar:
    VersionedType ::= Versioned MemoryType

From the typed-memory paper (Section 4):
    ChangeReason ::= Observation | Inference | Correction | Decay

This module provides:
- ChangeReason: Enum for tracking why memory changed
- VersionEntry[S]: A single version snapshot with reason and timestamp
- Timeline[S]: Ordered collection of version entries
- VersionedMem[S]: Full versioned memory with timeline and operations

The versioning system enables:
- Tracking the causal structure of belief revision
- Reasoning about why memory changed, not just that it did
- Maintaining agent continuity through belief evolution
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic
from uuid import uuid4

from pydantic import BaseModel, Field

from contextfs.types.base import BaseSchema, S

if TYPE_CHECKING:
    from contextfs.schemas import Memory
    from contextfs.types.memory import Mem


# =============================================================================
# ChangeReason Enum (Paper Section 4.1)
# =============================================================================


class ChangeReason(str, Enum):
    """
    Reason for a memory version change.

    From the typed-memory paper (Definition 4.2):
        ChangeReason ::= Observation | Inference | Correction | Decay

    Each reason captures a fundamental mode of belief change:
    - OBSERVATION: New information observed from external sources
    - INFERENCE: Derived from existing knowledge via reasoning
    - CORRECTION: Fixing an error in previous belief
    - DECAY: Knowledge becoming stale or confidence reducing

    The ChangeReason is tracked in VersionEntry and stored in
    edge metadata when using memory_lineage operations.
    """

    OBSERVATION = "observation"
    """Memory changed due to new external information (sensory/informational input)."""

    INFERENCE = "inference"
    """Memory changed due to reasoning from existing beliefs."""

    CORRECTION = "correction"
    """Memory changed to fix an identified error in previous state."""

    DECAY = "decay"
    """Memory changed due to staleness or confidence reduction (forgetting)."""

    @classmethod
    def from_string(cls, value: str) -> ChangeReason:
        """Parse ChangeReason from string, with fallback to OBSERVATION."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.OBSERVATION


# =============================================================================
# VersionEntry - Single Version Snapshot
# =============================================================================


class VersionEntry(BaseModel, Generic[S]):
    """
    A single version entry in the memory timeline.

    Captures a snapshot of memory state along with metadata about
    when and why the change occurred.

    Type Parameters:
        S: The schema type for the version content.

    Attributes:
        version_id: Unique identifier for this version.
        memory_id: ID of the memory this version belongs to.
        timestamp: When this version was created.
        reason: Why this version was created (ChangeReason).
        content: The schema data at this version.
        author: Who/what created this version.
        metadata: Additional version metadata.

    Example:
        >>> entry = VersionEntry[DecisionData](
        ...     memory_id="mem123",
        ...     reason=ChangeReason.CORRECTION,
        ...     content=DecisionData(decision="SQLite"),
        ...     author="claude"
        ... )
    """

    version_id: str = Field(default_factory=lambda: str(uuid4())[:12])
    memory_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reason: ChangeReason
    content: Any  # Actually S, but Pydantic needs Any for generics
    author: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    @property
    def content_dict(self) -> dict[str, Any]:
        """Get content as dictionary."""
        if isinstance(self.content, BaseModel):
            return self.content.model_dump(mode="json")
        return dict(self.content) if self.content else {}

    def to_storage_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "version_id": self.version_id,
            "memory_id": self.memory_id,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason.value,
            "content": self.content_dict,
            "author": self.author,
            "metadata": self.metadata,
        }

    @classmethod
    def from_storage_dict(
        cls,
        data: dict[str, Any],
        schema_type: type[S] | None = None,
    ) -> VersionEntry[S]:
        """Create from storage dictionary."""
        content = data.get("content", {})
        if schema_type is not None and content:
            content = schema_type.model_validate(content)

        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            version_id=data.get("version_id", str(uuid4())[:12]),
            memory_id=data["memory_id"],
            timestamp=timestamp or datetime.now(timezone.utc),
            reason=ChangeReason.from_string(data.get("reason", "observation")),
            content=content,
            author=data.get("author"),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Timeline - Ordered Version History
# =============================================================================


class Timeline(BaseModel, Generic[S]):
    """
    Timeline tracking version history for a memory.

    Provides ordered access to all versions of a memory with
    query operations for accessing historical states.

    Type Parameters:
        S: The schema type for version content.

    Attributes:
        entries: Ordered list of version entries (oldest first).

    Example:
        >>> timeline = Timeline[DecisionData]()
        >>> timeline.add(VersionEntry(
        ...     memory_id="mem123",
        ...     reason=ChangeReason.OBSERVATION,
        ...     content=DecisionData(decision="PostgreSQL")
        ... ))
        >>> timeline.current.content.decision
        'PostgreSQL'
    """

    entries: list[VersionEntry[S]] = Field(default_factory=list)

    model_config = {"extra": "allow"}

    @property
    def current(self) -> VersionEntry[S] | None:
        """Get the current (latest) version entry."""
        return self.entries[-1] if self.entries else None

    @property
    def root(self) -> VersionEntry[S] | None:
        """Get the original (first) version entry."""
        return self.entries[0] if self.entries else None

    @property
    def is_empty(self) -> bool:
        """Check if timeline has no entries."""
        return len(self.entries) == 0

    def add(self, entry: VersionEntry[S]) -> None:
        """
        Add a new version entry to the timeline.

        Entries are maintained in chronological order.

        Args:
            entry: The version entry to add.
        """
        self.entries.append(entry)

        # Sort by timestamp to maintain order (normalize to UTC for comparison)
        def normalize_ts(e: VersionEntry[S]) -> datetime:
            ts = e.timestamp
            if ts.tzinfo is None:
                return ts.replace(tzinfo=timezone.utc)
            return ts

        self.entries.sort(key=normalize_ts)

    def at(self, timestamp: datetime) -> VersionEntry[S] | None:
        """
        Get the version that was current at a specific point in time.

        Args:
            timestamp: The point in time to query.

        Returns:
            The most recent version entry before or at the timestamp,
            or None if no versions exist before that time.
        """
        # Normalize timestamp to UTC-aware for comparison
        ts = timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        result = None
        for entry in self.entries:
            entry_ts = entry.timestamp
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.replace(tzinfo=timezone.utc)
            if entry_ts <= ts:
                result = entry
            else:
                break
        return result

    def since(self, timestamp: datetime) -> list[VersionEntry[S]]:
        """Get all versions since a timestamp."""
        # Normalize timestamp for comparison
        ts = timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        def normalize_entry_ts(entry: VersionEntry[S]) -> datetime:
            entry_ts = entry.timestamp
            if entry_ts.tzinfo is None:
                return entry_ts.replace(tzinfo=timezone.utc)
            return entry_ts

        return [e for e in self.entries if normalize_entry_ts(e) > ts]

    def by_reason(self, reason: ChangeReason) -> list[VersionEntry[S]]:
        """Get all versions with a specific change reason."""
        return [e for e in self.entries if e.reason == reason]

    def get_version(self, version_id: str) -> VersionEntry[S] | None:
        """Get a specific version by ID."""
        for entry in self.entries:
            if entry.version_id == version_id:
                return entry
        return None

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[VersionEntry[S]]:
        return iter(self.entries)

    def __getitem__(self, index: int) -> VersionEntry[S]:
        return self.entries[index]


# =============================================================================
# VersionedMem - Full Versioned Memory Type
# =============================================================================


class VersionedMem(BaseModel, Generic[S]):
    """
    Versioned memory with full history tracking.

    Corresponds to: VersionedType ::= Versioned MemoryType

    Wraps a memory with timeline tracking, enabling:
    - Version evolution with change reasons
    - Historical state queries
    - Consistency checking
    - Agent continuity through belief revision

    Type Parameters:
        S: The schema type for memory content.

    Attributes:
        memory_id: ID of the underlying memory.
        timeline: Full version history.
        authoritative_version: ID of the authoritative version (if set).
        schema_type_name: Name of the schema type for serialization.

    Example:
        >>> versioned = VersionedMem.from_memory(memory, DecisionData)
        >>>
        >>> # Evolve with a reason
        >>> versioned.evolve(
        ...     new_content=DecisionData(decision="SQLite"),
        ...     reason=ChangeReason.CORRECTION,
        ...     author="claude"
        ... )
        >>>
        >>> # Query history
        >>> print(f"Version count: {len(versioned.timeline)}")
        >>> print(f"Original: {versioned.timeline.root.content.decision}")
        >>> print(f"Current: {versioned.timeline.current.content.decision}")
    """

    memory_id: str
    timeline: Timeline[S] = Field(default_factory=Timeline)
    authoritative_version: str | None = None
    schema_type_name: str | None = None

    # Internal: schema type for typed operations (not serialized)
    _schema_type: type[S] | None = None

    model_config = {"extra": "allow", "arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        """Initialize schema type from name if needed."""
        # Schema type resolution happens via registry when needed
        pass

    @classmethod
    def from_memory(
        cls,
        memory: Memory,
        schema_type: type[S],
    ) -> VersionedMem[S]:
        """
        Create a VersionedMem from an existing Memory.

        The memory's current state becomes the initial version
        with OBSERVATION as the change reason.

        Args:
            memory: The Memory to create versioned view for.
            schema_type: The schema type for typed access.

        Returns:
            New VersionedMem with the memory as initial version.
        """
        versioned = cls(
            memory_id=memory.id,
            schema_type_name=getattr(schema_type, "_schema_name", schema_type.__name__),
        )
        versioned._schema_type = schema_type

        # Create initial version from memory's structured_data
        if memory.structured_data:
            content = schema_type.model_validate(memory.structured_data)
            # Normalize timestamp to UTC-aware
            ts = memory.created_at
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            initial_entry = VersionEntry[S](
                memory_id=memory.id,
                timestamp=ts,
                reason=ChangeReason.OBSERVATION,
                content=content,
                metadata={"source": "initial_memory"},
            )
            versioned.timeline.add(initial_entry)

        return versioned

    @classmethod
    def from_lineage(
        cls,
        memory_id: str,
        schema_type: type[S],
        lineage_history: list[dict[str, Any]],
    ) -> VersionedMem[S]:
        """
        Create VersionedMem from memory lineage history.

        Reconstructs timeline from existing lineage graph data.

        Args:
            memory_id: The memory ID.
            schema_type: The schema type.
            lineage_history: List of version dicts from lineage system.

        Returns:
            VersionedMem with reconstructed timeline.
        """
        versioned = cls(
            memory_id=memory_id,
            schema_type_name=getattr(schema_type, "_schema_name", schema_type.__name__),
        )
        versioned._schema_type = schema_type

        for entry_data in lineage_history:
            entry = VersionEntry.from_storage_dict(entry_data, schema_type)
            versioned.timeline.add(entry)

        return versioned

    # =========================================================================
    # Current State Access
    # =========================================================================

    @property
    def current(self) -> Mem[S] | None:
        """
        Get the current state as a Mem[S].

        Returns:
            Mem wrapping current version, or None if no versions exist.
        """
        from contextfs.schemas import Memory
        from contextfs.types.memory import Mem

        entry = self.timeline.current
        if entry is None:
            return None

        # Reconstruct Memory from version entry
        content_dict = entry.content_dict
        memory = Memory(
            id=entry.memory_id,
            content=str(content_dict.get("decision", content_dict)),
            structured_data=content_dict,
        )

        return Mem(memory, self._schema_type or BaseSchema)

    @property
    def current_data(self) -> S | None:
        """Get current schema data directly."""
        entry = self.timeline.current
        if entry is None:
            return None
        return entry.content

    # =========================================================================
    # Evolution Operations
    # =========================================================================

    def evolve(
        self,
        new_content: S,
        reason: ChangeReason,
        author: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VersionEntry[S]:
        """
        Evolve to a new version with change tracking.

        Creates a new version entry and adds it to the timeline.
        The previous version remains accessible in history.

        Args:
            new_content: New schema data for this version.
            reason: Why this version was created.
            author: Who/what created this version.
            metadata: Additional version metadata.

        Returns:
            The new VersionEntry that was created.

        Example:
            >>> entry = versioned.evolve(
            ...     new_content=DecisionData(decision="SQLite"),
            ...     reason=ChangeReason.CORRECTION,
            ...     author="claude",
            ...     metadata={"ticket": "PROJ-123"}
            ... )
        """
        entry = VersionEntry[S](
            memory_id=self.memory_id,
            reason=reason,
            content=new_content,
            author=author,
            metadata=metadata or {},
        )
        self.timeline.add(entry)
        return entry

    def correct(
        self,
        new_content: S,
        author: str | None = None,
        correction_note: str | None = None,
    ) -> VersionEntry[S]:
        """
        Create a correction version.

        Convenience method for evolve() with CORRECTION reason.

        Args:
            new_content: Corrected schema data.
            author: Who made the correction.
            correction_note: Note explaining the correction.

        Returns:
            The new correction VersionEntry.
        """
        metadata = {}
        if correction_note:
            metadata["correction_note"] = correction_note
        return self.evolve(
            new_content=new_content,
            reason=ChangeReason.CORRECTION,
            author=author,
            metadata=metadata,
        )

    def infer(
        self,
        new_content: S,
        premises: list[str] | None = None,
        author: str | None = None,
    ) -> VersionEntry[S]:
        """
        Create an inference version.

        Convenience method for evolve() with INFERENCE reason.

        Args:
            new_content: Inferred schema data.
            premises: Memory IDs that served as inference premises.
            author: Who/what made the inference.

        Returns:
            The new inference VersionEntry.
        """
        metadata = {}
        if premises:
            metadata["premises"] = premises
        return self.evolve(
            new_content=new_content,
            reason=ChangeReason.INFERENCE,
            author=author,
            metadata=metadata,
        )

    # =========================================================================
    # Consistency and Validation
    # =========================================================================

    def is_consistent(self) -> bool:
        """
        Verify timeline consistency.

        Checks that:
        - Timestamps are monotonically increasing
        - All memory_ids match
        - No duplicate version_ids

        Returns:
            True if timeline is consistent, False otherwise.
        """
        if self.timeline.is_empty:
            return True

        version_ids = set()
        prev_timestamp = None

        for entry in self.timeline:
            # Check memory_id
            if entry.memory_id != self.memory_id:
                return False

            # Check timestamp ordering
            if prev_timestamp is not None and entry.timestamp < prev_timestamp:
                return False
            prev_timestamp = entry.timestamp

            # Check unique version_ids
            if entry.version_id in version_ids:
                return False
            version_ids.add(entry.version_id)

        return True

    def get_authoritative(self) -> VersionEntry[S] | None:
        """Get the authoritative version if set."""
        if self.authoritative_version is None:
            return None
        return self.timeline.get_version(self.authoritative_version)

    def set_authoritative(self, version_id: str) -> bool:
        """
        Mark a version as authoritative.

        Args:
            version_id: The version to mark as authoritative.

        Returns:
            True if version exists and was marked, False otherwise.
        """
        if self.timeline.get_version(version_id) is None:
            return False
        self.authoritative_version = version_id
        return True

    # =========================================================================
    # History Queries
    # =========================================================================

    def why_changed(self, version_id: str | None = None) -> ChangeReason | None:
        """
        Get the reason for a version change.

        Args:
            version_id: Version to query (current if None).

        Returns:
            The ChangeReason for the version, or None if not found.
        """
        if version_id is None:
            entry = self.timeline.current
        else:
            entry = self.timeline.get_version(version_id)
        return entry.reason if entry else None

    def corrections(self) -> list[VersionEntry[S]]:
        """Get all correction versions."""
        return self.timeline.by_reason(ChangeReason.CORRECTION)

    def inferences(self) -> list[VersionEntry[S]]:
        """Get all inference versions."""
        return self.timeline.by_reason(ChangeReason.INFERENCE)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_storage_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "memory_id": self.memory_id,
            "authoritative_version": self.authoritative_version,
            "schema_type_name": self.schema_type_name,
            "timeline": [e.to_storage_dict() for e in self.timeline],
        }

    @classmethod
    def from_storage_dict(
        cls,
        data: dict[str, Any],
        schema_type: type[S],
    ) -> VersionedMem[S]:
        """Create from storage dictionary."""
        versioned = cls(
            memory_id=data["memory_id"],
            authoritative_version=data.get("authoritative_version"),
            schema_type_name=data.get("schema_type_name"),
        )
        versioned._schema_type = schema_type

        for entry_data in data.get("timeline", []):
            entry = VersionEntry.from_storage_dict(entry_data, schema_type)
            versioned.timeline.add(entry)

        return versioned

    def __len__(self) -> int:
        """Number of versions in timeline."""
        return len(self.timeline)

    def __repr__(self) -> str:
        schema_name = self.schema_type_name or "Unknown"
        return f"VersionedMem[{schema_name}](id={self.memory_id!r}, versions={len(self)})"
