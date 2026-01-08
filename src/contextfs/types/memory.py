"""
Schema-indexed memory type for the ContextFS formal type system.

Implements from Definition 5.1 Type Grammar:
    MemoryType ::= Mem Schema

This module provides:
- Mem[S]: A schema-indexed memory wrapper that provides type-safe access
  to structured_data while maintaining full compatibility with Memory.

The Mem[S] type wraps the existing Memory class and adds:
- Type-safe access to structured_data via the `data` property
- Schema validation on construction
- Generic type parameter preservation for static type checking
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic

from pydantic import BaseModel

from contextfs.types.base import S

if TYPE_CHECKING:
    from contextfs.schemas import Memory, MemoryType
    from contextfs.types.versioned import VersionedMem


# =============================================================================
# Mem[S] - Schema-Indexed Memory Type
# =============================================================================


class Mem(Generic[S]):
    """
    Schema-indexed memory wrapper providing type-safe access to structured_data.

    Corresponds to: MemoryType ::= Mem Schema

    Mem[S] wraps a Memory instance and provides:
    - Type-safe access to structured_data via the `data` property
    - Schema validation ensuring structured_data matches type S
    - Full compatibility with Memory (delegation pattern)
    - Generic type parameter for static type checking

    The wrapped Memory is accessible via the `memory` property for
    operations requiring the underlying Memory object.

    Type Parameters:
        S: The schema type for structured_data, must be BaseSchema subclass.

    Attributes:
        memory: The underlying Memory instance.
        data: Type-safe access to structured_data as schema type S.

    Example:
        >>> from contextfs.types import Mem
        >>> from contextfs.schemas import DecisionData, Memory
        >>>
        >>> # Create via factory
        >>> mem: Mem[DecisionData] = Mem.create(
        ...     content="Database choice",
        ...     schema=DecisionData(decision="PostgreSQL")
        ... )
        >>> mem.data.decision  # Type-safe: str
        'PostgreSQL'
        >>>
        >>> # Wrap existing memory
        >>> memory = Memory(content="...", structured_data={"decision": "SQLite"})
        >>> typed = Mem.wrap(memory, DecisionData)
        >>> typed.data.decision
        'SQLite'
    """

    __slots__ = ("_memory", "_schema_type", "_cached_data")

    def __init__(
        self,
        memory: Memory,
        schema_type: type[S],
    ) -> None:
        """
        Create a Mem wrapper around an existing Memory.

        Args:
            memory: The Memory instance to wrap.
            schema_type: The schema type for type-safe access.

        Note:
            Prefer using Mem.create() or Mem.wrap() factory methods
            for clearer intent.
        """
        self._memory = memory
        self._schema_type = schema_type
        self._cached_data: S | None = None

    @classmethod
    def create(
        cls,
        content: str,
        schema: S,
        memory_type: MemoryType | None = None,
        **kwargs: Any,
    ) -> Mem[S]:
        """
        Create a new Mem with schema data.

        This is the primary factory method for creating typed memories.

        Args:
            content: The memory content string.
            schema: The typed schema data instance.
            memory_type: Optional MemoryType enum value.
            **kwargs: Additional Memory fields (tags, summary, etc.)

        Returns:
            New Mem[S] instance with the schema data.

        Example:
            >>> mem = Mem.create(
            ...     content="API design decision",
            ...     schema=DecisionData(
            ...         decision="REST over GraphQL",
            ...         rationale="Team familiarity"
            ...     ),
            ...     tags=["architecture", "api"]
            ... )
        """
        from contextfs.schemas import Memory
        from contextfs.schemas import MemoryType as MT

        # Convert schema to structured_data dict
        if isinstance(schema, BaseModel):
            structured_data = schema.model_dump(mode="json")
        else:
            structured_data = dict(schema) if hasattr(schema, "__iter__") else {}

        # Determine memory type from schema if not provided
        if memory_type is None:
            schema_name = getattr(schema, "_schema_name", None)
            if schema_name:
                try:
                    memory_type = MT(schema_name)
                except ValueError:
                    memory_type = MT.FACT

        memory = Memory(
            content=content,
            type=memory_type or MT.FACT,
            structured_data=structured_data,
            **kwargs,
        )

        return cls(memory, type(schema))

    @classmethod
    def wrap(
        cls,
        memory: Memory,
        schema_type: type[S],
        validate: bool = True,
    ) -> Mem[S]:
        """
        Wrap an existing Memory with typed access.

        Args:
            memory: The Memory to wrap.
            schema_type: The expected schema type.
            validate: Whether to validate structured_data matches schema.

        Returns:
            Mem[S] wrapper around the memory.

        Raises:
            ValueError: If validate=True and structured_data doesn't match schema.

        Example:
            >>> memory = ctx.recall("abc123")
            >>> typed = Mem.wrap(memory, DecisionData)
            >>> typed.data.decision
            'PostgreSQL'
        """
        if validate and memory.structured_data:
            # Validate by attempting to parse
            try:
                schema_type.model_validate(memory.structured_data)
            except Exception as e:
                raise ValueError(
                    f"structured_data does not match schema {schema_type.__name__}: {e}"
                ) from e

        return cls(memory, schema_type)

    # =========================================================================
    # Core Properties
    # =========================================================================

    @property
    def memory(self) -> Memory:
        """Get the underlying Memory instance."""
        return self._memory

    @property
    def schema_type(self) -> type[S]:
        """Get the schema type."""
        return self._schema_type

    @property
    def data(self) -> S:
        """
        Get structured_data as the typed schema.

        This property provides type-safe access to the memory's
        structured_data field, validated against schema type S.

        Returns:
            The structured_data parsed as schema type S.

        Raises:
            ValueError: If structured_data is None or invalid.
        """
        if self._cached_data is not None:
            return self._cached_data

        if self._memory.structured_data is None:
            raise ValueError("Memory has no structured_data")

        self._cached_data = self._schema_type.model_validate(self._memory.structured_data)
        return self._cached_data

    @property
    def data_or_none(self) -> S | None:
        """Get structured_data as typed schema, or None if not available."""
        try:
            return self.data
        except (ValueError, Exception):
            return None

    # =========================================================================
    # Delegated Properties (from Memory)
    # =========================================================================

    @property
    def id(self) -> str:
        """Memory ID."""
        return self._memory.id

    @property
    def content(self) -> str:
        """Memory content."""
        return self._memory.content

    @property
    def type(self) -> MemoryType:
        """Memory type."""
        return self._memory.type

    @property
    def tags(self) -> list[str]:
        """Memory tags."""
        return self._memory.tags

    @property
    def summary(self) -> str | None:
        """Memory summary."""
        return self._memory.summary

    @property
    def structured_data(self) -> dict[str, Any] | None:
        """Raw structured_data dict (prefer .data for typed access)."""
        return self._memory.structured_data

    # =========================================================================
    # Versioning Support
    # =========================================================================

    def as_versioned(self) -> VersionedMem[S]:
        """
        Get a versioned view of this memory.

        Returns:
            VersionedMem[S] wrapping this memory with timeline support.
        """
        from contextfs.types.versioned import VersionedMem

        return VersionedMem.from_memory(self._memory, self._schema_type)

    # =========================================================================
    # Mutation Methods
    # =========================================================================

    def with_data(self, new_data: S) -> Mem[S]:
        """
        Create a new Mem with updated schema data.

        This creates a copy with new structured_data while preserving
        other memory fields. The original Mem is not modified.

        Args:
            new_data: New schema data.

        Returns:
            New Mem[S] with updated data.
        """
        from contextfs.schemas import Memory

        new_structured = (
            new_data.model_dump(mode="json") if isinstance(new_data, BaseModel) else dict(new_data)
        )

        new_memory = Memory(
            id=self._memory.id,
            content=self._memory.content,
            type=self._memory.type,
            tags=self._memory.tags.copy(),
            summary=self._memory.summary,
            structured_data=new_structured,
            namespace_id=self._memory.namespace_id,
            source_file=self._memory.source_file,
            source_repo=self._memory.source_repo,
            source_tool=self._memory.source_tool,
            project=self._memory.project,
            session_id=self._memory.session_id,
            metadata=self._memory.metadata.copy(),
            authoritative=self._memory.authoritative,
        )

        return Mem(new_memory, self._schema_type)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary including schema type info."""
        return {
            **self._memory.model_dump(mode="json"),
            "_schema_type": self._schema_type.__name__,
        }

    def __repr__(self) -> str:
        return f"Mem[{self._schema_type.__name__}](id={self.id!r})"

    def __str__(self) -> str:
        return f"Mem[{self._schema_type.__name__}]: {self.content[:50]}..."


# =============================================================================
# Type Factory for Runtime Type Creation
# =============================================================================


def mem_type(schema_type: type[S]) -> type[Mem[S]]:
    """
    Create a concrete Mem type bound to a specific schema.

    This is useful when you need a type object for isinstance checks
    or for creating typed collections.

    Args:
        schema_type: The schema type to bind.

    Returns:
        A Mem class bound to the schema type.

    Example:
        >>> DecisionMem = mem_type(DecisionData)
        >>> mem = DecisionMem.create(content="...", schema=DecisionData(...))
    """

    class BoundMem(Mem[S]):
        _bound_schema_type = schema_type

        def __init__(self, memory: Memory) -> None:
            super().__init__(memory, schema_type)

        @classmethod
        def create(cls, content: str, schema: S, **kwargs: Any) -> BoundMem:
            base = Mem.create(content, schema, **kwargs)
            return cls(base.memory)

    BoundMem.__name__ = f"Mem[{schema_type.__name__}]"
    BoundMem.__qualname__ = f"Mem[{schema_type.__name__}]"

    return BoundMem
