"""
ContextFS Formal Type System.

Implements Definition 5.1 Type Grammar from the typed-memory paper:

    BaseType      ::= String | Int | Float | Bool | DateTime | UUID
    EntityType    ::= Entity Name Schema
    RefType       ::= Ref EntityType
    OptionType    ::= Option Type
    ListType      ::= List Type
    SetType       ::= Set Type where Ord Type
    MapType       ::= Map KeyType ValueType
    UnionType     ::= Type1 | Type2 | ... | Typen
    RecordType    ::= {f1: Type1, ..., fn: Typen}
    MemoryType    ::= Mem Schema
    VersionedType ::= Versioned MemoryType

This module provides a formal type-theoretic foundation for AI memory systems,
enabling both runtime (Pydantic) and static (mypy/pyright) type enforcement.

Key Features:
- Schema-indexed memory types: Mem[S] wraps Memory with typed structured_data
- Versioned memory with timeline: VersionedMem[S] tracks evolution history
- Typed entity references: Ref[E] provides lazy-loading typed references
- Change reason tracking: ChangeReason enum (Observation, Inference, Correction, Decay)

Example Usage:
    >>> from contextfs.types import Mem, VersionedMem, ChangeReason
    >>> from contextfs.schemas import DecisionData
    >>>
    >>> # Create schema-indexed memory
    >>> memory: Mem[DecisionData] = Mem(
    ...     content="Database selection",
    ...     schema=DecisionData(decision="PostgreSQL", rationale="ACID compliance")
    ... )
    >>> print(memory.data.decision)  # Type-safe access
    'PostgreSQL'
    >>>
    >>> # Get versioned view with timeline
    >>> versioned = memory.as_versioned()
    >>> versioned.evolve(
    ...     new_content=DecisionData(decision="SQLite", rationale="Simpler for MVP"),
    ...     reason=ChangeReason.CORRECTION
    ... )
"""

from contextfs.types.base import (
    S2,
    # Base classes
    BaseSchema,
    # Base type aliases
    Bool,
    DateTime,
    # Type variables
    E,
    Float,
    Int,
    K,
    # Helpers
    Option,
    S,
    String,
    T,
    UUIDType,
    V,
)
from contextfs.types.collections import TypedList, TypedMap, TypedSet
from contextfs.types.entity import Entity, Ref, RefList
from contextfs.types.memory import Mem, mem_type
from contextfs.types.registry import (
    SchemaRegistry,
    auto_register_schema,
    get_schema_for_memory_type,
)
from contextfs.types.versioned import (
    ChangeReason,
    Timeline,
    VersionedMem,
    VersionEntry,
)

__all__ = [
    # Base type aliases (Definition 5.1 BaseType)
    "String",
    "Int",
    "Float",
    "Bool",
    "DateTime",
    "UUIDType",
    # Type variables for Generic types
    "S",
    "S2",
    "E",
    "T",
    "K",
    "V",
    # Base classes
    "BaseSchema",
    # Helper functions
    "Option",
    # Entity and Ref types (Definition 5.1 EntityType, RefType)
    "Entity",
    "Ref",
    "RefList",
    # Schema-indexed memory (Definition 5.1 MemoryType)
    "Mem",
    "mem_type",
    # Versioned memory (Definition 5.1 VersionedType)
    "VersionedMem",
    "VersionEntry",
    "Timeline",
    "ChangeReason",
    # Collection types (Definition 5.1 ListType, SetType, MapType)
    "TypedList",
    "TypedSet",
    "TypedMap",
    # Schema registry
    "SchemaRegistry",
    "auto_register_schema",
    "get_schema_for_memory_type",
]
