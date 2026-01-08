# Formal Type System

ContextFS implements a formal type-theoretic memory system based on **Definition 5.1 Type Grammar** from the [typed-memory paper](../paper/typed-memory-paper.pdf).

## Overview

The type system provides both **runtime** (Pydantic) and **static** (mypy/pyright) type enforcement for AI memory systems. It enables type-safe access to structured data while maintaining full backward compatibility with existing code.

## Type Grammar (Definition 5.1)

```
BaseType      ::= String | Int | Float | Bool | DateTime | UUID
EntityType    ::= Entity Name Schema
RefType       ::= Ref EntityType
OptionType    ::= Option Type          (T | None)
ListType      ::= List Type            (list[T])
SetType       ::= Set Type             (frozenset[T])
MapType       ::= Map KeyType Value    (dict[K, V])
UnionType     ::= T1 | T2 | ... | Tn
RecordType    ::= {f1: T1, ..., fn: Tn} (Pydantic BaseModel)
MemoryType    ::= Mem Schema           (Generic[S])
VersionedType ::= Versioned MemoryType (with Timeline/ChangeReason)
```

## Core Types

### Mem[S] - Schema-Indexed Memory

The `Mem[S]` type wraps a Memory instance and provides type-safe access to `structured_data`:

```python
from contextfs.schemas import Memory, DecisionData
from contextfs.types import Mem

# Create a decision memory
memory = Memory.decision("Database choice", decision="PostgreSQL", rationale="ACID compliance")

# Convert to typed access
typed: Mem[DecisionData] = memory.as_typed(DecisionData)

# Type-safe property access
print(typed.data.decision)   # "PostgreSQL" - IDE knows this is str
print(typed.data.rationale)  # "ACID compliance" - also type-safe
```

#### Factory Methods

```python
# Create directly with schema
mem = Mem.create(
    content="API design decision",
    schema=DecisionData(
        decision="REST over GraphQL",
        rationale="Team familiarity"
    ),
    tags=["architecture", "api"]
)

# Wrap existing memory
typed = Mem.wrap(memory, DecisionData)
```

### VersionedMem[S] - Timeline-Based Versioning

Track memory evolution with formal change reasons:

```python
from contextfs.types import VersionedMem, ChangeReason

# Get versioned view
versioned: VersionedMem[DecisionData] = memory.as_versioned(DecisionData)

# Evolve with reason tracking
versioned.evolve(
    new_content=DecisionData(decision="SQLite", rationale="Simpler for MVP"),
    reason=ChangeReason.CORRECTION,
    author="claude"
)

# Query timeline
print(f"Version count: {len(versioned.timeline)}")
print(f"Original: {versioned.timeline.root.content.decision}")  # "PostgreSQL"
print(f"Current: {versioned.timeline.current.content.decision}")  # "SQLite"
```

### ChangeReason Enum

From Section 4 of the typed-memory paper, memory evolution has four formal reasons:

| Reason | Description | Example |
|--------|-------------|---------|
| `OBSERVATION` | New external information | User provided new requirements |
| `INFERENCE` | Derived from existing knowledge | Concluded from code analysis |
| `CORRECTION` | Fixing an identified error | Bug fix, wrong assumption |
| `DECAY` | Knowledge becoming stale | Outdated documentation |

```python
from contextfs.types import ChangeReason

# Observation - new information
versioned.evolve(data, reason=ChangeReason.OBSERVATION)

# Inference - derived knowledge
versioned.infer(data, premises=["mem123", "mem456"])

# Correction - fixing errors
versioned.correct(data, correction_note="Previous assumption was wrong")
```

### Entity[S] and Ref[E] - Typed References

For structured entities with lazy-loading references:

```python
from contextfs.types import Entity, Ref, BaseSchema

class UserSchema(BaseSchema):
    _schema_name = "user"
    username: str
    email: str

# Create entity
user = Entity[UserSchema](
    name="John Doe",
    schema_data=UserSchema(username="johnd", email="john@example.com")
)

# Create reference (lazy-loading)
user_ref: Ref[Entity[UserSchema]] = Ref(user.id, resolver=load_user)

# Resolve when needed
resolved = user_ref.resolve()  # Loads and caches
print(resolved.schema_data.username)
```

### Collection Types

Type-safe wrappers for Python collections:

```python
from contextfs.types import TypedList, TypedSet, TypedMap

# TypedList[T]
tags = TypedList[str].of("python", "memory", "ai")
tags.append("contextfs")

# TypedSet[T] - immutable
unique_tags = TypedSet[str].of("a", "b", "a")  # len = 2

# TypedMap[K, V]
scores = TypedMap[str, int].of({"alice": 100, "bob": 85})
scores["charlie"] = 90
```

## Schema Registry

Runtime schema resolution and discovery:

```python
from contextfs.types import SchemaRegistry, auto_register_schema

# Get built-in schemas
DecisionSchema = SchemaRegistry.get("decision")

# Register custom schema
@auto_register_schema
class CustomSchema(BaseSchema):
    _schema_name = "custom"
    field: str

# Resolve data to schema
data = {"type": "decision", "decision": "PostgreSQL"}
schema = SchemaRegistry.resolve(data)
```

## Integration with Memory Lineage

The `evolve()` operation in `MemoryLineage` now accepts a `ChangeReason`:

```python
from contextfs.memory_lineage import MemoryLineage
from contextfs.types import ChangeReason

lineage = MemoryLineage(storage)

# Evolve with reason
evolved = lineage.evolve(
    "memory_id",
    new_content="Updated content",
    reason=ChangeReason.CORRECTION
)

# Reason stored in metadata
print(evolved.metadata["change_reason"])  # "correction"
```

## Timeline Operations

Query version history:

```python
from datetime import datetime, timezone

versioned = memory.as_versioned(DecisionData)

# Get version at specific time
past_version = versioned.timeline.at(datetime(2024, 6, 1, tzinfo=timezone.utc))

# Get all versions since timestamp
recent = versioned.timeline.since(datetime(2024, 1, 1, tzinfo=timezone.utc))

# Filter by change reason
corrections = versioned.timeline.by_reason(ChangeReason.CORRECTION)
```

## Type Variables

For generic type parameters:

```python
from contextfs.types import S, E, T, K, V

# S - Schema types (bound to BaseSchema)
# E - Entity types (bound to BaseModel)
# T - General type parameter
# K - Key type for maps
# V - Value type for maps
```

## Module Structure

```
src/contextfs/types/
├── __init__.py      # Public exports
├── base.py          # BaseType aliases, TypeVars, BaseSchema
├── entity.py        # Entity[S], Ref[E], RefList[E]
├── memory.py        # Mem[S], mem_type()
├── versioned.py     # VersionedMem[S], Timeline, ChangeReason
├── collections.py   # TypedList, TypedSet, TypedMap
└── registry.py      # SchemaRegistry
```

## Further Reading

- [Typed Memory Paper (PDF)](../paper/typed-memory-paper.pdf) - Formal foundations
- [Theoretical Foundations](theory.md) - Type theory background
- [Type-Safe Context Engineering](typesafe-context.md) - Application principles
- [Memory Types](../architecture/memory-types.md) - Memory categorization
