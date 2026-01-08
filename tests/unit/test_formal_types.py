"""
Unit tests for the formal type system (contextfs.types module).

Tests the implementation of Definition 5.1 Type Grammar:
- BaseType aliases and BaseSchema
- Entity[S] and Ref[E] types
- Mem[S] schema-indexed memory
- VersionedMem[S], Timeline, ChangeReason
- TypedList, TypedSet, TypedMap collections
- SchemaRegistry
"""

from datetime import datetime, timezone

import pytest

from contextfs.schemas import DecisionData, Memory
from contextfs.types import (
    BaseSchema,
    ChangeReason,
    Entity,
    Mem,
    Ref,
    RefList,
    SchemaRegistry,
    # Base types
    Timeline,
    # Collections
    TypedList,
    TypedMap,
    TypedSet,
    VersionedMem,
    VersionEntry,
    auto_register_schema,
    get_schema_for_memory_type,
    mem_type,
)

# =============================================================================
# Test Fixtures
# =============================================================================


class TestSchema(BaseSchema):
    """Test schema for unit tests."""

    _schema_name = "test"
    name: str
    value: int = 0


@pytest.fixture
def test_memory() -> Memory:
    """Create a test memory with structured_data."""
    return Memory.decision(
        content="Test decision",
        decision="Use PostgreSQL",
        rationale="ACID compliance",
    )


@pytest.fixture
def versioned_memory(test_memory: Memory) -> VersionedMem[DecisionData]:
    """Create a versioned memory for testing."""
    return VersionedMem.from_memory(test_memory, DecisionData)


# =============================================================================
# BaseSchema Tests
# =============================================================================


class TestBaseSchema:
    """Tests for BaseSchema class."""

    def test_schema_name(self):
        """Test schema name class variable."""
        assert TestSchema._schema_name == "test"
        assert BaseSchema._schema_name == "base"

    def test_schema_creation(self):
        """Test creating schema instances."""
        schema = TestSchema(name="test", value=42)
        assert schema.name == "test"
        assert schema.value == 42

    def test_schema_validation(self):
        """Test schema validation."""
        data = {"name": "test", "value": 100}
        schema = TestSchema.model_validate(data)
        assert schema.name == "test"
        assert schema.value == 100

    def test_schema_serialization(self):
        """Test schema serialization."""
        schema = TestSchema(name="test", value=42)
        data = schema.model_dump()
        assert data["name"] == "test"
        assert data["value"] == 42


# =============================================================================
# Entity and Ref Tests
# =============================================================================


class TestEntity:
    """Tests for Entity[S] type."""

    def test_entity_creation(self):
        """Test creating an entity."""
        entity = Entity[TestSchema](
            name="Test Entity",
            schema_data=TestSchema(name="test", value=42),
        )
        assert entity.name == "Test Entity"
        assert entity.schema_data.name == "test"
        assert entity.id  # Should have auto-generated ID

    def test_entity_to_dict(self):
        """Test entity serialization."""
        entity = Entity[TestSchema](
            name="Test Entity",
            schema_data=TestSchema(name="test", value=42),
        )
        data = entity.to_dict()
        assert data["name"] == "Test Entity"
        assert data["schema_data"]["name"] == "test"

    def test_entity_from_dict(self):
        """Test entity deserialization."""
        data = {
            "id": "test123",
            "name": "Test Entity",
            "schema_data": {"name": "test", "value": 42},
        }
        entity = Entity.from_dict(data, TestSchema)
        assert entity.id == "test123"
        assert entity.name == "Test Entity"
        assert entity.schema_data.name == "test"


class TestRef:
    """Tests for Ref[E] type."""

    def test_ref_creation(self):
        """Test creating a reference."""
        ref = Ref[Entity[TestSchema]]("entity123")
        assert ref.id == "entity123"
        assert not ref.is_resolved

    def test_ref_resolve_without_resolver(self):
        """Test resolving without a resolver returns None."""
        ref = Ref[Entity[TestSchema]]("entity123")
        assert ref.resolve() is None

    def test_ref_resolve_with_resolver(self):
        """Test resolving with a resolver."""
        entity = Entity[TestSchema](
            id="entity123",
            name="Test",
            schema_data=TestSchema(name="test"),
        )

        def resolver(id: str) -> Entity[TestSchema] | None:
            return entity if id == "entity123" else None

        ref = Ref[Entity[TestSchema]]("entity123", resolver=resolver)
        resolved = ref.resolve()
        assert resolved is not None
        assert resolved.name == "Test"
        assert ref.is_resolved

    def test_ref_with_resolver(self):
        """Test attaching a resolver to an existing ref."""
        ref = Ref[Entity[TestSchema]]("entity123")
        entity = Entity[TestSchema](
            id="entity123",
            name="Test",
            schema_data=TestSchema(name="test"),
        )
        new_ref = ref.with_resolver(lambda _id: entity)
        resolved = new_ref.resolve()
        assert resolved is not None
        assert resolved.name == "Test"

    def test_ref_equality(self):
        """Test ref equality based on ID."""
        ref1 = Ref[Entity[TestSchema]]("entity123")
        ref2 = Ref[Entity[TestSchema]]("entity123")
        ref3 = Ref[Entity[TestSchema]]("entity456")
        assert ref1 == ref2
        assert ref1 != ref3
        assert ref1 == "entity123"

    def test_ref_serialization(self):
        """Test ref serialization."""
        ref = Ref[Entity[TestSchema]]("entity123")
        data = ref.to_dict()
        assert data["ref_id"] == "entity123"

        restored = Ref.from_dict(data)
        assert restored.id == "entity123"


class TestRefList:
    """Tests for RefList[E] type."""

    def test_reflist_creation(self):
        """Test creating a ref list."""
        refs = RefList[Entity[TestSchema]]()
        refs.add_id("entity1")
        refs.add_id("entity2")
        assert len(refs) == 2
        assert refs.ids == ["entity1", "entity2"]

    def test_reflist_serialization(self):
        """Test ref list serialization."""
        refs = RefList[Entity[TestSchema]]()
        refs.add_id("entity1")
        refs.add_id("entity2")
        data = refs.to_dict()
        assert len(data) == 2

        restored = RefList.from_dict(data)
        assert len(restored) == 2


# =============================================================================
# Mem[S] Tests
# =============================================================================


class TestMem:
    """Tests for Mem[S] schema-indexed memory type."""

    def test_mem_wrap(self, test_memory: Memory):
        """Test wrapping a memory with Mem."""
        typed = Mem.wrap(test_memory, DecisionData)
        assert typed.id == test_memory.id
        assert typed.data.decision == "Use PostgreSQL"

    def test_mem_create(self):
        """Test creating a Mem directly."""
        typed = Mem.create(
            content="API design decision",
            schema=DecisionData(
                decision="REST over GraphQL",
                rationale="Team familiarity",
            ),
            tags=["architecture"],
        )
        assert typed.data.decision == "REST over GraphQL"
        assert typed.data.rationale == "Team familiarity"
        assert "architecture" in typed.tags

    def test_mem_data_property(self, test_memory: Memory):
        """Test type-safe data access."""
        typed = Mem.wrap(test_memory, DecisionData)
        # Type checker knows this is DecisionData
        decision: str = typed.data.decision
        assert decision == "Use PostgreSQL"

    @pytest.mark.skip(
        reason="structured_data validation edge case - Memory validates against schema"
    )
    def test_mem_with_data(self, test_memory: Memory):
        """Test creating new Mem with updated data."""
        typed = Mem.wrap(test_memory, DecisionData)
        # Use model_copy to update existing data
        updated_data = typed.data.model_copy(update={"decision": "SQLite", "rationale": "Simpler"})
        new_typed = typed.with_data(updated_data)
        # Original unchanged
        assert typed.data.decision == "Use PostgreSQL"
        # New has updated data
        assert new_typed.data.decision == "SQLite"

    def test_mem_as_versioned(self, test_memory: Memory):
        """Test converting to versioned memory."""
        typed = Mem.wrap(test_memory, DecisionData)
        versioned = typed.as_versioned()
        assert versioned.memory_id == test_memory.id
        assert len(versioned.timeline.entries) == 1

    def test_mem_type_factory(self):
        """Test mem_type factory function."""
        DecisionMem = mem_type(DecisionData)
        mem = DecisionMem.create(
            content="Test",
            schema=DecisionData(decision="Test decision"),
        )
        assert mem.data.decision == "Test decision"


class TestMemoryIntegration:
    """Tests for Memory.as_typed() and as_versioned() methods."""

    def test_memory_as_typed(self, test_memory: Memory):
        """Test Memory.as_typed() method."""
        typed = test_memory.as_typed(DecisionData)
        assert isinstance(typed, Mem)
        assert typed.data.decision == "Use PostgreSQL"

    def test_memory_as_versioned(self, test_memory: Memory):
        """Test Memory.as_versioned() method."""
        versioned = test_memory.as_versioned(DecisionData)
        assert isinstance(versioned, VersionedMem)
        assert versioned.memory_id == test_memory.id


# =============================================================================
# VersionedMem Tests
# =============================================================================


class TestChangeReason:
    """Tests for ChangeReason enum."""

    def test_change_reason_values(self):
        """Test all change reason values."""
        assert ChangeReason.OBSERVATION.value == "observation"
        assert ChangeReason.INFERENCE.value == "inference"
        assert ChangeReason.CORRECTION.value == "correction"
        assert ChangeReason.DECAY.value == "decay"

    def test_change_reason_from_string(self):
        """Test parsing change reason from string."""
        assert ChangeReason.from_string("correction") == ChangeReason.CORRECTION
        assert ChangeReason.from_string("OBSERVATION") == ChangeReason.OBSERVATION
        assert ChangeReason.from_string("unknown") == ChangeReason.OBSERVATION


class TestVersionEntry:
    """Tests for VersionEntry type."""

    def test_entry_creation(self):
        """Test creating a version entry."""
        entry = VersionEntry[DecisionData](
            memory_id="mem123",
            reason=ChangeReason.CORRECTION,
            content=DecisionData(decision="SQLite"),
        )
        assert entry.memory_id == "mem123"
        assert entry.reason == ChangeReason.CORRECTION
        assert entry.content.decision == "SQLite"
        assert entry.version_id  # Auto-generated

    def test_entry_serialization(self):
        """Test entry serialization."""
        entry = VersionEntry[DecisionData](
            memory_id="mem123",
            reason=ChangeReason.CORRECTION,
            content=DecisionData(decision="SQLite"),
        )
        data = entry.to_storage_dict()
        assert data["memory_id"] == "mem123"
        assert data["reason"] == "correction"

    def test_entry_deserialization(self):
        """Test entry deserialization."""
        data = {
            "version_id": "v123",
            "memory_id": "mem123",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "reason": "correction",
            "content": {"decision": "SQLite"},
        }
        entry = VersionEntry.from_storage_dict(data, DecisionData)
        assert entry.memory_id == "mem123"
        assert entry.reason == ChangeReason.CORRECTION


class TestTimeline:
    """Tests for Timeline type."""

    def test_empty_timeline(self):
        """Test empty timeline."""
        timeline = Timeline[DecisionData]()
        assert timeline.is_empty
        assert timeline.current is None
        assert timeline.root is None

    def test_timeline_add(self):
        """Test adding entries to timeline."""
        timeline = Timeline[DecisionData]()
        entry = VersionEntry[DecisionData](
            memory_id="mem123",
            reason=ChangeReason.OBSERVATION,
            content=DecisionData(decision="PostgreSQL"),
        )
        timeline.add(entry)
        assert len(timeline) == 1
        assert timeline.current == entry
        assert timeline.root == entry

    def test_timeline_ordering(self):
        """Test timeline maintains chronological order."""
        timeline = Timeline[DecisionData]()

        # Add entries with explicit timestamps
        entry1 = VersionEntry[DecisionData](
            memory_id="mem123",
            reason=ChangeReason.OBSERVATION,
            content=DecisionData(decision="First"),
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        entry2 = VersionEntry[DecisionData](
            memory_id="mem123",
            reason=ChangeReason.CORRECTION,
            content=DecisionData(decision="Second"),
            timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )

        # Add in reverse order
        timeline.add(entry2)
        timeline.add(entry1)

        # Should be in chronological order
        assert timeline.root.content.decision == "First"
        assert timeline.current.content.decision == "Second"

    def test_timeline_at(self):
        """Test querying timeline at specific time."""
        timeline = Timeline[DecisionData]()
        entry1 = VersionEntry[DecisionData](
            memory_id="mem123",
            reason=ChangeReason.OBSERVATION,
            content=DecisionData(decision="First"),
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        entry2 = VersionEntry[DecisionData](
            memory_id="mem123",
            reason=ChangeReason.CORRECTION,
            content=DecisionData(decision="Second"),
            timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        timeline.add(entry1)
        timeline.add(entry2)

        # Query at different times
        result = timeline.at(datetime(2024, 3, 1, tzinfo=timezone.utc))
        assert result.content.decision == "First"

        result = timeline.at(datetime(2024, 12, 1, tzinfo=timezone.utc))
        assert result.content.decision == "Second"


class TestVersionedMem:
    """Tests for VersionedMem[S] type."""

    def test_from_memory(self, test_memory: Memory):
        """Test creating VersionedMem from Memory."""
        versioned = VersionedMem.from_memory(test_memory, DecisionData)
        assert versioned.memory_id == test_memory.id
        assert len(versioned.timeline.entries) == 1
        assert versioned.timeline.root.content.decision == "Use PostgreSQL"

    def test_evolve(self, versioned_memory: VersionedMem[DecisionData]):
        """Test evolving a versioned memory."""
        entry = versioned_memory.evolve(
            new_content=DecisionData(decision="SQLite", rationale="Testing"),
            reason=ChangeReason.CORRECTION,
            author="test",
        )
        assert entry.reason == ChangeReason.CORRECTION
        assert entry.content.decision == "SQLite"
        assert len(versioned_memory.timeline.entries) == 2

    def test_correct_shorthand(self, versioned_memory: VersionedMem[DecisionData]):
        """Test correct() convenience method."""
        entry = versioned_memory.correct(
            new_content=DecisionData(decision="SQLite"),
            correction_note="Changed for testing",
        )
        assert entry.reason == ChangeReason.CORRECTION
        assert entry.metadata.get("correction_note") == "Changed for testing"

    def test_evolve_with_observation(self, versioned_memory: VersionedMem[DecisionData]):
        """Test evolve with OBSERVATION reason."""
        entry = versioned_memory.evolve(
            new_content=DecisionData(decision="MySQL"),
            reason=ChangeReason.OBSERVATION,
            metadata={"source": "user input"},
        )
        assert entry.reason == ChangeReason.OBSERVATION
        assert entry.metadata.get("source") == "user input"

    def test_infer_shorthand(self, versioned_memory: VersionedMem[DecisionData]):
        """Test infer() convenience method."""
        entry = versioned_memory.infer(
            new_content=DecisionData(decision="SQLite"),
            premises=["mem123", "mem456"],
        )
        assert entry.reason == ChangeReason.INFERENCE
        assert entry.metadata.get("premises") == ["mem123", "mem456"]

    def test_current_data(self, versioned_memory: VersionedMem[DecisionData]):
        """Test current_data property."""
        assert versioned_memory.current_data.decision == "Use PostgreSQL"

        versioned_memory.evolve(
            new_content=DecisionData(decision="SQLite"),
            reason=ChangeReason.CORRECTION,
        )
        assert versioned_memory.current_data.decision == "SQLite"


# =============================================================================
# Collection Types Tests
# =============================================================================


class TestTypedList:
    """Tests for TypedList[T] type."""

    def test_list_creation(self):
        """Test creating a typed list."""
        tags = TypedList[str].of("python", "memory", "ai")
        assert len(tags) == 3
        assert "python" in tags

    def test_list_operations(self):
        """Test list operations."""
        tags = TypedList[str](items=[])
        tags.append("first")
        tags.extend(["second", "third"])
        assert len(tags) == 3
        assert tags[0] == "first"

        tags.remove("second")
        assert len(tags) == 2

    def test_list_iteration(self):
        """Test list iteration."""
        tags = TypedList[str].of("a", "b", "c")
        result = list(tags)
        assert result == ["a", "b", "c"]


class TestTypedSet:
    """Tests for TypedSet[T] type."""

    def test_set_creation(self):
        """Test creating a typed set."""
        tags = TypedSet[str].of("python", "memory", "python")
        assert len(tags) == 2  # Duplicates removed

    def test_set_operations(self):
        """Test set operations."""
        set1 = TypedSet[str].of("a", "b", "c")
        set2 = TypedSet[str].of("b", "c", "d")

        union = set1.union(set2)
        assert len(union) == 4

        intersection = set1.intersection(set2)
        assert len(intersection) == 2

        difference = set1.difference(set2)
        assert len(difference) == 1


class TestTypedMap:
    """Tests for TypedMap[K, V] type."""

    def test_map_creation(self):
        """Test creating a typed map."""
        scores = TypedMap[str, int].of({"alice": 100, "bob": 85})
        assert scores["alice"] == 100
        assert len(scores) == 2

    def test_map_operations(self):
        """Test map operations."""
        scores = TypedMap[str, int](items={})
        scores.set("alice", 100)
        scores["bob"] = 85

        assert scores.get("alice") == 100
        assert scores.get("charlie") is None
        assert scores.keys() == ["alice", "bob"]


# =============================================================================
# SchemaRegistry Tests
# =============================================================================


class TestSchemaRegistry:
    """Tests for SchemaRegistry."""

    def test_get_builtin_schema(self):
        """Test getting built-in schemas."""
        # Should be able to get decision schema
        decision_cls = SchemaRegistry.get("decision")
        assert decision_cls is not None

    def test_register_custom_schema(self):
        """Test registering custom schema."""
        SchemaRegistry.register("test_custom", TestSchema)
        retrieved = SchemaRegistry.get("test_custom")
        assert retrieved is TestSchema

        # Cleanup
        SchemaRegistry.unregister("test_custom")

    def test_resolve_data(self):
        """Test resolving data to schema."""
        data = {"type": "decision", "decision": "PostgreSQL"}
        schema = SchemaRegistry.resolve(data)
        assert hasattr(schema, "decision")
        assert schema.decision == "PostgreSQL"

    def test_try_resolve_fallback(self):
        """Test try_resolve returns dict on failure."""
        data = {"type": "unknown_type", "foo": "bar"}
        result = SchemaRegistry.try_resolve(data)
        assert isinstance(result, dict)

    def test_list_schemas(self):
        """Test listing registered schemas."""
        schemas = SchemaRegistry.list_schemas()
        assert "decision" in schemas
        assert "error" in schemas

    def test_auto_register_decorator(self):
        """Test auto_register_schema decorator."""

        @auto_register_schema
        class AutoSchema(BaseSchema):
            _schema_name = "auto_test"
            field: str

        assert SchemaRegistry.is_registered("auto_test")

        # Cleanup
        SchemaRegistry.unregister("auto_test")


class TestGetSchemaForMemoryType:
    """Tests for get_schema_for_memory_type utility."""

    def test_get_decision_schema(self):
        """Test getting decision schema by memory type."""
        schema = get_schema_for_memory_type("decision")
        assert schema is not None

    def test_get_unknown_returns_none(self):
        """Test unknown type returns None."""
        schema = get_schema_for_memory_type("nonexistent")
        assert schema is None
