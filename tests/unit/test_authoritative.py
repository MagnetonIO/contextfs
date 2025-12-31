"""Unit tests for Phase 3: Authoritative flag and lineage-aware queries."""

from contextfs.schemas import Memory, MemoryType


class TestAuthoritativeField:
    """Test authoritative field on Memory model."""

    def test_memory_default_not_authoritative(self):
        """Test that memories are not authoritative by default."""
        memory = Memory(content="Test memory", type=MemoryType.FACT)
        assert memory.authoritative is False

    def test_memory_can_be_authoritative(self):
        """Test creating an authoritative memory."""
        memory = Memory(
            content="Authoritative memory",
            type=MemoryType.DECISION,
            authoritative=True,
        )
        assert memory.authoritative is True

    def test_authoritative_with_structured_data(self):
        """Test authoritative memory with structured data."""
        memory = Memory(
            content="Database decision",
            type=MemoryType.DECISION,
            structured_data={"decision": "Use PostgreSQL"},
            authoritative=True,
        )
        assert memory.authoritative is True
        assert memory.structured_data["decision"] == "Use PostgreSQL"

    def test_factory_methods_accept_authoritative(self):
        """Test that factory methods accept authoritative kwarg."""
        memory = Memory.decision(
            content="Important decision",
            decision="Use Redis",
            authoritative=True,
        )
        assert memory.authoritative is True
        assert memory.type == MemoryType.DECISION
