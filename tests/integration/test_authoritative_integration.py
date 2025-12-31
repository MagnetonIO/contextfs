"""Integration tests for Phase 3: Authoritative flag and lineage-aware queries."""

import pytest

from contextfs.core import ContextFS
from contextfs.schemas import MemoryType


@pytest.fixture
def ctx(tmp_path):
    """Create a ContextFS instance with temporary data directory."""
    ctx = ContextFS(data_dir=tmp_path, auto_index=False)
    yield ctx
    ctx.close()


class TestAuthoritativeSaveRecall:
    """Test saving and recalling authoritative memories."""

    def test_save_authoritative_memory(self, ctx):
        """Test saving an authoritative memory."""
        memory = ctx.save(
            content="Authoritative decision",
            type=MemoryType.DECISION,
            structured_data={"decision": "Use PostgreSQL"},
            authoritative=True,
        )

        assert memory.authoritative is True

        # Verify persistence
        recalled = ctx.recall(memory.id)
        assert recalled is not None
        assert recalled.authoritative is True

    def test_save_non_authoritative_by_default(self, ctx):
        """Test that memories are not authoritative by default."""
        memory = ctx.save(
            content="Regular memory",
            type=MemoryType.FACT,
        )

        assert memory.authoritative is False

        recalled = ctx.recall(memory.id)
        assert recalled.authoritative is False

    def test_update_to_authoritative(self, ctx):
        """Test updating a memory to be authoritative."""
        memory = ctx.save(
            content="Initially not authoritative",
            type=MemoryType.DECISION,
            structured_data={"decision": "Initial"},
        )

        assert memory.authoritative is False

        # Update to authoritative
        updated = ctx.update(memory_id=memory.id, authoritative=True)
        assert updated.authoritative is True

        # Verify persistence
        recalled = ctx.recall(memory.id)
        assert recalled.authoritative is True


class TestSetAuthoritative:
    """Test set_authoritative method."""

    def test_set_authoritative_simple(self, ctx):
        """Test marking a memory as authoritative."""
        memory = ctx.save(
            content="Test memory",
            type=MemoryType.FACT,
        )

        result = ctx.set_authoritative(memory.id)
        assert result is not None
        assert result.authoritative is True

    def test_set_authoritative_nonexistent(self, ctx):
        """Test setting authoritative on nonexistent memory."""
        result = ctx.set_authoritative("nonexistent-id")
        assert result is None


class TestGetAuthoritative:
    """Test get_authoritative method."""

    def test_get_authoritative_when_exists(self, ctx):
        """Test finding authoritative memory in lineage."""
        # Create an authoritative memory
        memory = ctx.save(
            content="Authoritative version",
            type=MemoryType.DECISION,
            structured_data={"decision": "Final decision"},
            authoritative=True,
        )

        # Should find the authoritative memory
        auth = ctx.get_authoritative(memory.id)
        assert auth is not None
        assert auth.id == memory.id
        assert auth.authoritative is True

    def test_get_authoritative_when_none(self, ctx):
        """Test finding authoritative when none exists."""
        memory = ctx.save(
            content="Not authoritative",
            type=MemoryType.FACT,
        )

        # Should return None when no authoritative memory
        auth = ctx.get_authoritative(memory.id)
        assert auth is None


class TestSearchAuthoritative:
    """Test search_authoritative method."""

    def test_search_authoritative_only(self, ctx):
        """Test searching only authoritative memories."""
        # Create non-authoritative memories
        ctx.save(content="Database configuration regular", type=MemoryType.FACT)
        ctx.save(content="Database setup normal", type=MemoryType.FACT)

        # Create authoritative memory
        ctx.save(
            content="Database configuration authoritative",
            type=MemoryType.FACT,
            authoritative=True,
        )

        # Search for "database" - should only return authoritative
        results = ctx.search_authoritative("database configuration")

        # All results should be authoritative
        for r in results:
            assert r.memory.authoritative is True

    def test_search_authoritative_empty_results(self, ctx):
        """Test searching when no authoritative memories match."""
        ctx.save(content="Regular memory", type=MemoryType.FACT)

        results = ctx.search_authoritative("something completely different")
        assert len(results) == 0


class TestAuthoritativeWithLineage:
    """Test authoritative flag with memory lineage."""

    def test_evolve_preserves_authoritative(self, ctx):
        """Test that evolving doesn't automatically set authoritative."""
        original = ctx.save(
            content="Original decision",
            type=MemoryType.DECISION,
            structured_data={"decision": "v1"},
            authoritative=True,
        )

        # Evolve the memory
        evolved = ctx.evolve(
            memory_id=original.id,
            new_content="Evolved decision",
        )

        # Evolved memory should NOT be authoritative (original still is)
        assert evolved is not None
        recalled_evolved = ctx.recall(evolved.id)
        assert recalled_evolved.authoritative is False

        # Original should still be authoritative
        recalled_original = ctx.recall(original.id)
        assert recalled_original.authoritative is True

    def test_set_authoritative_exclusive(self, ctx):
        """Test that set_authoritative unmarks others in lineage."""
        original = ctx.save(
            content="Original",
            type=MemoryType.DECISION,
            structured_data={"decision": "v1"},
            authoritative=True,
        )

        # Evolve the memory
        evolved = ctx.evolve(
            memory_id=original.id,
            new_content="Evolved version",
        )

        # Set evolved as authoritative (exclusive=True by default)
        ctx.set_authoritative(evolved.id)

        # Evolved should now be authoritative
        recalled_evolved = ctx.recall(evolved.id)
        assert recalled_evolved.authoritative is True

        # Original should no longer be authoritative
        recalled_original = ctx.recall(original.id)
        assert recalled_original.authoritative is False
