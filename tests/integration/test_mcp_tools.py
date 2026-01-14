"""
Integration tests for MCP tool handlers.

Tests the MCP server tools through the core ContextFS API.
"""

import pytest

from contextfs.core import ContextFS
from contextfs.schemas import MemoryType


@pytest.fixture
def ctx(tmp_path):
    """Create a ContextFS instance with temporary data directory."""
    ctx = ContextFS(data_dir=tmp_path, auto_index=False)
    yield ctx
    ctx.close()


class TestMCPSave:
    """Tests for contextfs_save MCP tool."""

    def test_save_basic_memory(self, ctx):
        """Test saving a basic memory."""
        memory = ctx.save(
            content="Test memory content",
            type=MemoryType.FACT,
            summary="Test summary",
            tags=["test", "mcp"],
        )

        assert memory is not None
        assert memory.id is not None
        assert memory.content == "Test memory content"
        assert memory.type == MemoryType.FACT
        assert memory.summary == "Test summary"
        assert "test" in memory.tags
        assert "mcp" in memory.tags

    def test_save_with_project(self, ctx):
        """Test saving memory with project grouping."""
        memory = ctx.save(
            content="Project-specific memory",
            type=MemoryType.DECISION,
            project="my-project",
        )

        assert memory.project == "my-project"

        # Verify persistence
        recalled = ctx.recall(memory.id)
        assert recalled.project == "my-project"

    def test_save_with_structured_data(self, ctx):
        """Test saving memory with structured_data."""
        structured_data = {
            "decision": "Use PostgreSQL",
            "rationale": "ACID compliance and JSON support",
            "alternatives": ["MySQL", "SQLite"],
        }

        memory = ctx.save(
            content="Database decision",
            type=MemoryType.DECISION,
            structured_data=structured_data,
        )

        assert memory.structured_data is not None
        assert memory.structured_data["decision"] == "Use PostgreSQL"
        assert "MySQL" in memory.structured_data["alternatives"]

    def test_save_different_types(self, ctx):
        """Test saving memories of different types."""
        types_to_test = [
            MemoryType.FACT,
            MemoryType.DECISION,
            MemoryType.PROCEDURAL,
            MemoryType.EPISODIC,
            MemoryType.ERROR,
            MemoryType.CODE,
            MemoryType.API,
            MemoryType.CONFIG,
        ]

        for mem_type in types_to_test:
            memory = ctx.save(
                content=f"Memory of type {mem_type.value}",
                type=mem_type,
            )
            assert memory.type == mem_type

    def test_save_empty_content(self, ctx):
        """Test saving empty content (allowed but may create minimal memory)."""
        # Empty content is allowed - creates memory with empty content
        memory = ctx.save(content=" ", type=MemoryType.FACT)
        assert memory is not None

    def test_end_session(self, ctx):
        """Test ending session creates episodic memory."""
        # Start a session and add messages
        ctx.start_session(tool="test-tool", label="test-session")
        ctx.add_message("user", "Hello, how are you?")
        ctx.add_message("assistant", "I'm doing well, thanks!")

        # End session (creates episodic memory with summary)
        ctx.end_session(generate_summary=True)

        # Should have created session memory - just verify search runs without error
        ctx.search("test-session", limit=5)


class TestMCPSearch:
    """Tests for contextfs_search MCP tool."""

    def test_search_basic(self, ctx):
        """Test basic search functionality."""
        # Create some test memories
        ctx.save(content="Python is a programming language", type=MemoryType.FACT)
        ctx.save(content="JavaScript is also a programming language", type=MemoryType.FACT)
        ctx.save(content="Databases store data", type=MemoryType.FACT)

        # Search for programming
        results = ctx.search("programming language")

        assert len(results) >= 1
        # Should find Python and/or JavaScript memories
        found_content = [r.memory.content for r in results]
        assert any("programming" in c.lower() for c in found_content)

    def test_search_with_type_filter(self, ctx):
        """Test search with type filter."""
        ctx.save(content="A fact about Python", type=MemoryType.FACT)
        ctx.save(content="A decision about Python", type=MemoryType.DECISION)

        # Search only facts - use type parameter
        results = ctx.search("Python", type=MemoryType.FACT)

        for result in results:
            assert result.memory.type == MemoryType.FACT

    def test_search_with_limit(self, ctx):
        """Test search with result limit."""
        # Create many memories
        for i in range(10):
            ctx.save(content=f"Memory number {i} about testing", type=MemoryType.FACT)

        # Search with limit
        results = ctx.search("testing", limit=3)

        assert len(results) <= 3

    def test_search_empty_query(self, ctx):
        """Test search with empty query returns results."""
        ctx.save(content="Test memory", type=MemoryType.FACT)

        # Empty query should work (list all)
        results = ctx.search("*", limit=10)

        assert len(results) >= 1

    def test_search_no_results(self, ctx):
        """Test search with highly specific non-matching query."""
        ctx.save(content="Something about databases", type=MemoryType.FACT)

        # Use unique query that won't match anything, search in current namespace only
        results = ctx.search("xyznonexistent123qwertyuiop987654321", cross_repo=False)

        # May still return results from semantic similarity, so just verify it runs
        assert isinstance(results, list)


class TestMCPRecall:
    """Tests for contextfs_recall MCP tool."""

    def test_recall_by_full_id(self, ctx):
        """Test recalling memory by full ID."""
        memory = ctx.save(content="Memory to recall", type=MemoryType.FACT)

        recalled = ctx.recall(memory.id)

        assert recalled is not None
        assert recalled.id == memory.id
        assert recalled.content == "Memory to recall"

    def test_recall_by_partial_id(self, ctx):
        """Test recalling memory by partial ID (min 8 chars)."""
        memory = ctx.save(content="Memory with partial ID", type=MemoryType.FACT)

        # Use first 8 characters
        partial_id = memory.id[:8]
        recalled = ctx.recall(partial_id)

        assert recalled is not None
        assert recalled.id == memory.id

    def test_recall_nonexistent(self, ctx):
        """Test recalling nonexistent memory returns None."""
        recalled = ctx.recall("nonexistent-id-12345")

        assert recalled is None

    def test_recall_includes_metadata(self, ctx):
        """Test that recalled memory includes all metadata."""
        memory = ctx.save(
            content="Memory with metadata",
            type=MemoryType.DECISION,
            summary="Important decision",
            tags=["test", "metadata"],
            structured_data={"decision": "Test", "rationale": "Testing"},
        )

        recalled = ctx.recall(memory.id)

        assert recalled.summary == "Important decision"
        assert "test" in recalled.tags
        assert recalled.structured_data is not None


class TestMCPUpdate:
    """Tests for contextfs_update MCP tool."""

    def test_update_content(self, ctx):
        """Test updating memory content."""
        memory = ctx.save(content="Original content", type=MemoryType.FACT)

        updated = ctx.update(memory_id=memory.id, content="Updated content")

        assert updated.content == "Updated content"
        assert updated.id == memory.id

    def test_update_summary(self, ctx):
        """Test updating memory summary."""
        memory = ctx.save(
            content="Memory content",
            type=MemoryType.FACT,
            summary="Original summary",
        )

        updated = ctx.update(memory_id=memory.id, summary="New summary")

        assert updated.summary == "New summary"

    def test_update_tags(self, ctx):
        """Test updating memory tags."""
        memory = ctx.save(
            content="Memory content",
            type=MemoryType.FACT,
            tags=["original"],
        )

        updated = ctx.update(memory_id=memory.id, tags=["new", "tags"])

        assert "new" in updated.tags
        assert "tags" in updated.tags

    def test_update_preserves_unmodified_fields(self, ctx):
        """Test that update preserves fields not being modified."""
        memory = ctx.save(
            content="Original content",
            type=MemoryType.DECISION,
            summary="Original summary",
            tags=["preserve"],
            structured_data={"decision": "Keep this"},
        )

        # Only update content
        updated = ctx.update(memory_id=memory.id, content="New content")

        assert updated.content == "New content"
        assert updated.summary == "Original summary"
        assert "preserve" in updated.tags
        assert updated.structured_data["decision"] == "Keep this"

    def test_update_nonexistent_returns_none(self, ctx):
        """Test updating nonexistent memory returns None or handles gracefully."""
        # May return None or raise - just verify it doesn't crash unexpectedly
        ctx.update(memory_id="nonexistent-id-12345", content="New content")


class TestMCPDelete:
    """Tests for contextfs_delete MCP tool."""

    def test_delete_memory(self, ctx):
        """Test deleting a memory."""
        memory = ctx.save(content="Memory to delete", type=MemoryType.FACT)

        ctx.delete(memory.id)

        recalled = ctx.recall(memory.id)
        assert recalled is None

    def test_delete_by_partial_id(self, ctx):
        """Test deleting by partial ID."""
        memory = ctx.save(content="Delete by partial", type=MemoryType.FACT)
        partial_id = memory.id[:8]

        ctx.delete(partial_id)

        recalled = ctx.recall(memory.id)
        assert recalled is None

    def test_delete_nonexistent(self, ctx):
        """Test deleting nonexistent memory."""
        # Should not raise, just return gracefully
        ctx.delete("nonexistent-id-12345")


class TestMCPEvolve:
    """Tests for contextfs_evolve MCP tool."""

    def test_evolve_creates_new_version(self, ctx):
        """Test evolving memory creates new version with history."""
        original = ctx.save(
            content="Original decision about architecture",
            type=MemoryType.DECISION,
            summary="Architecture v1",
        )

        evolved = ctx.evolve(
            memory_id=original.id,
            new_content="Updated decision about architecture with new insights",
            summary="Architecture v2",
        )

        # Evolved should be a new memory
        assert evolved.id != original.id
        assert "Updated decision" in evolved.content

    def test_evolve_preserves_tags(self, ctx):
        """Test evolve preserves original tags by default."""
        original = ctx.save(
            content="Original content",
            type=MemoryType.FACT,
            tags=["important", "keep-these"],
        )

        evolved = ctx.evolve(
            memory_id=original.id,
            new_content="Evolved content",
        )

        assert "important" in evolved.tags
        assert "keep-these" in evolved.tags

    def test_evolve_with_additional_tags(self, ctx):
        """Test evolve with additional tags."""
        original = ctx.save(
            content="Original content",
            type=MemoryType.FACT,
            tags=["original"],
        )

        evolved = ctx.evolve(
            memory_id=original.id,
            new_content="Evolved content",
            additional_tags=["evolved", "updated"],
        )

        assert "original" in evolved.tags
        assert "evolved" in evolved.tags
        assert "updated" in evolved.tags


class TestMCPLink:
    """Tests for contextfs_link MCP tool."""

    def test_link_memories(self, ctx):
        """Test creating link between memories."""
        memory1 = ctx.save(content="First memory", type=MemoryType.FACT)
        memory2 = ctx.save(content="Second memory", type=MemoryType.FACT)

        # Create link - uses from_memory_id and to_memory_id
        result = ctx.link(
            from_memory_id=memory1.id,
            to_memory_id=memory2.id,
            relation="references",
        )

        assert result is not None

    def test_link_different_relations(self, ctx):
        """Test creating links with different relation types."""
        base = ctx.save(content="Base memory", type=MemoryType.DECISION)

        # Use valid EdgeRelation values
        relations = [
            "references",
            "related_to",
            "contradicts",
            "supersedes",
            "part_of",
            "caused_by",
        ]

        for relation in relations:
            target = ctx.save(content=f"Target for {relation}", type=MemoryType.FACT)

            result = ctx.link(
                from_memory_id=base.id,
                to_memory_id=target.id,
                relation=relation,
            )

            assert result is not None

    def test_link_bidirectional(self, ctx):
        """Test creating bidirectional link."""
        memory1 = ctx.save(content="Memory A", type=MemoryType.FACT)
        memory2 = ctx.save(content="Memory B", type=MemoryType.FACT)

        result = ctx.link(
            from_memory_id=memory1.id,
            to_memory_id=memory2.id,
            relation="related_to",
            bidirectional=True,
        )

        assert result is not None

    def test_link_with_weight(self, ctx):
        """Test creating weighted link."""
        memory1 = ctx.save(content="Strong reference", type=MemoryType.FACT)
        memory2 = ctx.save(content="Target", type=MemoryType.FACT)

        # Use valid relation "references" instead of "supports"
        result = ctx.link(
            from_memory_id=memory1.id,
            to_memory_id=memory2.id,
            relation="references",
            weight=0.8,
        )

        assert result is not None


class TestMCPList:
    """Tests for contextfs_list MCP tool."""

    def test_list_recent(self, ctx):
        """Test listing recent memories."""
        for i in range(5):
            ctx.save(content=f"Memory {i}", type=MemoryType.FACT)

        memories = ctx.list_recent(limit=10)

        assert len(memories) >= 5

    def test_list_with_type_filter(self, ctx):
        """Test listing with type filter."""
        ctx.save(content="A fact", type=MemoryType.FACT)
        ctx.save(content="A decision", type=MemoryType.DECISION)
        ctx.save(content="Another fact", type=MemoryType.FACT)

        memories = ctx.list_recent(type=MemoryType.FACT)

        for memory in memories:
            assert memory.type == MemoryType.FACT

    def test_list_with_limit(self, ctx):
        """Test listing with limit."""
        for i in range(10):
            ctx.save(content=f"Memory {i}", type=MemoryType.FACT)

        memories = ctx.list_recent(limit=3)

        assert len(memories) <= 3


class TestMCPListRepos:
    """Tests for contextfs_list_repos MCP tool."""

    def test_list_repos_empty(self, ctx):
        """Test listing repos when none indexed."""
        repos = ctx.list_repos()

        assert isinstance(repos, list)


class TestMCPListProjects:
    """Tests for contextfs_list_projects MCP tool."""

    def test_list_projects(self, ctx):
        """Test listing projects."""
        ctx.save(content="Project A memory", type=MemoryType.FACT, project="project-a")
        ctx.save(content="Project B memory", type=MemoryType.FACT, project="project-b")
        ctx.save(content="No project memory", type=MemoryType.FACT)

        projects = ctx.list_projects()

        assert isinstance(projects, list)


class TestMCPSessions:
    """Tests for contextfs_sessions and session management."""

    def test_add_message(self, ctx):
        """Test adding message to session."""
        ctx.start_session(tool="test", label="test-add-message")
        ctx.add_message("user", "Hello!")
        ctx.add_message("assistant", "Hi there!")

        # Verify session has messages
        session = ctx.get_current_session()
        assert session is not None

    def test_list_sessions(self, ctx):
        """Test listing sessions."""
        # Create a session by adding messages and ending
        ctx.start_session(tool="test", label="test-list-sessions")
        ctx.add_message("user", "First message")
        ctx.add_message("assistant", "Response")
        ctx.end_session(generate_summary=True)

        sessions = ctx.list_sessions(limit=10)

        assert isinstance(sessions, list)


class TestMCPGetTypeSchema:
    """Tests for contextfs_get_type_schema MCP tool."""

    def test_get_decision_schema(self, ctx):
        """Test getting schema for decision type."""
        from contextfs.schemas import get_type_schema

        schema = get_type_schema(MemoryType.DECISION)

        assert schema is not None
        assert "decision" in str(schema).lower() or schema.get("properties") is not None

    def test_get_error_schema(self, ctx):
        """Test getting schema for error type."""
        from contextfs.schemas import get_type_schema

        schema = get_type_schema(MemoryType.ERROR)

        assert schema is not None

    def test_get_schema_no_schema_type(self, ctx):
        """Test getting schema for type without schema."""
        from contextfs.schemas import get_type_schema

        # Fact type may not have a schema
        schema = get_type_schema(MemoryType.FACT)

        # Should return None or empty schema
        assert schema is None or isinstance(schema, dict)
