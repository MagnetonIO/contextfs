"""Integration tests for typed schema validation.

Tests structured_data through CLI, MCP, and API interfaces.
"""

import json

import pytest

from contextfs.core import ContextFS
from contextfs.schemas import MemoryType


@pytest.fixture
def ctx(tmp_path):
    """Create a ContextFS instance with temporary data directory."""
    ctx = ContextFS(data_dir=tmp_path, auto_index=False)
    yield ctx
    ctx.close()


class TestCLIStructuredData:
    """Test CLI with structured_data."""

    def test_save_with_structured_data(self, ctx):
        """Test saving memory with structured_data via core API."""
        structured_data = {
            "decision": "Use PostgreSQL",
            "rationale": "ACID compliance",
            "alternatives": ["MySQL", "MongoDB"],
        }

        memory = ctx.save(
            content="Database decision for project",
            type=MemoryType.DECISION,
            structured_data=structured_data,
        )

        assert memory.structured_data == structured_data

        # Verify persistence
        recalled = ctx.recall(memory.id)
        assert recalled is not None
        assert recalled.structured_data == structured_data

    def test_save_with_invalid_structured_data(self, ctx):
        """Test that invalid structured_data raises error."""
        with pytest.raises(ValueError) as exc_info:
            ctx.save(
                content="Missing decision field",
                type=MemoryType.DECISION,
                structured_data={"rationale": "No decision"},  # missing 'decision'
            )

        assert "validation" in str(exc_info.value).lower()

    def test_save_procedural_with_steps(self, ctx):
        """Test saving procedural memory with steps."""
        structured_data = {
            "steps": [
                "Clone the repository",
                "Install dependencies",
                "Run tests",
            ],
            "prerequisites": ["Python 3.10+", "Git"],
        }

        memory = ctx.save(
            content="Setup procedure for new developers",
            type=MemoryType.PROCEDURAL,
            structured_data=structured_data,
        )

        assert memory.structured_data["steps"] == structured_data["steps"]

        recalled = ctx.recall(memory.id)
        assert recalled.structured_data == structured_data

    def test_save_error_with_structured_data(self, ctx):
        """Test saving error memory with structured error info."""
        structured_data = {
            "error_type": "ConnectionError",
            "message": "Failed to connect to database",
            "file": "db.py",
            "line": 42,
            "resolution": "Check database credentials",
        }

        memory = ctx.save(
            content="Database connection error encountered",
            type=MemoryType.ERROR,
            structured_data=structured_data,
        )

        assert memory.structured_data["error_type"] == "ConnectionError"
        assert memory.structured_data["line"] == 42

    def test_search_with_structured_data(self, ctx):
        """Test that search returns memories with structured_data."""
        # Save a memory with structured data
        memory = ctx.save(
            content="JWT authentication decision",
            type=MemoryType.DECISION,
            structured_data={
                "decision": "Use JWT",
                "rationale": "Stateless",
            },
        )

        # Search should return it with structured_data intact
        results = ctx.search("JWT authentication")
        assert len(results) > 0

        found = next((r for r in results if r.memory.id == memory.id), None)
        assert found is not None
        assert found.memory.structured_data is not None
        assert found.memory.structured_data["decision"] == "Use JWT"

    def test_update_preserves_structured_data(self, ctx):
        """Test that updating memory preserves structured_data."""
        memory = ctx.save(
            content="Original content",
            type=MemoryType.DECISION,
            structured_data={"decision": "Original decision"},
        )

        # Update content but not structured_data
        updated = ctx.update(
            memory_id=memory.id,
            content="Updated content",
        )

        # structured_data should be preserved
        assert updated.structured_data["decision"] == "Original decision"

    def test_type_without_schema(self, ctx):
        """Test that types without schemas accept any structured_data."""
        # 'fact' type has no schema, should accept anything
        memory = ctx.save(
            content="Random fact",
            type=MemoryType.FACT,
            structured_data={
                "custom_field": "anything",
                "nested": {"works": True},
            },
        )

        assert memory.structured_data["custom_field"] == "anything"


class TestMCPStructuredData:
    """Test MCP server with structured_data."""

    def test_mcp_save_with_structured_data(self, ctx):
        """Test MCP save tool with structured_data parameter."""
        # This tests the MCP tool indirectly via the core API
        # since the MCP server just passes through to ctx.save()

        # Simulate MCP tool call arguments
        arguments = {
            "content": "MCP decision memory",
            "type": "decision",
            "structured_data": {
                "decision": "Via MCP",
                "rationale": "Testing MCP integration",
            },
        }

        memory = ctx.save(
            content=arguments["content"],
            type=MemoryType(arguments["type"]),
            structured_data=arguments.get("structured_data"),
        )

        assert memory.structured_data is not None
        assert memory.structured_data["decision"] == "Via MCP"


class TestAPIStructuredData:
    """Test API endpoints with structured_data."""

    def test_api_create_memory_with_structured_data(self, ctx):
        """Test API memory creation with structured_data."""
        # This tests the serialization/deserialization of structured_data
        from contextfs.web.server import serialize_memory

        memory = ctx.save(
            content="API test memory",
            type=MemoryType.DECISION,
            structured_data={
                "decision": "Use FastAPI",
                "rationale": "Async support, automatic OpenAPI",
            },
        )

        # Serialize like the API does
        serialized = serialize_memory(memory)

        assert "structured_data" in serialized
        assert serialized["structured_data"]["decision"] == "Use FastAPI"

        # Verify it's JSON-serializable
        json_str = json.dumps(serialized)
        parsed = json.loads(json_str)
        assert parsed["structured_data"]["decision"] == "Use FastAPI"

    def test_api_response_includes_structured_data(self, ctx):
        """Test that API responses include structured_data."""
        from contextfs.schemas import SearchResult
        from contextfs.web.server import serialize_search_result

        memory = ctx.save(
            content="Searchable decision",
            type=MemoryType.DECISION,
            structured_data={"decision": "Test", "rationale": "Testing"},
        )

        # Simulate search result
        search_result = SearchResult(memory=memory, score=0.95)
        serialized = serialize_search_result(search_result)

        assert serialized["memory"]["structured_data"]["decision"] == "Test"


class TestDatabaseMigration:
    """Test that structured_data column is added correctly."""

    def test_structured_data_persists_after_restart(self, tmp_path):
        """Test structured_data survives database close/reopen."""
        # Create and save
        ctx1 = ContextFS(data_dir=tmp_path, auto_index=False)
        memory = ctx1.save(
            content="Persistent decision",
            type=MemoryType.DECISION,
            structured_data={"decision": "Survives restart"},
        )
        memory_id = memory.id
        ctx1.close()

        # Reopen and verify
        ctx2 = ContextFS(data_dir=tmp_path, auto_index=False)
        recalled = ctx2.recall(memory_id)

        assert recalled is not None
        assert recalled.structured_data is not None
        assert recalled.structured_data["decision"] == "Survives restart"
        ctx2.close()

    def test_existing_memories_have_null_structured_data(self, tmp_path):
        """Test that existing memories without structured_data work."""
        # Create memory without structured_data
        ctx = ContextFS(data_dir=tmp_path, auto_index=False)
        memory = ctx.save(
            content="No structured data",
            type=MemoryType.FACT,
        )

        recalled = ctx.recall(memory.id)
        assert recalled.structured_data is None
        ctx.close()


class TestEdgeCases:
    """Test edge cases for structured_data."""

    def test_empty_structured_data(self, ctx):
        """Test saving empty dict as structured_data."""
        # Empty dict should still be stored (vs None)
        memory = ctx.save(
            content="Empty structured",
            type=MemoryType.FACT,  # No schema, accepts anything
            structured_data={},
        )

        recalled = ctx.recall(memory.id)
        assert recalled.structured_data == {}

    def test_nested_structured_data(self, ctx):
        """Test deeply nested structured_data."""
        nested_data = {
            "decision": "Complex structure",
            "metadata": {
                "author": {
                    "name": "Test",
                    "team": "Engineering",
                },
                "tags": ["nested", "test"],
            },
        }

        memory = ctx.save(
            content="Nested test",
            type=MemoryType.DECISION,
            structured_data=nested_data,
        )

        recalled = ctx.recall(memory.id)
        assert recalled.structured_data["metadata"]["author"]["name"] == "Test"

    def test_unicode_in_structured_data(self, ctx):
        """Test unicode characters in structured_data."""
        memory = ctx.save(
            content="Unicode test",
            type=MemoryType.DECISION,
            structured_data={
                "decision": "支持Unicode 🎉",
                "rationale": "Émojis and spëcial characters",
            },
        )

        recalled = ctx.recall(memory.id)
        assert "🎉" in recalled.structured_data["decision"]
        assert "ë" in recalled.structured_data["rationale"]

    def test_large_structured_data(self, ctx):
        """Test large structured_data."""
        large_data = {
            "decision": "Large data test",
            "alternatives": [f"Alternative {i}" for i in range(100)],
            "constraints": [f"Constraint {i}" for i in range(100)],
        }

        memory = ctx.save(
            content="Large structured data",
            type=MemoryType.DECISION,
            structured_data=large_data,
        )

        recalled = ctx.recall(memory.id)
        assert len(recalled.structured_data["alternatives"]) == 100
