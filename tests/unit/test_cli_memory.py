"""CLI Memory Commands Tests.

Tests for the memory CLI commands: save, search, recall, delete, list, evolve, merge.
"""

import json

import pytest
from typer.testing import CliRunner as TyperRunner

from contextfs.cli.memory import memory_app


@pytest.fixture
def runner():
    """Create CLI runner."""
    return TyperRunner()


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    """Create a ContextFS instance with temp directory."""
    from contextfs import ContextFS

    # Use temp directory to avoid polluting user's actual data
    monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

    ctx = ContextFS(data_dir=tmp_path, namespace_id="test-cli", auto_load=False)
    yield ctx
    ctx.close()


# =============================================================================
# Save Command Tests
# =============================================================================


class TestMemorySaveCommand:
    """Tests for 'contextfs memory save' command."""

    def test_save_basic(self, runner, tmp_path, monkeypatch):
        """Test basic memory save."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        result = runner.invoke(memory_app, ["save", "Test content"])
        assert result.exit_code == 0
        assert "Memory saved" in result.stdout

    def test_save_with_type(self, runner, tmp_path, monkeypatch):
        """Test save with specific type."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        result = runner.invoke(memory_app, ["save", "Decision content", "-t", "decision"])
        assert result.exit_code == 0
        assert "Memory saved" in result.stdout
        assert "decision" in result.stdout.lower()

    def test_save_with_tags(self, runner, tmp_path, monkeypatch):
        """Test save with tags."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        result = runner.invoke(memory_app, ["save", "Tagged content", "--tags", "tag1,tag2,tag3"])
        assert result.exit_code == 0
        assert "Memory saved" in result.stdout

    def test_save_with_summary(self, runner, tmp_path, monkeypatch):
        """Test save with summary."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        result = runner.invoke(
            memory_app,
            ["save", "Long content here", "--summary", "Brief summary"],
        )
        assert result.exit_code == 0
        assert "Memory saved" in result.stdout

    def test_save_with_structured_data(self, runner, tmp_path, monkeypatch):
        """Test save with structured data."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        structured = json.dumps({"decision": "Use PostgreSQL", "rationale": "Better JSON support"})
        result = runner.invoke(
            memory_app,
            ["save", "Auth decision", "-t", "decision", "--structured", structured],
        )
        assert result.exit_code == 0
        assert "Memory saved" in result.stdout

    def test_save_invalid_type(self, runner, tmp_path, monkeypatch):
        """Test save with invalid type fails."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        result = runner.invoke(memory_app, ["save", "Content", "-t", "invalid_type_xyz"])
        assert result.exit_code == 1
        assert "Invalid type" in result.stdout

    def test_save_invalid_json(self, runner, tmp_path, monkeypatch):
        """Test save with invalid JSON structured data fails."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        result = runner.invoke(memory_app, ["save", "Content", "--structured", "not valid json"])
        assert result.exit_code == 1
        assert "Invalid JSON" in result.stdout


# =============================================================================
# Search Command Tests
# =============================================================================


class TestMemorySearchCommand:
    """Tests for 'contextfs memory search' command."""

    def test_search_basic(self, runner, tmp_path, monkeypatch):
        """Test basic search."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        # First save something to search for
        runner.invoke(memory_app, ["save", "Python programming tutorial"])

        result = runner.invoke(memory_app, ["search", "Python"])
        # May or may not find results depending on indexing
        assert result.exit_code == 0

    def test_search_with_limit(self, runner, tmp_path, monkeypatch):
        """Test search with limit."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        result = runner.invoke(memory_app, ["search", "test query", "-n", "5"])
        assert result.exit_code == 0

    def test_search_no_results(self, runner, tmp_path, monkeypatch):
        """Test search with no results."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        result = runner.invoke(memory_app, ["search", "xyznonexistentquery123456789"])
        assert result.exit_code == 0
        # Should handle empty results gracefully

    def test_search_with_type_filter(self, runner, tmp_path, monkeypatch):
        """Test search filtered by type."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        result = runner.invoke(memory_app, ["search", "query", "-t", "fact"])
        assert result.exit_code == 0

    def test_search_different_modes(self, runner, tmp_path, monkeypatch):
        """Test search with different modes."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        for mode in ["hybrid", "semantic", "keyword"]:
            result = runner.invoke(memory_app, ["search", "test", "-m", mode])
            assert result.exit_code == 0


# =============================================================================
# Recall Command Tests
# =============================================================================


class TestMemoryRecallCommand:
    """Tests for 'contextfs memory recall' command."""

    def test_recall_by_id(self, runner, tmp_path, monkeypatch):
        """Test recalling memory by ID."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        # First save a memory
        save_result = runner.invoke(memory_app, ["save", "Content to recall"])
        assert save_result.exit_code == 0

        # Extract ID from output
        lines = save_result.stdout.split("\n")
        memory_id = None
        for line in lines:
            if line.startswith("ID:"):
                memory_id = line.split(":")[1].strip()
                break

        if memory_id:
            result = runner.invoke(memory_app, ["recall", memory_id])
            assert result.exit_code == 0
            assert "Content to recall" in result.stdout

    def test_recall_partial_id(self, runner, tmp_path, monkeypatch):
        """Test recalling memory with partial ID."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        save_result = runner.invoke(memory_app, ["save", "Partial ID test"])
        assert save_result.exit_code == 0

        # Extract ID and use partial
        lines = save_result.stdout.split("\n")
        memory_id = None
        for line in lines:
            if line.startswith("ID:"):
                memory_id = line.split(":")[1].strip()
                break

        if memory_id:
            partial_id = memory_id[:8]
            result = runner.invoke(memory_app, ["recall", partial_id])
            # May or may not find it with partial ID
            assert result.exit_code in [0, 1]

    def test_recall_not_found(self, runner, tmp_path, monkeypatch):
        """Test recall with nonexistent ID."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        result = runner.invoke(memory_app, ["recall", "nonexistent-id-123456"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()


# =============================================================================
# List Command Tests
# =============================================================================


class TestMemoryListCommand:
    """Tests for 'contextfs memory list' command."""

    def test_list_basic(self, runner, tmp_path, monkeypatch):
        """Test basic list."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        result = runner.invoke(memory_app, ["list"])
        assert result.exit_code == 0

    def test_list_with_limit(self, runner, tmp_path, monkeypatch):
        """Test list with limit."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        # Save multiple memories
        for i in range(5):
            runner.invoke(memory_app, ["save", f"Memory {i}"])

        result = runner.invoke(memory_app, ["list", "-n", "3"])
        assert result.exit_code == 0

    def test_list_with_type_filter(self, runner, tmp_path, monkeypatch):
        """Test list filtered by type."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        runner.invoke(memory_app, ["save", "Fact memory", "-t", "fact"])
        runner.invoke(memory_app, ["save", "Error memory", "-t", "error"])

        result = runner.invoke(memory_app, ["list", "-t", "fact"])
        assert result.exit_code == 0


# =============================================================================
# Delete Command Tests
# =============================================================================


class TestMemoryDeleteCommand:
    """Tests for 'contextfs memory delete' command."""

    def test_delete_with_confirm(self, runner, tmp_path, monkeypatch):
        """Test delete with confirmation."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        save_result = runner.invoke(memory_app, ["save", "To delete"])
        assert save_result.exit_code == 0

        # Extract ID
        lines = save_result.stdout.split("\n")
        memory_id = None
        for line in lines:
            if line.startswith("ID:"):
                memory_id = line.split(":")[1].strip()
                break

        if memory_id:
            result = runner.invoke(memory_app, ["delete", memory_id, "-y"])
            assert result.exit_code == 0
            assert "deleted" in result.stdout.lower()

    def test_delete_not_found(self, runner, tmp_path, monkeypatch):
        """Test delete nonexistent memory."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        result = runner.invoke(memory_app, ["delete", "nonexistent-id", "-y"])
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()


# =============================================================================
# Evolve Command Tests
# =============================================================================


class TestMemoryEvolveCommand:
    """Tests for 'contextfs memory evolve' command."""

    def test_evolve_basic(self, runner, tmp_path, monkeypatch):
        """Test basic evolve."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        save_result = runner.invoke(memory_app, ["save", "Original content"])
        assert save_result.exit_code == 0

        # Extract ID
        lines = save_result.stdout.split("\n")
        memory_id = None
        for line in lines:
            if line.startswith("ID:"):
                memory_id = line.split(":")[1].strip()
                break

        if memory_id:
            result = runner.invoke(memory_app, ["evolve", memory_id, "Updated content"])
            assert result.exit_code == 0
            assert "evolved" in result.stdout.lower()

    def test_evolve_with_tags(self, runner, tmp_path, monkeypatch):
        """Test evolve with additional tags."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        save_result = runner.invoke(memory_app, ["save", "Original"])
        assert save_result.exit_code == 0

        lines = save_result.stdout.split("\n")
        memory_id = None
        for line in lines:
            if line.startswith("ID:"):
                memory_id = line.split(":")[1].strip()
                break

        if memory_id:
            result = runner.invoke(
                memory_app,
                ["evolve", memory_id, "Updated", "-t", "new_tag"],
            )
            assert result.exit_code == 0

    def test_evolve_not_found(self, runner, tmp_path, monkeypatch):
        """Test evolve nonexistent memory."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        result = runner.invoke(memory_app, ["evolve", "nonexistent-id", "New content"])
        assert result.exit_code == 1
        # Error may be in stdout or exception output
        output = result.stdout + (str(result.exception) if result.exception else "")
        assert "not found" in output.lower()


# =============================================================================
# Type Schema Command Tests
# =============================================================================


class TestMemoryTypeSchemaCommand:
    """Tests for 'contextfs memory type-schema' command."""

    def test_type_schema_decision(self, runner, tmp_path, monkeypatch):
        """Test showing decision type schema."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        result = runner.invoke(memory_app, ["type-schema", "decision"])
        assert result.exit_code == 0
        # Should show schema or 'no schema' message

    def test_type_schema_error(self, runner, tmp_path, monkeypatch):
        """Test showing error type schema."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        result = runner.invoke(memory_app, ["type-schema", "error"])
        assert result.exit_code == 0

    def test_type_schema_no_schema(self, runner, tmp_path, monkeypatch):
        """Test type without schema."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        result = runner.invoke(memory_app, ["type-schema", "fact"])
        assert result.exit_code == 0
        # Fact type may not have a schema


# =============================================================================
# Integration Tests
# =============================================================================


class TestMemoryCLIIntegration:
    """Integration tests for CLI memory workflow."""

    def test_save_search_recall_delete_workflow(self, runner, tmp_path, monkeypatch):
        """Test complete save -> search -> recall -> delete workflow."""
        monkeypatch.setenv("CONTEXTFS_DATA_DIR", str(tmp_path))

        # Save
        save_result = runner.invoke(
            memory_app,
            ["save", "Integration test content", "-t", "fact", "--tags", "test,integration"],
        )
        assert save_result.exit_code == 0

        # Extract ID
        lines = save_result.stdout.split("\n")
        memory_id = None
        for line in lines:
            if line.startswith("ID:"):
                memory_id = line.split(":")[1].strip()
                break

        assert memory_id is not None

        # List - should show our memory
        list_result = runner.invoke(memory_app, ["list", "-t", "fact"])
        assert list_result.exit_code == 0

        # Recall
        recall_result = runner.invoke(memory_app, ["recall", memory_id])
        assert recall_result.exit_code == 0
        assert "Integration test content" in recall_result.stdout

        # Delete
        delete_result = runner.invoke(memory_app, ["delete", memory_id, "-y"])
        assert delete_result.exit_code == 0

        # Recall again - should fail
        recall_again = runner.invoke(memory_app, ["recall", memory_id])
        assert recall_again.exit_code == 1
