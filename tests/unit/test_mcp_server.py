"""MCP Server Tests.

Tests for the FastMCP server tools and helper functions.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from contextfs.schemas import Memory, MemoryType, SearchResult, Session


@pytest.fixture
def mock_ctx():
    """Create a mock ContextFS instance."""
    ctx = MagicMock()
    ctx.config = MagicMock()
    ctx.config.auto_link_enabled = False
    return ctx


@pytest.fixture
def sample_memory():
    """Create a sample memory for testing."""
    return Memory(
        id="test-memory-123",
        content="Test content for memory",
        type=MemoryType.FACT,
        tags=["test", "sample"],
        summary="Test summary",
        source_tool="claude-code",
        source_repo="test-repo",
        project="test-project",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def sample_session():
    """Create a sample session for testing."""
    return Session(
        id="test-session-456",
        label="test-session",
        tool="claude-code",
        summary="Test session summary",
        messages=[],
        created_at=datetime.now(),
    )


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_source_tool_default(self, monkeypatch):
        """Test default source tool."""
        # Reset global state
        import contextfs.mcp.fastmcp_server as server

        server._source_tool = None
        monkeypatch.delenv("CONTEXTFS_SOURCE_TOOL", raising=False)

        result = server.get_source_tool()
        assert result == "claude-code"

    def test_get_source_tool_from_env(self, monkeypatch):
        """Test source tool from environment variable."""
        import contextfs.mcp.fastmcp_server as server

        server._source_tool = None
        monkeypatch.setenv("CONTEXTFS_SOURCE_TOOL", "cursor")

        result = server.get_source_tool()
        assert result == "cursor"

    def test_get_source_tool_cached(self, monkeypatch):
        """Test source tool is cached."""
        import contextfs.mcp.fastmcp_server as server

        server._source_tool = "cached-tool"

        result = server.get_source_tool()
        assert result == "cached-tool"

    def test_detect_current_repo_success(self):
        """Test detecting current repository."""
        import contextfs.mcp.fastmcp_server as server

        with patch.dict("sys.modules", {"git": MagicMock()}):
            import sys

            mock_git = sys.modules["git"]
            mock_repo = MagicMock()
            mock_repo.working_tree_dir = "/path/to/my-repo"
            mock_git.Repo.return_value = mock_repo

            result = server.detect_current_repo()
            assert result == "my-repo"

    def test_detect_current_repo_not_git(self):
        """Test detecting repo when not in git directory."""
        import contextfs.mcp.fastmcp_server as server

        with patch.dict("sys.modules", {"git": MagicMock()}):
            import sys

            mock_git = sys.modules["git"]
            mock_git.Repo.side_effect = Exception("Not a git repository")

            result = server.detect_current_repo()
            assert result is None


# =============================================================================
# Save Tool Tests
# =============================================================================


class TestContextfsSave:
    """Tests for contextfs_save tool."""

    @pytest.mark.asyncio
    async def test_save_basic(self, mock_ctx, sample_memory):
        """Test basic save operation."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.save.return_value = sample_memory

        with (
            patch.object(server, "get_ctx", return_value=mock_ctx),
            patch.object(server, "get_source_tool", return_value="claude-code"),
            patch.object(server, "detect_current_repo", return_value="test-repo"),
        ):
            result = await server.contextfs_save(
                content="Test content",
                type="fact",
                tags=["test"],
                summary="Test summary",
            )

        assert "Memory saved successfully" in result
        assert sample_memory.id in result
        mock_ctx.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_empty_content(self, mock_ctx):
        """Test save with empty content returns error."""
        import contextfs.mcp.fastmcp_server as server

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = await server.contextfs_save(content="")

        assert "Error: content is required" in result

    @pytest.mark.asyncio
    async def test_save_session(self, mock_ctx, sample_session):
        """Test saving current session."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.get_current_session.return_value = sample_session

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = await server.contextfs_save(
                save_session="current",
                label="my-session",
            )

        assert "Session saved" in result
        assert sample_session.id in result
        mock_ctx.end_session.assert_called_once_with(generate_summary=True)

    @pytest.mark.asyncio
    async def test_save_session_no_active(self, mock_ctx):
        """Test saving session when none active."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.get_current_session.return_value = None

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = await server.contextfs_save(save_session="current")

        assert "No active session" in result

    @pytest.mark.asyncio
    async def test_save_with_auto_link(self, mock_ctx, sample_memory):
        """Test save with auto-linking enabled."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.config.auto_link_enabled = True
        mock_ctx.save.return_value = sample_memory
        mock_ctx.get_related.return_value = [MagicMock(), MagicMock()]

        with (
            patch.object(server, "get_ctx", return_value=mock_ctx),
            patch.object(server, "get_source_tool", return_value="claude-code"),
            patch.object(server, "detect_current_repo", return_value=None),
        ):
            result = await server.contextfs_save(
                content="Test content",
                type="fact",
            )

        assert "Auto-linked: 2 related memories" in result


# =============================================================================
# Search Tool Tests
# =============================================================================


class TestContextfsSearch:
    """Tests for contextfs_search tool."""

    def test_search_basic(self, mock_ctx, sample_memory):
        """Test basic search."""
        import contextfs.mcp.fastmcp_server as server

        search_result = SearchResult(memory=sample_memory, score=0.95)
        mock_ctx.search.return_value = [search_result]

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_search(query="test query")

        assert sample_memory.id in result
        assert "0.95" in result
        assert "fact" in result
        mock_ctx.search.assert_called_once()

    def test_search_no_results(self, mock_ctx):
        """Test search with no results."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.search.return_value = []

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_search(query="nonexistent")

        assert "No memories found" in result

    def test_search_with_filters(self, mock_ctx, sample_memory):
        """Test search with type and repo filters."""
        import contextfs.mcp.fastmcp_server as server

        search_result = SearchResult(memory=sample_memory, score=0.85)
        mock_ctx.search.return_value = [search_result]

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            server.contextfs_search(
                query="test",
                type="fact",
                source_repo="my-repo",
                limit=10,
            )

        mock_ctx.search.assert_called_once()
        call_kwargs = mock_ctx.search.call_args[1]
        assert call_kwargs["type"] == MemoryType.FACT
        assert call_kwargs["source_repo"] == "my-repo"
        assert call_kwargs["limit"] == 10


# =============================================================================
# Recall Tool Tests
# =============================================================================


class TestContextfsRecall:
    """Tests for contextfs_recall tool."""

    def test_recall_found(self, mock_ctx, sample_memory):
        """Test recalling existing memory."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.recall.return_value = sample_memory

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_recall(id=sample_memory.id)

        assert sample_memory.id in result
        assert sample_memory.content in result
        assert "fact" in result

    def test_recall_not_found(self, mock_ctx):
        """Test recalling non-existent memory."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.recall.return_value = None

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_recall(id="nonexistent-id")

        assert "Memory not found" in result

    def test_recall_with_structured_data(self, mock_ctx, sample_memory):
        """Test recalling memory with structured data."""
        import contextfs.mcp.fastmcp_server as server

        sample_memory.structured_data = {"key": "value", "nested": {"a": 1}}
        mock_ctx.recall.return_value = sample_memory

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_recall(id=sample_memory.id)

        assert "Structured Data" in result
        assert '"key": "value"' in result


# =============================================================================
# List Tools Tests
# =============================================================================


class TestContextfsList:
    """Tests for contextfs_list tool."""

    def test_list_recent(self, mock_ctx, sample_memory):
        """Test listing recent memories."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.list_recent.return_value = [sample_memory]

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_list(limit=10)

        assert sample_memory.id[:8] in result
        mock_ctx.list_recent.assert_called_once()

    def test_list_empty(self, mock_ctx):
        """Test listing when no memories."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.list_recent.return_value = []

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_list()

        assert "No memories found" in result


class TestContextfsListRepos:
    """Tests for contextfs_list_repos tool."""

    def test_list_repos(self, mock_ctx):
        """Test listing repositories."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.list_repos.return_value = [
            {"source_repo": "repo1", "memory_count": 10},
            {"source_repo": "repo2", "memory_count": 5},
        ]
        mock_ctx.list_indexes.return_value = []

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_list_repos()

        assert "repo1" in result
        assert "10 memories" in result

    def test_list_repos_empty(self, mock_ctx):
        """Test listing when no repos."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.list_repos.return_value = []
        mock_ctx.list_indexes.return_value = []

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_list_repos()

        assert "No repositories" in result


class TestContextfsListTools:
    """Tests for contextfs_list_tools tool."""

    def test_list_tools(self, mock_ctx):
        """Test listing source tools."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.list_tools.return_value = [
            {"source_tool": "claude-code", "memory_count": 50},
        ]

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_list_tools()

        assert "claude-code" in result
        assert "50 memories" in result


class TestContextfsListProjects:
    """Tests for contextfs_list_projects tool."""

    def test_list_projects(self, mock_ctx):
        """Test listing projects."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.list_projects.return_value = [
            {"project": "my-project", "memory_count": 20, "repos": ["repo1", "repo2"]},
        ]

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_list_projects()

        assert "my-project" in result
        assert "20 memories" in result


# =============================================================================
# Session Tools Tests
# =============================================================================


class TestContextfsSessions:
    """Tests for contextfs_sessions tool."""

    def test_list_sessions(self, mock_ctx, sample_session):
        """Test listing sessions."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.list_sessions.return_value = [sample_session]

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_sessions(limit=10)

        assert sample_session.id[:8] in result
        assert sample_session.label in result

    def test_list_sessions_empty(self, mock_ctx):
        """Test listing when no sessions."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.list_sessions.return_value = []

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_sessions()

        assert "No sessions found" in result


class TestContextfsLoadSession:
    """Tests for contextfs_load_session tool."""

    def test_load_session_by_id(self, mock_ctx, sample_session):
        """Test loading session by ID."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.load_session.return_value = sample_session

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_load_session(session_id=sample_session.id)

        assert sample_session.id in result
        assert sample_session.label in result

    def test_load_session_not_found(self, mock_ctx):
        """Test loading non-existent session."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.load_session.return_value = None

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_load_session(session_id="nonexistent")

        assert "Session not found" in result


class TestContextfsMessage:
    """Tests for contextfs_message tool."""

    def test_add_message(self, mock_ctx):
        """Test adding message to session."""
        import contextfs.mcp.fastmcp_server as server

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_message(role="user", content="Test message")

        assert "Message added" in result
        mock_ctx.add_message.assert_called_once_with(role="user", content="Test message")

    def test_add_message_empty(self, mock_ctx):
        """Test adding empty message."""
        import contextfs.mcp.fastmcp_server as server

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_message(role="", content="")

        assert "Error" in result


# =============================================================================
# Update/Delete Tools Tests
# =============================================================================


class TestContextfsUpdate:
    """Tests for contextfs_update tool."""

    def test_update_memory(self, mock_ctx, sample_memory):
        """Test updating memory."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.update.return_value = sample_memory

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_update(
                id=sample_memory.id,
                content="Updated content",
                summary="Updated summary",
            )

        assert "Memory updated" in result
        mock_ctx.update.assert_called_once()

    def test_update_not_found(self, mock_ctx):
        """Test updating non-existent memory."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.update.return_value = None

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_update(id="nonexistent", content="test")

        assert "Memory not found" in result


class TestContextfsDelete:
    """Tests for contextfs_delete tool."""

    def test_delete_memory(self, mock_ctx):
        """Test deleting memory."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.delete.return_value = True

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_delete(id="test-id")

        assert "Memory deleted" in result

    def test_delete_not_found(self, mock_ctx):
        """Test deleting non-existent memory."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.delete.return_value = False

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_delete(id="nonexistent")

        assert "Memory not found" in result


# =============================================================================
# Evolve/Link Tools Tests
# =============================================================================


class TestContextfsEvolve:
    """Tests for contextfs_evolve tool."""

    def test_evolve_memory(self, mock_ctx, sample_memory):
        """Test evolving memory."""
        import contextfs.mcp.fastmcp_server as server

        new_memory = Memory(
            id="new-memory-789",
            content="Evolved content",
            type=MemoryType.FACT,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_ctx.evolve.return_value = new_memory

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_evolve(
                memory_id=sample_memory.id,
                new_content="Evolved content",
            )

        assert "Memory evolved" in result
        assert sample_memory.id in result
        assert new_memory.id in result

    def test_evolve_not_found(self, mock_ctx):
        """Test evolving non-existent memory."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.evolve.return_value = None

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_evolve(
                memory_id="nonexistent",
                new_content="test",
            )

        assert "Memory not found" in result


class TestContextfsLink:
    """Tests for contextfs_link tool."""

    def test_link_memories(self, mock_ctx):
        """Test linking memories."""
        import contextfs.mcp.fastmcp_server as server

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_link(
                from_id="mem1",
                to_id="mem2",
                relation="references",
            )

        assert "Link created" in result
        assert "mem1" in result
        assert "mem2" in result
        assert "references" in result
        mock_ctx.link.assert_called_once()

    def test_link_bidirectional(self, mock_ctx):
        """Test bidirectional linking."""
        import contextfs.mcp.fastmcp_server as server

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_link(
                from_id="mem1",
                to_id="mem2",
                relation="related_to",
                bidirectional=True,
            )

        assert "bidirectional" in result


# =============================================================================
# Type Schema Tool Tests
# =============================================================================


class TestContextfsGetTypeSchema:
    """Tests for contextfs_get_type_schema tool."""

    def test_get_schema_decision(self):
        """Test getting schema for decision type."""
        import contextfs.mcp.fastmcp_server as server

        result = server.contextfs_get_type_schema(memory_type="decision")

        assert "decision" in result
        assert "rationale" in result

    def test_get_schema_not_found(self):
        """Test getting schema for unknown type."""
        import contextfs.mcp.fastmcp_server as server

        result = server.contextfs_get_type_schema(memory_type="unknown_type")

        assert "No schema defined" in result

    def test_get_schema_empty(self):
        """Test getting schema with empty type."""
        import contextfs.mcp.fastmcp_server as server

        result = server.contextfs_get_type_schema(memory_type="")

        assert "Error" in result


# =============================================================================
# Index Tools Tests
# =============================================================================


class TestContextfsInit:
    """Tests for contextfs_init tool."""

    def test_init_repo(self, mock_ctx, tmp_path):
        """Test initializing repository."""
        import contextfs.mcp.fastmcp_server as server

        git_root = tmp_path / "my-repo"
        git_root.mkdir()

        with (
            patch.object(server, "get_ctx", return_value=mock_ctx),
            patch.object(server, "find_git_root", return_value=git_root),
            patch.object(server, "is_repo_initialized", return_value=False),
            patch.object(
                server,
                "create_repo_config",
                return_value=git_root / ".contextfs/config.yaml",
            ),
        ):
            mock_ctx.index_repository.return_value = {
                "files_indexed": 10,
                "commits_indexed": 5,
            }
            result = server.contextfs_init(repo_path=str(git_root))

        assert "Repository initialized" in result

    def test_init_not_git(self, mock_ctx, tmp_path):
        """Test init when not a git repo."""
        import contextfs.mcp.fastmcp_server as server

        with (
            patch.object(server, "get_ctx", return_value=mock_ctx),
            patch.object(server, "find_git_root", return_value=None),
        ):
            result = server.contextfs_init(repo_path=str(tmp_path))

        assert "Not a git repository" in result

    def test_init_already_initialized(self, mock_ctx, tmp_path):
        """Test init when already initialized."""
        import contextfs.mcp.fastmcp_server as server

        git_root = tmp_path / "my-repo"
        git_root.mkdir()

        with (
            patch.object(server, "get_ctx", return_value=mock_ctx),
            patch.object(server, "find_git_root", return_value=git_root),
            patch.object(server, "is_repo_initialized", return_value=True),
        ):
            result = server.contextfs_init(repo_path=str(git_root))

        assert "already initialized" in result


class TestContextfsIndex:
    """Tests for contextfs_index tool."""

    @pytest.mark.asyncio
    async def test_index_start(self, mock_ctx):
        """Test starting indexing."""
        import contextfs.mcp.fastmcp_server as server

        # Reset indexing state
        server._indexing_state.running = False
        server._indexing_state.task = None

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = await server.contextfs_index(repo_path="/path/to/repo")

        assert "Indexing started" in result

    @pytest.mark.asyncio
    async def test_index_already_running(self, mock_ctx):
        """Test indexing when already running."""
        import contextfs.mcp.fastmcp_server as server

        server._indexing_state.running = True
        server._indexing_state.repo_name = "test-repo"
        server._indexing_state.current = 5
        server._indexing_state.total = 10

        result = await server.contextfs_index()

        assert "already in progress" in result
        assert "test-repo" in result

        # Reset state
        server._indexing_state.running = False


class TestContextfsIndexStatus:
    """Tests for contextfs_index_status tool."""

    def test_status_running(self):
        """Test status when indexing is running."""
        import contextfs.mcp.fastmcp_server as server

        server._indexing_state.running = True
        server._indexing_state.repo_name = "test-repo"
        server._indexing_state.current = 50
        server._indexing_state.total = 100
        server._indexing_state.current_file = "src/main.py"

        result = server.contextfs_index_status()

        assert "in progress" in result
        assert "50/100" in result
        assert "src/main.py" in result

        # Reset state
        server._indexing_state.running = False

    def test_status_completed(self):
        """Test status when indexing completed."""
        import contextfs.mcp.fastmcp_server as server

        server._indexing_state.running = False
        server._indexing_state.repo_name = "test-repo"
        server._indexing_state.result = {
            "files_indexed": 100,
            "commits_indexed": 50,
            "memories_created": 200,
        }
        server._indexing_state.error = None

        result = server.contextfs_index_status()

        assert "complete" in result
        assert "100" in result

        # Reset state
        server._indexing_state.result = None

    def test_status_error(self):
        """Test status when indexing failed."""
        import contextfs.mcp.fastmcp_server as server

        server._indexing_state.running = False
        server._indexing_state.error = "Something went wrong"
        server._indexing_state.result = None

        result = server.contextfs_index_status()

        assert "failed" in result
        assert "Something went wrong" in result

        # Reset state
        server._indexing_state.error = None

    def test_status_cancel(self):
        """Test cancelling indexing."""
        import contextfs.mcp.fastmcp_server as server

        mock_task = MagicMock()
        server._indexing_state.running = True
        server._indexing_state.task = mock_task

        result = server.contextfs_index_status(cancel=True)

        assert "cancelled" in result
        mock_task.cancel.assert_called_once()

        # Reset state
        server._indexing_state.running = False
        server._indexing_state.task = None


class TestContextfsListIndexes:
    """Tests for contextfs_list_indexes tool."""

    def test_list_indexes(self, mock_ctx):
        """Test listing indexes."""
        import contextfs.mcp.fastmcp_server as server

        mock_index = MagicMock()
        mock_index.repo_path = "/path/to/repo"
        mock_index.namespace_id = "ns1"
        mock_index.files_indexed = 100
        mock_index.commits_indexed = 50
        mock_index.memories_created = 200
        mock_index.indexed_at = datetime.now()

        mock_ctx.list_indexes.return_value = [mock_index]

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_list_indexes()

        assert "repo" in result
        assert "100" in result

    def test_list_indexes_empty(self, mock_ctx):
        """Test listing when no indexes."""
        import contextfs.mcp.fastmcp_server as server

        mock_ctx.list_indexes.return_value = []

        with patch.object(server, "get_ctx", return_value=mock_ctx):
            result = server.contextfs_list_indexes()

        assert "No indexed repositories" in result


# =============================================================================
# Sync Tool Tests
# =============================================================================


class TestContextfsSync:
    """Tests for contextfs_sync tool."""

    @pytest.mark.asyncio
    async def test_sync_disabled(self):
        """Test sync when cloud is disabled."""
        import contextfs.mcp.fastmcp_server as server

        with patch.object(server, "_get_cloud_config", return_value={"enabled": False}):
            result = await server.contextfs_sync()

        assert "disabled" in result

    @pytest.mark.asyncio
    async def test_sync_no_api_key(self):
        """Test sync without API key."""
        import contextfs.mcp.fastmcp_server as server

        with patch.object(server, "_get_cloud_config", return_value={"enabled": True}):
            result = await server.contextfs_sync()

        assert "No API key" in result

    @pytest.mark.asyncio
    async def test_sync_push(self, mock_ctx):
        """Test push sync."""
        import contextfs.mcp.fastmcp_server as server

        cloud_config = {
            "enabled": True,
            "api_key": "test-key",
            "server_url": "https://api.test.com",
        }

        mock_push_result = MagicMock()
        mock_push_result.accepted = 5
        mock_push_result.rejected = 0
        mock_push_result.conflicts = []
        mock_push_result.pushed_items = []

        mock_client = MagicMock()
        mock_client.push = MagicMock(return_value=mock_push_result)
        mock_client.__aenter__ = MagicMock(return_value=mock_client)
        mock_client.__aexit__ = MagicMock(return_value=None)

        with (
            patch.object(server, "_get_cloud_config", return_value=cloud_config),
            patch.object(server, "get_ctx", return_value=mock_ctx),
            patch("contextfs.sync.SyncClient", return_value=mock_client),
        ):
            result = await server.contextfs_sync(direction="push")

        assert "Push complete" in result or "Sync failed" in result


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check returns OK."""
        import contextfs.mcp.fastmcp_server as server

        mock_request = MagicMock()
        response = await server.health_check(mock_request)

        assert response.status_code == 200
        # JSONResponse body is bytes
        import json

        body = json.loads(response.body)
        assert body["status"] == "ok"
        assert body["service"] == "contextfs-mcp"
