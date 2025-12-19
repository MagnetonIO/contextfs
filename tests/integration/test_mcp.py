"""
Integration tests for MCP server.
"""

import subprocess
import time
from pathlib import Path

import pytest


class TestIndexStatus:
    """Tests for IndexStatus attribute access (bug fix verification)."""

    def test_index_status_has_attributes(self):
        """Test that IndexStatus object has expected attributes (not dict methods)."""
        from contextfs.autoindex import IndexStatus

        status = IndexStatus(
            namespace_id="test-ns",
            indexed=True,
            files_indexed=10,
        )

        # These should work (attribute access)
        assert status.indexed is True
        assert status.files_indexed == 10
        assert status.namespace_id == "test-ns"

        # This should NOT work (dict access) - verifies it's not a dict
        assert not hasattr(status, "get")

    def test_get_index_status_returns_object(self, temp_dir: Path):
        """Test that ContextFS.get_index_status returns IndexStatus object."""
        from contextfs.autoindex import IndexStatus
        from contextfs.core import ContextFS

        data_dir = temp_dir / "contextfs_data"
        ctx = ContextFS(data_dir=data_dir, auto_index=False)

        status = ctx.get_index_status()

        # Status may be None if not indexed, or IndexStatus if indexed
        if status is not None:
            assert isinstance(status, IndexStatus)
            assert hasattr(status, "indexed")
            assert hasattr(status, "files_indexed")
            assert not hasattr(status, "get")  # Not a dict

        ctx.close()


class TestCLIBackgroundIndex:
    """Tests for CLI background indexing flag."""

    def test_background_flag_returns_immediately(self, temp_dir: Path):
        """Test that --background flag spawns subprocess and returns quickly."""
        import sys

        # Create a directory with some files
        repo_dir = temp_dir / "test-repo"
        repo_dir.mkdir()
        (repo_dir / "test.py").write_text("def foo(): pass")

        # Initialize git
        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"], cwd=repo_dir, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"], cwd=repo_dir, capture_output=True
        )

        start_time = time.time()

        # Run with --background flag
        result = subprocess.run(
            [sys.executable, "-m", "contextfs.cli", "index", "--background", "--quiet"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,  # Should return within 5 seconds
        )

        elapsed = time.time() - start_time

        # Should return quickly (< 2 seconds) since it spawns background process
        assert elapsed < 2.0, f"Background index took too long: {elapsed}s"
        assert result.returncode == 0


class TestMCPServerTools:
    """Tests for MCP server tool functions."""

    @pytest.fixture
    def git_repo(self, temp_dir: Path, sample_python_code: str):
        """Create a temporary git repo for testing."""
        repo_dir = temp_dir / "test-repo"
        repo_dir.mkdir()

        # Initialize git
        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"], cwd=repo_dir, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"], cwd=repo_dir, capture_output=True
        )

        # Add sample files
        (repo_dir / "app.py").write_text(sample_python_code)
        (repo_dir / "utils.py").write_text("def helper(): return 42")

        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, capture_output=True)

        return repo_dir

    def test_detect_current_repo(self, git_repo: Path):
        """Test detect_current_repo function."""
        import os

        from contextfs.mcp_server import detect_current_repo

        original_cwd = os.getcwd()
        os.chdir(git_repo)

        try:
            repo_name = detect_current_repo()
            assert repo_name == "test-repo"
        finally:
            os.chdir(original_cwd)

    def test_detect_current_repo_not_in_repo(self, temp_dir: Path):
        """Test detect_current_repo returns None outside git repo."""
        import os

        from contextfs.mcp_server import detect_current_repo

        # Create a non-git directory
        non_git_dir = temp_dir / "not-a-repo"
        non_git_dir.mkdir()

        original_cwd = os.getcwd()
        os.chdir(non_git_dir)

        try:
            repo_name = detect_current_repo()
            assert repo_name is None
        finally:
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_list_tools_includes_index(self):
        """Test that list_tools includes contextfs_index."""
        from contextfs.mcp_server import list_tools

        tools = await list_tools()
        tool_names = [t.name for t in tools]

        assert "contextfs_index" in tool_names
        assert "contextfs_save" in tool_names

    @pytest.mark.asyncio
    async def test_list_prompts_includes_save_memory(self):
        """Test that list_prompts includes new prompts."""
        from contextfs.mcp_server import list_prompts

        prompts = await list_prompts()
        prompt_names = [p.name for p in prompts]

        assert "contextfs-save-memory" in prompt_names
        assert "contextfs-index" in prompt_names

    @pytest.mark.asyncio
    async def test_get_prompt_save_memory(self):
        """Test contextfs-save-memory prompt content."""
        from contextfs.mcp_server import get_prompt

        result = await get_prompt(
            "contextfs-save-memory", {"content": "Test content", "type": "fact"}
        )

        assert result.description == "Save Memory to ContextFS"
        assert len(result.messages) == 1
        assert "Test content" in result.messages[0].content.text
        assert "fact" in result.messages[0].content.text

    @pytest.mark.asyncio
    async def test_get_prompt_index(self):
        """Test contextfs-index prompt content."""
        from contextfs.mcp_server import get_prompt

        result = await get_prompt("contextfs-index", {})

        assert result.description == "Index Repository"
        assert "contextfs_index" in result.messages[0].content.text

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_call_tool_index(self, git_repo: Path, temp_dir: Path):
        """Test contextfs_index tool call."""
        import asyncio
        import os

        import contextfs.mcp_server as mcp_module
        from contextfs.core import ContextFS

        # Create a fresh ContextFS instance with test data dir
        data_dir = temp_dir / "contextfs_data"
        data_dir.mkdir(parents=True, exist_ok=True)

        original_cwd = os.getcwd()
        os.chdir(git_repo)

        try:
            # Create isolated ContextFS instance
            test_ctx = ContextFS(data_dir=data_dir, auto_index=False)
            mcp_module._ctx = test_ctx
            mcp_module._session_started = False
            mcp_module._indexing_state = mcp_module.IndexingState()

            from contextfs.mcp_server import call_tool

            # Start indexing (runs in background)
            result = await call_tool("contextfs_index", {"incremental": True})
            assert len(result) == 1
            text = result[0].text
            assert "Started indexing" in text or "already indexed" in text

            # Wait for background indexing to complete
            if "Started indexing" in text:
                for _ in range(30):  # Wait up to 30 seconds
                    await asyncio.sleep(1)
                    status = await call_tool("contextfs_index_status", {})
                    status_text = status[0].text
                    if "complete" in status_text or "No indexing" in status_text:
                        break
                assert "complete" in status_text or "No indexing" in status_text
        finally:
            os.chdir(original_cwd)
            if mcp_module._ctx:
                mcp_module._ctx.close()
            mcp_module._ctx = None
            mcp_module._session_started = False
            mcp_module._indexing_state = mcp_module.IndexingState()

    @pytest.mark.asyncio
    async def test_call_tool_index_not_in_repo(self, temp_dir: Path):
        """Test contextfs_index fails gracefully outside git repo."""
        import os

        import contextfs.mcp_server as mcp_module
        from contextfs.mcp_server import call_tool

        mcp_module._ctx = None

        non_git_dir = temp_dir / "not-a-repo"
        non_git_dir.mkdir()

        original_cwd = os.getcwd()
        os.chdir(non_git_dir)

        try:
            result = await call_tool("contextfs_index", {})

            assert len(result) == 1
            assert "Not in a git repository" in result[0].text
        finally:
            os.chdir(original_cwd)
            mcp_module._ctx = None

    @pytest.mark.asyncio
    async def test_list_tools_includes_rebuild_and_reindex(self):
        """Test that list_tools includes rebuild and reindex tools."""
        from contextfs.mcp_server import list_tools

        tools = await list_tools()
        tool_names = [t.name for t in tools]

        assert "contextfs_rebuild_chroma" in tool_names
        assert "contextfs_reindex" in tool_names


class TestReindexAllRepos:
    """Tests for reindex_all_repos core method."""

    def test_reindex_all_repos_empty(self, temp_dir: Path):
        """Test reindex_all_repos with no repos in database."""
        from contextfs.core import ContextFS

        data_dir = temp_dir / "contextfs_data"
        ctx = ContextFS(data_dir=data_dir, auto_index=False)

        result = ctx.reindex_all_repos()

        assert result["success"] is True
        assert result["repos_found"] == 0
        assert result["repos_indexed"] == 0

        ctx.close()

    def test_reindex_all_repos_with_repo(self, temp_dir: Path, sample_python_code: str):
        """Test reindex_all_repos with an existing indexed repo."""
        import subprocess

        from contextfs.core import ContextFS

        # Create and index a repo first
        repo_dir = temp_dir / "test-repo"
        repo_dir.mkdir()

        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"], cwd=repo_dir, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"], cwd=repo_dir, capture_output=True
        )
        (repo_dir / "app.py").write_text(sample_python_code)
        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=repo_dir, capture_output=True)

        data_dir = temp_dir / "contextfs_data"
        ctx = ContextFS(data_dir=data_dir, auto_index=False)

        # Index the repo first
        ctx.index_repository(repo_path=repo_dir)

        # Now try to reindex all - but it won't find the repo since temp path
        # isn't in the common search paths
        result = ctx.reindex_all_repos()

        # Will report the repo but fail to find path (since temp dir not in common paths)
        assert result["repos_found"] >= 1

        ctx.close()
