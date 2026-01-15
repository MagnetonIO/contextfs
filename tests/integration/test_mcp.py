"""
Integration tests for MCP server.
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest


def get_python_executable() -> str:
    """Get the Python executable that has contextfs installed.

    When running under uv, sys.executable may point to pyenv Python,
    but the actual venv Python is at sys.prefix/bin/python.
    """
    venv_python = Path(sys.prefix) / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


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

        # Run with --background flag (now: contextfs index index --background)
        result = subprocess.run(
            [
                get_python_executable(),
                "-m",
                "contextfs.cli",
                "index",
                "index",
                "--background",
                "--quiet",
            ],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,  # Should return within 5 seconds
        )

        elapsed = time.time() - start_time

        # Should return quickly (< 2 seconds) since it spawns background process
        assert elapsed < 2.0, f"Background index took too long: {elapsed}s"
        assert result.returncode == 0


class TestDetectCurrentRepo:
    """Tests for detect_current_repo function."""

    @pytest.fixture
    def git_repo(self, temp_dir: Path):
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
        (repo_dir / "app.py").write_text("def foo(): pass")
        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, capture_output=True)

        return repo_dir

    def test_detect_current_repo(self, git_repo: Path):
        """Test detect_current_repo function."""
        import os

        from contextfs.mcp.server import detect_current_repo

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

        from contextfs.mcp.server import detect_current_repo

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
