"""
Integration tests for auto-indexing module.
"""

import subprocess
from pathlib import Path

import pytest


class TestAutoIndexer:
    """Tests for AutoIndexer class."""

    def test_init_creates_tables(self, temp_dir: Path):
        """Test that initialization creates required tables."""
        import sqlite3

        from contextfs.autoindex import AutoIndexer

        db_path = temp_dir / "test.db"
        AutoIndexer(db_path=db_path)  # Creates tables on init

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check index_status table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='index_status'")
        assert cursor.fetchone() is not None

        # Check indexed_files table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='indexed_files'")
        assert cursor.fetchone() is not None

        conn.close()

    def test_is_indexed_false_initially(self, temp_dir: Path):
        """Test that is_indexed returns False for new namespace."""
        from contextfs.autoindex import AutoIndexer

        indexer = AutoIndexer(db_path=temp_dir / "test.db")
        assert indexer.is_indexed("test-namespace") is False

    def test_should_index_true_for_new_namespace(self, temp_dir: Path):
        """Test that should_index returns True for new namespace."""
        from contextfs.autoindex import AutoIndexer

        indexer = AutoIndexer(db_path=temp_dir / "test.db")
        assert indexer.should_index("test-namespace") is True

    def test_discover_files_finds_python(self, temp_dir: Path, sample_python_code: str):
        """Test that discover_files finds Python files."""
        from contextfs.autoindex import AutoIndexer

        # Create test file
        test_file = temp_dir / "app.py"
        test_file.write_text(sample_python_code)

        indexer = AutoIndexer(db_path=temp_dir / "test.db")
        files = indexer.discover_files(temp_dir)

        assert len(files) == 1
        assert files[0].name == "app.py"

    def test_discover_files_respects_ignore_patterns(self, temp_dir: Path, sample_python_code: str):
        """Test that discover_files ignores node_modules, etc."""
        from contextfs.autoindex import AutoIndexer

        # Create files in ignored directory
        node_modules = temp_dir / "node_modules"
        node_modules.mkdir()
        (node_modules / "package.js").write_text("export const x = 1;")

        # Create normal file
        (temp_dir / "app.py").write_text(sample_python_code)

        indexer = AutoIndexer(db_path=temp_dir / "test.db")
        files = indexer.discover_files(temp_dir)

        # Should only find app.py, not node_modules
        assert len(files) == 1
        assert files[0].name == "app.py"

    def test_discover_files_respects_gitignore(self, temp_dir: Path, sample_python_code: str):
        """Test that discover_files respects .gitignore patterns."""
        from contextfs.autoindex import AutoIndexer

        # Create .gitignore
        (temp_dir / ".gitignore").write_text("ignored/\n*.log")

        # Create ignored files
        ignored_dir = temp_dir / "ignored"
        ignored_dir.mkdir()
        (ignored_dir / "secret.py").write_text("x = 1")
        (temp_dir / "debug.log").write_text("log content")

        # Create normal file
        (temp_dir / "app.py").write_text(sample_python_code)

        indexer = AutoIndexer(db_path=temp_dir / "test.db")
        files = indexer.discover_files(temp_dir, respect_gitignore=True)

        # Should only find app.py
        assert len(files) == 1
        assert files[0].name == "app.py"

    def test_discover_files_multiple_languages(
        self,
        temp_dir: Path,
        sample_python_code: str,
        sample_typescript_code: str,
        sample_go_code: str,
    ):
        """Test discovery across multiple programming languages."""
        from contextfs.autoindex import AutoIndexer

        (temp_dir / "main.py").write_text(sample_python_code)
        (temp_dir / "app.ts").write_text(sample_typescript_code)
        (temp_dir / "server.go").write_text(sample_go_code)

        indexer = AutoIndexer(db_path=temp_dir / "test.db")
        files = indexer.discover_files(temp_dir)

        assert len(files) == 3
        extensions = {f.suffix for f in files}
        assert extensions == {".py", ".ts", ".go"}

    def test_discover_files_skips_large_files(self, temp_dir: Path):
        """Test that files over 1MB are skipped."""
        from contextfs.autoindex import AutoIndexer

        # Create large file (> 1MB)
        large_file = temp_dir / "large.py"
        large_file.write_text("x = 1\n" * 200000)  # ~1.2MB

        # Create normal file
        (temp_dir / "small.py").write_text("x = 1")

        indexer = AutoIndexer(db_path=temp_dir / "test.db")
        files = indexer.discover_files(temp_dir)

        # Should only find small.py
        assert len(files) == 1
        assert files[0].name == "small.py"

    @pytest.mark.slow
    def test_index_repository(self, temp_dir: Path, rag_backend, sample_python_code: str):
        """Test full repository indexing."""
        from contextfs.autoindex import AutoIndexer

        # Create test files
        (temp_dir / "main.py").write_text(sample_python_code)
        (temp_dir / "utils.py").write_text("def helper(): return 42")

        indexer = AutoIndexer(db_path=temp_dir / "test.db")

        progress_calls = []

        def on_progress(current, total, file):
            progress_calls.append((current, total, file))

        stats = indexer.index_repository(
            repo_path=temp_dir,
            namespace_id="test-namespace",
            rag_backend=rag_backend,
            on_progress=on_progress,
            incremental=False,
        )

        assert stats["files_indexed"] >= 1
        assert stats["memories_created"] >= 1
        assert len(progress_calls) >= 1

    @pytest.mark.slow
    def test_index_repository_marks_as_indexed(
        self, temp_dir: Path, rag_backend, sample_python_code: str
    ):
        """Test that indexing marks namespace as indexed."""
        from contextfs.autoindex import AutoIndexer

        (temp_dir / "app.py").write_text(sample_python_code)

        indexer = AutoIndexer(db_path=temp_dir / "test.db")

        # Initially not indexed
        assert indexer.is_indexed("test-namespace") is False

        indexer.index_repository(
            repo_path=temp_dir,
            namespace_id="test-namespace",
            rag_backend=rag_backend,
        )

        # Now indexed
        assert indexer.is_indexed("test-namespace") is True

    @pytest.mark.slow
    def test_index_repository_incremental(
        self, temp_dir: Path, rag_backend, sample_python_code: str
    ):
        """Test incremental indexing skips unchanged files."""
        from contextfs.autoindex import AutoIndexer

        (temp_dir / "app.py").write_text(sample_python_code)

        indexer = AutoIndexer(db_path=temp_dir / "test.db")

        # First index
        stats1 = indexer.index_repository(
            repo_path=temp_dir,
            namespace_id="test-namespace",
            rag_backend=rag_backend,
            incremental=True,
        )
        files1 = stats1["files_indexed"]

        # Reset index status but keep file tracking
        import sqlite3

        conn = sqlite3.connect(temp_dir / "test.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE index_status SET indexed = 0")
        conn.commit()
        conn.close()

        # Second index - should skip unchanged files
        stats2 = indexer.index_repository(
            repo_path=temp_dir,
            namespace_id="test-namespace",
            rag_backend=rag_backend,
            incremental=True,
        )

        assert stats2["skipped"] >= files1

    @pytest.mark.slow
    def test_index_repository_incremental_detects_changes(
        self, temp_dir: Path, rag_backend, sample_python_code: str
    ):
        """Test incremental indexing detects file changes."""
        from contextfs.autoindex import AutoIndexer

        test_file = temp_dir / "app.py"
        test_file.write_text(sample_python_code)

        indexer = AutoIndexer(db_path=temp_dir / "test.db")

        # First index
        indexer.index_repository(
            repo_path=temp_dir,
            namespace_id="test-namespace",
            rag_backend=rag_backend,
            incremental=True,
        )

        # Modify file
        test_file.write_text(sample_python_code + "\n# Modified!")

        # Reset index status
        import sqlite3

        conn = sqlite3.connect(temp_dir / "test.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE index_status SET indexed = 0")
        conn.commit()
        conn.close()

        # Second index - should re-index modified file
        stats = indexer.index_repository(
            repo_path=temp_dir,
            namespace_id="test-namespace",
            rag_backend=rag_backend,
            incremental=True,
        )

        # Modified file should be indexed, not skipped
        assert stats["files_indexed"] >= 1

    def test_get_status(self, temp_dir: Path):
        """Test getting index status."""
        from contextfs.autoindex import AutoIndexer

        indexer = AutoIndexer(db_path=temp_dir / "test.db")

        # No status initially
        status = indexer.get_status("test-namespace")
        assert status is None

    def test_clear_index(self, temp_dir: Path, rag_backend, sample_python_code: str):
        """Test clearing index status."""
        from contextfs.autoindex import AutoIndexer

        (temp_dir / "app.py").write_text(sample_python_code)

        indexer = AutoIndexer(db_path=temp_dir / "test.db")

        # Index first
        indexer.index_repository(
            repo_path=temp_dir,
            namespace_id="test-namespace",
            rag_backend=rag_backend,
        )
        assert indexer.is_indexed("test-namespace") is True

        # Clear
        indexer.clear_index("test-namespace")
        assert indexer.is_indexed("test-namespace") is False

    def test_list_all_indexes(self, temp_dir: Path, rag_backend, sample_python_code: str):
        """Test listing all indexed repositories."""
        from contextfs.autoindex import AutoIndexer

        (temp_dir / "app.py").write_text(sample_python_code)

        indexer = AutoIndexer(db_path=temp_dir / "test.db")

        # No indexes initially
        indexes = indexer.list_all_indexes()
        assert indexes == []

        # Index a repository
        indexer.index_repository(
            repo_path=temp_dir,
            namespace_id="test-namespace-1",
            rag_backend=rag_backend,
        )

        # Should have one index
        indexes = indexer.list_all_indexes()
        assert len(indexes) == 1
        assert indexes[0].namespace_id == "test-namespace-1"
        assert indexes[0].indexed is True
        assert indexes[0].files_indexed >= 1

        # Index another "repository" (same dir, different namespace)
        indexer.index_repository(
            repo_path=temp_dir,
            namespace_id="test-namespace-2",
            rag_backend=rag_backend,
        )

        # Should have two indexes
        indexes = indexer.list_all_indexes()
        assert len(indexes) == 2
        namespace_ids = {idx.namespace_id for idx in indexes}
        assert "test-namespace-1" in namespace_ids
        assert "test-namespace-2" in namespace_ids


class TestCodebaseSummary:
    """Tests for codebase summary generation."""

    def test_create_codebase_summary(
        self,
        temp_dir: Path,
        sample_python_code: str,
        sample_typescript_code: str,
    ):
        """Test creating codebase summary."""
        from contextfs.autoindex import create_codebase_summary

        # Create test files
        src = temp_dir / "src"
        src.mkdir()
        (src / "main.py").write_text(sample_python_code)
        (src / "app.ts").write_text(sample_typescript_code)

        docs = temp_dir / "docs"
        docs.mkdir()
        (docs / "README.md").write_text("# Docs")

        memory = create_codebase_summary(temp_dir)

        assert memory.type.value == "fact"
        assert "Codebase Summary" in memory.content
        assert temp_dir.name in memory.content
        assert "src" in memory.content or "docs" in memory.content
        assert "codebase-summary" in memory.tags
        assert "auto-indexed" in memory.tags


class TestContextFSAutoIndex:
    """Tests for auto-indexing integration with ContextFS."""

    @pytest.mark.slow
    def test_auto_index_on_first_save(self, temp_dir: Path, sample_python_code: str):
        """Test that first save triggers auto-indexing."""
        from contextfs.core import ContextFS
        from contextfs.schemas import MemoryType

        # Create a mini repo structure
        repo_dir = temp_dir / "my-repo"
        repo_dir.mkdir()
        (repo_dir / "app.py").write_text(sample_python_code)

        # Initialize git repo for namespace detection
        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)

        # Create ContextFS with custom data dir
        import os

        original_cwd = os.getcwd()
        os.chdir(repo_dir)

        try:
            ctx = ContextFS(
                data_dir=temp_dir / "contextfs_data",
                auto_index=True,
            )

            # First save should trigger auto-indexing
            ctx.save(
                content="Test memory",
                type=MemoryType.FACT,
            )

            # Verify auto-indexing occurred
            status = ctx.get_index_status()
            assert status is not None
            assert status.indexed is True

        finally:
            os.chdir(original_cwd)
            ctx.close()

    @pytest.mark.slow
    def test_auto_index_disabled(self, temp_dir: Path, sample_python_code: str):
        """Test that auto_index=False disables auto-indexing."""
        from contextfs.core import ContextFS
        from contextfs.schemas import MemoryType

        repo_dir = temp_dir / "my-repo"
        repo_dir.mkdir()
        (repo_dir / "app.py").write_text(sample_python_code)
        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)

        import os

        original_cwd = os.getcwd()
        os.chdir(repo_dir)

        try:
            ctx = ContextFS(
                data_dir=temp_dir / "contextfs_data",
                auto_index=False,
            )

            ctx.save(
                content="Test memory",
                type=MemoryType.FACT,
            )

            # Auto-indexing should not have occurred
            status = ctx.get_index_status()
            assert status is None

        finally:
            os.chdir(original_cwd)
            ctx.close()

    @pytest.mark.slow
    def test_manual_index_repository(self, temp_dir: Path, sample_python_code: str):
        """Test manual index_repository method."""
        from contextfs.core import ContextFS

        repo_dir = temp_dir / "my-repo"
        repo_dir.mkdir()
        (repo_dir / "main.py").write_text(sample_python_code)
        (repo_dir / "utils.py").write_text("def helper(): return 1")
        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)

        import os

        original_cwd = os.getcwd()
        os.chdir(repo_dir)

        try:
            ctx = ContextFS(
                data_dir=temp_dir / "contextfs_data",
                auto_index=False,
            )

            progress_updates = []

            def on_progress(current, total, file):
                progress_updates.append(file)

            stats = ctx.index_repository(on_progress=on_progress)

            assert stats["files_indexed"] >= 1
            assert len(progress_updates) >= 1

        finally:
            os.chdir(original_cwd)
            ctx.close()


class TestGitHistoryIndexing:
    """Tests for git history indexing."""

    def test_index_git_history(self, temp_dir: Path, rag_backend):
        """Test indexing git commit history."""
        from contextfs.autoindex import AutoIndexer

        # Create a git repo with commits
        repo_dir = temp_dir / "my-repo"
        repo_dir.mkdir()

        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"], cwd=repo_dir, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"], cwd=repo_dir, capture_output=True
        )

        # Create files and commits
        (repo_dir / "app.py").write_text("print('hello')")
        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "feat: Initial commit"], cwd=repo_dir, capture_output=True
        )

        (repo_dir / "utils.py").write_text("def helper(): pass")
        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "fix: Add helper function"], cwd=repo_dir, capture_output=True
        )

        indexer = AutoIndexer(db_path=temp_dir / "test.db")

        stats = indexer.index_git_history(
            repo_path=repo_dir,
            namespace_id="test-namespace",
            rag_backend=rag_backend,
            max_commits=10,
        )

        assert stats["commits_indexed"] == 2
        assert stats["memories_created"] == 2

    def test_extract_commit_tags_conventional(self, temp_dir: Path):
        """Test extracting tags from conventional commits."""
        from contextfs.autoindex import AutoIndexer

        indexer = AutoIndexer(db_path=temp_dir / "test.db")

        # Feature commit
        tags = indexer._extract_commit_tags(
            {
                "hash": "abc123",
                "author": "Test",
                "date": "2024-01-01",
                "message": "feat: Add new feature",
                "files": ["app.py"],
            }
        )
        assert "feature" in tags

        # Bugfix commit
        tags = indexer._extract_commit_tags(
            {
                "hash": "def456",
                "author": "Test",
                "date": "2024-01-01",
                "message": "fix: Fix bug",
                "files": ["utils.ts"],
            }
        )
        assert "bugfix" in tags

    def test_git_history_with_file_changes(self, temp_dir: Path, rag_backend):
        """Test git history includes file change information."""
        from contextfs.autoindex import AutoIndexer

        repo_dir = temp_dir / "my-repo"
        repo_dir.mkdir()

        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"], cwd=repo_dir, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"], cwd=repo_dir, capture_output=True
        )

        # Create initial commit first
        (repo_dir / "init.py").write_text("# Init")
        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, capture_output=True)

        # Create multiple files in second commit (so diff-tree works)
        (repo_dir / "main.py").write_text("# Main")
        (repo_dir / "utils.py").write_text("# Utils")
        (repo_dir / "config.json").write_text("{}")
        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "chore: Add multiple files"], cwd=repo_dir, capture_output=True
        )

        indexer = AutoIndexer(db_path=temp_dir / "test.db")
        commits = indexer._get_git_commits(repo_dir, max_commits=10)

        assert len(commits) == 2
        # The second commit should show the 3 new files
        assert len(commits[0]["files"]) == 3  # Most recent commit first

    @pytest.mark.slow
    def test_full_index_includes_git_history(
        self, temp_dir: Path, rag_backend, sample_python_code: str
    ):
        """Test full repository index includes git history."""
        from contextfs.autoindex import AutoIndexer

        repo_dir = temp_dir / "my-repo"
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
        subprocess.run(
            ["git", "commit", "-m", "feat: Initial implementation"],
            cwd=repo_dir,
            capture_output=True,
        )

        indexer = AutoIndexer(db_path=temp_dir / "test.db")

        stats = indexer.index_repository(
            repo_path=repo_dir,
            namespace_id="test-namespace",
            rag_backend=rag_backend,
        )

        assert stats["files_indexed"] >= 1
        assert stats["commits_indexed"] >= 1
        assert "commits_indexed" in stats


class TestLanguageDetection:
    """Tests for intelligent language detection."""

    def test_detects_python_files(self, temp_dir: Path, sample_python_code: str):
        """Test Python file detection."""
        from contextfs.autoindex import AutoIndexer

        (temp_dir / "app.py").write_text(sample_python_code)
        (temp_dir / "utils.py").write_text("def foo(): pass")

        indexer = AutoIndexer(db_path=temp_dir / "test.db")
        files = indexer.discover_files(temp_dir)

        assert len(files) == 2
        assert all(f.suffix == ".py" for f in files)

    def test_detects_web_files(self, temp_dir: Path, sample_typescript_code: str):
        """Test web technology file detection."""
        from contextfs.autoindex import AutoIndexer

        (temp_dir / "app.tsx").write_text(sample_typescript_code)
        (temp_dir / "styles.css").write_text("body { margin: 0; }")
        (temp_dir / "index.html").write_text("<html></html>")

        indexer = AutoIndexer(db_path=temp_dir / "test.db")
        files = indexer.discover_files(temp_dir)

        extensions = {f.suffix for f in files}
        assert ".tsx" in extensions
        assert ".css" in extensions
        assert ".html" in extensions

    def test_detects_config_files(self, temp_dir: Path, sample_json: str):
        """Test config file detection."""
        from contextfs.autoindex import AutoIndexer

        (temp_dir / "package.json").write_text(sample_json)
        (temp_dir / "config.yaml").write_text("key: value")
        (temp_dir / "settings.toml").write_text("[section]\nkey = 'value'")

        indexer = AutoIndexer(db_path=temp_dir / "test.db")
        files = indexer.discover_files(temp_dir)

        extensions = {f.suffix for f in files}
        assert ".json" in extensions
        assert ".yaml" in extensions
        assert ".toml" in extensions

    def test_detects_systems_languages(
        self,
        temp_dir: Path,
        sample_go_code: str,
        sample_rust_code: str,
    ):
        """Test systems programming language detection."""
        from contextfs.autoindex import AutoIndexer

        (temp_dir / "main.go").write_text(sample_go_code)
        (temp_dir / "lib.rs").write_text(sample_rust_code)
        (temp_dir / "helper.c").write_text("int main() { return 0; }")
        (temp_dir / "utils.cpp").write_text("int main() { return 0; }")

        indexer = AutoIndexer(db_path=temp_dir / "test.db")
        files = indexer.discover_files(temp_dir)

        extensions = {f.suffix for f in files}
        assert ".go" in extensions
        assert ".rs" in extensions
        assert ".c" in extensions
        assert ".cpp" in extensions
