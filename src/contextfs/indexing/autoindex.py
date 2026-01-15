"""
Auto-indexing module for ContextFS.

Automatically indexes repository files on first memory save,
creating a searchable knowledge base of the codebase.
"""

import logging
import os
import sqlite3
import subprocess
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from pathlib import Path

from contextfs.config import Config
from contextfs.filetypes.integration import SmartDocumentProcessor
from contextfs.filetypes.registry import FileTypeRegistry
from contextfs.indexing.discovery import (
    DEFAULT_INDEX_EXTENSIONS,
    discover_git_repos,
)
from contextfs.rag import RAGBackend
from contextfs.schemas import Memory, MemoryType
from contextfs.storage_router import StorageRouter


class IndexMode(Enum):
    """Indexing modes for selective indexing."""

    ALL = "all"  # Index both files and commits
    FILES_ONLY = "files_only"  # Only index files
    COMMITS_ONLY = "commits_only"  # Only index git commits


logger = logging.getLogger(__name__)


# Default directories and files to ignore
DEFAULT_IGNORE_PATTERNS = {
    # Package managers & dependencies (all languages)
    "node_modules",
    "vendor",
    "packages",
    ".pnpm",
    "bower_components",
    "jspm_packages",
    ".yarn",
    ".npm",
    "site-packages",
    "Pods",  # iOS CocoaPods
    "Carthage",  # iOS Carthage
    # Build outputs (all languages)
    "dist",
    "build",
    "out",
    "target",
    "_build",
    "bin",
    "obj",
    ".next",
    ".nuxt",
    ".output",
    ".svelte-kit",
    ".vercel",
    ".netlify",
    ".parcel-cache",
    ".turbo",
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    "coverage",
    ".coverage",
    "htmlcov",
    ".nyc_output",
    # Lock files (large, auto-generated)
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "composer.lock",
    "Gemfile.lock",
    "poetry.lock",
    "Cargo.lock",
    "go.sum",
    # IDE/Editor
    ".idea",
    ".vscode",
    ".vs",
    "*.swp",
    "*.swo",
    # Version control
    ".git",
    ".svn",
    ".hg",
    # Virtual environments
    "venv",
    ".venv",
    "env",
    ".env",
    "virtualenv",
    # Dependencies & caches
    ".tox",
    ".nox",
    ".eggs",
    "*.egg-info",
    ".cache",
    ".gradle",
    ".m2",
    ".ivy2",
    # Misc
    ".DS_Store",
    "Thumbs.db",
    "*.log",
    # Large/minified files
    "*.min.js",
    "*.min.css",
    "*.bundle.js",
    "*.chunk.js",
    "*.map",  # Source maps
    # Database files
    "*.db",
    "*.sqlite",
    "*.sqlite3",
    # Generated documentation
    "docs/_build",
    "site",  # MkDocs output
    "_site",  # Jekyll output
    # TeX/LaTeX build artifacts
    "*.aux",
    "*.bbl",
    "*.blg",
    "*.fdb_latexmk",
    "*.fls",
    "*.lof",
    "*.lot",
    "*.out",
    "*.toc",
    "*.synctex.gz",
    "*.bcf",
    "*.run.xml",
    "_minted*",  # minted package output
}


class IndexStatus:
    """Tracks indexing status for a namespace."""

    def __init__(
        self,
        namespace_id: str,
        indexed: bool = False,
        indexed_at: datetime | None = None,
        files_indexed: int = 0,
        commits_indexed: int = 0,
        memories_created: int = 0,
        repo_path: str | None = None,
        commit_hash: str | None = None,
    ):
        self.namespace_id = namespace_id
        self.indexed = indexed
        self.indexed_at = indexed_at
        self.files_indexed = files_indexed
        self.commits_indexed = commits_indexed
        self.memories_created = memories_created
        self.repo_path = repo_path
        self.commit_hash = commit_hash


class AutoIndexer:
    """
    Automatic codebase indexing on first memory save.

    Features:
    - Intelligent file discovery (respects .gitignore)
    - Incremental indexing (only new/changed files)
    - Progress callbacks for UI integration
    - Configurable ignore patterns
    """

    def __init__(
        self,
        config: Config | None = None,
        db_path: Path | None = None,
        ignore_patterns: set[str] | None = None,
        extensions: set[str] | None = None,
    ):
        """
        Initialize auto-indexer.

        Args:
            config: ContextFS configuration
            db_path: Path to SQLite database
            ignore_patterns: Patterns to ignore (directories/files)
            extensions: File extensions to index
        """
        self.config = config or Config()
        self.db_path = db_path or (self.config.data_dir / "context.db")
        self.ignore_patterns = ignore_patterns or DEFAULT_IGNORE_PATTERNS
        self.extensions = extensions or DEFAULT_INDEX_EXTENSIONS
        self.processor = SmartDocumentProcessor()
        self.registry = FileTypeRegistry()

        self._init_db()

    def _get_max_commits(self) -> int:
        """Get max commits from config (0 means unlimited, returns large number)."""
        max_commits = self.config.max_commits
        return max_commits if max_commits > 0 else 999999

    def _init_db(self) -> None:
        """Initialize index tracking table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS index_status (
                namespace_id TEXT PRIMARY KEY,
                indexed INTEGER DEFAULT 0,
                indexed_at TEXT,
                files_indexed INTEGER DEFAULT 0,
                commits_indexed INTEGER DEFAULT 0,
                memories_created INTEGER DEFAULT 0,
                repo_path TEXT,
                commit_hash TEXT,
                metadata TEXT,
                namespace_source TEXT,
                remote_url TEXT
            )
        """)

        # Migration: add columns if missing
        cursor.execute("PRAGMA table_info(index_status)")
        columns = {row[1] for row in cursor.fetchall()}
        if "commits_indexed" not in columns:
            cursor.execute("ALTER TABLE index_status ADD COLUMN commits_indexed INTEGER DEFAULT 0")
        if "namespace_source" not in columns:
            cursor.execute("ALTER TABLE index_status ADD COLUMN namespace_source TEXT")
        if "remote_url" not in columns:
            cursor.execute("ALTER TABLE index_status ADD COLUMN remote_url TEXT")

        # Track indexed files for incremental updates
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indexed_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                memories_created INTEGER DEFAULT 0,
                UNIQUE(namespace_id, file_path)
            )
        """)

        # Track indexed commits for incremental updates
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indexed_commits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace_id TEXT NOT NULL,
                commit_hash TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                UNIQUE(namespace_id, commit_hash)
            )
        """)

        conn.commit()
        conn.close()

    def is_indexed(self, namespace_id: str) -> bool:
        """Check if namespace has been indexed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT indexed FROM index_status WHERE namespace_id = ?", (namespace_id,))
        row = cursor.fetchone()
        conn.close()

        return bool(row and row[0])

    def get_status(self, namespace_id: str) -> IndexStatus | None:
        """Get indexing status for namespace."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT namespace_id, indexed, indexed_at, files_indexed,
                   commits_indexed, memories_created, repo_path, commit_hash
            FROM index_status WHERE namespace_id = ?
        """,
            (namespace_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return IndexStatus(
            namespace_id=row[0],
            indexed=bool(row[1]),
            indexed_at=datetime.fromisoformat(row[2]) if row[2] else None,
            files_indexed=row[3],
            commits_indexed=row[4] or 0,
            memories_created=row[5],
            repo_path=row[6],
            commit_hash=row[7],
        )

    def list_all_indexes(self) -> list[IndexStatus]:
        """List all indexed repositories."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT namespace_id, indexed, indexed_at, files_indexed,
                   commits_indexed, memories_created, repo_path, commit_hash
            FROM index_status
            WHERE indexed = 1
            ORDER BY indexed_at DESC
        """)
        rows = cursor.fetchall()
        conn.close()

        return [
            IndexStatus(
                namespace_id=row[0],
                indexed=bool(row[1]),
                indexed_at=datetime.fromisoformat(row[2]) if row[2] else None,
                files_indexed=row[3],
                commits_indexed=row[4] or 0,
                memories_created=row[5],
                repo_path=row[6],
                commit_hash=row[7],
            )
            for row in rows
        ]

    def cleanup_stale_indexes(self, dry_run: bool = False, require_git: bool = True) -> dict:
        """
        Remove indexes for repositories that no longer exist on disk.

        Args:
            dry_run: If True, only report what would be deleted without deleting
            require_git: If True, also remove indexes for paths without .git directory

        Returns:
            Dict with 'removed' (list of removed indexes) and 'kept' (list of valid indexes)
        """
        indexes = self.list_all_indexes()
        removed = []
        kept = []

        for idx in indexes:
            repo_path = Path(idx.repo_path) if idx.repo_path else None
            is_stale = False
            reason = None

            if not idx.repo_path:
                # No path stored - mark as stale
                is_stale = True
                reason = "no_path"
            elif not repo_path.exists():
                # Path doesn't exist anymore
                is_stale = True
                reason = "path_missing"
            elif require_git and not (repo_path / ".git").exists():
                # Not a git repo (might be a subdir that was indexed)
                is_stale = True
                reason = "not_git_repo"

            if is_stale:
                removed.append((idx, reason))
            else:
                kept.append(idx)

        if not dry_run:
            for idx, reason in removed:
                self.delete_index(idx.namespace_id)

        return {
            "removed": [
                {
                    "namespace_id": idx.namespace_id,
                    "repo_path": idx.repo_path,
                    "files_indexed": idx.files_indexed,
                    "commits_indexed": idx.commits_indexed,
                    "reason": reason,
                }
                for idx, reason in removed
            ],
            "kept": [
                {
                    "namespace_id": idx.namespace_id,
                    "repo_path": idx.repo_path,
                    "files_indexed": idx.files_indexed,
                    "commits_indexed": idx.commits_indexed,
                }
                for idx in kept
            ],
        }

    def delete_index(self, namespace_id: str) -> bool:
        """
        Delete an index by namespace ID.

        Args:
            namespace_id: The namespace ID to delete

        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if it exists
        cursor.execute(
            "SELECT namespace_id FROM index_status WHERE namespace_id = ?",
            (namespace_id,),
        )
        if not cursor.fetchone():
            conn.close()
            return False

        # Delete from index_status
        cursor.execute("DELETE FROM index_status WHERE namespace_id = ?", (namespace_id,))

        # Delete from indexed_files
        cursor.execute("DELETE FROM indexed_files WHERE namespace_id = ?", (namespace_id,))

        # Delete from indexed_commits
        cursor.execute("DELETE FROM indexed_commits WHERE namespace_id = ?", (namespace_id,))

        conn.commit()
        conn.close()

        logger.info(f"Deleted index for namespace: {namespace_id}")
        return True

    def delete_index_by_path(self, repo_path: str | Path) -> bool:
        """
        Delete an index by repository path.

        Args:
            repo_path: The repository path to find and delete

        Returns:
            True if deleted, False if not found
        """
        repo_path_str = str(Path(repo_path).resolve())

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT namespace_id FROM index_status WHERE repo_path = ?",
            (repo_path_str,),
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return self.delete_index(row[0])
        return False

    def update_repo_path(
        self,
        namespace_id: str | None = None,
        old_path: str | Path | None = None,
        new_path: str | Path | None = None,
    ) -> dict:
        """
        Update the repository path for an existing index.

        Use this when a repository has been moved to a new location.
        The namespace_id remains the same, preserving all indexed memories.

        Args:
            namespace_id: The namespace ID to update (preferred)
            old_path: The old repository path to find the index (alternative)
            new_path: The new repository path

        Returns:
            Dict with status and updated info
        """
        if not new_path:
            return {"success": False, "error": "new_path is required"}

        new_path_str = str(Path(new_path).resolve())

        # Verify new path exists and is a git repo
        new_path_obj = Path(new_path_str)
        if not new_path_obj.exists():
            return {"success": False, "error": f"New path does not exist: {new_path_str}"}
        if not (new_path_obj / ".git").exists():
            return {"success": False, "error": f"New path is not a git repository: {new_path_str}"}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Find the index to update
        if namespace_id:
            cursor.execute(
                "SELECT namespace_id, repo_path, files_indexed, commits_indexed, memories_created FROM index_status WHERE namespace_id = ?",
                (namespace_id,),
            )
        elif old_path:
            old_path_str = str(Path(old_path).resolve())
            cursor.execute(
                "SELECT namespace_id, repo_path, files_indexed, commits_indexed, memories_created FROM index_status WHERE repo_path = ?",
                (old_path_str,),
            )
        else:
            conn.close()
            return {"success": False, "error": "Either namespace_id or old_path is required"}

        row = cursor.fetchone()
        if not row:
            conn.close()
            return {"success": False, "error": "Index not found"}

        found_namespace_id = row[0]
        old_path_stored = row[1]
        files_indexed = row[2]
        commits_indexed = row[3]
        memories_created = row[4]

        # Update the repo_path
        cursor.execute(
            "UPDATE index_status SET repo_path = ? WHERE namespace_id = ?",
            (new_path_str, found_namespace_id),
        )

        conn.commit()
        conn.close()

        logger.info(
            f"Updated repo path for {found_namespace_id}: {old_path_stored} -> {new_path_str}"
        )

        return {
            "success": True,
            "namespace_id": found_namespace_id,
            "old_path": old_path_stored,
            "new_path": new_path_str,
            "files_indexed": files_indexed,
            "commits_indexed": commits_indexed or 0,
            "memories_created": memories_created,
        }

    def find_index_by_repo_name(self, repo_name: str) -> list[dict]:
        """
        Find indexes that match a repository name.

        Useful when you know the repo name but not the exact path.

        Args:
            repo_name: Repository directory name to search for

        Returns:
            List of matching indexes with their info
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Search for repos where the path ends with the given name
        cursor.execute(
            """
            SELECT namespace_id, repo_path, files_indexed, commits_indexed, memories_created, indexed_at
            FROM index_status
            WHERE repo_path LIKE ?
            """,
            (f"%/{repo_name}",),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "namespace_id": row[0],
                    "repo_path": row[1],
                    "files_indexed": row[2],
                    "commits_indexed": row[3] or 0,
                    "memories_created": row[4],
                    "indexed_at": row[5],
                }
            )

        conn.close()
        return results

    def migrate_namespace(
        self,
        old_namespace_id: str,
        new_namespace_id: str,
        new_source: str | None = None,
        new_remote_url: str | None = None,
        storage: "StorageRouter | None" = None,
    ) -> dict:
        """
        Migrate an index from one namespace ID to another.

        Updates:
        - index_status table
        - indexed_files table
        - indexed_commits table
        - memories table (namespace_id)
        - ChromaDB metadata (via storage)

        Args:
            old_namespace_id: Current namespace ID
            new_namespace_id: New namespace ID to migrate to
            new_source: Namespace derivation source (explicit, git_remote, path)
            new_remote_url: Git remote URL if available
            storage: StorageRouter for ChromaDB updates

        Returns:
            Migration statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check old namespace exists
        cursor.execute(
            "SELECT repo_path, files_indexed, commits_indexed, memories_created FROM index_status WHERE namespace_id = ?",
            (old_namespace_id,),
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            return {"success": False, "error": f"Namespace {old_namespace_id} not found"}

        repo_path = row[0]
        # Unused but kept for reference: files_indexed, commits_indexed, memories_created
        _, _, _ = row[1], row[2] or 0, row[3]

        # Check new namespace doesn't already exist
        cursor.execute(
            "SELECT namespace_id FROM index_status WHERE namespace_id = ?",
            (new_namespace_id,),
        )
        if cursor.fetchone():
            conn.close()
            return {
                "success": False,
                "error": f"Target namespace {new_namespace_id} already exists",
            }

        try:
            # Update index_status
            cursor.execute(
                """
                UPDATE index_status
                SET namespace_id = ?, namespace_source = ?, remote_url = ?
                WHERE namespace_id = ?
                """,
                (new_namespace_id, new_source, new_remote_url, old_namespace_id),
            )

            # Update indexed_files
            cursor.execute(
                "UPDATE indexed_files SET namespace_id = ? WHERE namespace_id = ?",
                (new_namespace_id, old_namespace_id),
            )
            files_updated = cursor.rowcount

            # Update indexed_commits
            cursor.execute(
                "UPDATE indexed_commits SET namespace_id = ? WHERE namespace_id = ?",
                (new_namespace_id, old_namespace_id),
            )
            commits_updated = cursor.rowcount

            # Update memories table
            cursor.execute(
                "UPDATE memories SET namespace_id = ? WHERE namespace_id = ?",
                (new_namespace_id, old_namespace_id),
            )
            memories_updated = cursor.rowcount

            conn.commit()

            # Update ChromaDB if storage provided
            chroma_updated = 0
            if storage:
                try:
                    chroma_updated = storage.update_namespace_in_chroma(
                        old_namespace_id, new_namespace_id
                    )
                except Exception as e:
                    logger.warning(f"ChromaDB namespace update failed: {e}")

            logger.info(
                f"Migrated namespace {old_namespace_id} -> {new_namespace_id}: "
                f"{memories_updated} memories, {files_updated} files, {commits_updated} commits"
            )

            return {
                "success": True,
                "old_namespace_id": old_namespace_id,
                "new_namespace_id": new_namespace_id,
                "repo_path": repo_path,
                "namespace_source": new_source,
                "remote_url": new_remote_url,
                "memories_migrated": memories_updated,
                "files_migrated": files_updated,
                "commits_migrated": commits_updated,
                "chroma_updated": chroma_updated,
            }

        except Exception as e:
            conn.rollback()
            conn.close()
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    def get_migration_candidates(self) -> list[dict]:
        """
        Find indexes that should be migrated from path-based to git-remote-based.

        Returns list of indexes where:
        - namespace_id starts with 'repo-' (old path-based)
        - repo_path exists and has a git remote
        - Current namespace_id doesn't match what git remote would generate
        """
        from contextfs.schemas import (
            NamespaceSource,
            _get_git_remote_url,
            _normalize_git_url,
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT namespace_id, repo_path, files_indexed, commits_indexed, memories_created, namespace_source
            FROM index_status
            WHERE indexed = 1 AND repo_path IS NOT NULL
        """)

        candidates = []
        for row in cursor.fetchall():
            old_ns_id = row[0]
            repo_path = row[1]
            files_indexed = row[2]
            commits_indexed = row[3] or 0
            memories_created = row[4]
            current_source = row[5]

            # Skip if already git_remote based
            if current_source == NamespaceSource.GIT_REMOTE:
                continue

            # Check if repo path exists and has git remote
            if not Path(repo_path).exists():
                continue

            remote_url = _get_git_remote_url(repo_path)
            if not remote_url:
                continue

            # Calculate what the new namespace ID should be
            normalized_url = _normalize_git_url(remote_url)
            import hashlib

            new_ns_id = f"repo-{hashlib.sha256(normalized_url.encode()).hexdigest()[:12]}"

            # Only add if namespace would change
            if new_ns_id != old_ns_id:
                candidates.append(
                    {
                        "old_namespace_id": old_ns_id,
                        "new_namespace_id": new_ns_id,
                        "repo_path": repo_path,
                        "remote_url": remote_url,
                        "files_indexed": files_indexed,
                        "commits_indexed": commits_indexed,
                        "memories_created": memories_created,
                        "current_source": current_source or "path",
                    }
                )

        conn.close()
        return candidates

    def migrate_all_to_git_remote(
        self, storage: "StorageRouter | None" = None, dry_run: bool = False
    ) -> dict:
        """
        Migrate all path-based namespaces to git-remote-based where possible.

        Args:
            storage: StorageRouter for ChromaDB updates
            dry_run: If True, only report what would be migrated

        Returns:
            Migration summary
        """
        from contextfs.schemas import NamespaceSource

        candidates = self.get_migration_candidates()

        if not candidates:
            return {
                "success": True,
                "migrated": 0,
                "failed": 0,
                "candidates": [],
                "message": "No migration candidates found",
            }

        if dry_run:
            return {
                "success": True,
                "dry_run": True,
                "candidates": candidates,
                "message": f"Would migrate {len(candidates)} namespace(s)",
            }

        migrated = []
        failed = []

        for candidate in candidates:
            result = self.migrate_namespace(
                old_namespace_id=candidate["old_namespace_id"],
                new_namespace_id=candidate["new_namespace_id"],
                new_source=NamespaceSource.GIT_REMOTE,
                new_remote_url=candidate["remote_url"],
                storage=storage,
            )

            if result["success"]:
                migrated.append(result)
            else:
                failed.append(
                    {
                        "old_namespace_id": candidate["old_namespace_id"],
                        "error": result.get("error"),
                    }
                )

        return {
            "success": len(failed) == 0,
            "migrated": len(migrated),
            "failed": len(failed),
            "results": migrated,
            "errors": failed,
        }

    def should_index(self, namespace_id: str, repo_path: Path | None = None) -> bool:
        """
        Determine if indexing should occur.

        Returns True if:
        - Namespace has never been indexed
        - Repo has new commits since last index
        """
        status = self.get_status(namespace_id)

        if not status or not status.indexed:
            return True

        # Check for new commits if we have commit hash
        if status.commit_hash and repo_path:
            current_hash = self._get_commit_hash(repo_path)
            if current_hash and current_hash != status.commit_hash:
                return True

        return False

    def _get_commit_hash(self, repo_path: Path) -> str | None:
        """Get current HEAD commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _get_gitignore_patterns(self, repo_path: Path) -> set[str]:
        """Parse .gitignore for additional ignore patterns."""
        patterns = set()
        gitignore = repo_path / ".gitignore"

        if gitignore.exists():
            try:
                for line in gitignore.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Convert glob patterns to simple patterns
                        pattern = line.lstrip("/").rstrip("/")
                        if pattern:
                            patterns.add(pattern)
            except Exception as e:
                logger.warning(f"Failed to parse .gitignore: {e}")

        return patterns

    def _should_ignore(self, path: Path, ignore_patterns: set[str]) -> bool:
        """Check if path should be ignored."""
        path_str = str(path)
        name = path.name

        for pattern in ignore_patterns:
            # Check exact name match
            if name == pattern:
                return True
            # Check if pattern appears in path
            if pattern in path_str:
                return True
            # Check glob-like patterns
            if pattern.startswith("*") and name.endswith(pattern[1:]):
                return True

        return False

    def discover_files(
        self,
        repo_path: Path,
        respect_gitignore: bool = True,
    ) -> list[Path]:
        """
        Discover indexable files in repository.

        Args:
            repo_path: Root path to scan
            respect_gitignore: Honor .gitignore patterns

        Returns:
            List of file paths to index
        """
        ignore_patterns = self.ignore_patterns.copy()

        if respect_gitignore:
            ignore_patterns.update(self._get_gitignore_patterns(repo_path))

        files = []

        for path in repo_path.rglob("*"):
            # Skip directories
            if not path.is_file():
                continue

            # Skip ignored paths
            rel_path = path.relative_to(repo_path)
            if self._should_ignore(rel_path, ignore_patterns):
                continue

            # Check extension
            if path.suffix.lower() not in self.extensions:
                continue

            # Skip very large files (> 1MB)
            try:
                if path.stat().st_size > 1_000_000:
                    continue
            except OSError:
                continue

            files.append(path)

        return sorted(files)

    def _file_hash(self, file_path: Path) -> str:
        """Get simple hash of file for change detection."""
        import hashlib

        try:
            content = file_path.read_bytes()
            return hashlib.md5(content).hexdigest()[:16]
        except Exception:
            return ""

    def index_repository(
        self,
        repo_path: Path,
        namespace_id: str,
        rag_backend: RAGBackend | None = None,
        on_progress: Callable[[int, int, str], None] | None = None,
        incremental: bool = True,
        project: str | None = None,
        source_repo: str | None = None,
        storage: StorageRouter | None = None,
        mode: IndexMode = IndexMode.ALL,
        parallel_workers: int | None = None,
        max_commits: int | None = None,
    ) -> dict:
        """
        Index all files in repository to both SQLite and ChromaDB.

        Args:
            repo_path: Repository root path
            namespace_id: Namespace for memories
            rag_backend: DEPRECATED - use storage instead
            on_progress: Callback for progress updates (current, total, file)
            incremental: Only index new/changed files
            project: Project name for grouping memories across repos
            source_repo: Repository name (default: repo_path.name)
            storage: StorageRouter for unified SQLite + ChromaDB storage
            mode: IndexMode.ALL, FILES_ONLY, or COMMITS_ONLY
            parallel_workers: Number of parallel workers (None = auto)
            max_commits: Maximum commits to index (None = unlimited)

        Returns:
            Indexing statistics
        """
        # Handle storage parameter (new) vs rag_backend (deprecated)
        # If neither provided, this is an error
        if storage is None and rag_backend is None:
            raise ValueError("Either storage or rag_backend must be provided")

        # Default source_repo to directory name
        if source_repo is None:
            source_repo = repo_path.name
        logger.info(
            f"Starting indexing for {repo_path} (namespace: {namespace_id}, mode: {mode.value})"
        )

        # Clear existing memories for this namespace when doing full re-index
        # This prevents duplicate memories when using --force
        if not incremental:
            if storage is not None:
                deleted = storage.delete_by_namespace(namespace_id)
            elif rag_backend is not None:
                deleted = rag_backend.delete_by_namespace(namespace_id)
            else:
                deleted = 0
            if deleted > 0:
                logger.info(f"Cleared {deleted} existing memories for full re-index")
            self.clear_index(namespace_id)

        files_indexed = 0
        memories_created = 0
        skipped = 0
        errors = []
        total_files = 0
        commits_indexed = 0

        # Index files (unless COMMITS_ONLY mode)
        if mode != IndexMode.COMMITS_ONLY:
            # Discover files
            files = self.discover_files(repo_path)
            total_files = len(files)

            if total_files == 0:
                logger.info("No indexable files found")
            else:
                # Get already indexed files for incremental mode
                indexed_hashes = {}
                if incremental:
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT file_path, file_hash FROM indexed_files WHERE namespace_id = ?",
                        (namespace_id,),
                    )
                    indexed_hashes = {row[0]: row[1] for row in cursor.fetchall()}
                    conn.close()

                # Determine number of workers
                if parallel_workers is None:
                    # Auto: use half of CPU cores for I/O bound work
                    parallel_workers = max(1, (os.cpu_count() or 4) // 2)

                # Process files (with optional parallelism for file reading)
                for idx, file_path in enumerate(files):
                    rel_path = str(file_path.relative_to(repo_path))

                    # Progress callback - show file being processed
                    if on_progress:
                        on_progress(idx + 1, total_files, rel_path)

                    # Check if file changed (incremental mode)
                    if incremental:
                        current_hash = self._file_hash(file_path)
                        if rel_path in indexed_hashes and indexed_hashes[rel_path] == current_hash:
                            skipped += 1
                            continue

                    try:
                        # Process file
                        chunks = self.processor.process_file(file_path)

                        if not chunks:
                            skipped += 1
                            continue

                        # Create memories from chunks (batch for speed)
                        file_memory_list = []
                        for chunk in chunks:
                            memory = Memory(
                                content=chunk["content"],
                                type=MemoryType.CODE,
                                tags=self._extract_tags(chunk, rel_path),
                                summary=chunk["metadata"].get("summary") or f"Code from {rel_path}",
                                namespace_id=namespace_id,
                                source_file=rel_path,
                                source_repo=source_repo,
                                source_tool="auto-index",
                                project=project,
                                metadata={
                                    "chunk_index": chunk["metadata"].get("chunk_index"),
                                    "total_chunks": chunk["metadata"].get("total_chunks"),
                                    "file_type": chunk["metadata"].get("file_type"),
                                    "language": chunk["metadata"].get("language"),
                                    "auto_indexed": True,
                                },
                            )
                            file_memory_list.append(memory)

                        # Batch add all memories for this file (much faster)
                        try:
                            if storage is not None:
                                # Use unified storage (saves to both SQLite and ChromaDB)
                                added = storage.save_batch(file_memory_list)
                                memories_created += added
                                file_memories = added
                            elif hasattr(rag_backend, "add_memories_batch"):
                                # DEPRECATED: ChromaDB-only path (for backwards compatibility)
                                added = rag_backend.add_memories_batch(file_memory_list)
                                memories_created += added
                                file_memories = added
                            else:
                                # Fallback to individual adds
                                file_memories = 0
                                for memory in file_memory_list:
                                    try:
                                        if storage is not None:
                                            storage.save(memory)
                                        else:
                                            rag_backend.add_memory(memory)
                                        file_memories += 1
                                        memories_created += 1
                                    except Exception as e:
                                        logger.warning(f"Failed to save memory for {rel_path}: {e}")
                        except Exception as e:
                            logger.warning(f"Failed to batch save memories for {rel_path}: {e}")
                            file_memories = 0

                        # Track indexed file
                        if file_memories > 0:
                            self._record_indexed_file(
                                namespace_id,
                                rel_path,
                                self._file_hash(file_path),
                                file_memories,
                            )
                            files_indexed += 1

                    except Exception as e:
                        logger.warning(f"Failed to index {rel_path}: {e}")
                        errors.append({"file": rel_path, "error": str(e)})

        # Index git history (unless FILES_ONLY mode)
        if mode != IndexMode.FILES_ONLY:
            git_stats = self.index_git_history(
                repo_path=repo_path,
                namespace_id=namespace_id,
                rag_backend=rag_backend,
                max_commits=max_commits if max_commits is not None else self._get_max_commits(),
                on_progress=on_progress,
                incremental=incremental,
                project=project,
                source_repo=source_repo,
                storage=storage,
            )
            memories_created += git_stats["memories_created"]
            commits_indexed = git_stats["commits_indexed"]

        # Update index status only if we indexed something new
        # Don't overwrite existing status with zeros from incremental skips
        if files_indexed > 0 or commits_indexed > 0 or memories_created > 0 or not incremental:
            self._update_status(
                namespace_id,
                repo_path,
                files_indexed,
                commits_indexed,
                memories_created,
                incremental=incremental,
            )
        else:
            logger.debug("Skipping status update - no new files indexed (incremental mode)")

        logger.info(
            f"Indexing complete: {files_indexed} files, "
            f"{commits_indexed} commits, {memories_created} memories"
        )

        return {
            "files_discovered": total_files,
            "files_indexed": files_indexed,
            "commits_indexed": commits_indexed,
            "memories_created": memories_created,
            "skipped": skipped,
            "errors": errors,
            "mode": mode.value,
        }

    def _extract_tags(self, chunk: dict, rel_path: str) -> list[str]:
        """Extract tags from chunk metadata."""
        tags = ["auto-indexed"]
        metadata = chunk.get("metadata", {})

        # File type tag
        if metadata.get("file_type"):
            tags.append(f"type:{metadata['file_type']}")

        # Language tag
        if metadata.get("language"):
            tags.append(f"lang:{metadata['language']}")

        # Extension tag
        ext = Path(rel_path).suffix.lower()
        if ext:
            tags.append(f"ext:{ext.lstrip('.')}")

        # Directory path tags (first two levels)
        parts = Path(rel_path).parts[:-1]  # Exclude filename
        for part in parts[:2]:
            if part and not part.startswith("."):
                tags.append(f"dir:{part}")

        # Keywords from chunk
        if metadata.get("keywords"):
            tags.extend(metadata["keywords"][:3])

        return tags

    def _record_indexed_file(
        self,
        namespace_id: str,
        file_path: str,
        file_hash: str,
        memories_created: int,
    ) -> None:
        """Record indexed file for incremental updates."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO indexed_files
            (namespace_id, file_path, file_hash, indexed_at, memories_created)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                namespace_id,
                file_path,
                file_hash,
                datetime.now().isoformat(),
                memories_created,
            ),
        )

        conn.commit()
        conn.close()

    def index_git_history(
        self,
        repo_path: Path,
        namespace_id: str,
        rag_backend: RAGBackend | None = None,
        max_commits: int = 100,
        on_progress: Callable[[int, int, str], None] | None = None,
        incremental: bool = True,
        project: str | None = None,
        source_repo: str | None = None,
        storage: StorageRouter | None = None,
    ) -> dict:
        """
        Index git commit history to both SQLite and ChromaDB.

        Args:
            repo_path: Repository root path
            namespace_id: Namespace for memories
            rag_backend: DEPRECATED - use storage instead
            max_commits: Maximum commits to index
            on_progress: Progress callback
            incremental: Only index new commits
            project: Project name for grouping memories
            source_repo: Repository name
            storage: StorageRouter for unified storage

        Returns:
            Indexing statistics
        """
        # Default source_repo to directory name
        if source_repo is None:
            source_repo = repo_path.name
        logger.info(f"Indexing git history for {repo_path} (incremental={incremental})")

        # Get already indexed commits for incremental mode
        indexed_commits = set()
        if incremental:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT commit_hash FROM indexed_commits WHERE namespace_id = ?",
                (namespace_id,),
            )
            indexed_commits = {row[0] for row in cursor.fetchall()}
            conn.close()

        # Get latest commits (max_commits limit)
        # For incremental: we only want NEW commits at the head, not old ones
        all_commits = self._get_git_commits(repo_path, max_commits)

        if not all_commits:
            logger.info("No git commits found")
            return {"commits_indexed": 0, "memories_created": 0, "skipped": 0}

        # Filter to unindexed commits only
        if incremental:
            commits = [c for c in all_commits if c["hash"] not in indexed_commits]
            skipped = len(all_commits) - len(commits)
        else:
            commits = all_commits
            skipped = 0

        if not commits:
            logger.info(f"All {len(all_commits)} recent commits already indexed")
            return {"commits_indexed": 0, "memories_created": 0, "skipped": skipped}

        commits_indexed = 0
        memories_created = 0

        for idx, commit in enumerate(commits):
            if on_progress:
                on_progress(idx + 1, len(commits), commit["hash"][:8])

            # Create memory for each commit
            content = f"""Git Commit: {commit["hash"][:8]}
Author: {commit["author"]}
Date: {commit["date"]}

{commit["message"]}

Files changed: {", ".join(commit["files"][:10])}{"..." if len(commit["files"]) > 10 else ""}
"""

            memory = Memory(
                content=content,
                type=MemoryType.COMMIT,
                tags=["git-history", *self._extract_commit_tags(commit)],
                summary=commit["message"].split("\n")[0][:100],
                namespace_id=namespace_id,
                source_repo=source_repo,
                source_tool="auto-index",
                project=project,
                metadata={
                    "commit_hash": commit["hash"],
                    "author": commit["author"],
                    "date": commit["date"],
                    "files_changed": len(commit["files"]),
                    "auto_indexed": True,
                    "source_type": "git-history",
                },
            )

            try:
                if storage is not None:
                    storage.save(memory)
                elif rag_backend is not None:
                    rag_backend.add_memory(memory)
                else:
                    raise ValueError("Either storage or rag_backend must be provided")
                memories_created += 1
                commits_indexed += 1
                # Record indexed commit
                self._record_indexed_commit(namespace_id, commit["hash"])
            except Exception as e:
                logger.warning(f"Failed to index commit {commit['hash'][:8]}: {e}")

        logger.info(f"Indexed {commits_indexed} git commits ({skipped} skipped)")
        return {
            "commits_indexed": commits_indexed,
            "memories_created": memories_created,
            "skipped": skipped,
        }

    def _record_indexed_commit(self, namespace_id: str, commit_hash: str) -> None:
        """Record indexed commit for incremental updates."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO indexed_commits
            (namespace_id, commit_hash, indexed_at)
            VALUES (?, ?, ?)
        """,
            (namespace_id, commit_hash, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

    def _get_git_commits(self, repo_path: Path, max_commits: int) -> list[dict]:
        """Get git commit history."""
        try:
            # Get commit log with format: hash|author|date|message
            result = subprocess.run(
                [
                    "git",
                    "log",
                    f"-{max_commits}",
                    "--format=%H|%an|%ad|%s",
                    "--date=short",
                ],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return []

            commits = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                parts = line.split("|", 3)
                if len(parts) < 4:
                    continue

                commit_hash, author, date, message = parts

                # Get files changed for this commit
                files_result = subprocess.run(
                    ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                files = (
                    files_result.stdout.strip().split("\n") if files_result.returncode == 0 else []
                )

                commits.append(
                    {
                        "hash": commit_hash,
                        "author": author,
                        "date": date,
                        "message": message,
                        "files": [f for f in files if f],
                    }
                )

            return commits

        except Exception as e:
            logger.warning(f"Failed to get git commits: {e}")
            return []

    def _extract_commit_tags(self, commit: dict) -> list[str]:
        """Extract tags from commit for categorization."""
        tags = []
        message = commit["message"].lower()

        # Conventional commit prefixes
        prefixes = {
            "feat": "feature",
            "fix": "bugfix",
            "docs": "documentation",
            "style": "style",
            "refactor": "refactor",
            "test": "testing",
            "chore": "maintenance",
            "perf": "performance",
            "ci": "ci-cd",
        }

        for prefix, tag in prefixes.items():
            if message.startswith(f"{prefix}:") or message.startswith(f"{prefix}("):
                tags.append(tag)
                break

        # File type tags from changed files
        for file in commit["files"][:5]:
            ext = Path(file).suffix.lower()
            if ext in {".py", ".js", ".ts", ".go", ".rs", ".java"}:
                tags.append(f"lang:{ext.lstrip('.')}")
                break

        return tags

    def _update_status(
        self,
        namespace_id: str,
        repo_path: Path,
        files_indexed: int,
        commits_indexed: int,
        memories_created: int,
        incremental: bool = True,
    ) -> None:
        """Update index status.

        For incremental indexing, accumulates counts.
        For full indexing, replaces counts.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        commit_hash = self._get_commit_hash(repo_path)

        if incremental:
            # Check if record exists
            cursor.execute(
                "SELECT files_indexed, commits_indexed, memories_created FROM index_status WHERE namespace_id = ?",
                (namespace_id,),
            )
            row = cursor.fetchone()

            if row:
                # Accumulate counts for incremental indexing
                files_indexed += row[0] or 0
                commits_indexed += row[1] or 0
                memories_created += row[2] or 0

        cursor.execute(
            """
            INSERT OR REPLACE INTO index_status
            (namespace_id, indexed, indexed_at, files_indexed, commits_indexed, memories_created, repo_path, commit_hash)
            VALUES (?, 1, ?, ?, ?, ?, ?, ?)
        """,
            (
                namespace_id,
                datetime.now().isoformat(),
                files_indexed,
                commits_indexed,
                memories_created,
                str(repo_path),
                commit_hash,
            ),
        )

        conn.commit()
        conn.close()

    def clear_index(self, namespace_id: str) -> None:
        """Clear indexing status and records for namespace."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM index_status WHERE namespace_id = ?", (namespace_id,))
        cursor.execute("DELETE FROM indexed_files WHERE namespace_id = ?", (namespace_id,))
        cursor.execute("DELETE FROM indexed_commits WHERE namespace_id = ?", (namespace_id,))

        conn.commit()
        conn.close()

        logger.info(f"Cleared index for namespace: {namespace_id}")

    def index_directory(
        self,
        root_dir: Path,
        rag_backend: RAGBackend | None = None,
        max_depth: int = 5,
        on_progress: Callable[[int, int, str], None] | None = None,
        on_repo_start: Callable[[str, str | None], None] | None = None,
        on_repo_complete: Callable[[str, dict], None] | None = None,
        incremental: bool = True,
        project_override: str | None = None,
        storage: StorageRouter | None = None,
        repo_filter: Callable[[Path], bool] | None = None,
    ) -> dict:
        """
        Recursively scan a directory for git repos and index each.

        Args:
            root_dir: Root directory to scan
            rag_backend: DEPRECATED - use storage instead
            max_depth: Maximum depth to search for repos
            on_progress: Progress callback for file indexing (current, total, file)
            on_repo_start: Callback when starting a repo (repo_name, project)
            on_repo_complete: Callback when repo completes (repo_name, stats)
            incremental: Only index new/changed files
            project_override: Override detected project name for all repos
            storage: StorageRouter for unified SQLite + ChromaDB storage
            repo_filter: Optional callback to filter repos (return True to include)

        Returns:
            Summary statistics for all repos indexed
        """
        logger.info(f"Scanning {root_dir} for git repositories (max_depth={max_depth})")

        # Discover all repos
        repos = discover_git_repos(root_dir, max_depth=max_depth)

        # Apply repo filter if provided
        if repo_filter:
            repos = [r for r in repos if repo_filter(Path(r["path"]))]

        if not repos:
            logger.info("No git repositories found")
            return {
                "repos_found": 0,
                "repos_indexed": 0,
                "total_files": 0,
                "total_memories": 0,
                "repos": [],
            }

        logger.info(f"Found {len(repos)} git repositories")

        total_stats = {
            "repos_found": len(repos),
            "repos_indexed": 0,
            "total_files": 0,
            "total_memories": 0,
            "total_commits": 0,
            "repos": [],
        }

        for repo_info in repos:
            repo_path = repo_info["path"]
            repo_name = repo_info["name"]
            project = project_override or repo_info["project"]
            suggested_tags = repo_info["suggested_tags"]

            if on_repo_start:
                on_repo_start(repo_name, project)

            # Get namespace for this repo
            from contextfs.schemas import Namespace

            namespace_id = Namespace.for_repo(str(repo_path)).id

            logger.info(f"Indexing {repo_name} (project={project}, namespace={namespace_id[:12]})")

            try:
                # Clear existing memories for this namespace when doing full re-index
                if not incremental:
                    if storage is not None:
                        deleted = storage.delete_by_namespace(namespace_id)
                    elif rag_backend is not None:
                        deleted = rag_backend.delete_by_namespace(namespace_id)
                    else:
                        deleted = 0
                    if deleted > 0:
                        logger.info(f"Cleared {deleted} existing memories for {repo_name}")
                    self.clear_index(namespace_id)

                # Index the repository
                stats = self.index_repository(
                    repo_path=repo_path,
                    namespace_id=namespace_id,
                    rag_backend=rag_backend,
                    on_progress=on_progress,
                    incremental=incremental,
                    project=project,
                    source_repo=repo_name,
                    storage=storage,
                )

                # Create a summary memory with project and auto-detected tags
                if stats["files_indexed"] > 0 or stats["commits_indexed"] > 0:
                    summary_memory = Memory(
                        content=f"Repository {repo_name} indexed with {stats['files_indexed']} files and {stats['commits_indexed']} commits.",
                        type=MemoryType.FACT,
                        tags=["repo-index", *suggested_tags],
                        summary=f"Indexed repository: {repo_name}",
                        namespace_id=namespace_id,
                        source_repo=repo_name,
                        source_tool="auto-index",
                        project=project,
                        metadata={
                            "auto_indexed": True,
                            "files_indexed": stats["files_indexed"],
                            "commits_indexed": stats["commits_indexed"],
                            "remote_url": repo_info.get("remote_url"),
                        },
                    )
                    if storage is not None:
                        storage.save(summary_memory)
                    elif rag_backend is not None:
                        rag_backend.add_memory(summary_memory)

                repo_result = {
                    "name": repo_name,
                    "path": str(repo_path),
                    "project": project,
                    "tags": suggested_tags,
                    "files_indexed": stats["files_indexed"],
                    "commits_indexed": stats.get("commits_indexed", 0),
                    "memories_created": stats["memories_created"],
                }

                total_stats["repos_indexed"] += 1
                total_stats["total_files"] += stats["files_indexed"]
                total_stats["total_memories"] += stats["memories_created"]
                total_stats["total_commits"] += stats.get("commits_indexed", 0)
                total_stats["repos"].append(repo_result)

                if on_repo_complete:
                    on_repo_complete(repo_name, repo_result)

            except Exception as e:
                logger.warning(f"Failed to index {repo_name}: {e}")
                total_stats["repos"].append(
                    {
                        "name": repo_name,
                        "path": str(repo_path),
                        "error": str(e),
                    }
                )

        logger.info(
            f"Directory indexing complete: {total_stats['repos_indexed']} repos, "
            f"{total_stats['total_files']} files, {total_stats['total_memories']} memories"
        )

        return total_stats
