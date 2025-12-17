"""
Auto-indexing module for ContextFS.

Automatically indexes repository files on first memory save,
creating a searchable knowledge base of the codebase.
"""

import logging
import sqlite3
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from contextfs.config import Config
from contextfs.filetypes.integration import SmartDocumentProcessor
from contextfs.filetypes.registry import FileTypeRegistry
from contextfs.rag import RAGBackend
from contextfs.schemas import Memory, MemoryType

logger = logging.getLogger(__name__)


# Default directories and files to ignore
DEFAULT_IGNORE_PATTERNS = {
    # Package managers
    "node_modules",
    "vendor",
    "packages",
    ".pnpm",
    "bower_components",
    # Build outputs
    "dist",
    "build",
    "out",
    "target",
    "_build",
    ".next",
    ".nuxt",
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    "coverage",
    ".coverage",
    "htmlcov",
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
    # Dependencies
    ".tox",
    ".nox",
    ".eggs",
    "*.egg-info",
    # Misc
    ".DS_Store",
    "Thumbs.db",
    "*.log",
    "*.lock",
    "*.lockb",
    # Large binary files
    "*.min.js",
    "*.min.css",
    "*.bundle.js",
    "*.chunk.js",
    # Database files
    "*.db",
    "*.sqlite",
    "*.sqlite3",
}

# Extensions to index by default
DEFAULT_INDEX_EXTENSIONS = {
    # Programming languages
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".kt",
    ".scala",
    ".go",
    ".rs",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".fs",
    ".vb",
    ".rb",
    ".php",
    ".swift",
    ".m",
    ".mm",
    ".lua",
    ".pl",
    ".pm",
    ".r",
    ".R",
    ".ex",
    ".exs",
    ".erl",
    ".hrl",
    ".clj",
    ".cljs",
    ".cljc",
    ".hs",
    ".ml",
    ".mli",
    ".jl",
    ".nim",
    ".zig",
    ".d",
    ".v",
    ".sv",
    ".vhd",
    ".vhdl",
    # Web
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".vue",
    ".svelte",
    # Config
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".xml",
    ".plist",
    # Documentation
    ".md",
    ".rst",
    ".txt",
    ".adoc",
    # Data
    ".sql",
    ".graphql",
    ".gql",
    # Shell
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".ps1",
    # Templates
    ".jinja",
    ".j2",
    ".ejs",
    ".hbs",
    ".pug",
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
                metadata TEXT
            )
        """)

        # Migration: add commits_indexed column if missing
        cursor.execute("PRAGMA table_info(index_status)")
        columns = {row[1] for row in cursor.fetchall()}
        if "commits_indexed" not in columns:
            cursor.execute("ALTER TABLE index_status ADD COLUMN commits_indexed INTEGER DEFAULT 0")

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
        rag_backend: RAGBackend,
        on_progress: Callable[[int, int, str], None] | None = None,
        incremental: bool = True,
        project: str | None = None,
        source_repo: str | None = None,
    ) -> dict:
        """
        Index all files in repository to ChromaDB (vector store).

        Args:
            repo_path: Repository root path
            namespace_id: Namespace for memories
            rag_backend: RAG backend (ChromaDB) for vector storage
            on_progress: Callback for progress updates (current, total, file)
            incremental: Only index new/changed files
            project: Project name for grouping memories across repos
            source_repo: Repository name (default: repo_path.name)

        Returns:
            Indexing statistics
        """
        # Default source_repo to directory name
        if source_repo is None:
            source_repo = repo_path.name
        logger.info(f"Starting indexing for {repo_path} (namespace: {namespace_id})")

        # Discover files
        files = self.discover_files(repo_path)
        total_files = len(files)

        if total_files == 0:
            logger.info("No indexable files found")
            return {
                "files_discovered": 0,
                "files_indexed": 0,
                "memories_created": 0,
                "skipped": 0,
            }

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

        files_indexed = 0
        memories_created = 0
        skipped = 0
        errors = []

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
                    if hasattr(rag_backend, "add_memories_batch"):
                        added = rag_backend.add_memories_batch(file_memory_list)
                        memories_created += added
                        file_memories = added
                    else:
                        # Fallback to individual adds
                        file_memories = 0
                        for memory in file_memory_list:
                            try:
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

        # Index git history
        git_stats = self.index_git_history(
            repo_path=repo_path,
            namespace_id=namespace_id,
            rag_backend=rag_backend,
            max_commits=100,
            on_progress=on_progress,
            incremental=incremental,
            project=project,
            source_repo=source_repo,
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
            )
        else:
            logger.debug("Skipping status update - no new files indexed (incremental mode)")

        logger.info(
            f"Indexing complete: {files_indexed} files, "
            f"{git_stats['commits_indexed']} commits, {memories_created} memories"
        )

        return {
            "files_discovered": total_files,
            "files_indexed": files_indexed,
            "commits_indexed": git_stats["commits_indexed"],
            "memories_created": memories_created,
            "skipped": skipped,
            "errors": errors,
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
        rag_backend: RAGBackend,
        max_commits: int = 100,
        on_progress: Callable[[int, int, str], None] | None = None,
        incremental: bool = True,
        project: str | None = None,
        source_repo: str | None = None,
    ) -> dict:
        """
        Index git commit history to RAG backend.

        Args:
            repo_path: Repository root path
            namespace_id: Namespace for memories
            rag_backend: RAG backend for vector storage
            max_commits: Maximum commits to index
            on_progress: Progress callback
            incremental: Only index new commits
            project: Project name for grouping memories
            source_repo: Repository name

        Returns:
            Indexing statistics
        """
        # Default source_repo to directory name
        if source_repo is None:
            source_repo = repo_path.name
        logger.info(f"Indexing git history for {repo_path} (incremental={incremental})")

        commits = self._get_git_commits(repo_path, max_commits)
        if not commits:
            logger.info("No git commits found")
            return {"commits_indexed": 0, "memories_created": 0, "skipped": 0}

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

        commits_indexed = 0
        memories_created = 0
        skipped = 0

        for idx, commit in enumerate(commits):
            if on_progress:
                on_progress(idx + 1, len(commits), commit["hash"][:8])

            # Skip already indexed commits in incremental mode
            if incremental and commit["hash"] in indexed_commits:
                skipped += 1
                continue

            # Create memory for each commit
            content = f"""Git Commit: {commit["hash"][:8]}
Author: {commit["author"]}
Date: {commit["date"]}

{commit["message"]}

Files changed: {", ".join(commit["files"][:10])}{"..." if len(commit["files"]) > 10 else ""}
"""

            memory = Memory(
                content=content,
                type=MemoryType.EPISODIC,
                tags=["git-history", "commit", *self._extract_commit_tags(commit)],
                summary=commit["message"].split("\n")[0][:100],
                namespace_id=namespace_id,
                source_repo=source_repo,
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
                rag_backend.add_memory(memory)
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
    ) -> None:
        """Update index status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        commit_hash = self._get_commit_hash(repo_path)

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
        rag_backend: RAGBackend,
        max_depth: int = 5,
        on_progress: Callable[[int, int, str], None] | None = None,
        on_repo_start: Callable[[str, str | None], None] | None = None,
        on_repo_complete: Callable[[str, dict], None] | None = None,
        incremental: bool = True,
        project_override: str | None = None,
    ) -> dict:
        """
        Recursively scan a directory for git repos and index each.

        Args:
            root_dir: Root directory to scan
            rag_backend: RAG backend for vector storage
            max_depth: Maximum depth to search for repos
            on_progress: Progress callback for file indexing (current, total, file)
            on_repo_start: Callback when starting a repo (repo_name, project)
            on_repo_complete: Callback when repo completes (repo_name, stats)
            incremental: Only index new/changed files
            project_override: Override detected project name for all repos

        Returns:
            Summary statistics for all repos indexed
        """
        logger.info(f"Scanning {root_dir} for git repositories (max_depth={max_depth})")

        # Discover all repos
        repos = discover_git_repos(root_dir, max_depth=max_depth)

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
                    deleted = rag_backend.delete_by_namespace(namespace_id)
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
                        project=project,
                        metadata={
                            "auto_indexed": True,
                            "files_indexed": stats["files_indexed"],
                            "commits_indexed": stats["commits_indexed"],
                            "remote_url": repo_info.get("remote_url"),
                        },
                    )
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


class RepoInfo(TypedDict):
    """Information about a discovered git repository."""

    path: Path
    name: str
    project: str | None
    suggested_tags: list[str]
    remote_url: str | None
    relative_path: str


@dataclass
class LanguageDetector:
    """Detects programming languages based on indicator files."""

    language: str
    indicators: list[str]

    def matches(self, repo_path: Path) -> bool:
        """Check if any indicator file exists in the repo."""
        for indicator in self.indicators:
            if "*" in indicator:
                if list(repo_path.glob(indicator)):
                    return True
            elif (repo_path / indicator).exists():
                return True
        return False


@dataclass
class FrameworkDetector:
    """Detects frameworks from package dependencies or config files."""

    framework: str
    config_files: list[str] = field(default_factory=list)
    npm_packages: list[str] = field(default_factory=list)
    pip_packages: list[str] = field(default_factory=list)

    def matches(self, repo_path: Path, npm_deps: set[str], pip_content: str) -> bool:
        """Check if framework is detected."""
        # Check config files
        for cf in self.config_files:
            if (repo_path / cf).exists():
                return True
        # Check npm packages
        for pkg in self.npm_packages:
            if pkg in npm_deps:
                return True
        # Check pip packages
        return any(pkg in pip_content for pkg in self.pip_packages)


@dataclass
class TypeIndicator:
    """Detects project type based on file/directory presence."""

    tag: str
    paths: list[str]

    def matches(self, repo_path: Path) -> bool:
        """Check if any path exists."""
        return any((repo_path / p).exists() for p in self.paths)


# Registry of language detectors
LANGUAGE_DETECTORS = [
    LanguageDetector("python", ["setup.py", "pyproject.toml", "requirements.txt", "Pipfile"]),
    LanguageDetector("javascript", ["package.json"]),
    LanguageDetector("typescript", ["tsconfig.json"]),
    LanguageDetector("rust", ["Cargo.toml"]),
    LanguageDetector("go", ["go.mod"]),
    LanguageDetector("java", ["pom.xml", "build.gradle"]),
    LanguageDetector("ruby", ["Gemfile"]),
    LanguageDetector("php", ["composer.json"]),
    LanguageDetector("swift", ["Package.swift"]),
    LanguageDetector("csharp", ["*.csproj", "*.sln"]),
]

# Registry of framework detectors
FRAMEWORK_DETECTORS = [
    FrameworkDetector("react", npm_packages=["react"]),
    FrameworkDetector(
        "vue", config_files=["vue.config.js", "nuxt.config.js"], npm_packages=["vue"]
    ),
    FrameworkDetector("angular", config_files=["angular.json"], npm_packages=["@angular/core"]),
    FrameworkDetector(
        "nextjs", config_files=["next.config.js", "next.config.mjs"], npm_packages=["next"]
    ),
    FrameworkDetector("express", npm_packages=["express"]),
    FrameworkDetector("fastify", npm_packages=["fastify"]),
    FrameworkDetector("django", pip_packages=["django"]),
    FrameworkDetector("flask", pip_packages=["flask"]),
    FrameworkDetector("fastapi", pip_packages=["fastapi"]),
    FrameworkDetector("rails", config_files=["config/routes.rb"]),
]

# Registry of project type indicators
TYPE_INDICATORS = [
    TypeIndicator("type:containerized", ["Dockerfile"]),
    TypeIndicator("type:docker-compose", ["docker-compose.yml", "docker-compose.yaml"]),
    TypeIndicator("ci:github-actions", [".github/workflows"]),
    TypeIndicator("ci:gitlab", [".gitlab-ci.yml"]),
    TypeIndicator("has-tests", ["tests", "test"]),
]

# Directories that indicate a project container
PROJECT_CONTAINER_NAMES = {"projects", "repos", "work", "workspace", "dev", "development"}

# Workspace config files that indicate a project container
WORKSPACE_CONFIG_FILES = [
    "pnpm-workspace.yaml",
    "lerna.json",
    ".workspace",
    "Cargo.toml",  # Rust workspace
    "go.work",  # Go workspace
]


def discover_git_repos(
    root_dir: Path,
    max_depth: int = 5,
    ignore_patterns: set[str] | None = None,
) -> list[RepoInfo]:
    """
    Recursively discover all git repositories under a directory.

    Args:
        root_dir: Root directory to scan
        max_depth: Maximum depth to search (default: 5)
        ignore_patterns: Directory names to skip (default: common ignores)

    Returns:
        List of RepoInfo dicts with repo metadata
    """
    if ignore_patterns is None:
        ignore_patterns = {
            "node_modules",
            ".git",
            "vendor",
            "__pycache__",
            "venv",
            ".venv",
            "dist",
            "build",
            ".cache",
        }

    repos: list[RepoInfo] = []
    root_dir = root_dir.resolve()

    def _scan(current: Path, depth: int, parent_project: str | None = None) -> None:
        if depth > max_depth:
            return

        try:
            entries = list(current.iterdir())
        except PermissionError:
            return

        for entry in entries:
            if not entry.is_dir():
                continue

            if entry.name in ignore_patterns:
                continue

            if entry.name.startswith(".") and entry.name != ".git":
                continue

            # Check if this is a git repo
            if (entry / ".git").exists():
                repo_info = _analyze_repo(entry, root_dir, parent_project)
                repos.append(repo_info)
                # Don't recurse into git repos (they're self-contained)
                continue

            # If current dir looks like a project container, use it as project name
            project_name = parent_project
            if _is_project_container(entry):
                project_name = entry.name

            # Recurse into subdirectory
            _scan(entry, depth + 1, project_name)

    _scan(root_dir, 0)
    return repos


def _is_project_container(path: Path) -> bool:
    """
    Check if a directory looks like a project container (monorepo parent).

    Uses PROJECT_CONTAINER_NAMES and WORKSPACE_CONFIG_FILES registries.
    """
    if path.name.lower() in PROJECT_CONTAINER_NAMES:
        return True

    return any((path / wf).exists() for wf in WORKSPACE_CONFIG_FILES)


def _analyze_repo(repo_path: Path, root_dir: Path, parent_project: str | None) -> RepoInfo:
    """
    Analyze a git repository and extract metadata.

    Returns:
        RepoInfo with path, name, project, suggested_tags, etc.
    """
    import subprocess

    name = repo_path.name
    rel_path = repo_path.relative_to(root_dir)

    # Determine project name
    project = parent_project
    if project is None and len(rel_path.parts) > 1:
        potential_project = rel_path.parts[0]
        if _is_project_container(root_dir / potential_project):
            project = potential_project

    # Detect tags based on repo contents
    suggested_tags = _detect_repo_tags(repo_path)

    # Get remote URL
    remote_url = None
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            remote_url = result.stdout.strip()
    except Exception:
        pass

    return RepoInfo(
        path=repo_path,
        name=name,
        project=project,
        suggested_tags=suggested_tags,
        remote_url=remote_url,
        relative_path=str(rel_path),
    )


def _detect_repo_tags(repo_path: Path) -> list[str]:
    """
    Detect tags using registered detectors.

    Uses LANGUAGE_DETECTORS, FRAMEWORK_DETECTORS, and TYPE_INDICATORS.
    """
    tags: list[str] = []

    # Detect languages
    for detector in LANGUAGE_DETECTORS:
        if detector.matches(repo_path):
            tags.append(f"lang:{detector.language}")

    # Load npm dependencies once (if package.json exists)
    npm_deps: set[str] = set()
    pkg_json = repo_path / "package.json"
    if pkg_json.exists():
        try:
            import json

            pkg = json.loads(pkg_json.read_text())
            npm_deps = set(pkg.get("dependencies", {}).keys())
            npm_deps.update(pkg.get("devDependencies", {}).keys())
        except Exception:
            pass

    # Load pip content once (requirements.txt or pyproject.toml)
    pip_content = ""
    for pyfile in [repo_path / "requirements.txt", repo_path / "pyproject.toml"]:
        if pyfile.exists():
            try:
                pip_content = pyfile.read_text().lower()
            except Exception:
                pass
            break

    # Detect frameworks
    for detector in FRAMEWORK_DETECTORS:
        if detector.matches(repo_path, npm_deps, pip_content):
            tags.append(f"framework:{detector.framework}")

    # Detect project types
    for indicator in TYPE_INDICATORS:
        if indicator.matches(repo_path):
            tags.append(indicator.tag)

    return list(set(tags))  # Deduplicate


def create_codebase_summary(repo_path: Path) -> Memory:
    """
    Create a summary memory of the codebase structure.

    Returns a single memory with high-level codebase overview.
    """
    # Count files by extension
    extension_counts: dict[str, int] = {}
    total_files = 0
    total_lines = 0

    ignore_dirs = {"node_modules", ".git", "venv", ".venv", "__pycache__", "dist", "build"}

    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue

        # Skip ignored directories
        if any(d in path.parts for d in ignore_dirs):
            continue

        ext = path.suffix.lower()
        if ext in DEFAULT_INDEX_EXTENSIONS:
            extension_counts[ext] = extension_counts.get(ext, 0) + 1
            total_files += 1

            # Count lines (sample for large repos)
            if total_files <= 100:
                try:
                    total_lines += len(path.read_text(errors="ignore").splitlines())
                except Exception:
                    pass

    # Get top directories
    top_dirs = []
    for item in sorted(repo_path.iterdir()):
        if item.is_dir() and item.name not in ignore_dirs and not item.name.startswith("."):
            top_dirs.append(item.name)

    # Build summary
    top_extensions = sorted(extension_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    ext_summary = ", ".join(f"{ext}({count})" for ext, count in top_extensions)

    summary_content = f"""Codebase Summary for {repo_path.name}

Structure:
- Top-level directories: {", ".join(top_dirs[:10])}
- Total indexable files: {total_files}
- Estimated lines of code: {total_lines}

File types: {ext_summary}

This codebase was auto-indexed on {datetime.now().strftime("%Y-%m-%d %H:%M")}.
Search for specific files, functions, or patterns to explore the code."""

    return Memory(
        content=summary_content,
        type=MemoryType.FACT,
        tags=["codebase-summary", "auto-indexed", repo_path.name],
        summary=f"Codebase summary: {repo_path.name}",
        metadata={
            "total_files": total_files,
            "extensions": dict(top_extensions),
            "top_dirs": top_dirs[:10],
        },
    )
