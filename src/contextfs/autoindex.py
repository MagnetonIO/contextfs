"""
Auto-indexing module for ContextFS.

Automatically indexes repository files on first memory save,
creating a searchable knowledge base of the codebase.
"""

import logging
import sqlite3
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

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
        memories_created: int = 0,
        repo_path: str | None = None,
        commit_hash: str | None = None,
    ):
        self.namespace_id = namespace_id
        self.indexed = indexed
        self.indexed_at = indexed_at
        self.files_indexed = files_indexed
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
                memories_created INTEGER DEFAULT 0,
                repo_path TEXT,
                commit_hash TEXT,
                metadata TEXT
            )
        """)

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

        cursor.execute("SELECT * FROM index_status WHERE namespace_id = ?", (namespace_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return IndexStatus(
            namespace_id=row[0],
            indexed=bool(row[1]),
            indexed_at=datetime.fromisoformat(row[2]) if row[2] else None,
            files_indexed=row[3],
            memories_created=row[4],
            repo_path=row[5],
            commit_hash=row[6],
        )

    def list_all_indexes(self) -> list[IndexStatus]:
        """List all indexed repositories."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT namespace_id, indexed, indexed_at, files_indexed,
                   memories_created, repo_path, commit_hash
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
                memories_created=row[4],
                repo_path=row[5],
                commit_hash=row[6],
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
    ) -> dict:
        """
        Index all files in repository to ChromaDB (vector store).

        Args:
            repo_path: Repository root path
            namespace_id: Namespace for memories
            rag_backend: RAG backend (ChromaDB) for vector storage
            on_progress: Callback for progress updates (current, total, file)
            incremental: Only index new/changed files

        Returns:
            Indexing statistics
        """
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
        )
        memories_created += git_stats["memories_created"]

        # Update index status
        self._update_status(
            namespace_id,
            repo_path,
            files_indexed,
            memories_created,
        )

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

        Returns:
            Indexing statistics
        """
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
        memories_created: int,
    ) -> None:
        """Update index status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        commit_hash = self._get_commit_hash(repo_path)

        cursor.execute(
            """
            INSERT OR REPLACE INTO index_status
            (namespace_id, indexed, indexed_at, files_indexed, memories_created, repo_path, commit_hash)
            VALUES (?, 1, ?, ?, ?, ?, ?)
        """,
            (
                namespace_id,
                datetime.now().isoformat(),
                files_indexed,
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

        conn.commit()
        conn.close()

        logger.info(f"Cleared index for namespace: {namespace_id}")


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
