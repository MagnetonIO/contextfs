"""
Core ContextFS class - main interface for memory operations.

Memory lineage (evolve, merge, split) is a CORE FEATURE that works
automatically based on .env configuration. No user code required.
"""

import json
import logging
import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextfs.config import get_config
from contextfs.memory_lineage import MemoryLineage, MergeStrategy
from contextfs.rag import RAGBackend
from contextfs.schemas import (
    Memory,
    MemoryType,
    Namespace,
    SearchResult,
    Session,
    SessionMessage,
)
from contextfs.storage_protocol import EdgeRelation
from contextfs.storage_router import StorageRouter

logger = logging.getLogger(__name__)


class ContextFS:
    """
    Universal AI Memory Layer.

    Provides:
    - Semantic search with RAG
    - Cross-repo namespace isolation
    - Session management
    - Git-aware context
    - Memory lineage (evolve, merge, split) - CORE FEATURE

    Memory lineage works automatically based on .env configuration.
    Configure CONTEXTFS_BACKEND to select storage backend.
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        namespace_id: str | None = None,
        auto_load: bool = True,
        auto_index: bool = True,
    ):
        """
        Initialize ContextFS.

        Args:
            data_dir: Data directory (default: ~/.contextfs)
            namespace_id: Default namespace (default: global or auto-detect from repo)
            auto_load: Load memories on startup
            auto_index: Auto-index repository on first memory save
        """
        self.config = get_config()
        self.data_dir = data_dir or self.config.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect namespace from current repo
        self._repo_path: Path | None = None
        if namespace_id is None:
            namespace_id, self._repo_path = self._detect_namespace_and_repo()
        self.namespace_id = namespace_id

        # Initialize storage using backend factory
        self._db_path = self.data_dir / self.config.sqlite_filename
        self._init_db()

        # Initialize RAG backend with configurable embedding backend
        # When chroma_host is set, uses HttpClient (server mode) instead of PersistentClient
        # When chroma_auto_server is True, auto-starts server on corruption
        self.rag = RAGBackend(
            data_dir=self.data_dir,
            embedding_model=self.config.embedding_model,
            embedding_backend=self.config.embedding_backend,
            use_gpu=self.config.use_gpu,
            parallel_workers=self.config.embedding_parallel_workers,
            chroma_host=self.config.chroma_host,
            chroma_port=self.config.chroma_port,
            chroma_auto_server=self.config.chroma_auto_server,
        )

        # Initialize FTS and Hybrid search backends
        from contextfs.fts import FTSBackend, HybridSearch

        self._fts = FTSBackend(self._db_path)
        self._hybrid = HybridSearch(fts_backend=self._fts, rag_backend=self.rag)

        # Initialize graph backend if configured
        self._graph = self._init_graph_backend()

        # Initialize unified storage router (keeps all backends in sync)
        self._storage = StorageRouter(
            db_path=self._db_path,
            rag_backend=self.rag,
            graph_backend=self._graph,
        )

        # Alias for backwards compatibility
        self.storage = self._storage

        # Initialize memory lineage (CORE FEATURE)
        self._lineage = MemoryLineage(self._storage, self._graph)

        # Auto-indexing
        self._auto_index = auto_index
        self._auto_indexer = None
        self._indexing_triggered = False

        # Current session
        self._current_session: Session | None = None

        # Auto-load memories
        if auto_load and self.config.auto_load_on_startup:
            self._load_startup_context()

    def _init_graph_backend(self):
        """Initialize graph backend based on configuration."""
        if not self.config.falkordb_enabled:
            return None

        try:
            from contextfs.graph_backend import FalkorDBBackend

            graph = FalkorDBBackend(
                host=self.config.falkordb_host,
                port=self.config.falkordb_port,
                password=self.config.falkordb_password,
                graph_name=self.config.falkordb_graph_name,
            )
            logger.info(
                f"FalkorDB graph backend enabled: {self.config.falkordb_host}:{self.config.falkordb_port}"
            )
            return graph
        except ImportError:
            logger.warning("FalkorDB not installed. Graph features using SQLite fallback.")
            return None
        except Exception as e:
            logger.warning(f"FalkorDB connection failed: {e}. Using SQLite fallback.")
            return None

    def _detect_namespace(self) -> str:
        """Detect namespace from current git repo or use global."""
        namespace_id, _ = self._detect_namespace_and_repo()
        return namespace_id

    def _detect_namespace_and_repo(self) -> tuple[str, Path | None]:
        """Detect namespace and repo path from current git repo."""
        cwd = Path.cwd()

        # Walk up to find .git
        for parent in [cwd] + list(cwd.parents):
            if (parent / ".git").exists():
                return Namespace.for_repo(str(parent)).id, parent

        return "global", None

    def _init_db(self) -> None:
        """Initialize SQLite database with Alembic migrations."""
        from contextfs.migrations.runner import run_migrations, stamp_database

        db_exists = self._db_path.exists()

        if db_exists:
            # Check if database has alembic_version table
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='alembic_version'"
            )
            has_alembic = cursor.fetchone() is not None
            conn.close()

            if not has_alembic:
                # Existing database without migrations - stamp it first
                logger.info("Stamping existing database with migration baseline")
                stamp_database(self._db_path, "001")

        # Run any pending migrations
        try:
            run_migrations(self._db_path)
        except Exception as e:
            logger.warning(f"Migration failed, falling back to legacy init: {e}")
            self._init_db_legacy()

    def _init_db_legacy(self) -> None:
        """Legacy database initialization (fallback)."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # Memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                type TEXT NOT NULL,
                tags TEXT,
                summary TEXT,
                namespace_id TEXT NOT NULL,
                source_file TEXT,
                source_repo TEXT,
                source_tool TEXT,
                project TEXT,
                session_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            )
        """)

        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                label TEXT,
                namespace_id TEXT NOT NULL,
                tool TEXT NOT NULL,
                repo_path TEXT,
                branch TEXT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                summary TEXT,
                metadata TEXT
            )
        """)

        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        # Namespaces table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS namespaces (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                parent_id TEXT,
                repo_path TEXT,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
        """)

        # FTS for text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id, content, summary, tags,
                content='memories',
                content_rowid='rowid'
            )
        """)

        # Indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace_id)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_namespace ON sessions(namespace_id)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_label ON sessions(label)")

        conn.commit()
        conn.close()

    def _load_startup_context(self) -> None:
        """Load relevant context on startup."""
        # This could load recent memories, active session, etc.
        pass

    def check_and_repair_chromadb(self) -> bool:
        """Check if ChromaDB needs repair and rebuild if necessary.

        This is called automatically when corruption is detected during
        initialization. Returns True if repair was performed.
        """
        import logging

        logger = logging.getLogger(__name__)

        if not self.rag.needs_rebuild:
            return False

        logger.info("ChromaDB was auto-recovered from corruption. Rebuilding from SQLite...")

        try:
            # Rebuild ChromaDB from SQLite (our memories are safe there)
            stats = self.rebuild_chromadb()
            logger.info(f"ChromaDB rebuilt successfully: {stats.get('rebuilt', 0)} memories")
            self.rag.mark_rebuilt()
            return True
        except Exception as e:
            logger.error(f"Failed to rebuild ChromaDB: {e}")
            return False

    # ==================== Auto-Indexing ====================

    def _get_auto_indexer(self):
        """Lazy-load the auto-indexer."""
        if self._auto_indexer is None:
            from contextfs.autoindex import AutoIndexer

            self._auto_indexer = AutoIndexer(
                config=self.config,
                db_path=self._db_path,
            )
        return self._auto_indexer

    def _maybe_auto_index(self) -> dict | None:
        """
        Trigger auto-indexing on first memory save if applicable.

        Returns indexing stats if indexing occurred, None otherwise.
        """
        if not self._auto_index or self._indexing_triggered:
            return None

        if not self._repo_path or not self._repo_path.exists():
            return None

        self._indexing_triggered = True

        indexer = self._get_auto_indexer()

        # Check if already indexed
        if indexer.is_indexed(self.namespace_id):
            logger.debug(f"Namespace {self.namespace_id} already indexed")
            return None

        # Index repository
        logger.info(f"Auto-indexing repository: {self._repo_path}")

        def on_progress(current: int, total: int, file: str) -> None:
            if current % 10 == 0 or current == total:
                logger.info(f"Indexing: {current}/{total} - {file}")

        try:
            stats = indexer.index_repository(
                repo_path=self._repo_path,
                namespace_id=self.namespace_id,
                storage=self.storage,
                on_progress=on_progress,
                incremental=True,
            )
            logger.info(
                f"Auto-indexing complete: {stats['files_indexed']} files, "
                f"{stats['memories_created']} memories"
            )
            return stats
        except Exception as e:
            logger.warning(f"Auto-indexing failed: {e}")
            return None

    def _namespace_for_path(self, repo_path: Path) -> str:
        """Get namespace ID for a repository path."""
        from contextfs.schemas import Namespace

        return Namespace.for_repo(str(repo_path)).id

    def index_repository(
        self,
        repo_path: Path | None = None,
        on_progress: Callable[[int, int, str], None] | None = None,
        incremental: bool = True,
        project: str | None = None,
        source_repo: str | None = None,
        mode: str = "all",
    ) -> dict:
        """
        Manually index a repository to ChromaDB.

        Args:
            repo_path: Repository path (default: current repo)
            on_progress: Progress callback (current, total, file)
            incremental: Only index new/changed files
            project: Project name for grouping memories across repos
            source_repo: Repository name (default: repo directory name)
            mode: "all", "files_only", or "commits_only"

        Returns:
            Indexing statistics
        """
        from contextfs.autoindex import IndexMode

        path = repo_path or self._repo_path
        if not path:
            raise ValueError("No repository path available")

        # Use namespace derived from the repo being indexed, not ctx's namespace
        namespace_id = self._namespace_for_path(Path(path))

        # Default source_repo to directory name
        if source_repo is None:
            source_repo = Path(path).name

        # Convert string mode to IndexMode enum
        index_mode = IndexMode(mode) if isinstance(mode, str) else mode

        indexer = self._get_auto_indexer()
        return indexer.index_repository(
            repo_path=path,
            namespace_id=namespace_id,
            storage=self.storage,
            on_progress=on_progress,
            incremental=incremental,
            project=project,
            source_repo=source_repo,
            mode=index_mode,
        )

    def get_index_status(self, repo_path: Path | None = None):
        """Get indexing status for a repository.

        Args:
            repo_path: Repository path (default: current working directory's repo)
        """
        if repo_path:
            namespace_id = self._namespace_for_path(repo_path)
        else:
            # Detect from current working directory
            namespace_id, _ = self._detect_namespace_and_repo()
        return self._get_auto_indexer().get_status(namespace_id)

    def clear_index(self, repo_path: Path | None = None) -> None:
        """Clear indexing status for a repository.

        Args:
            repo_path: Repository path (default: current working directory's repo)
        """
        if repo_path:
            namespace_id = self._namespace_for_path(repo_path)
        else:
            namespace_id, _ = self._detect_namespace_and_repo()
        self._get_auto_indexer().clear_index(namespace_id)
        self._indexing_triggered = False

    def list_indexes(self) -> list:
        """List all indexed repositories."""
        return self._get_auto_indexer().list_all_indexes()

    def cleanup_indexes(self, dry_run: bool = False) -> dict:
        """
        Remove indexes for repositories that no longer exist on disk.

        Args:
            dry_run: If True, only report what would be deleted without deleting

        Returns:
            Dict with 'removed' (list of removed indexes) and 'kept' (list of valid indexes)
        """
        return self._get_auto_indexer().cleanup_stale_indexes(dry_run=dry_run)

    def delete_index(self, namespace_id: str | None = None, repo_path: str | None = None) -> bool:
        """
        Delete a specific index by namespace ID or repository path.

        Args:
            namespace_id: The namespace ID to delete
            repo_path: The repository path to delete

        Returns:
            True if deleted, False if not found
        """
        indexer = self._get_auto_indexer()
        if namespace_id:
            return indexer.delete_index(namespace_id)
        elif repo_path:
            return indexer.delete_index_by_path(repo_path)
        return False

    def index_directory(
        self,
        root_dir: Path,
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
            root_dir: Root directory to scan for git repositories
            max_depth: Maximum directory depth to search (default: 5)
            on_progress: Progress callback for file indexing (current, total, file)
            on_repo_start: Callback when starting a repo (repo_name, project)
            on_repo_complete: Callback when repo completes (repo_name, stats)
            incremental: Only index new/changed files
            project_override: Override auto-detected project name for all repos

        Returns:
            Summary statistics including repos found, files indexed, etc.
        """
        indexer = self._get_auto_indexer()
        return indexer.index_directory(
            root_dir=root_dir,
            storage=self.storage,
            max_depth=max_depth,
            on_progress=on_progress,
            on_repo_start=on_repo_start,
            on_repo_complete=on_repo_complete,
            incremental=incremental,
            project_override=project_override,
        )

    def discover_repos(self, root_dir: Path, max_depth: int = 5) -> list[dict]:
        """
        Discover git repositories without indexing them.

        Args:
            root_dir: Root directory to scan
            max_depth: Maximum directory depth to search

        Returns:
            List of repo info dicts with path, name, project, suggested_tags
        """
        from contextfs.autoindex import discover_git_repos

        return discover_git_repos(root_dir, max_depth=max_depth)

    # ==================== Memory Operations ====================

    def save(
        self,
        content: str,
        type: MemoryType = MemoryType.FACT,
        tags: list[str] | None = None,
        summary: str | None = None,
        namespace_id: str | None = None,
        source_tool: str | None = None,
        source_repo: str | None = None,
        project: str | None = None,
        metadata: dict | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        id: str | None = None,
    ) -> Memory:
        """
        Save content to memory.

        Args:
            content: Content to save
            type: Memory type
            tags: Tags for categorization
            summary: Brief summary
            namespace_id: Namespace (default: current)
            source_tool: Tool that created memory (claude-code, claude-desktop, gemini, etc.)
            source_repo: Repository name/path
            project: Project name for grouping memories across repos
            metadata: Additional metadata
            created_at: Original creation timestamp (for sync, defaults to now)
            updated_at: Original update timestamp (for sync, defaults to now)
            id: Memory ID (for sync, auto-generated if not provided)

        Returns:
            Saved Memory object
        """
        # Trigger auto-indexing on first save (indexes codebase to ChromaDB)
        self._maybe_auto_index()

        # Auto-detect source_repo from repo_path
        if source_repo is None and self._repo_path:
            source_repo = self._repo_path.name

        # Auto-set project from source_repo if not provided
        if project is None and source_repo:
            project = source_repo

        memory_kwargs = {
            "content": content,
            "type": type,
            "tags": tags or [],
            "summary": summary,
            "namespace_id": namespace_id or self.namespace_id,
            "source_tool": source_tool,
            "source_repo": source_repo,
            "project": project,
            "session_id": self._current_session.id if self._current_session else None,
            "metadata": metadata or {},
            "created_at": created_at or datetime.now(timezone.utc),
            "updated_at": updated_at or datetime.now(timezone.utc),
        }
        if id is not None:
            memory_kwargs["id"] = id
        memory = Memory(**memory_kwargs)

        # Save to SQLite
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO memories (id, content, type, tags, summary, namespace_id,
                                  source_file, source_repo, source_tool, project, session_id, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                memory.id,
                memory.content,
                memory.type.value,
                json.dumps(memory.tags),
                memory.summary,
                memory.namespace_id,
                memory.source_file,
                memory.source_repo,
                memory.source_tool,
                memory.project,
                memory.session_id,
                memory.created_at.isoformat(),
                memory.updated_at.isoformat(),
                json.dumps(memory.metadata),
            ),
        )

        # Update FTS index
        # For content-linked FTS (content='memories'), we need to use the special rebuild command
        # First try the direct approach, fall back to rebuild if it fails
        try:
            # Get the rowid for this memory
            cursor.execute("SELECT rowid FROM memories WHERE id = ?", (memory.id,))
            row = cursor.fetchone()
            if row:
                rowid = row[0]
                # Delete old FTS entry using the special 'delete' command
                try:
                    cursor.execute(
                        "INSERT INTO memories_fts(memories_fts, rowid) VALUES('delete', ?)",
                        (rowid,),
                    )
                except sqlite3.OperationalError:
                    pass  # May not exist yet
                # Insert new FTS entry
                cursor.execute(
                    """
                    INSERT INTO memories_fts(rowid, id, content, summary, tags, type, namespace_id, source_repo, source_tool, project)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rowid,
                        memory.id,
                        memory.content,
                        memory.summary,
                        " ".join(memory.tags),
                        memory.type.value if hasattr(memory.type, "value") else memory.type,
                        memory.namespace_id,
                        memory.source_repo,
                        memory.source_tool,
                        memory.project,
                    ),
                )
        except sqlite3.DatabaseError:
            # If FTS is corrupted, rebuild it
            try:
                cursor.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            except sqlite3.DatabaseError:
                pass  # FTS rebuild failed, will be handled later

        conn.commit()
        conn.close()

        # Add to RAG index
        self.rag.add_memory(memory)

        return memory

    def save_batch(
        self,
        memories: list[Memory],
        skip_rag: bool = False,
    ) -> int:
        """
        Save multiple memories in a single transaction for efficiency.

        Used by sync operations to bulk-insert pulled memories.

        Args:
            memories: List of Memory objects to save
            skip_rag: If True, skip adding to RAG index (caller will rebuild)

        Returns:
            Number of memories saved
        """
        if not memories:
            return 0

        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        count = 0
        for memory in memories:
            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO memories (id, content, type, tags, summary, namespace_id,
                                      source_file, source_repo, source_tool, project, session_id, created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory.id,
                        memory.content,
                        memory.type.value if hasattr(memory.type, "value") else memory.type,
                        json.dumps(memory.tags),
                        memory.summary,
                        memory.namespace_id,
                        memory.source_file,
                        memory.source_repo,
                        memory.source_tool,
                        memory.project,
                        memory.session_id,
                        memory.created_at.isoformat()
                        if hasattr(memory.created_at, "isoformat")
                        else memory.created_at,
                        memory.updated_at.isoformat()
                        if hasattr(memory.updated_at, "isoformat")
                        else memory.updated_at,
                        json.dumps(memory.metadata) if memory.metadata else "{}",
                    ),
                )
                count += 1
            except Exception as e:
                logger.warning(f"Failed to save memory {memory.id}: {e}")

        # Commit all changes
        conn.commit()

        # Rebuild FTS index once (much faster than individual updates)
        try:
            cursor.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            conn.commit()
        except sqlite3.DatabaseError as e:
            logger.warning(f"FTS rebuild failed: {e}")

        conn.close()

        # Add to RAG index in batch
        if not skip_rag:
            for memory in memories:
                try:
                    self.rag.add_memory(memory)
                except Exception as e:
                    logger.warning(f"Failed to add memory {memory.id} to RAG: {e}")

        return count

    def search(
        self,
        query: str,
        limit: int = 10,
        type: MemoryType | None = None,
        tags: list[str] | None = None,
        namespace_id: str | None = None,
        source_tool: str | None = None,
        source_repo: str | None = None,
        project: str | None = None,
        cross_repo: bool = False,
        mode: str = "hybrid",
        use_semantic: bool | None = None,  # Deprecated, use mode instead
    ) -> list[SearchResult]:
        """
        Search memories.

        Args:
            query: Search query
            limit: Maximum results
            type: Filter by type
            tags: Filter by tags
            namespace_id: Filter by namespace (None with cross_repo=True searches all)
            source_tool: Filter by source tool (claude-code, claude-desktop, gemini, etc.)
            source_repo: Filter by source repository name
            project: Filter by project name (groups memories across repos)
            cross_repo: If True, search across all namespaces/repos
            mode: Search mode - "hybrid" (default), "semantic", "keyword", "smart"
                  - hybrid: Combines FTS + RAG using Reciprocal Rank Fusion
                  - semantic: RAG vector search only (good for conceptual queries)
                  - keyword: FTS5 keyword search only (fast, good for exact terms)
                  - smart: Routes to optimal backend based on memory type
            use_semantic: DEPRECATED - use mode="semantic" or mode="keyword" instead

        Returns:
            List of SearchResult objects
        """
        import re

        # Handle deprecated use_semantic parameter
        if use_semantic is not None:
            mode = "semantic" if use_semantic else "keyword"

        # Auto-detect memory ID pattern (8+ hex chars)
        # If query looks like a memory ID, use recall() instead of vector search
        if re.match(r"^[a-f0-9]{8,}$", query.lower().strip()):
            memory = self.recall(query.strip())
            if memory:
                # Check type filter if specified
                if type and memory.type != type:
                    return []
                return [SearchResult(memory=memory, score=1.0)]
            # If not found by ID, fall through to regular search

        # For cross-repo or project search, don't filter by namespace
        effective_namespace = (
            None if (cross_repo or project) else (namespace_id or self.namespace_id)
        )

        # Determine fetch limit (over-fetch for post-filtering)
        fetch_limit = limit * 2 if (source_tool or source_repo or project) else limit

        # Execute search based on mode
        if mode == "semantic":
            results = self.rag.search(
                query=query,
                limit=fetch_limit,
                type=type,
                tags=tags,
                namespace_id=effective_namespace,
            )
        elif mode == "keyword":
            results = self._fts.search(
                query=query,
                limit=fetch_limit,
                type=type,
                tags=tags,
                namespace_id=effective_namespace,
            )
        elif mode == "smart":
            results = self._hybrid.smart_search(
                query=query,
                limit=fetch_limit,
                type=type,
                tags=tags,
                namespace_id=effective_namespace,
            )
        else:  # Default: hybrid
            results = self._hybrid.search(
                query=query,
                limit=fetch_limit,
                type=type,
                tags=tags,
                namespace_id=effective_namespace,
            )

        # Post-filter by source_tool, source_repo, and project if specified
        if source_tool or source_repo or project:
            filtered = []
            for r in results:
                if source_tool and r.memory.source_tool != source_tool:
                    continue
                if source_repo and r.memory.source_repo != source_repo:
                    continue
                if project and r.memory.project != project:
                    continue
                filtered.append(r)
            results = filtered[:limit]

        return results

    def search_global(
        self,
        query: str,
        limit: int = 10,
        type: MemoryType | None = None,
        source_tool: str | None = None,
        source_repo: str | None = None,
    ) -> list[SearchResult]:
        """
        Search memories across all repos and namespaces.

        Args:
            query: Search query
            limit: Maximum results
            type: Filter by type
            source_tool: Filter by source tool
            source_repo: Filter by source repository

        Returns:
            List of SearchResult objects from all repos
        """
        return self.search(
            query=query,
            limit=limit,
            type=type,
            source_tool=source_tool,
            source_repo=source_repo,
            cross_repo=True,
        )

    def list_repos(self) -> list[dict]:
        """
        List all repositories with memories.

        Returns:
            List of dicts with repo info (name, namespace_id, memory_count)
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT source_repo, namespace_id, COUNT(*) as count
            FROM memories
            WHERE source_repo IS NOT NULL
            GROUP BY source_repo, namespace_id
            ORDER BY count DESC
        """)

        repos = []
        for row in cursor.fetchall():
            repos.append(
                {
                    "source_repo": row[0],
                    "namespace_id": row[1],
                    "memory_count": row[2],
                }
            )

        conn.close()
        return repos

    def list_tools(self) -> list[dict]:
        """
        List all source tools with memories.

        Returns:
            List of dicts with tool info (name, memory_count)
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT source_tool, COUNT(*) as count
            FROM memories
            WHERE source_tool IS NOT NULL
            GROUP BY source_tool
            ORDER BY count DESC
        """)

        tools = []
        for row in cursor.fetchall():
            tools.append(
                {
                    "source_tool": row[0],
                    "memory_count": row[1],
                }
            )

        conn.close()
        return tools

    def list_projects(self) -> list[dict]:
        """
        List all projects with memories.

        Returns:
            List of dicts with project info (name, repos, memory_count)
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT project, GROUP_CONCAT(DISTINCT source_repo) as repos, COUNT(*) as count
            FROM memories
            WHERE project IS NOT NULL
            GROUP BY project
            ORDER BY count DESC
        """)

        projects = []
        for row in cursor.fetchall():
            projects.append(
                {
                    "project": row[0],
                    "repos": row[1].split(",") if row[1] else [],
                    "memory_count": row[2],
                }
            )

        conn.close()
        return projects

    def search_project(
        self,
        project: str,
        query: str | None = None,
        limit: int = 10,
        type: MemoryType | None = None,
    ) -> list[SearchResult]:
        """
        Search memories within a project (across all repos in the project).

        Args:
            project: Project name
            query: Optional search query (if None, returns recent memories)
            limit: Maximum results
            type: Filter by type

        Returns:
            List of SearchResult objects
        """
        if query:
            return self.search(
                query=query,
                limit=limit,
                type=type,
                project=project,
                cross_repo=True,
            )
        else:
            # Return recent memories for project
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()

            sql = "SELECT * FROM memories WHERE project = ?"
            params = [project]

            if type:
                sql += " AND type = ?"
                params.append(type.value)

            sql += f" ORDER BY created_at DESC LIMIT {limit}"

            cursor.execute(sql, params)
            rows = cursor.fetchall()
            conn.close()

            return [SearchResult(memory=self._row_to_memory(row), score=1.0) for row in rows]

    def _fts_search(
        self,
        query: str,
        limit: int,
        type: MemoryType | None,
        tags: list[str] | None,
        namespace_id: str | None,
    ) -> list[SearchResult]:
        """Full-text search fallback."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        sql = """
            SELECT m.* FROM memories m
            JOIN memories_fts fts ON m.id = fts.id
            WHERE memories_fts MATCH ?
        """
        params = [query]

        if namespace_id:
            sql += " AND m.namespace_id = ?"
            params.append(namespace_id)

        if type:
            sql += " AND m.type = ?"
            params.append(type.value)

        sql += f" LIMIT {limit}"

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            memory = self._row_to_memory(row)
            results.append(SearchResult(memory=memory, score=0.8))

        return results

    def recall(self, memory_id: str) -> Memory | None:
        """
        Recall a specific memory by ID.

        Checks SQLite first, then falls back to ChromaDB for indexed memories.

        Args:
            memory_id: Memory ID (can be partial, at least 8 chars)

        Returns:
            Memory or None
        """
        # Use StorageRouter for unified recall (SQLite + ChromaDB fallback)
        return self.storage.recall(memory_id)

    def list_recent(
        self,
        limit: int = 10,
        type: MemoryType | None = None,
        namespace_id: str | None = None,
        source_tool: str | None = None,
        project: str | None = None,
    ) -> list[Memory]:
        """List recent memories."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        sql = "SELECT * FROM memories WHERE 1=1"
        params: list = []

        if namespace_id:
            sql += " AND namespace_id = ?"
            params.append(namespace_id)

        if type:
            sql += " AND type = ?"
            params.append(type.value)

        if source_tool:
            sql += " AND source_tool = ?"
            params.append(source_tool)

        if project:
            sql += " AND project = ?"
            params.append(project)

        sql += f" ORDER BY created_at DESC LIMIT {limit}"

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_memory(row) for row in rows]

    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # Support partial ID matching
        cursor.execute("SELECT id FROM memories WHERE id LIKE ?", (f"{memory_id}%",))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False

        full_id = row[0]
        cursor.execute("DELETE FROM memories WHERE id = ?", (full_id,))
        deleted = cursor.rowcount > 0
        cursor.execute("DELETE FROM memories_fts WHERE id = ?", (full_id,))

        conn.commit()
        conn.close()

        if deleted:
            self.rag.remove_memory(full_id)

        return deleted

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        type: MemoryType | None = None,
        tags: list[str] | None = None,
        summary: str | None = None,
        project: str | None = None,
        metadata: dict | None = None,
    ) -> Memory | None:
        """
        Update an existing memory.

        Args:
            memory_id: Memory ID (can be partial, at least 8 chars)
            content: New content (optional)
            type: New type (optional)
            tags: New tags (optional)
            summary: New summary (optional)
            project: New project (optional)
            metadata: New metadata (optional)

        Returns:
            Updated Memory or None if not found
        """
        # First, recall the existing memory
        memory = self.recall(memory_id)
        if not memory:
            return None

        # Update fields if provided
        if content is not None:
            memory.content = content
        if type is not None:
            memory.type = type
        if tags is not None:
            memory.tags = tags
        if summary is not None:
            memory.summary = summary
        if project is not None:
            memory.project = project
        if metadata is not None:
            memory.metadata = metadata

        memory.updated_at = datetime.now(timezone.utc)

        # Update in database
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE memories SET
                content = ?,
                type = ?,
                tags = ?,
                summary = ?,
                project = ?,
                updated_at = ?,
                metadata = ?
            WHERE id = ?
        """,
            (
                memory.content,
                memory.type.value,
                json.dumps(memory.tags),
                memory.summary,
                memory.project,
                memory.updated_at.isoformat(),
                json.dumps(memory.metadata),
                memory.id,
            ),
        )

        # Update FTS
        cursor.execute("DELETE FROM memories_fts WHERE id = ?", (memory.id,))
        cursor.execute(
            """
            INSERT INTO memories_fts (id, content, summary, tags)
            VALUES (?, ?, ?, ?)
        """,
            (memory.id, memory.content, memory.summary, " ".join(memory.tags)),
        )

        conn.commit()
        conn.close()

        # Update RAG index
        self.rag.remove_memory(memory.id)
        self.rag.add_memory(memory)

        return memory

    def _row_to_memory(self, row) -> Memory:
        """Convert database row to Memory object."""
        # DB schema (after migration 002):
        # id, content, type, tags, summary, namespace_id, source_file,
        # source_repo, source_tool, project, session_id, created_at,
        # updated_at, metadata
        return Memory(
            id=row[0],
            content=row[1],
            type=MemoryType(row[2]),
            tags=json.loads(row[3]) if row[3] else [],
            summary=row[4],
            namespace_id=row[5],
            source_file=row[6],
            source_repo=row[7],
            source_tool=row[8],
            project=row[9],
            session_id=row[10],
            created_at=datetime.fromisoformat(row[11]),
            updated_at=datetime.fromisoformat(row[12]),
            metadata=json.loads(row[13]) if row[13] else {},
        )

    # ==================== Memory Lineage Operations (CORE FEATURE) ====================

    def evolve(
        self,
        memory_id: str,
        new_content: str,
        summary: str | None = None,
        preserve_tags: bool | None = None,
        additional_tags: list[str] | None = None,
    ) -> Memory:
        """
        Evolve a memory by creating an updated version while preserving history.

        The original memory remains unchanged. A new memory is created with
        an EVOLVED_FROM relationship to the original. This is a CORE FEATURE
        that tracks memory changes over time.

        Args:
            memory_id: ID of memory to evolve
            new_content: Updated content for new memory
            summary: Optional new summary
            preserve_tags: Whether to copy tags (default from config)
            additional_tags: Additional tags for new memory

        Returns:
            New evolved Memory object

        Example:
            >>> mem = ctx.save("Initial documentation")
            >>> evolved = ctx.evolve(mem.id, "Updated documentation with examples")
            >>> # Original still exists, evolved has EVOLVED_FROM relationship
        """
        if preserve_tags is None:
            preserve_tags = self.config.lineage_preserve_tags

        return self._lineage.evolve(
            memory_id=memory_id,
            new_content=new_content,
            summary=summary,
            preserve_tags=preserve_tags,
            additional_tags=additional_tags,
        )

    def merge(
        self,
        memory_ids: list[str],
        merged_content: str | None = None,
        summary: str | None = None,
        strategy: str | MergeStrategy | None = None,
        memory_type: MemoryType | None = None,
    ) -> Memory:
        """
        Merge multiple memories into a single memory.

        Creates a new memory with MERGED_FROM relationships to all originals.
        Original memories are not modified. This is a CORE FEATURE for
        consolidating related information.

        Args:
            memory_ids: List of memory IDs to merge (minimum 2)
            merged_content: Content for merged memory (auto-generated if None)
            summary: Summary for merged memory
            strategy: Merge strategy (union, intersection, latest, oldest)
            memory_type: Type for merged memory

        Returns:
            New merged Memory object

        Example:
            >>> m1 = ctx.save("Auth uses JWT")
            >>> m2 = ctx.save("Auth requires 2FA")
            >>> merged = ctx.merge([m1.id, m2.id], summary="Auth documentation")
        """
        # Convert string strategy to enum
        if strategy is None:
            strategy = MergeStrategy(self.config.lineage_merge_strategy.value)
        elif isinstance(strategy, str):
            strategy = MergeStrategy(strategy)

        return self._lineage.merge(
            memory_ids=memory_ids,
            merged_content=merged_content,
            summary=summary,
            strategy=strategy,
            memory_type=memory_type,
        )

    def split(
        self,
        memory_id: str,
        parts: list[str],
        summaries: list[str] | None = None,
        preserve_tags: bool | None = None,
    ) -> list[Memory]:
        """
        Split a memory into multiple parts.

        Creates new memories with SPLIT_FROM relationships to the original.
        Original memory is not modified. This is a CORE FEATURE for
        breaking down complex information.

        Args:
            memory_id: ID of memory to split
            parts: List of content strings for each part (minimum 2)
            summaries: Optional summaries for each part
            preserve_tags: Whether to copy tags from original

        Returns:
            List of new Memory objects

        Example:
            >>> mem = ctx.save("Auth: JWT tokens. Sessions expire in 1h. 2FA required.")
            >>> parts = ctx.split(mem.id, [
            ...     "Auth uses JWT tokens",
            ...     "Sessions expire in 1 hour",
            ...     "2FA is required",
            ... ])
        """
        if preserve_tags is None:
            preserve_tags = self.config.lineage_preserve_tags

        return self._lineage.split(
            memory_id=memory_id,
            parts=parts,
            summaries=summaries,
            preserve_tags=preserve_tags,
        )

    def get_lineage(
        self,
        memory_id: str,
        direction: str = "both",
    ) -> dict[str, Any]:
        """
        Get the evolution lineage of a memory.

        Traces EVOLVED_FROM, MERGED_FROM, SPLIT_FROM relationships to
        find ancestors and descendants. This is a CORE FEATURE for
        understanding memory history.

        Args:
            memory_id: Memory ID to trace
            direction: "ancestors", "descendants", or "both"

        Returns:
            Dict with:
                - root: ID of original ancestor
                - memory: Current memory object
                - ancestors: List of ancestor memories with depths
                - descendants: List of descendant memories with depths
                - timeline: Chronologically ordered history

        Example:
            >>> lineage = ctx.get_lineage("abc123")
            >>> print(f"Root: {lineage['root']}")
            >>> for a in lineage['ancestors']:
            ...     print(f"  <- {a['memory_id']} ({a['relation']})")
        """
        return self._lineage.get_history(memory_id)

    def link(
        self,
        from_memory_id: str,
        to_memory_id: str,
        relation: str | EdgeRelation = EdgeRelation.REFERENCES,
        weight: float = 1.0,
        bidirectional: bool = False,
    ) -> bool:
        """
        Create a relationship link between two memories.

        Args:
            from_memory_id: Source memory ID
            to_memory_id: Target memory ID
            relation: Type of relationship (default: REFERENCES)
            weight: Relationship strength (0.0-1.0)
            bidirectional: Whether to create inverse edge

        Returns:
            True if link created successfully

        Example:
            >>> ctx.link("mem1", "mem2", relation="references")
            >>> ctx.link("auth_doc", "login_doc", relation="related_to", bidirectional=True)
        """
        if isinstance(relation, str):
            relation = EdgeRelation(relation)

        # Resolve partial IDs
        from_mem = self._storage.recall(from_memory_id)
        to_mem = self._storage.recall(to_memory_id)

        if not from_mem or not to_mem:
            return False

        # Use storage router directly (has SQLite fallback)
        edge = self._storage.add_edge(
            from_id=from_mem.id,
            to_id=to_mem.id,
            relation=relation,
            weight=weight,
        )

        # Create bidirectional edge if requested
        if bidirectional and edge:
            inverse_relation = self._get_inverse_relation(relation)
            self._storage.add_edge(
                from_id=to_mem.id,
                to_id=from_mem.id,
                relation=inverse_relation,
                weight=weight,
            )

        return edge is not None

    def _get_inverse_relation(self, relation: EdgeRelation) -> EdgeRelation:
        """Get inverse relation for bidirectional links."""
        inverses = {
            EdgeRelation.REFERENCES: EdgeRelation.REFERENCED_BY,
            EdgeRelation.REFERENCED_BY: EdgeRelation.REFERENCES,
            EdgeRelation.RELATED_TO: EdgeRelation.RELATED_TO,
            EdgeRelation.CONTRADICTS: EdgeRelation.CONTRADICTS,
            EdgeRelation.SUPERSEDES: EdgeRelation.SUPERSEDED_BY,
            EdgeRelation.SUPERSEDED_BY: EdgeRelation.SUPERSEDES,
            EdgeRelation.PARENT_OF: EdgeRelation.CHILD_OF,
            EdgeRelation.CHILD_OF: EdgeRelation.PARENT_OF,
            EdgeRelation.PART_OF: EdgeRelation.CONTAINS,
            EdgeRelation.CONTAINS: EdgeRelation.PART_OF,
            EdgeRelation.CAUSED_BY: EdgeRelation.CAUSES,
            EdgeRelation.CAUSES: EdgeRelation.CAUSED_BY,
        }
        return inverses.get(relation, relation)

    def get_related(
        self,
        memory_id: str,
        relation: str | EdgeRelation | None = None,
        max_depth: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Find memories related to a given memory.

        Uses graph traversal to find connected memories.

        Args:
            memory_id: Starting memory ID
            relation: Filter by relation type (None = all)
            max_depth: Maximum traversal depth

        Returns:
            List of related memories with relationship info

        Example:
            >>> related = ctx.get_related("auth_doc", max_depth=2)
            >>> for r in related:
            ...     print(f"{r['memory'].id}: {r['relation']} (depth {r['depth']})")
        """
        if isinstance(relation, str):
            relation = EdgeRelation(relation)

        # Resolve partial ID
        mem = self._storage.recall(memory_id)
        if not mem:
            return []

        # Use storage router directly (has SQLite fallback)
        results = self._storage.get_related(
            memory_id=mem.id,
            relation=relation,
            max_depth=max_depth,
        )

        # Convert GraphTraversalResult to dict format
        return [
            {
                "id": r.memory.id,
                "memory": r.memory,
                "relation": r.relation.value,
                "depth": r.depth,
                "content": r.memory.content,
                "summary": r.memory.summary,
            }
            for r in results
        ]

    def has_graph(self) -> bool:
        """Check if graph backend is available for advanced lineage operations."""
        return self._graph is not None or self._storage.has_graph()

    # ==================== Session Operations ====================

    def start_session(
        self,
        tool: str = "contextfs",
        label: str | None = None,
        repo_path: str | None = None,
        branch: str | None = None,
    ) -> Session:
        """Start a new session."""
        # End current session if exists
        if self._current_session:
            self.end_session()

        session = Session(
            tool=tool,
            label=label,
            namespace_id=self.namespace_id,
            repo_path=repo_path or str(Path.cwd()),
            branch=branch or self._get_current_branch(),
        )

        # Save to database
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO sessions (id, label, namespace_id, tool, repo_path, branch,
                                  started_at, ended_at, summary, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                session.id,
                session.label,
                session.namespace_id,
                session.tool,
                session.repo_path,
                session.branch,
                session.started_at.isoformat(),
                None,
                None,
                json.dumps(session.metadata),
            ),
        )

        conn.commit()
        conn.close()

        self._current_session = session
        return session

    def end_session(self, generate_summary: bool = True) -> None:
        """End the current session."""
        if not self._current_session:
            return

        self._current_session.end()

        # Update in database
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE sessions SET ended_at = ?, summary = ?
            WHERE id = ?
        """,
            (
                self._current_session.ended_at.isoformat(),
                self._current_session.summary,
                self._current_session.id,
            ),
        )

        conn.commit()
        conn.close()

        # Save session as episodic memory
        if generate_summary and self._current_session.messages:
            self.save(
                content=self._format_session_summary(),
                type=MemoryType.EPISODIC,
                tags=["session", self._current_session.tool],
                summary=f"Session {self._current_session.id[:8]}",
                metadata={"session_id": self._current_session.id},
            )

        self._current_session = None

    def add_message(self, role: str, content: str) -> SessionMessage:
        """Add a message to current session."""
        if not self._current_session:
            self.start_session()

        msg = self._current_session.add_message(role, content)

        # Save to database
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO messages (id, session_id, role, content, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                msg.id,
                self._current_session.id,
                msg.role,
                msg.content,
                msg.timestamp.isoformat(),
                json.dumps(msg.metadata),
            ),
        )

        conn.commit()
        conn.close()

        return msg

    def load_session(
        self,
        session_id: str | None = None,
        label: str | None = None,
    ) -> Session | None:
        """Load a session by ID or label."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        if session_id:
            cursor.execute("SELECT * FROM sessions WHERE id LIKE ?", (f"{session_id}%",))
        elif label:
            cursor.execute("SELECT * FROM sessions WHERE label = ?", (label,))
        else:
            return None

        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        session = Session(
            id=row[0],
            label=row[1],
            namespace_id=row[2],
            tool=row[3],
            repo_path=row[4],
            branch=row[5],
            started_at=datetime.fromisoformat(row[6]),
            ended_at=datetime.fromisoformat(row[7]) if row[7] else None,
            summary=row[8],
            metadata=json.loads(row[9]) if row[9] else {},
        )

        # Load messages
        cursor.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp", (session.id,)
        )
        for msg_row in cursor.fetchall():
            session.messages.append(
                SessionMessage(
                    id=msg_row[0],
                    role=msg_row[2],
                    content=msg_row[3],
                    timestamp=datetime.fromisoformat(msg_row[4]),
                    metadata=json.loads(msg_row[5]) if msg_row[5] else {},
                )
            )

        conn.close()
        return session

    def list_sessions(
        self,
        limit: int = 10,
        offset: int = 0,
        tool: str | None = None,
        label: str | None = None,
        all_namespaces: bool = False,
    ) -> list[Session]:
        """List recent sessions."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        if all_namespaces:
            sql = "SELECT * FROM sessions WHERE 1=1"
            params: list = []
        else:
            sql = "SELECT * FROM sessions WHERE namespace_id = ?"
            params = [self.namespace_id]

        if tool:
            sql += " AND tool = ?"
            params.append(tool)

        if label:
            sql += " AND label LIKE ?"
            params.append(f"%{label}%")

        sql += f" ORDER BY started_at DESC LIMIT {limit} OFFSET {offset}"

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        sessions = []
        for row in rows:
            sessions.append(
                Session(
                    id=row[0],
                    label=row[1],
                    namespace_id=row[2],
                    tool=row[3],
                    repo_path=row[4],
                    branch=row[5],
                    started_at=datetime.fromisoformat(row[6]),
                    ended_at=datetime.fromisoformat(row[7]) if row[7] else None,
                    summary=row[8],
                    metadata=json.loads(row[9]) if row[9] else {},
                )
            )

        return sessions

    def update_session(
        self,
        session_id: str,
        label: str | None = None,
        summary: str | None = None,
    ) -> Session | None:
        """
        Update an existing session.

        Args:
            session_id: Session ID (can be partial)
            label: New label (optional)
            summary: New summary (optional)

        Returns:
            Updated Session or None if not found
        """
        session = self.load_session(session_id=session_id)
        if not session:
            return None

        # Update fields if provided
        if label is not None:
            session.label = label
        if summary is not None:
            session.summary = summary

        # Update in database
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE sessions SET label = ?, summary = ?
            WHERE id = ?
        """,
            (session.label, session.summary, session.id),
        )

        conn.commit()
        conn.close()

        return session

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its messages.

        Args:
            session_id: Session ID (can be partial)

        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # Support partial ID matching
        cursor.execute("SELECT id FROM sessions WHERE id LIKE ?", (f"{session_id}%",))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return False

        full_id = row[0]

        # Delete messages first
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (full_id,))

        # Delete session
        cursor.execute("DELETE FROM sessions WHERE id = ?", (full_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return deleted

    def _format_session_summary(self) -> str:
        """Format session messages for episodic memory."""
        if not self._current_session:
            return ""

        lines = [f"Session with {self._current_session.tool}"]
        for msg in self._current_session.messages[-10:]:  # Last 10 messages
            lines.append(f"{msg.role}: {msg.content[:200]}...")

        return "\n".join(lines)

    def _get_current_branch(self) -> str | None:
        """Get current git branch."""
        try:
            head_path = Path.cwd() / ".git" / "HEAD"
            if head_path.exists():
                content = head_path.read_text().strip()
                if content.startswith("ref: refs/heads/"):
                    return content[16:]
        except Exception:
            pass
        return None

    # ==================== Context Helpers ====================

    def get_context_for_task(self, task: str, limit: int = 5) -> list[str]:
        """Get relevant context strings for a task."""
        results = self.search(task, limit=limit)
        return [r.memory.to_context_string() for r in results]

    def get_current_session(self) -> Session | None:
        """Get current active session."""
        return self._current_session

    # ==================== Cleanup ====================

    def reset_chromadb(self) -> bool:
        """
        Reset the ChromaDB database.

        Use this when ChromaDB becomes corrupted (e.g., after version upgrades).
        This will delete all vector embeddings but SQLite data remains intact.
        After reset, you should re-index to rebuild the ChromaDB database.

        Returns:
            True if reset successful, False otherwise
        """
        return self.rag.reset_database()

    def rebuild_chromadb(
        self,
        on_progress: callable = None,
    ) -> dict:
        """
        Rebuild ChromaDB from SQLite data.

        Use this to restore search capability after ChromaDB corruption
        without needing to re-index from source files.

        This is MUCH faster than re-indexing because:
        - No file scanning needed
        - No file content processing
        - Memories already exist in SQLite

        Args:
            on_progress: Callback for progress updates (current, total)

        Returns:
            Statistics dict with count of memories rebuilt
        """
        return self.storage.rebuild_chromadb_from_sqlite(on_progress=on_progress)

    def reindex_all_repos(
        self,
        incremental: bool = True,
        mode: str = "all",
        on_progress: Callable[[str, int, int], None] | None = None,
    ) -> dict:
        """
        Reindex all repositories that have been previously indexed.

        Uses stored repo paths from index_status table to reindex.
        Useful for rebuilding indexes after ChromaDB corruption or upgrades.

        Args:
            incremental: Only index new/changed files (default: True)
            mode: "all", "files_only", or "commits_only"
            on_progress: Callback (repo_name, current_repo, total_repos)

        Returns:
            Statistics dict with repos processed, files indexed, etc.
        """
        # Get repo paths from index_status table
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        # Check if index_status table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='index_status'")
        if not cursor.fetchone():
            conn.close()
            return {
                "success": True,
                "repos_found": 0,
                "repos_indexed": 0,
                "repos_failed": 0,
                "total_files": 0,
                "total_memories": 0,
                "errors": [],
            }

        cursor.execute("""
            SELECT namespace_id, repo_path
            FROM index_status
            WHERE repo_path IS NOT NULL AND repo_path != ''
        """)

        repos = []
        for row in cursor.fetchall():
            repos.append(
                {
                    "namespace_id": row[0],
                    "repo_path": row[1],
                }
            )

        conn.close()

        if not repos:
            return {
                "success": True,
                "repos_found": 0,
                "repos_indexed": 0,
                "repos_failed": 0,
                "total_files": 0,
                "total_memories": 0,
                "errors": [],
            }

        successful = 0
        failed = 0
        total_files = 0
        total_memories = 0
        errors = []

        for i, repo_info in enumerate(repos):
            repo_path = Path(repo_info["repo_path"])
            repo_name = repo_path.name

            if on_progress:
                on_progress(repo_name, i + 1, len(repos))

            # Verify path exists and is a git repo
            if not repo_path.exists():
                errors.append(f"{repo_name}: Path no longer exists: {repo_path}")
                failed += 1
                continue

            if not (repo_path / ".git").exists():
                errors.append(f"{repo_name}: Not a git repository: {repo_path}")
                failed += 1
                continue

            try:
                result = self.index_repository(
                    repo_path=repo_path,
                    incremental=incremental,
                    mode=mode,
                )
                total_files += result.get("files_indexed", 0)
                total_memories += result.get("memories_created", 0)
                successful += 1
            except Exception as e:
                errors.append(f"{repo_name}: {e!s}")
                failed += 1

        return {
            "success": failed == 0,
            "repos_found": len(repos),
            "repos_indexed": successful,
            "repos_failed": failed,
            "total_files": total_files,
            "total_memories": total_memories,
            "errors": errors,
        }

    def close(self) -> None:
        """Clean shutdown."""
        if self._current_session:
            self.end_session()
        self.rag.close()
