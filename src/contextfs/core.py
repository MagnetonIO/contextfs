"""
Core ContextFS class - main interface for memory operations.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from contextfs.config import get_config
from contextfs.rag import RAGBackend
from contextfs.schemas import (
    Memory,
    MemoryType,
    Namespace,
    SearchResult,
    Session,
    SessionMessage,
)


class ContextFS:
    """
    Universal AI Memory Layer.

    Provides:
    - Semantic search with RAG
    - Cross-repo namespace isolation
    - Session management
    - Git-aware context
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        namespace_id: str | None = None,
        auto_load: bool = True,
    ):
        """
        Initialize ContextFS.

        Args:
            data_dir: Data directory (default: ~/.contextfs)
            namespace_id: Default namespace (default: global or auto-detect from repo)
            auto_load: Load memories on startup
        """
        self.config = get_config()
        self.data_dir = data_dir or self.config.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect namespace from current repo
        if namespace_id is None:
            namespace_id = self._detect_namespace()
        self.namespace_id = namespace_id

        # Initialize storage
        self._db_path = self.data_dir / "context.db"
        self._init_db()

        # Initialize RAG backend
        self.rag = RAGBackend(
            data_dir=self.data_dir,
            embedding_model=self.config.embedding_model,
        )

        # Current session
        self._current_session: Session | None = None

        # Auto-load memories
        if auto_load and self.config.auto_load_on_startup:
            self._load_startup_context()

    def _detect_namespace(self) -> str:
        """Detect namespace from current git repo or use global."""
        cwd = Path.cwd()

        # Walk up to find .git
        for parent in [cwd] + list(cwd.parents):
            if (parent / ".git").exists():
                return Namespace.for_repo(str(parent)).id

        return "global"

    def _init_db(self) -> None:
        """Initialize SQLite database."""
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

    # ==================== Memory Operations ====================

    def save(
        self,
        content: str,
        type: MemoryType = MemoryType.FACT,
        tags: list[str] | None = None,
        summary: str | None = None,
        namespace_id: str | None = None,
        metadata: dict | None = None,
    ) -> Memory:
        """
        Save content to memory.

        Args:
            content: Content to save
            type: Memory type
            tags: Tags for categorization
            summary: Brief summary
            namespace_id: Namespace (default: current)
            metadata: Additional metadata

        Returns:
            Saved Memory object
        """
        memory = Memory(
            content=content,
            type=type,
            tags=tags or [],
            summary=summary,
            namespace_id=namespace_id or self.namespace_id,
            session_id=self._current_session.id if self._current_session else None,
            metadata=metadata or {},
        )

        # Save to SQLite
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO memories (id, content, type, tags, summary, namespace_id,
                                  source_file, source_repo, session_id, created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                memory.session_id,
                memory.created_at.isoformat(),
                memory.updated_at.isoformat(),
                json.dumps(memory.metadata),
            ),
        )

        # Update FTS
        cursor.execute(
            """
            INSERT INTO memories_fts (id, content, summary, tags)
            VALUES (?, ?, ?, ?)
        """,
            (memory.id, memory.content, memory.summary, " ".join(memory.tags)),
        )

        conn.commit()
        conn.close()

        # Add to RAG index
        self.rag.add_memory(memory)

        return memory

    def search(
        self,
        query: str,
        limit: int = 10,
        type: MemoryType | None = None,
        tags: list[str] | None = None,
        namespace_id: str | None = None,
        use_semantic: bool = True,
    ) -> list[SearchResult]:
        """
        Search memories.

        Args:
            query: Search query
            limit: Maximum results
            type: Filter by type
            tags: Filter by tags
            namespace_id: Filter by namespace
            use_semantic: Use semantic search (vs FTS only)

        Returns:
            List of SearchResult objects
        """
        if use_semantic:
            return self.rag.search(
                query=query,
                limit=limit,
                type=type,
                tags=tags,
                namespace_id=namespace_id or self.namespace_id,
            )
        else:
            return self._fts_search(query, limit, type, tags, namespace_id)

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

        Args:
            memory_id: Memory ID (can be partial, at least 8 chars)

        Returns:
            Memory or None
        """
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM memories WHERE id LIKE ?", (f"{memory_id}%",))
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_memory(row)
        return None

    def list_recent(
        self,
        limit: int = 10,
        type: MemoryType | None = None,
        namespace_id: str | None = None,
    ) -> list[Memory]:
        """List recent memories."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        sql = "SELECT * FROM memories WHERE 1=1"
        params = []

        if namespace_id:
            sql += " AND namespace_id = ?"
            params.append(namespace_id)

        if type:
            sql += " AND type = ?"
            params.append(type.value)

        sql += f" ORDER BY created_at DESC LIMIT {limit}"

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_memory(row) for row in rows]

    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        cursor.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        if deleted:
            self.rag.remove_memory(memory_id)

        return deleted

    def _row_to_memory(self, row) -> Memory:
        """Convert database row to Memory object."""
        return Memory(
            id=row[0],
            content=row[1],
            type=MemoryType(row[2]),
            tags=json.loads(row[3]) if row[3] else [],
            summary=row[4],
            namespace_id=row[5],
            source_file=row[6],
            source_repo=row[7],
            session_id=row[8],
            created_at=datetime.fromisoformat(row[9]),
            updated_at=datetime.fromisoformat(row[10]),
            metadata=json.loads(row[11]) if row[11] else {},
        )

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
        tool: str | None = None,
        label: str | None = None,
    ) -> list[Session]:
        """List recent sessions."""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()

        sql = "SELECT * FROM sessions WHERE namespace_id = ?"
        params = [self.namespace_id]

        if tool:
            sql += " AND tool = ?"
            params.append(tool)

        if label:
            sql += " AND label LIKE ?"
            params.append(f"%{label}%")

        sql += f" ORDER BY started_at DESC LIMIT {limit}"

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

    def close(self) -> None:
        """Clean shutdown."""
        if self._current_session:
            self.end_session()
        self.rag.close()
