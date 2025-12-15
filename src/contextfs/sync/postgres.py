"""
PostgreSQL Sync Service for ContextFS.

Provides:
- Sync local SQLite memories to PostgreSQL
- Global memory view across multiple machines
- Memory references and linking
- Conflict resolution with last-write-wins
"""

import asyncio
import contextlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from contextfs.schemas import Memory, MemoryType, SearchResult

logger = logging.getLogger(__name__)


class PostgresSync:
    """
    Sync service for PostgreSQL.

    Features:
    - Bidirectional sync between local SQLite and PostgreSQL
    - Incremental sync based on updated_at timestamp
    - Conflict resolution (configurable: last-write-wins, local-wins, remote-wins)
    - Memory references for cross-repo linking
    - Full-text search on PostgreSQL side
    """

    def __init__(
        self,
        connection_string: str | None = None,
        conflict_resolution: str = "last-write-wins",
    ):
        """
        Initialize PostgreSQL sync.

        Args:
            connection_string: PostgreSQL connection string
                Default: CONTEXTFS_POSTGRES_URL env var or
                         postgresql://contextfs:contextfs@localhost:5432/contextfs
            conflict_resolution: Strategy for conflicts
                - "last-write-wins": Most recent updated_at wins
                - "local-wins": Local changes always win
                - "remote-wins": Remote changes always win
        """
        self.connection_string = connection_string or os.environ.get(
            "CONTEXTFS_POSTGRES_URL", "postgresql://contextfs:contextfs@localhost:5432/contextfs"
        )
        self.conflict_resolution = conflict_resolution
        self._pool = None

    async def _ensure_pool(self):
        """Ensure connection pool is initialized."""
        if self._pool is not None:
            return

        try:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=10,
            )
            await self._init_schema()
        except ImportError:
            raise ImportError("asyncpg not installed. Install with: pip install asyncpg")

    async def _init_schema(self) -> None:
        """Initialize PostgreSQL schema."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL,
                    tags JSONB DEFAULT '[]',
                    summary TEXT,
                    namespace_id TEXT NOT NULL,
                    source_file TEXT,
                    source_repo TEXT,
                    session_id TEXT,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    machine_id TEXT,
                    sync_version INTEGER DEFAULT 1
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    label TEXT,
                    namespace_id TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    repo_path TEXT,
                    branch TEXT,
                    started_at TIMESTAMPTZ NOT NULL,
                    ended_at TIMESTAMPTZ,
                    summary TEXT,
                    metadata JSONB DEFAULT '{}',
                    machine_id TEXT,
                    sync_version INTEGER DEFAULT 1
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    metadata JSONB DEFAULT '{}'
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_references (
                    id TEXT PRIMARY KEY,
                    source_memory_id TEXT NOT NULL REFERENCES memories(id),
                    target_memory_id TEXT NOT NULL REFERENCES memories(id),
                    reference_type TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(source_memory_id, target_memory_id, reference_type)
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_state (
                    machine_id TEXT PRIMARY KEY,
                    last_sync_at TIMESTAMPTZ,
                    sync_version INTEGER DEFAULT 0
                )
            """)

            # Indexes
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace_id)"
            )
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_machine ON memories(machine_id)"
            )

            # Full-text search
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_fts ON memories
                USING gin(to_tsvector('english', content || ' ' || COALESCE(summary, '')))
            """)

    async def sync_all(self, ctx) -> int:
        """
        Sync all memories from local to PostgreSQL.

        Args:
            ctx: ContextFS instance

        Returns:
            Number of memories synced
        """
        await self._ensure_pool()

        # Get machine ID
        machine_id = self._get_machine_id()

        # Get all local memories
        memories = ctx.list_recent(limit=100000)

        synced = 0
        async with self._pool.acquire() as conn:
            for memory in memories:
                try:
                    await self._upsert_memory(conn, memory, machine_id)
                    synced += 1
                except Exception as e:
                    logger.error(f"Failed to sync memory {memory.id}: {e}")

        logger.info(f"Synced {synced} memories to PostgreSQL")
        return synced

    async def sync_incremental(self, ctx, since: datetime | None = None) -> int:
        """
        Incremental sync - only sync memories updated since last sync.

        Args:
            ctx: ContextFS instance
            since: Sync memories updated after this time (default: last sync time)

        Returns:
            Number of memories synced
        """
        await self._ensure_pool()

        machine_id = self._get_machine_id()

        # Get last sync time
        if since is None:
            since = await self._get_last_sync_time(machine_id)

        # Get updated memories from local
        memories = ctx.list_recent(limit=100000)
        if since:
            memories = [m for m in memories if m.updated_at > since]

        synced = 0
        async with self._pool.acquire() as conn:
            for memory in memories:
                try:
                    await self._upsert_memory(conn, memory, machine_id)
                    synced += 1
                except Exception as e:
                    logger.error(f"Failed to sync memory {memory.id}: {e}")

            # Update sync state
            await self._update_sync_state(conn, machine_id)

        logger.info(f"Incrementally synced {synced} memories")
        return synced

    async def pull(self, ctx, namespace_id: str | None = None) -> int:
        """
        Pull memories from PostgreSQL to local.

        Args:
            ctx: ContextFS instance
            namespace_id: Filter by namespace (optional)

        Returns:
            Number of memories pulled
        """
        await self._ensure_pool()

        machine_id = self._get_machine_id()

        async with self._pool.acquire() as conn:
            sql = """
                SELECT id, content, type, tags, summary, namespace_id,
                       source_file, source_repo, session_id, created_at,
                       updated_at, metadata
                FROM memories
                WHERE machine_id != $1
            """
            params = [machine_id]

            if namespace_id:
                sql += " AND namespace_id = $2"
                params.append(namespace_id)

            rows = await conn.fetch(sql, *params)

        pulled = 0
        for row in rows:
            try:
                # Check if memory exists locally
                existing = ctx.recall(row["id"])

                should_update = False
                if existing is None or self.conflict_resolution == "remote-wins":
                    should_update = True
                elif self.conflict_resolution == "last-write-wins":
                    remote_updated = row["updated_at"]
                    if remote_updated > existing.updated_at:
                        should_update = True

                if should_update:
                    # Save to local
                    ctx.save(
                        content=row["content"],
                        type=MemoryType(row["type"]),
                        tags=row["tags"] or [],
                        summary=row["summary"],
                        namespace_id=row["namespace_id"],
                        metadata=row["metadata"] or {},
                    )
                    pulled += 1

            except Exception as e:
                logger.error(f"Failed to pull memory {row['id']}: {e}")

        logger.info(f"Pulled {pulled} memories from PostgreSQL")
        return pulled

    async def search_global(
        self,
        query: str,
        limit: int = 20,
        type: MemoryType | None = None,
        namespace_id: str | None = None,
    ) -> list[SearchResult]:
        """
        Search across all machines in PostgreSQL.

        Args:
            query: Search query
            limit: Maximum results
            type: Filter by type
            namespace_id: Filter by namespace

        Returns:
            List of SearchResult objects
        """
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            sql = """
                SELECT m.*,
                       ts_rank(to_tsvector('english', content || ' ' || COALESCE(summary, '')),
                               plainto_tsquery('english', $1)) as rank
                FROM memories m
                WHERE to_tsvector('english', content || ' ' || COALESCE(summary, ''))
                      @@ plainto_tsquery('english', $1)
            """
            params = [query]

            if namespace_id:
                sql += f" AND namespace_id = ${len(params) + 1}"
                params.append(namespace_id)

            if type:
                sql += f" AND type = ${len(params) + 1}"
                params.append(type.value)

            sql += f" ORDER BY rank DESC LIMIT ${len(params) + 1}"
            params.append(limit)

            rows = await conn.fetch(sql, *params)

        results = []
        for row in rows:
            memory = Memory(
                id=row["id"],
                content=row["content"],
                type=MemoryType(row["type"]),
                tags=row["tags"] or [],
                summary=row["summary"],
                namespace_id=row["namespace_id"],
                source_file=row["source_file"],
                source_repo=row["source_repo"],
                session_id=row["session_id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata=row["metadata"] or {},
            )
            results.append(
                SearchResult(
                    memory=memory,
                    score=float(row["rank"]),
                )
            )

        return results

    async def add_reference(
        self,
        source_id: str,
        target_id: str,
        reference_type: str = "related",
        metadata: dict | None = None,
    ) -> None:
        """
        Add a reference between two memories.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            reference_type: Type of reference (related, derived, supersedes, etc.)
            metadata: Additional metadata
        """
        await self._ensure_pool()

        import uuid

        ref_id = str(uuid.uuid4())[:12]

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory_references (id, source_memory_id, target_memory_id,
                                               reference_type, metadata)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (source_memory_id, target_memory_id, reference_type)
                DO UPDATE SET metadata = $5
            """,
                ref_id,
                source_id,
                target_id,
                reference_type,
                json.dumps(metadata or {}),
            )

    async def get_references(
        self,
        memory_id: str,
        direction: str = "both",
    ) -> list[dict]:
        """
        Get references for a memory.

        Args:
            memory_id: Memory ID
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of reference dicts
        """
        await self._ensure_pool()

        references = []
        async with self._pool.acquire() as conn:
            if direction in ("outgoing", "both"):
                rows = await conn.fetch(
                    """
                    SELECT r.*, m.content, m.type, m.summary
                    FROM memory_references r
                    JOIN memories m ON r.target_memory_id = m.id
                    WHERE r.source_memory_id = $1
                """,
                    memory_id,
                )
                for row in rows:
                    references.append(
                        {
                            "direction": "outgoing",
                            "reference_type": row["reference_type"],
                            "memory_id": row["target_memory_id"],
                            "content_preview": row["content"][:200],
                            "type": row["type"],
                        }
                    )

            if direction in ("incoming", "both"):
                rows = await conn.fetch(
                    """
                    SELECT r.*, m.content, m.type, m.summary
                    FROM memory_references r
                    JOIN memories m ON r.source_memory_id = m.id
                    WHERE r.target_memory_id = $1
                """,
                    memory_id,
                )
                for row in rows:
                    references.append(
                        {
                            "direction": "incoming",
                            "reference_type": row["reference_type"],
                            "memory_id": row["source_memory_id"],
                            "content_preview": row["content"][:200],
                            "type": row["type"],
                        }
                    )

        return references

    async def get_global_stats(self) -> dict:
        """Get global statistics across all machines."""
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM memories")
            by_type = await conn.fetch("SELECT type, COUNT(*) as count FROM memories GROUP BY type")
            by_machine = await conn.fetch(
                "SELECT machine_id, COUNT(*) as count FROM memories GROUP BY machine_id"
            )
            namespaces = await conn.fetch("SELECT DISTINCT namespace_id FROM memories")
            sessions = await conn.fetchval("SELECT COUNT(*) FROM sessions")
            references = await conn.fetchval("SELECT COUNT(*) FROM memory_references")

        return {
            "total_memories": total,
            "memories_by_type": {row["type"]: row["count"] for row in by_type},
            "memories_by_machine": {row["machine_id"]: row["count"] for row in by_machine},
            "namespaces": [row["namespace_id"] for row in namespaces],
            "total_sessions": sessions,
            "total_references": references,
        }

    # ==================== Internal Methods ====================

    async def _upsert_memory(self, conn, memory: Memory, machine_id: str) -> None:
        """Upsert a memory to PostgreSQL."""
        await conn.execute(
            """
            INSERT INTO memories (id, content, type, tags, summary, namespace_id,
                                  source_file, source_repo, session_id, created_at,
                                  updated_at, metadata, machine_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                type = EXCLUDED.type,
                tags = EXCLUDED.tags,
                summary = EXCLUDED.summary,
                updated_at = EXCLUDED.updated_at,
                metadata = EXCLUDED.metadata,
                sync_version = memories.sync_version + 1
        """,
            memory.id,
            memory.content,
            memory.type.value,
            json.dumps(memory.tags),
            memory.summary,
            memory.namespace_id,
            memory.source_file,
            memory.source_repo,
            memory.session_id,
            memory.created_at,
            memory.updated_at,
            json.dumps(memory.metadata),
            machine_id,
        )

    async def _get_last_sync_time(self, machine_id: str) -> datetime | None:
        """Get last sync time for machine."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT last_sync_at FROM sync_state WHERE machine_id = $1", machine_id
            )
            return row["last_sync_at"] if row else None

    async def _update_sync_state(self, conn, machine_id: str) -> None:
        """Update sync state for machine."""
        await conn.execute(
            """
            INSERT INTO sync_state (machine_id, last_sync_at, sync_version)
            VALUES ($1, NOW(), 1)
            ON CONFLICT (machine_id) DO UPDATE SET
                last_sync_at = NOW(),
                sync_version = sync_state.sync_version + 1
        """,
            machine_id,
        )

    def _get_machine_id(self) -> str:
        """Get unique machine identifier."""
        import socket
        import uuid

        # Try to use hostname + MAC for consistent ID
        try:
            hostname = socket.gethostname()
            mac = uuid.getnode()
            return f"{hostname}-{mac:012x}"[:32]
        except Exception:
            # Fallback to random UUID (stored in config)
            config_path = Path.home() / ".contextfs" / "machine_id"
            if config_path.exists():
                return config_path.read_text().strip()
            else:
                machine_id = str(uuid.uuid4())[:12]
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(machine_id)
                return machine_id

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None


class SyncDaemon:
    """
    Background daemon for automatic syncing.

    Runs periodic sync in the background.
    """

    def __init__(
        self,
        ctx,
        sync: PostgresSync,
        interval: int = 300,  # 5 minutes
    ):
        """
        Initialize sync daemon.

        Args:
            ctx: ContextFS instance
            sync: PostgresSync instance
            interval: Sync interval in seconds
        """
        self.ctx = ctx
        self.sync = sync
        self.interval = interval
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the sync daemon."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info(f"Sync daemon started (interval: {self.interval}s)")

    async def stop(self) -> None:
        """Stop the sync daemon."""
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        logger.info("Sync daemon stopped")

    async def _run(self) -> None:
        """Run sync loop."""
        while self._running:
            try:
                await self.sync.sync_incremental(self.ctx)
            except Exception as e:
                logger.error(f"Sync error: {e}")

            await asyncio.sleep(self.interval)
