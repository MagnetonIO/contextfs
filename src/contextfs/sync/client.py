"""Client-side sync service.

Provides the SyncClient class for syncing local memories with a
remote sync server. Integrates with the local ContextFS instance
for SQLite operations.

Features:
- Vector clock conflict resolution
- Content hashing for deduplication
- Soft deletes (never hard delete during sync)
- Incremental sync based on timestamps
- Path normalization for cross-machine sync
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import socket
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from contextfs.sync.path_resolver import PathResolver, PortablePath
from contextfs.sync.protocol import (
    DeviceInfo,
    DeviceRegistration,
    EntityManifestEntry,
    SyncDiffResponse,
    SyncedEdge,
    SyncedMemory,
    SyncedSession,
    SyncManifestRequest,
    SyncPullRequest,
    SyncPullResponse,
    SyncPushRequest,
    SyncPushResponse,
    SyncResult,
    SyncStatusRequest,
    SyncStatusResponse,
)
from contextfs.sync.vector_clock import DeviceTracker, VectorClock

if TYPE_CHECKING:
    from contextfs import ContextFS
    from contextfs.schemas import Memory

logger = logging.getLogger(__name__)


def _ensure_tz_aware(dt: datetime | None) -> datetime | None:
    """Ensure datetime is timezone-aware.

    SQLite stores datetimes as local time without timezone info.
    This function converts naive datetimes (assumed local) to UTC.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Naive datetime from SQLite is local time - convert to UTC
        return dt.astimezone(timezone.utc)
    return dt


class SyncClient:
    """
    Client for syncing local memories to a remote sync server.

    Integrates with the local ContextFS instance to read/write
    SQLite data and sync with the remote PostgreSQL server.

    Features:
    - Vector clock conflict resolution
    - Content hashing for deduplication
    - Soft deletes (never hard delete during sync)
    - Incremental sync based on timestamps
    - Path normalization for cross-machine sync

    Usage:
        from contextfs import ContextFS
        from contextfs.sync import SyncClient

        ctx = ContextFS()
        client = SyncClient("http://localhost:8766", ctx=ctx)

        await client.register_device("My Laptop", "darwin")
        result = await client.sync_all()
    """

    def __init__(
        self,
        server_url: str,
        ctx: ContextFS | None = None,
        device_id: str | None = None,
        timeout: float = 30.0,
    ):
        """
        Initialize sync client.

        Args:
            server_url: Base URL of sync server (e.g., http://localhost:8766)
            ctx: ContextFS instance for local storage (auto-created if not provided)
            device_id: Unique device identifier (auto-generated if not provided)
            timeout: HTTP request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self._ctx = ctx
        self.device_id = device_id or self._get_or_create_device_id()
        self._client = httpx.AsyncClient(timeout=timeout)
        self._path_resolver = PathResolver()
        self._device_tracker = DeviceTracker()
        self._server_url = server_url

        # Sync state (loaded from SQLite)
        self._last_sync: datetime | None = None
        self._last_push: datetime | None = None
        self._last_pull: datetime | None = None
        self._load_sync_state()

    @property
    def ctx(self) -> ContextFS:
        """Get ContextFS instance, creating if needed."""
        if self._ctx is None:
            from contextfs import ContextFS

            self._ctx = ContextFS()
        return self._ctx

    # =========================================================================
    # Device Management
    # =========================================================================

    def _get_or_create_device_id(self) -> str:
        """Get or create a persistent device ID."""
        config_path = Path.home() / ".contextfs" / "device_id"
        if config_path.exists():
            return config_path.read_text().strip()

        # Generate unique device ID
        hostname = socket.gethostname()
        mac = uuid.getnode()
        device_id = f"{hostname}-{mac:012x}"[:32]

        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(device_id)
        return device_id

    def _get_db_path(self) -> Path:
        """Get path to SQLite database (same as ContextFS)."""
        return self.ctx._db_path

    def _ensure_sync_state_table(self, conn: sqlite3.Connection) -> None:
        """Ensure sync_state table exists."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sync_state (
                id INTEGER PRIMARY KEY,
                device_id TEXT NOT NULL UNIQUE,
                server_url TEXT,
                last_sync_at TIMESTAMP,
                last_push_at TIMESTAMP,
                last_pull_at TIMESTAMP,
                device_tracker TEXT DEFAULT '{}',
                registered_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

    def _load_sync_state(self) -> None:
        """Load sync state from SQLite database."""
        db_path = self._get_db_path()
        if not db_path.exists():
            return

        try:
            conn = sqlite3.connect(db_path)
            self._ensure_sync_state_table(conn)

            cursor = conn.execute(
                "SELECT last_sync_at, last_push_at, last_pull_at, device_tracker "
                "FROM sync_state WHERE device_id = ?",
                (self.device_id,),
            )
            row = cursor.fetchone()
            if row:
                if row[0]:
                    self._last_sync = datetime.fromisoformat(row[0])
                if row[1]:
                    self._last_push = datetime.fromisoformat(row[1])
                if row[2]:
                    self._last_pull = datetime.fromisoformat(row[2])
                if row[3]:
                    self._device_tracker = DeviceTracker.from_dict(json.loads(row[3]))
            conn.close()
        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load sync state: {e}")

    def _save_sync_state(self) -> None:
        """Save sync state to SQLite database."""
        db_path = self._get_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            conn = sqlite3.connect(db_path)
            self._ensure_sync_state_table(conn)

            conn.execute(
                """
                INSERT INTO sync_state (device_id, server_url, last_sync_at, last_push_at, last_pull_at, device_tracker, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(device_id) DO UPDATE SET
                    server_url = excluded.server_url,
                    last_sync_at = excluded.last_sync_at,
                    last_push_at = excluded.last_push_at,
                    last_pull_at = excluded.last_pull_at,
                    device_tracker = excluded.device_tracker,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    self.device_id,
                    self._server_url,
                    self._last_sync.isoformat() if self._last_sync else None,
                    self._last_push.isoformat() if self._last_push else None,
                    self._last_pull.isoformat() if self._last_pull else None,
                    json.dumps(self._device_tracker.to_dict()),
                ),
            )
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            logger.warning(f"Failed to save sync state: {e}")

    async def register_device(
        self,
        device_name: str | None = None,
        device_platform: str | None = None,
    ) -> DeviceInfo:
        """
        Register this device with the sync server.

        Args:
            device_name: Human-readable device name (defaults to hostname)
            device_platform: Platform name (defaults to current platform)

        Returns:
            DeviceInfo with registration details
        """
        registration = DeviceRegistration(
            device_id=self.device_id,
            device_name=device_name or socket.gethostname(),
            platform=device_platform or platform.system().lower(),
            client_version="0.1.0",
        )

        response = await self._client.post(
            f"{self.server_url}/api/sync/register",
            json=registration.model_dump(mode="json"),
        )
        response.raise_for_status()

        info = DeviceInfo.model_validate(response.json())
        self._device_tracker.update(self.device_id)
        self._save_sync_state()

        logger.info(f"Device registered: {info.device_id}")
        return info

    # =========================================================================
    # Content Hashing
    # =========================================================================

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute SHA-256 hash of content for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # =========================================================================
    # Path Normalization
    # =========================================================================

    def _normalize_memory_paths(self, memory: Memory) -> dict[str, str | None]:
        """Normalize memory paths for portable sync."""
        result: dict[str, str | None] = {
            "repo_url": None,
            "repo_name": None,
            "relative_path": None,
        }

        source_file = getattr(memory, "source_file", None)
        if source_file and Path(source_file).is_absolute():
            portable = self._path_resolver.normalize(source_file)
            if portable.is_valid():
                result["repo_url"] = portable.repo_url
                result["repo_name"] = portable.repo_name
                result["relative_path"] = portable.relative_path

        return result

    def _resolve_memory_paths(self, memory: SyncedMemory) -> dict[str, str | None]:
        """Resolve portable paths to local paths."""
        result: dict[str, str | None] = {
            "source_file": None,
            "source_repo": None,
        }

        if memory.repo_url and memory.relative_path:
            portable = PortablePath(
                repo_url=memory.repo_url,
                repo_name=memory.repo_name,
                relative_path=memory.relative_path,
            )
            local_path = self._path_resolver.resolve(portable)
            if local_path:
                result["source_file"] = str(local_path)
                result["source_repo"] = str(local_path.parent)

        return result

    # =========================================================================
    # Push Operations
    # =========================================================================

    async def push(
        self,
        memories: list[Memory] | None = None,
        namespace_ids: list[str] | None = None,
        push_all: bool = False,
    ) -> SyncPushResponse:
        """
        Push local changes to server.

        Args:
            memories: List of Memory objects to sync (queries local if not provided)
            namespace_ids: Namespace filter for querying local memories
            push_all: If True, push all memories regardless of last sync time

        Returns:
            SyncPushResponse with accepted/rejected counts and conflicts
        """
        if memories is None:
            # Query local memories changed since last sync (or all if push_all)
            memories = self._get_local_changes(namespace_ids, push_all=push_all)

        synced_memories = []
        memory_clocks: dict[str, VectorClock] = {}  # Track clocks for updating after push

        # Get embeddings from ChromaDB for all memories being pushed
        memory_embeddings = self._get_embeddings_from_chroma([m.id for m in memories])

        for m in memories:
            # Get vector clock from metadata (stored after previous sync)
            clock_data = m.metadata.get("_vector_clock") if m.metadata else None
            if isinstance(clock_data, str):
                clock = VectorClock.from_json(clock_data)
            elif isinstance(clock_data, dict):
                clock = VectorClock.from_dict(clock_data)
            else:
                clock = VectorClock()

            clock = clock.increment(self.device_id)
            memory_clocks[m.id] = clock  # Save for later update

            # Normalize paths
            paths = self._normalize_memory_paths(m)

            synced_memories.append(
                SyncedMemory(
                    id=m.id,
                    content=m.content,
                    type=m.type.value if hasattr(m.type, "value") else str(m.type),
                    tags=m.tags,
                    summary=m.summary,
                    namespace_id=m.namespace_id,
                    repo_url=paths["repo_url"],
                    repo_name=paths["repo_name"],
                    relative_path=paths["relative_path"],
                    source_file=m.source_file,
                    source_repo=m.source_repo,
                    source_tool=getattr(m, "source_tool", None),
                    project=getattr(m, "project", None),
                    session_id=getattr(m, "session_id", None),
                    created_at=m.created_at,
                    updated_at=m.updated_at,
                    vector_clock=clock.to_dict(),
                    content_hash=self.compute_content_hash(m.content),
                    deleted_at=getattr(m, "deleted_at", None),
                    metadata=m.metadata,
                    # Include embedding for sync to server
                    embedding=memory_embeddings.get(m.id),
                )
            )

        # Get sessions and edges to push
        sessions = self._get_sessions_to_push(push_all=push_all)
        edges = self._get_edges_to_push(push_all=push_all)

        request = SyncPushRequest(
            device_id=self.device_id,
            memories=synced_memories,
            sessions=sessions,
            edges=edges,
            last_sync_timestamp=self._last_sync,
        )

        response = await self._client.post(
            f"{self.server_url}/api/sync/push",
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()

        result = SyncPushResponse.model_validate(response.json())
        self._last_sync = result.server_timestamp
        self._last_push = result.server_timestamp
        self._save_sync_state()

        # Update local memories with new vector clocks after successful push
        if result.accepted > 0:
            self._update_local_vector_clocks(memories, memory_clocks)

        logger.info(
            f"Push complete: {result.accepted} accepted, "
            f"{result.rejected} rejected, {len(result.conflicts)} conflicts, "
            f"{len(sessions)} sessions, {len(edges)} edges"
        )
        return result

    def _update_local_vector_clocks(
        self, memories: list[Memory], clocks: dict[str, VectorClock]
    ) -> None:
        """Update local memories with vector clocks after successful push."""
        conn = sqlite3.connect(self._get_db_path())
        cursor = conn.cursor()

        for m in memories:
            if m.id in clocks:
                # Update metadata with vector clock
                metadata = m.metadata.copy() if m.metadata else {}
                metadata["_vector_clock"] = clocks[m.id].to_dict()
                metadata["_content_hash"] = self.compute_content_hash(m.content)

                cursor.execute(
                    "UPDATE memories SET metadata = ? WHERE id = ?",
                    (json.dumps(metadata), m.id),
                )

        conn.commit()
        conn.close()

    def _get_local_changes(
        self,
        namespace_ids: list[str] | None = None,
        push_all: bool = False,
    ) -> list[Memory]:
        """Get local memories changed since last sync.

        Args:
            namespace_ids: Filter by namespaces
            push_all: If True, return all memories regardless of last sync time
        """
        # Query local database for changed memories
        # No limit for push_all to ensure all memories are synced
        limit = None if push_all else 10000
        memories = self.ctx.list_recent(limit=limit) if limit else self._get_all_memories()

        # Filter by namespace if specified
        if namespace_ids:
            memories = [m for m in memories if m.namespace_id in namespace_ids]

        # Filter by last sync time if we have one (unless push_all)
        if self._last_sync and not push_all:
            last_sync_aware = _ensure_tz_aware(self._last_sync)
            memories = [m for m in memories if _ensure_tz_aware(m.updated_at) > last_sync_aware]

        return memories

    def _get_all_memories(self) -> list[Memory]:
        """Get all memories from local database without limit."""
        import sqlite3

        conn = sqlite3.connect(self.ctx._db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM memories ORDER BY updated_at DESC")
        rows = cursor.fetchall()
        conn.close()

        return [self.ctx._row_to_memory(row) for row in rows]

    def _get_sessions_to_push(self, push_all: bool = False) -> list[SyncedSession]:
        """Get local sessions changed since last sync."""
        import sqlite3

        conn = sqlite3.connect(self.ctx._db_path)
        conn.row_factory = sqlite3.Row

        # Check if sessions table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        )
        if not cursor.fetchone():
            conn.close()
            return []

        cursor = conn.execute("SELECT * FROM sessions ORDER BY started_at DESC")
        rows = cursor.fetchall()
        conn.close()

        sessions = []
        for row in rows:
            # Sessions use started_at as the reference timestamp
            timestamp = None
            if "started_at" in row.keys():  # noqa: SIM118 - sqlite3.Row requires .keys()
                timestamp = datetime.fromisoformat(row["started_at"]) if row["started_at"] else None

            # Filter by last sync if not push_all
            if self._last_sync and not push_all and timestamp:
                last_sync_aware = _ensure_tz_aware(self._last_sync)
                if _ensure_tz_aware(timestamp) <= last_sync_aware:
                    continue

            # Get vector clock from metadata or column
            try:
                clock_data = row["vector_clock"] or "{}"
            except (KeyError, IndexError):
                clock_data = "{}"
            if isinstance(clock_data, str):
                clock = VectorClock.from_json(clock_data) if clock_data else VectorClock()
            else:
                clock = VectorClock()
            clock = clock.increment(self.device_id)

            sessions.append(
                SyncedSession(
                    id=row["id"],
                    label=row["label"],
                    namespace_id=row["namespace_id"] or "global",
                    tool=row["tool"] or "contextfs",
                    repo_path=row["repo_path"],
                    branch=row["branch"],
                    started_at=datetime.fromisoformat(row["started_at"])
                    if row["started_at"]
                    else datetime.now(timezone.utc),
                    ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
                    summary=row["summary"],
                    vector_clock=clock.to_dict(),
                )
            )

        return sessions

    def _get_edges_to_push(self, push_all: bool = False) -> list[SyncedEdge]:
        """Get local edges changed since last sync."""
        import sqlite3

        conn = sqlite3.connect(self.ctx._db_path)
        conn.row_factory = sqlite3.Row

        # Check if memory_edges table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_edges'"
        )
        if not cursor.fetchone():
            conn.close()
            return []

        cursor = conn.execute("SELECT * FROM memory_edges ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()

        edges = []
        for row in rows:
            # Edges use created_at as the reference timestamp
            timestamp = None
            if "created_at" in row.keys():  # noqa: SIM118 - sqlite3.Row requires .keys()
                timestamp = datetime.fromisoformat(row["created_at"]) if row["created_at"] else None

            # Filter by last sync if not push_all
            if self._last_sync and not push_all and timestamp:
                last_sync_aware = _ensure_tz_aware(self._last_sync)
                if _ensure_tz_aware(timestamp) <= last_sync_aware:
                    continue

            # Get vector clock
            try:
                clock_data = row["vector_clock"] or "{}"
            except (KeyError, IndexError):
                clock_data = "{}"
            if isinstance(clock_data, str):
                clock = VectorClock.from_json(clock_data) if clock_data else VectorClock()
            else:
                clock = VectorClock()
            clock = clock.increment(self.device_id)

            edges.append(
                SyncedEdge(
                    id=f"{row['from_id']}:{row['to_id']}:{row['relation']}",
                    from_id=row["from_id"],
                    to_id=row["to_id"],
                    relation=row["relation"],
                    weight=row["weight"] or 1.0,
                    vector_clock=clock.to_dict(),
                )
            )

        return edges

    # =========================================================================
    # Pull Operations
    # =========================================================================

    async def pull(
        self,
        since: datetime | None = None,
        namespace_ids: list[str] | None = None,
        offset: int = 0,
        update_sync_state: bool = True,
    ) -> SyncPullResponse:
        """
        Pull changes from server.

        Args:
            since: Only pull changes after this timestamp (use _UNSET to use _last_sync)
            namespace_ids: Filter by namespaces
            offset: Offset for pagination
            update_sync_state: Whether to update _last_sync after this pull (set False for pagination)

        Returns:
            SyncPullResponse with memories and sessions
        """
        # Use _last_sync only if since is not explicitly provided
        # For pagination (offset > 0), caller should pass the same since value
        since_timestamp = since if since is not None else self._last_sync

        request = SyncPullRequest(
            device_id=self.device_id,
            since_timestamp=since_timestamp,
            namespace_ids=namespace_ids,
            offset=offset,
        )

        response = await self._client.post(
            f"{self.server_url}/api/sync/pull",
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()

        result = SyncPullResponse.model_validate(response.json())

        # Apply pulled changes to local database
        await self._apply_pulled_changes(result)

        # Only update sync state on final page
        if update_sync_state:
            self._last_sync = result.server_timestamp
            self._save_sync_state()

        logger.info(
            f"Pull complete: {len(result.memories)} memories, {len(result.sessions)} sessions"
        )
        return result

    # =========================================================================
    # Content-Addressed Sync (Merkle-style)
    # =========================================================================

    def _build_manifest(
        self,
        namespace_ids: list[str] | None = None,
    ) -> SyncManifestRequest:
        """Build a manifest of all local entities for content-addressed sync.

        Returns a manifest containing {id, content_hash} for all local
        memories, sessions, and edges.
        """
        conn = sqlite3.connect(self._get_db_path())
        conn.row_factory = sqlite3.Row

        memory_entries: list[EntityManifestEntry] = []
        session_entries: list[EntityManifestEntry] = []
        edge_entries: list[EntityManifestEntry] = []

        # Get all memories
        query = "SELECT id, content, updated_at FROM memories"
        params: list[Any] = []
        if namespace_ids:
            placeholders = ",".join("?" * len(namespace_ids))
            query += f" WHERE namespace_id IN ({placeholders})"
            params.extend(namespace_ids)

        for row in conn.execute(query, params):
            content_hash = (
                self.compute_content_hash(row["content"]) if row["content"] is not None else None
            )
            updated_at = None
            if row["updated_at"]:
                try:
                    updated_at = datetime.fromisoformat(row["updated_at"])
                except (ValueError, TypeError):
                    pass
            memory_entries.append(
                EntityManifestEntry(
                    id=row["id"],
                    content_hash=content_hash,
                    updated_at=updated_at,
                )
            )

        # Get all sessions
        for row in conn.execute("SELECT id, started_at FROM sessions"):
            started_at = None
            if row["started_at"]:
                try:
                    started_at = datetime.fromisoformat(row["started_at"])
                except (ValueError, TypeError):
                    pass
            session_entries.append(
                EntityManifestEntry(
                    id=row["id"],
                    content_hash=None,  # Sessions don't have content hash
                    updated_at=started_at,
                )
            )

        # Get all edges
        for row in conn.execute("SELECT from_id, to_id, relation, created_at FROM memory_edges"):
            edge_id = f"{row['from_id']}:{row['to_id']}:{row['relation']}"
            created_at = None
            if row["created_at"]:
                try:
                    created_at = datetime.fromisoformat(row["created_at"])
                except (ValueError, TypeError):
                    pass
            edge_entries.append(
                EntityManifestEntry(
                    id=edge_id,
                    content_hash=None,
                    updated_at=created_at,
                )
            )

        conn.close()

        return SyncManifestRequest(
            device_id=self.device_id,
            memories=memory_entries,
            sessions=session_entries,
            edges=edge_entries,
            namespace_ids=namespace_ids,
        )

    async def pull_diff(
        self,
        namespace_ids: list[str] | None = None,
    ) -> SyncDiffResponse:
        """Pull changes using content-addressed sync (Merkle-style).

        This is idempotent - run it 100 times, always correct result.
        Instead of relying on timestamps, compares actual content hashes.

        Args:
            namespace_ids: Filter by namespaces

        Returns:
            SyncDiffResponse with missing, updated, and deleted entities
        """
        # Build manifest of all local entities
        manifest = self._build_manifest(namespace_ids=namespace_ids)

        logger.info(
            f"Sending manifest: {len(manifest.memories)} memories, "
            f"{len(manifest.sessions)} sessions, {len(manifest.edges)} edges"
        )

        # Send manifest to server, get diff back
        response = await self._client.post(
            f"{self.server_url}/api/sync/diff",
            json=manifest.model_dump(mode="json"),
        )
        response.raise_for_status()

        diff = SyncDiffResponse.model_validate(response.json())

        # Apply the diff to local database
        if diff.missing_memories or diff.missing_sessions or diff.missing_edges:
            # Create a SyncPullResponse to reuse existing apply logic
            pull_response = SyncPullResponse(
                success=True,
                memories=diff.missing_memories,
                sessions=diff.missing_sessions,
                edges=diff.missing_edges,
                server_timestamp=diff.server_timestamp,
            )
            await self._apply_pulled_changes(pull_response)

        # Handle deletions
        for memory_id in diff.deleted_memory_ids:
            try:
                self.ctx.delete(memory_id)
            except Exception:
                pass  # Already deleted

        for session_id in diff.deleted_session_ids:
            try:
                self.ctx.delete_session(session_id)
            except Exception:
                pass

        # Note: Edge deletion would go here if implemented

        # Update sync state
        self._last_sync = diff.server_timestamp
        self._save_sync_state()

        logger.info(
            f"Diff sync complete: {len(diff.missing_memories)} memories, "
            f"{len(diff.missing_sessions)} sessions, "
            f"{len(diff.deleted_memory_ids)} deleted"
        )
        return diff

    async def _apply_pulled_changes(self, response: SyncPullResponse) -> None:
        """Apply pulled changes to local SQLite database."""
        from contextfs.schemas import Memory, MemoryType

        # Collect memories for batch save
        memories_to_save: list[Memory] = []
        deletes_count = 0

        for synced in response.memories:
            # Resolve portable paths to local paths
            paths = self._resolve_memory_paths(synced)

            if synced.deleted_at:
                # Soft delete locally
                try:
                    self.ctx.delete(synced.id)
                    deletes_count += 1
                except Exception:
                    pass  # Already deleted or doesn't exist
            else:
                # Prepare metadata with sync info
                metadata = synced.metadata.copy() if synced.metadata else {}
                metadata["_vector_clock"] = synced.vector_clock
                metadata["_content_hash"] = synced.content_hash

                # Create Memory object for batch save
                memory = Memory(
                    id=synced.id,
                    content=synced.content,
                    type=MemoryType(synced.type) if synced.type else MemoryType.FACT,
                    tags=synced.tags or [],
                    summary=synced.summary,
                    namespace_id=synced.namespace_id or "global",
                    source_repo=paths.get("source_repo") or synced.source_repo,
                    source_tool=synced.source_tool,
                    project=synced.project,
                    metadata=metadata,
                    created_at=synced.created_at,
                    updated_at=synced.updated_at,
                )
                memories_to_save.append(memory)

        # Batch save all memories (much faster than individual saves)
        if memories_to_save:
            saved_count = self.ctx.save_batch(memories_to_save, skip_rag=True)
            logger.info(f"Batch saved {saved_count} memories, deleted {deletes_count}")

        # Store synced embeddings in ChromaDB (avoids rebuild)
        embeddings_to_add = [
            (synced.id, synced.embedding, synced.content, synced.summary, synced.tags, synced.type)
            for synced in response.memories
            if synced.embedding is not None and not synced.deleted_at
        ]
        if embeddings_to_add:
            self._add_embeddings_to_chroma(embeddings_to_add)
            logger.info(f"Added {len(embeddings_to_add)} embeddings to ChromaDB")

        # Apply pulled sessions
        if response.sessions:
            self._apply_pulled_sessions(response.sessions)

        # Apply pulled edges
        if response.edges:
            self._apply_pulled_edges(response.edges)

    def _add_embeddings_to_chroma(
        self,
        embeddings: list[tuple[str, list[float], str, str | None, list[str], str]],
    ) -> None:
        """Add synced embeddings to ChromaDB."""
        try:
            # Ensure RAG is initialized (lazy init)
            self.ctx.rag._ensure_initialized()
            collection = self.ctx.rag._collection
            if collection is None:
                logger.warning("ChromaDB collection not available, skipping embedding sync")
                return

            # Prepare batch data
            ids = []
            embedding_vectors = []
            documents = []
            metadatas = []

            import json

            for memory_id, embedding, content, summary, tags, mem_type in embeddings:
                ids.append(memory_id)
                embedding_vectors.append(embedding)
                documents.append(content)
                metadatas.append(
                    {
                        "summary": summary or "",
                        "tags": json.dumps(tags) if tags else "[]",
                        "type": mem_type,
                    }
                )

            # Upsert to ChromaDB
            collection.upsert(
                ids=ids,
                embeddings=embedding_vectors,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as e:
            logger.warning(f"Failed to add embeddings to ChromaDB: {e}")

    def _apply_pulled_sessions(self, sessions: list[SyncedSession]) -> None:
        """Apply pulled sessions to local SQLite database."""
        import sqlite3

        conn = sqlite3.connect(self.ctx._db_path)
        cursor = conn.cursor()

        # Check if sessions table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
        if not cursor.fetchone():
            conn.close()
            logger.warning("Sessions table doesn't exist, skipping session sync")
            return

        saved_count = 0
        for session in sessions:
            if session.deleted_at:
                # Delete session
                cursor.execute("DELETE FROM sessions WHERE id = ?", (session.id,))
            else:
                # Upsert session
                cursor.execute(
                    """
                    INSERT INTO sessions (id, label, namespace_id, tool, repo_path, branch,
                                         started_at, ended_at, summary, vector_clock)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        label = excluded.label,
                        namespace_id = excluded.namespace_id,
                        tool = excluded.tool,
                        repo_path = excluded.repo_path,
                        branch = excluded.branch,
                        started_at = excluded.started_at,
                        ended_at = excluded.ended_at,
                        summary = excluded.summary,
                        vector_clock = excluded.vector_clock
                    """,
                    (
                        session.id,
                        session.label,
                        session.namespace_id,
                        session.tool,
                        session.repo_path,
                        session.branch,
                        session.started_at.isoformat() if session.started_at else None,
                        session.ended_at.isoformat() if session.ended_at else None,
                        session.summary,
                        json.dumps(session.vector_clock) if session.vector_clock else "{}",
                    ),
                )
                saved_count += 1

        conn.commit()
        conn.close()
        logger.info(f"Applied {saved_count} pulled sessions")

    def _apply_pulled_edges(self, edges: list[SyncedEdge]) -> None:
        """Apply pulled edges to local SQLite database."""
        import sqlite3

        conn = sqlite3.connect(self.ctx._db_path)
        cursor = conn.cursor()

        # Check if memory_edges table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_edges'")
        if not cursor.fetchone():
            conn.close()
            logger.warning("Memory edges table doesn't exist, skipping edge sync")
            return

        saved_count = 0
        for edge in edges:
            if edge.deleted_at:
                # Delete edge
                cursor.execute(
                    "DELETE FROM memory_edges WHERE from_id = ? AND to_id = ? AND relation = ?",
                    (edge.from_id, edge.to_id, edge.relation),
                )
            else:
                # Upsert edge
                cursor.execute(
                    """
                    INSERT INTO memory_edges (from_id, to_id, relation, weight, vector_clock, created_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(from_id, to_id, relation) DO UPDATE SET
                        weight = excluded.weight,
                        vector_clock = excluded.vector_clock
                    """,
                    (
                        edge.from_id,
                        edge.to_id,
                        edge.relation,
                        edge.weight,
                        json.dumps(edge.vector_clock) if edge.vector_clock else "{}",
                    ),
                )
                saved_count += 1

        conn.commit()
        conn.close()
        logger.info(f"Applied {saved_count} pulled edges")

    def _get_embeddings_from_chroma(self, memory_ids: list[str]) -> dict[str, list[float]]:
        """Get embeddings from ChromaDB for the given memory IDs."""
        embeddings: dict[str, list[float]] = {}
        if not memory_ids:
            return embeddings

        try:
            # Ensure RAG is initialized (lazy init)
            self.ctx.rag._ensure_initialized()
            collection = self.ctx.rag._collection
            if collection is None:
                return embeddings

            # Get embeddings in batches (ChromaDB has limits)
            batch_size = 100
            for i in range(0, len(memory_ids), batch_size):
                batch_ids = memory_ids[i : i + batch_size]
                try:
                    result = collection.get(
                        ids=batch_ids,
                        include=["embeddings"],
                    )
                    # result["embeddings"] may be a numpy array, so check length
                    result_embeddings = result.get("embeddings")
                    if result and result_embeddings is not None and len(result_embeddings) > 0:
                        for mem_id, emb in zip(result["ids"], result_embeddings, strict=False):
                            if emb is not None:
                                # Convert numpy array to list for JSON serialization
                                embeddings[mem_id] = list(emb) if hasattr(emb, "tolist") else emb
                except Exception as e:
                    logger.debug(f"Failed to get embeddings for batch: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Failed to get embeddings from ChromaDB: {e}")

        return embeddings

    # =========================================================================
    # Full Sync
    # =========================================================================

    async def sync_all(
        self,
        namespace_ids: list[str] | None = None,
    ) -> SyncResult:
        """
        Full bidirectional sync using content-addressed approach.

        1. Get diff from server (what client needs + what server needs)
        2. Push only what server is missing
        3. Apply pulled changes locally

        Args:
            namespace_ids: Filter by namespaces

        Returns:
            SyncResult with push and pull responses
        """
        import time

        start = time.time()
        errors: list[str] = []

        # Step 1: Get diff (content-addressed - tells us both directions)
        try:
            diff_result = await self._get_diff(namespace_ids=namespace_ids)
        except Exception as e:
            logger.error(f"Diff failed: {e}")
            errors.append(f"Diff failed: {e}")
            # Return empty result on diff failure
            return SyncResult(
                pushed=SyncPushResponse(
                    success=False,
                    accepted=0,
                    rejected=0,
                    conflicts=[],
                    server_timestamp=datetime.now(timezone.utc),
                    message=str(e),
                ),
                pulled=SyncPullResponse(
                    success=False,
                    memories=[],
                    sessions=[],
                    edges=[],
                    server_timestamp=datetime.now(timezone.utc),
                ),
                duration_ms=(time.time() - start) * 1000,
                errors=errors,
            )

        # Step 2: Push only what server is missing (content-addressed push)
        push_result: SyncPushResponse
        if diff_result.total_server_missing > 0:
            try:
                push_result = await self._push_by_ids(
                    memory_ids=diff_result.server_missing_memory_ids,
                    session_ids=diff_result.server_missing_session_ids,
                    edge_ids=diff_result.server_missing_edge_ids,
                )
            except Exception as e:
                logger.error(f"Push failed: {e}")
                errors.append(f"Push failed: {e}")
                push_result = SyncPushResponse(
                    success=False,
                    accepted=0,
                    rejected=0,
                    conflicts=[],
                    server_timestamp=datetime.now(timezone.utc),
                    message=str(e),
                )
        else:
            # Nothing to push
            push_result = SyncPushResponse(
                success=True,
                accepted=0,
                rejected=0,
                conflicts=[],
                server_timestamp=diff_result.server_timestamp,
            )

        # Step 3: Apply pulled changes (already done in _get_diff via pull_diff)
        # Convert SyncDiffResponse to SyncPullResponse for compatibility
        pull_result = SyncPullResponse(
            success=diff_result.success,
            memories=diff_result.missing_memories,
            sessions=diff_result.missing_sessions,
            edges=diff_result.missing_edges,
            server_timestamp=diff_result.server_timestamp,
        )

        duration_ms = (time.time() - start) * 1000

        result = SyncResult(
            pushed=push_result,
            pulled=pull_result,
            duration_ms=duration_ms,
            errors=errors,
        )

        logger.info(
            f"Sync complete in {duration_ms:.0f}ms: "
            f"pushed {push_result.accepted}, pulled {len(pull_result.memories)}"
        )

        return result

    async def _get_diff(
        self,
        namespace_ids: list[str] | None = None,
    ) -> SyncDiffResponse:
        """Get diff without applying changes (for sync_all coordination)."""
        # Build manifest of all local entities
        manifest = self._build_manifest(namespace_ids=namespace_ids)

        logger.info(
            f"Sending manifest: {len(manifest.memories)} memories, "
            f"{len(manifest.sessions)} sessions, {len(manifest.edges)} edges"
        )

        # Send manifest to server, get diff back
        response = await self._client.post(
            f"{self.server_url}/api/sync/diff",
            json=manifest.model_dump(mode="json"),
        )
        response.raise_for_status()

        diff = SyncDiffResponse.model_validate(response.json())

        # Apply the diff to local database (pull)
        if diff.missing_memories or diff.missing_sessions or diff.missing_edges:
            pull_response = SyncPullResponse(
                success=True,
                memories=diff.missing_memories,
                sessions=diff.missing_sessions,
                edges=diff.missing_edges,
                server_timestamp=diff.server_timestamp,
            )
            await self._apply_pulled_changes(pull_response)

        # Handle deletions
        for memory_id in diff.deleted_memory_ids:
            try:
                self.ctx.delete(memory_id)
            except Exception:
                pass

        for session_id in diff.deleted_session_ids:
            try:
                self.ctx.delete_session(session_id)
            except Exception:
                pass

        logger.info(
            f"Diff result: {len(diff.missing_memories)} to pull, "
            f"{diff.total_server_missing} server needs"
        )
        return diff

    async def _push_by_ids(
        self,
        memory_ids: list[str],
        session_ids: list[str],
        edge_ids: list[str],
    ) -> SyncPushResponse:
        """Push specific items by ID (content-addressed push)."""
        # Load memories by ID
        memories = self._get_memories_by_ids(memory_ids) if memory_ids else []

        # Convert to SyncedMemory format
        synced_memories = []
        memory_clocks: dict[str, VectorClock] = {}
        memory_embeddings = self._get_embeddings_from_chroma([m.id for m in memories])

        for m in memories:
            clock_data = m.metadata.get("_vector_clock") if m.metadata else None
            if isinstance(clock_data, str):
                clock = VectorClock.from_json(clock_data)
            elif isinstance(clock_data, dict):
                clock = VectorClock.from_dict(clock_data)
            else:
                clock = VectorClock()

            clock = clock.increment(self.device_id)
            memory_clocks[m.id] = clock

            paths = self._normalize_memory_paths(m)

            synced_memories.append(
                SyncedMemory(
                    id=m.id,
                    content=m.content,
                    type=m.type.value if hasattr(m.type, "value") else str(m.type),
                    tags=m.tags,
                    summary=m.summary,
                    namespace_id=m.namespace_id,
                    repo_url=paths["repo_url"],
                    repo_name=paths["repo_name"],
                    relative_path=paths["relative_path"],
                    source_file=m.source_file,
                    source_repo=m.source_repo,
                    source_tool=getattr(m, "source_tool", None),
                    project=getattr(m, "project", None),
                    session_id=getattr(m, "session_id", None),
                    created_at=m.created_at,
                    updated_at=m.updated_at,
                    vector_clock=clock.to_dict(),
                    content_hash=self.compute_content_hash(m.content),
                    deleted_at=getattr(m, "deleted_at", None),
                    metadata=m.metadata,
                    embedding=memory_embeddings.get(m.id),
                )
            )

        # Load sessions by ID
        sessions = self._get_sessions_by_ids(session_ids) if session_ids else []

        # Load edges by ID
        edges = self._get_edges_by_ids(edge_ids) if edge_ids else []

        request = SyncPushRequest(
            device_id=self.device_id,
            memories=synced_memories,
            sessions=sessions,
            edges=edges,
            last_sync_timestamp=self._last_sync,
        )

        response = await self._client.post(
            f"{self.server_url}/api/sync/push",
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()

        result = SyncPushResponse.model_validate(response.json())
        self._last_sync = result.server_timestamp
        self._last_push = result.server_timestamp
        self._save_sync_state()

        if result.accepted > 0:
            self._update_local_vector_clocks(memories, memory_clocks)

        logger.info(
            f"Content-addressed push: {result.accepted} accepted, "
            f"{result.rejected} rejected, {len(result.conflicts)} conflicts"
        )
        return result

    def _get_memories_by_ids(self, memory_ids: list[str]) -> list[Memory]:
        """Get memories by specific IDs."""
        if not memory_ids:
            return []

        conn = sqlite3.connect(self._get_db_path())
        conn.row_factory = sqlite3.Row

        placeholders = ",".join("?" * len(memory_ids))
        cursor = conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})",
            memory_ids,
        )
        rows = cursor.fetchall()
        conn.close()

        return [self.ctx._row_to_memory(row) for row in rows]

    def _get_sessions_by_ids(self, session_ids: list[str]) -> list[SyncedSession]:
        """Get sessions by specific IDs."""
        if not session_ids:
            return []

        conn = sqlite3.connect(self._get_db_path())
        conn.row_factory = sqlite3.Row

        placeholders = ",".join("?" * len(session_ids))
        cursor = conn.execute(
            f"SELECT * FROM sessions WHERE id IN ({placeholders})",
            session_ids,
        )
        rows = cursor.fetchall()
        conn.close()

        sessions = []
        for row in rows:
            try:
                clock_data = row["vector_clock"] or "{}"
            except (KeyError, IndexError):
                clock_data = "{}"
            if isinstance(clock_data, str):
                clock = VectorClock.from_json(clock_data) if clock_data else VectorClock()
            else:
                clock = VectorClock()
            clock = clock.increment(self.device_id)

            sessions.append(
                SyncedSession(
                    id=row["id"],
                    label=row["label"],
                    namespace_id=row["namespace_id"] or "global",
                    tool=row["tool"] or "contextfs",
                    repo_path=row["repo_path"],
                    branch=row["branch"],
                    started_at=datetime.fromisoformat(row["started_at"])
                    if row["started_at"]
                    else datetime.now(timezone.utc),
                    ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
                    summary=row["summary"],
                    vector_clock=clock.to_dict(),
                )
            )

        return sessions

    def _get_edges_by_ids(self, edge_ids: list[str]) -> list[SyncedEdge]:
        """Get edges by specific IDs (format: from_id:to_id:relation)."""
        if not edge_ids:
            return []

        conn = sqlite3.connect(self._get_db_path())
        conn.row_factory = sqlite3.Row

        edges = []
        for edge_id in edge_ids:
            parts = edge_id.split(":")
            if len(parts) >= 3:
                from_id, to_id, relation = parts[0], parts[1], ":".join(parts[2:])
                cursor = conn.execute(
                    "SELECT * FROM memory_edges WHERE from_id = ? AND to_id = ? AND relation = ?",
                    (from_id, to_id, relation),
                )
                row = cursor.fetchone()
                if row:
                    try:
                        clock_data = row["vector_clock"] or "{}"
                    except (KeyError, IndexError):
                        clock_data = "{}"
                    if isinstance(clock_data, str):
                        clock = VectorClock.from_json(clock_data) if clock_data else VectorClock()
                    else:
                        clock = VectorClock()
                    clock = clock.increment(self.device_id)

                    edges.append(
                        SyncedEdge(
                            id=edge_id,
                            from_id=row["from_id"],
                            to_id=row["to_id"],
                            relation=row["relation"],
                            weight=row["weight"] or 1.0,
                            vector_clock=clock.to_dict(),
                        )
                    )

        conn.close()
        return edges

    # =========================================================================
    # Status
    # =========================================================================

    async def status(self) -> SyncStatusResponse:
        """Get sync status from server."""
        request = SyncStatusRequest(device_id=self.device_id)

        response = await self._client.post(
            f"{self.server_url}/api/sync/status",
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()

        return SyncStatusResponse.model_validate(response.json())

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> SyncClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()
