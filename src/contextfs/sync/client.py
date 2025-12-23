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
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from contextfs.sync.path_resolver import PathResolver, PortablePath
from contextfs.sync.protocol import (
    DeviceInfo,
    DeviceRegistration,
    SyncedMemory,
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

        # Sync state
        self._last_sync: datetime | None = None
        self._sync_state_path = self._get_sync_state_path()
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

    def _get_sync_state_path(self) -> Path:
        """Get path to sync state file."""
        return Path.home() / ".contextfs" / "sync_state.json"

    def _load_sync_state(self) -> None:
        """Load sync state from file."""
        if self._sync_state_path.exists():
            try:
                with open(self._sync_state_path) as f:
                    data = json.load(f)
                if data.get("last_sync"):
                    self._last_sync = datetime.fromisoformat(data["last_sync"])
                self._device_tracker = DeviceTracker.from_dict(data.get("device_tracker"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load sync state: {e}")

    def _save_sync_state(self) -> None:
        """Save sync state to file."""
        self._sync_state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "device_tracker": self._device_tracker.to_dict(),
        }
        with open(self._sync_state_path, "w") as f:
            json.dump(data, f, indent=2)

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
        for m in memories:
            # Get or create vector clock
            clock_data = getattr(m, "vector_clock", None)
            if isinstance(clock_data, str):
                clock = VectorClock.from_json(clock_data)
            elif isinstance(clock_data, dict):
                clock = VectorClock.from_dict(clock_data)
            else:
                clock = VectorClock()

            clock = clock.increment(self.device_id)

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
                )
            )

        request = SyncPushRequest(
            device_id=self.device_id,
            memories=synced_memories,
            last_sync_timestamp=self._last_sync,
        )

        response = await self._client.post(
            f"{self.server_url}/api/sync/push",
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()

        result = SyncPushResponse.model_validate(response.json())
        self._last_sync = result.server_timestamp
        self._save_sync_state()

        logger.info(
            f"Push complete: {result.accepted} accepted, "
            f"{result.rejected} rejected, {len(result.conflicts)} conflicts"
        )
        return result

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
        # Use list_recent to get memories (limit high for sync)
        memories = self.ctx.list_recent(limit=10000)

        # Filter by namespace if specified
        if namespace_ids:
            memories = [m for m in memories if m.namespace_id in namespace_ids]

        # Filter by last sync time if we have one (unless push_all)
        if self._last_sync and not push_all:
            memories = [m for m in memories if m.updated_at > self._last_sync]

        return memories

    # =========================================================================
    # Pull Operations
    # =========================================================================

    async def pull(
        self,
        since: datetime | None = None,
        namespace_ids: list[str] | None = None,
    ) -> SyncPullResponse:
        """
        Pull changes from server.

        Args:
            since: Only pull changes after this timestamp
            namespace_ids: Filter by namespaces

        Returns:
            SyncPullResponse with memories and sessions
        """
        request = SyncPullRequest(
            device_id=self.device_id,
            since_timestamp=since or self._last_sync,
            namespace_ids=namespace_ids,
        )

        response = await self._client.post(
            f"{self.server_url}/api/sync/pull",
            json=request.model_dump(mode="json"),
        )
        response.raise_for_status()

        result = SyncPullResponse.model_validate(response.json())

        # Apply pulled changes to local database
        await self._apply_pulled_changes(result)

        self._last_sync = result.server_timestamp
        self._save_sync_state()

        logger.info(
            f"Pull complete: {len(result.memories)} memories, " f"{len(result.sessions)} sessions"
        )
        return result

    async def _apply_pulled_changes(self, response: SyncPullResponse) -> None:
        """Apply pulled changes to local SQLite database."""
        from contextfs.schemas import MemoryType

        for synced in response.memories:
            # Resolve portable paths to local paths
            paths = self._resolve_memory_paths(synced)

            if synced.deleted_at:
                # Soft delete locally
                try:
                    self.ctx.delete(synced.id)
                except Exception:
                    pass  # Already deleted or doesn't exist
            else:
                # Prepare metadata with sync info
                metadata = synced.metadata.copy() if synced.metadata else {}
                metadata["_vector_clock"] = synced.vector_clock
                metadata["_content_hash"] = synced.content_hash

                # Save using the save() method parameters
                self.ctx.save(
                    content=synced.content,
                    type=MemoryType(synced.type) if synced.type else MemoryType.FACT,
                    tags=synced.tags,
                    summary=synced.summary,
                    namespace_id=synced.namespace_id,
                    source_repo=paths.get("source_repo") or synced.source_repo,
                    metadata=metadata,
                )

    # =========================================================================
    # Full Sync
    # =========================================================================

    async def sync_all(
        self,
        namespace_ids: list[str] | None = None,
    ) -> SyncResult:
        """
        Full bidirectional sync.

        1. Push local changes to server
        2. Pull server changes to local
        3. Handle any conflicts

        Args:
            namespace_ids: Filter by namespaces

        Returns:
            SyncResult with push and pull responses
        """
        import time

        start = time.time()
        errors: list[str] = []

        # Push local changes
        try:
            push_result = await self.push(namespace_ids=namespace_ids)
        except Exception as e:
            logger.error(f"Push failed: {e}")
            errors.append(f"Push failed: {e}")
            push_result = SyncPushResponse(
                success=False,
                accepted=0,
                rejected=0,
                conflicts=[],
                server_timestamp=datetime.now(),
                message=str(e),
            )

        # Pull remote changes
        try:
            pull_result = await self.pull(namespace_ids=namespace_ids)
        except Exception as e:
            logger.error(f"Pull failed: {e}")
            errors.append(f"Pull failed: {e}")
            pull_result = SyncPullResponse(
                success=False,
                memories=[],
                sessions=[],
                edges=[],
                server_timestamp=datetime.now(),
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
