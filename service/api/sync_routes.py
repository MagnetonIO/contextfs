"""Sync API endpoints.

Implements push/pull endpoints for sync operations with
vector clock conflict resolution.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from contextfs.sync.protocol import (
    ConflictInfo,
    DeviceInfo,
    DeviceRegistration,
    SyncDiffResponse,
    SyncedEdge,
    SyncedMemory,
    SyncedSession,
    SyncManifestRequest,
    SyncPullRequest,
    SyncPullResponse,
    SyncPushRequest,
    SyncPushResponse,
    SyncStatus,
    SyncStatusRequest,
    SyncStatusResponse,
)
from contextfs.sync.vector_clock import VectorClock
from service.db.models import (
    Device,
    SyncedEdgeModel,
    SyncedMemoryModel,
    SyncedSessionModel,
    SyncState,
)
from service.db.session import get_session_dependency

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sync", tags=["sync"])


# =============================================================================
# Device Registration
# =============================================================================


@router.post("/register", response_model=DeviceInfo)
async def register_device(
    registration: DeviceRegistration,
    session: AsyncSession = Depends(get_session_dependency),
) -> DeviceInfo:
    """Register a new device for sync."""
    # Check if device already exists
    result = await session.execute(select(Device).where(Device.device_id == registration.device_id))
    existing = result.scalar_one_or_none()

    if existing:
        # Update existing device
        existing.device_name = registration.device_name
        existing.platform = registration.platform
        existing.client_version = registration.client_version
        device = existing
    else:
        # Create new device
        device = Device(
            device_id=registration.device_id,
            device_name=registration.device_name,
            platform=registration.platform,
            client_version=registration.client_version,
        )
        session.add(device)

    await session.commit()

    return DeviceInfo(
        device_id=device.device_id,
        device_name=device.device_name,
        platform=device.platform,
        client_version=device.client_version,
        registered_at=device.registered_at,
        last_sync_at=device.last_sync_at,
        sync_cursor=device.sync_cursor,
    )


# =============================================================================
# Push Changes
# =============================================================================


@router.post("/push", response_model=SyncPushResponse)
async def push_changes(
    request: SyncPushRequest,
    session: AsyncSession = Depends(get_session_dependency),
) -> SyncPushResponse:
    """
    Push local changes to server.

    Conflict resolution using vector clocks:
    1. If client vector_clock happens-before server: reject (stale)
    2. If server vector_clock happens-before client: accept
    3. If concurrent: conflict (return for manual resolution)
    """
    accepted = 0
    rejected = 0
    conflicts: list[ConflictInfo] = []
    server_timestamp = datetime.now(timezone.utc)

    # Process memories
    for memory in request.memories:
        result = await _process_memory_push(session, memory, request.device_id, conflicts)
        if result == "accepted":
            accepted += 1
        elif result == "rejected":
            rejected += 1
        # conflicts are added directly to the list

    # Process sessions
    for sess in request.sessions:
        result = await _process_session_push(session, sess, request.device_id, conflicts)
        if result == "accepted":
            accepted += 1
        elif result == "rejected":
            rejected += 1

    # Process edges
    for edge in request.edges:
        result = await _process_edge_push(session, edge, request.device_id, conflicts)
        if result == "accepted":
            accepted += 1
        elif result == "rejected":
            rejected += 1

    # Update device sync state
    await _update_device_sync_state(session, request.device_id, push_at=server_timestamp)

    await session.commit()

    status = SyncStatus.SUCCESS
    if conflicts:
        status = SyncStatus.CONFLICT
    elif rejected > 0:
        status = SyncStatus.PARTIAL

    return SyncPushResponse(
        success=len(conflicts) == 0,
        status=status,
        accepted=accepted,
        rejected=rejected,
        conflicts=conflicts,
        server_timestamp=server_timestamp,
    )


async def _process_memory_push(
    session: AsyncSession,
    memory: SyncedMemory,
    device_id: str,
    conflicts: list[ConflictInfo],
) -> str:
    """Process a single memory push. Returns 'accepted', 'rejected', or 'conflict'."""
    result = await session.execute(
        select(SyncedMemoryModel).where(SyncedMemoryModel.id == memory.id)
    )
    existing = result.scalar_one_or_none()

    client_clock = VectorClock.from_dict(memory.vector_clock)

    if existing is None:
        # New memory - accept
        new_memory = SyncedMemoryModel(
            id=memory.id,
            content=memory.content,
            type=memory.type,
            tags=memory.tags,
            summary=memory.summary,
            namespace_id=memory.namespace_id,
            repo_url=memory.repo_url,
            repo_name=memory.repo_name,
            relative_path=memory.relative_path,
            source_file=memory.source_file,
            source_repo=memory.source_repo,
            source_tool=memory.source_tool,
            project=memory.project,
            session_id=memory.session_id,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            vector_clock=client_clock.increment(device_id).to_dict(),
            content_hash=memory.content_hash,
            deleted_at=memory.deleted_at,
            last_modified_by=device_id,
            metadata=memory.metadata,
        )
        # Store embedding if provided (for sync to other clients)
        if hasattr(SyncedMemoryModel, "embedding") and memory.embedding:
            new_memory.embedding = memory.embedding
        session.add(new_memory)
        return "accepted"

    server_clock = VectorClock.from_dict(existing.vector_clock or {})

    if server_clock.happens_before(client_clock) or server_clock.equal_to(client_clock):
        # Server is behind or equal - accept update
        existing.content = memory.content
        existing.type = memory.type
        existing.tags = memory.tags
        existing.summary = memory.summary
        existing.repo_url = memory.repo_url
        existing.repo_name = memory.repo_name
        existing.relative_path = memory.relative_path
        existing.updated_at = memory.updated_at
        # Don't increment - client already incremented before sending
        existing.vector_clock = client_clock.merge(server_clock).to_dict()
        existing.content_hash = memory.content_hash
        existing.deleted_at = memory.deleted_at
        existing.last_modified_by = device_id
        existing.metadata = memory.metadata
        # Update embedding if provided
        if hasattr(existing, "embedding") and memory.embedding:
            existing.embedding = memory.embedding
        return "accepted"

    elif client_clock.happens_before(server_clock):
        # Client is behind - reject (stale)
        return "rejected"

    else:
        # Concurrent changes - conflict
        conflicts.append(
            ConflictInfo(
                entity_id=memory.id,
                entity_type="memory",
                client_clock=client_clock.to_dict(),
                server_clock=server_clock.to_dict(),
                client_content=memory.content,
                server_content=existing.content,
                client_updated_at=memory.updated_at,
                server_updated_at=existing.updated_at,
            )
        )
        return "conflict"


async def _process_session_push(
    session: AsyncSession,
    sess: SyncedSession,
    device_id: str,
    conflicts: list[ConflictInfo],
) -> str:
    """Process a single session push."""
    result = await session.execute(
        select(SyncedSessionModel).where(SyncedSessionModel.id == sess.id)
    )
    existing = result.scalar_one_or_none()

    client_clock = VectorClock.from_dict(sess.vector_clock)

    if existing is None:
        new_session = SyncedSessionModel(
            id=sess.id,
            label=sess.label,
            namespace_id=sess.namespace_id,
            tool=sess.tool,
            repo_url=sess.repo_url,
            repo_name=sess.repo_name,
            repo_path=sess.repo_path,
            branch=sess.branch,
            started_at=sess.started_at,
            ended_at=sess.ended_at,
            summary=sess.summary,
            created_at=sess.created_at,
            updated_at=sess.updated_at,
            vector_clock=client_clock.increment(device_id).to_dict(),
            content_hash=sess.content_hash,
            deleted_at=sess.deleted_at,
            last_modified_by=device_id,
            metadata=sess.metadata,
        )
        session.add(new_session)
        return "accepted"

    server_clock = VectorClock.from_dict(existing.vector_clock or {})

    if server_clock.happens_before(client_clock) or server_clock.equal_to(client_clock):
        existing.label = sess.label
        existing.summary = sess.summary
        existing.ended_at = sess.ended_at
        existing.updated_at = sess.updated_at
        # Don't increment - client already incremented before sending
        existing.vector_clock = client_clock.merge(server_clock).to_dict()
        existing.deleted_at = sess.deleted_at
        existing.last_modified_by = device_id
        existing.metadata = sess.metadata
        return "accepted"

    elif client_clock.happens_before(server_clock):
        return "rejected"

    else:
        conflicts.append(
            ConflictInfo(
                entity_id=sess.id,
                entity_type="session",
                client_clock=client_clock.to_dict(),
                server_clock=server_clock.to_dict(),
                client_content=sess.summary,
                server_content=existing.summary,
                client_updated_at=sess.updated_at,
                server_updated_at=existing.updated_at,
            )
        )
        return "conflict"


async def _process_edge_push(
    session: AsyncSession,
    edge: SyncedEdge,
    device_id: str,
    conflicts: list[ConflictInfo],
) -> str:
    """Process a single edge push."""
    result = await session.execute(
        select(SyncedEdgeModel).where(
            and_(
                SyncedEdgeModel.from_id == edge.from_id,
                SyncedEdgeModel.to_id == edge.to_id,
                SyncedEdgeModel.relation == edge.relation,
            )
        )
    )
    existing = result.scalar_one_or_none()

    client_clock = VectorClock.from_dict(edge.vector_clock)

    if existing is None:
        new_edge = SyncedEdgeModel(
            from_id=edge.from_id,
            to_id=edge.to_id,
            relation=edge.relation,
            weight=edge.weight,
            created_by=edge.created_by,
            created_at=edge.created_at,
            updated_at=edge.updated_at,
            vector_clock=client_clock.increment(device_id).to_dict(),
            deleted_at=edge.deleted_at,
            last_modified_by=device_id,
            metadata=edge.metadata,
        )
        session.add(new_edge)
        return "accepted"

    server_clock = VectorClock.from_dict(existing.vector_clock or {})

    if server_clock.happens_before(client_clock) or server_clock.equal_to(client_clock):
        existing.weight = edge.weight
        existing.updated_at = edge.updated_at
        # Don't increment - client already incremented before sending
        existing.vector_clock = client_clock.merge(server_clock).to_dict()
        existing.deleted_at = edge.deleted_at
        existing.last_modified_by = device_id
        existing.metadata = edge.metadata
        return "accepted"

    elif client_clock.happens_before(server_clock):
        return "rejected"

    else:
        conflicts.append(
            ConflictInfo(
                entity_id=edge.id,
                entity_type="edge",
                client_clock=client_clock.to_dict(),
                server_clock=server_clock.to_dict(),
                client_content=None,
                server_content=None,
                client_updated_at=edge.updated_at,
                server_updated_at=existing.updated_at,
            )
        )
        return "conflict"


# =============================================================================
# Pull Changes
# =============================================================================


@router.post("/pull", response_model=SyncPullResponse)
async def pull_changes(
    request: SyncPullRequest,
    session: AsyncSession = Depends(get_session_dependency),
) -> SyncPullResponse:
    """
    Pull changes from server.

    Returns all changes since last sync timestamp, including soft-deleted items
    (so clients can apply the deletion).
    """
    server_timestamp = datetime.now(timezone.utc)

    # Query memories
    memories = await _pull_memories(session, request)

    # Query sessions
    sessions = await _pull_sessions(session, request)

    # Query edges
    edges = await _pull_edges(session, request)

    # Update device sync state
    await _update_device_sync_state(session, request.device_id, pull_at=server_timestamp)

    # Check if there are more results
    has_more = (
        len(memories) >= request.limit
        or len(sessions) >= request.limit
        or len(edges) >= request.limit
    )

    # Calculate next offset for pagination
    next_offset = request.offset + len(memories) + len(sessions) + len(edges)

    await session.commit()

    return SyncPullResponse(
        success=True,
        memories=memories,
        sessions=sessions,
        edges=edges,
        server_timestamp=server_timestamp,
        has_more=has_more,
        next_offset=next_offset if has_more else 0,
    )


async def _pull_memories(
    session: AsyncSession,
    request: SyncPullRequest,
) -> list[SyncedMemory]:
    """Pull memories from database."""
    query = select(SyncedMemoryModel)

    conditions = []
    if request.since_timestamp:
        conditions.append(SyncedMemoryModel.updated_at > request.since_timestamp)
    if request.namespace_ids:
        conditions.append(SyncedMemoryModel.namespace_id.in_(request.namespace_ids))

    if conditions:
        query = query.where(and_(*conditions))

    query = (
        query.order_by(SyncedMemoryModel.updated_at.asc(), SyncedMemoryModel.id.asc())
        .offset(request.offset)
        .limit(request.limit)
    )

    result = await session.execute(query)
    rows = result.scalars().all()

    return [
        SyncedMemory(
            id=m.id,
            content=m.content,
            type=m.type,
            tags=m.tags or [],
            summary=m.summary,
            namespace_id=m.namespace_id,
            repo_url=m.repo_url,
            repo_name=m.repo_name,
            relative_path=m.relative_path,
            source_file=m.source_file,
            source_repo=m.source_repo,
            source_tool=m.source_tool,
            project=m.project,
            session_id=m.session_id,
            created_at=m.created_at,
            updated_at=m.updated_at,
            vector_clock=m.vector_clock or {},
            content_hash=m.content_hash,
            deleted_at=m.deleted_at,
            last_modified_by=m.last_modified_by,
            metadata=dict(m.extra_metadata) if m.extra_metadata else {},
            # Include embedding if available (for sync to client ChromaDB)
            embedding=list(m.embedding)
            if hasattr(m, "embedding") and m.embedding is not None
            else None,
        )
        for m in rows
    ]


async def _pull_sessions(
    session: AsyncSession,
    request: SyncPullRequest,
) -> list[SyncedSession]:
    """Pull sessions from database."""
    query = select(SyncedSessionModel)

    conditions = []
    if request.since_timestamp:
        conditions.append(SyncedSessionModel.updated_at > request.since_timestamp)
    if request.namespace_ids:
        conditions.append(SyncedSessionModel.namespace_id.in_(request.namespace_ids))

    if conditions:
        query = query.where(and_(*conditions))

    query = (
        query.order_by(SyncedSessionModel.updated_at.asc(), SyncedSessionModel.id.asc())
        .offset(request.offset)
        .limit(request.limit)
    )

    result = await session.execute(query)
    rows = result.scalars().all()

    return [
        SyncedSession(
            id=s.id,
            label=s.label,
            namespace_id=s.namespace_id,
            tool=s.tool,
            repo_url=s.repo_url,
            repo_name=s.repo_name,
            repo_path=s.repo_path,
            branch=s.branch,
            started_at=s.started_at,
            ended_at=s.ended_at,
            summary=s.summary,
            created_at=s.created_at,
            updated_at=s.updated_at,
            vector_clock=s.vector_clock or {},
            content_hash=s.content_hash,
            deleted_at=s.deleted_at,
            last_modified_by=s.last_modified_by,
            metadata=dict(s.extra_metadata) if s.extra_metadata else {},
        )
        for s in rows
    ]


async def _pull_edges(
    session: AsyncSession,
    request: SyncPullRequest,
) -> list[SyncedEdge]:
    """Pull edges from database."""
    query = select(SyncedEdgeModel)

    if request.since_timestamp:
        query = query.where(SyncedEdgeModel.updated_at > request.since_timestamp)

    query = (
        query.order_by(
            SyncedEdgeModel.updated_at.asc(),
            SyncedEdgeModel.from_id.asc(),
            SyncedEdgeModel.to_id.asc(),
        )
        .offset(request.offset)
        .limit(request.limit)
    )

    result = await session.execute(query)
    rows = result.scalars().all()

    return [
        SyncedEdge(
            id=e.id,
            from_id=e.from_id,
            to_id=e.to_id,
            relation=e.relation,
            weight=e.weight,
            created_by=e.created_by,
            created_at=e.created_at,
            updated_at=e.updated_at,
            vector_clock=e.vector_clock or {},
            deleted_at=e.deleted_at,
            last_modified_by=e.last_modified_by,
            metadata=dict(e.extra_metadata) if e.extra_metadata else {},
        )
        for e in rows
    ]


# =============================================================================
# Sync Status
# =============================================================================


@router.post("/status", response_model=SyncStatusResponse)
async def get_sync_status(
    request: SyncStatusRequest,
    session: AsyncSession = Depends(get_session_dependency),
) -> SyncStatusResponse:
    """Get sync status for a device."""
    # Get device info
    result = await session.execute(select(Device).where(Device.device_id == request.device_id))
    device = result.scalar_one_or_none()

    if not device:
        raise HTTPException(status_code=404, detail="Device not registered")

    # Count pending changes
    pending_pull = 0
    if device.sync_cursor:
        # Count memories updated after cursor
        result = await session.execute(
            select(SyncedMemoryModel)
            .where(SyncedMemoryModel.updated_at > device.sync_cursor)
            .limit(1)
        )
        if result.scalar_one_or_none():
            # There are pending changes, get actual count
            from sqlalchemy import func

            result = await session.execute(
                select(func.count())
                .select_from(SyncedMemoryModel)
                .where(SyncedMemoryModel.updated_at > device.sync_cursor)
            )
            pending_pull = result.scalar() or 0

    return SyncStatusResponse(
        device_id=request.device_id,
        last_sync_at=device.last_sync_at,
        pending_push_count=0,  # Server doesn't know client state
        pending_pull_count=pending_pull,
        server_timestamp=datetime.now(timezone.utc),
    )


# =============================================================================
# Content-Addressed Sync (Merkle-style)
# =============================================================================


@router.post("/diff", response_model=SyncDiffResponse)
async def compute_diff(
    request: SyncManifestRequest,
    session: AsyncSession = Depends(get_session_dependency),
) -> SyncDiffResponse:
    """
    Content-addressed sync: compare client manifest with server state.

    This is idempotent - run it 100 times, always correct result.
    Client sends list of {id, content_hash} for all their entities.
    Server returns:
      - What client is missing (for pull)
      - What server is missing (for push)
      - What was deleted
    """
    server_timestamp = datetime.now(timezone.utc)

    # Build lookup sets from client manifest
    client_memory_map = {e.id: e.content_hash for e in request.memories}
    client_session_ids = {e.id for e in request.sessions}
    client_edge_ids = {e.id for e in request.edges}

    missing_memories: list[SyncedMemory] = []
    missing_sessions: list[SyncedSession] = []
    missing_edges: list[SyncedEdge] = []
    deleted_memory_ids: list[str] = []
    deleted_session_ids: list[str] = []
    deleted_edge_ids: list[str] = []
    server_missing_memory_ids: list[str] = []
    server_missing_session_ids: list[str] = []
    server_missing_edge_ids: list[str] = []
    updated_count = 0

    # Query all server memories (optionally filtered by namespace)
    memory_query = select(SyncedMemoryModel)
    if request.namespace_ids:
        memory_query = memory_query.where(SyncedMemoryModel.namespace_id.in_(request.namespace_ids))
    result = await session.execute(memory_query)
    server_memories = result.scalars().all()

    # Build server memory map for reverse lookup
    server_memory_ids = set()

    for m in server_memories:
        server_memory_ids.add(m.id)
        client_hash = client_memory_map.get(m.id)

        if m.deleted_at:
            # Server has this as deleted
            if m.id in client_memory_map:
                deleted_memory_ids.append(m.id)
        elif client_hash is None:
            # Client doesn't have this memory at all
            missing_memories.append(_memory_model_to_synced(m))
        elif client_hash != m.content_hash:
            # Client has different content (outdated)
            missing_memories.append(_memory_model_to_synced(m))
            updated_count += 1

    # Find what server is missing (client has, server doesn't)
    for client_id in client_memory_map:
        if client_id not in server_memory_ids:
            server_missing_memory_ids.append(client_id)

    # Query all server sessions
    session_query = select(SyncedSessionModel)
    if request.namespace_ids:
        session_query = session_query.where(
            SyncedSessionModel.namespace_id.in_(request.namespace_ids)
        )
    result = await session.execute(session_query)
    server_sessions = result.scalars().all()

    # Build server session map for reverse lookup
    server_session_ids = set()

    for s in server_sessions:
        server_session_ids.add(s.id)
        if s.deleted_at:
            if s.id in client_session_ids:
                deleted_session_ids.append(s.id)
        elif s.id not in client_session_ids:
            missing_sessions.append(_session_model_to_synced(s))

    # Find what server is missing (sessions)
    for client_id in client_session_ids:
        if client_id not in server_session_ids:
            server_missing_session_ids.append(client_id)

    # Query all server edges
    result = await session.execute(select(SyncedEdgeModel))
    server_edges = result.scalars().all()

    # Build server edge map for reverse lookup
    server_edge_ids = set()

    for e in server_edges:
        edge_id = f"{e.from_id}:{e.to_id}:{e.relation}"
        server_edge_ids.add(edge_id)
        if e.deleted_at:
            if edge_id in client_edge_ids:
                deleted_edge_ids.append(edge_id)
        elif edge_id not in client_edge_ids:
            missing_edges.append(_edge_model_to_synced(e))

    # Find what server is missing (edges)
    for client_id in client_edge_ids:
        if client_id not in server_edge_ids:
            server_missing_edge_ids.append(client_id)

    # Update device sync state
    await _update_device_sync_state(session, request.device_id, pull_at=server_timestamp)
    await session.commit()

    total_missing = len(missing_memories) + len(missing_sessions) + len(missing_edges)
    total_deleted = len(deleted_memory_ids) + len(deleted_session_ids) + len(deleted_edge_ids)
    total_server_missing = (
        len(server_missing_memory_ids)
        + len(server_missing_session_ids)
        + len(server_missing_edge_ids)
    )

    logger.info(
        f"Diff computed for {request.device_id}: "
        f"{total_missing} missing, {updated_count} updated, {total_deleted} deleted, "
        f"{total_server_missing} server needs"
    )

    return SyncDiffResponse(
        success=True,
        missing_memories=missing_memories,
        missing_sessions=missing_sessions,
        missing_edges=missing_edges,
        deleted_memory_ids=deleted_memory_ids,
        deleted_session_ids=deleted_session_ids,
        deleted_edge_ids=deleted_edge_ids,
        server_missing_memory_ids=server_missing_memory_ids,
        server_missing_session_ids=server_missing_session_ids,
        server_missing_edge_ids=server_missing_edge_ids,
        total_missing=total_missing,
        total_updated=updated_count,
        total_deleted=total_deleted,
        total_server_missing=total_server_missing,
        server_timestamp=server_timestamp,
    )


def _memory_model_to_synced(m: SyncedMemoryModel) -> SyncedMemory:
    """Convert database model to sync protocol model."""
    return SyncedMemory(
        id=m.id,
        content=m.content,
        type=m.type,
        tags=m.tags or [],
        summary=m.summary,
        namespace_id=m.namespace_id,
        repo_url=m.repo_url,
        repo_name=m.repo_name,
        relative_path=m.relative_path,
        source_file=m.source_file,
        source_repo=m.source_repo,
        source_tool=m.source_tool,
        project=m.project,
        session_id=m.session_id,
        created_at=m.created_at,
        updated_at=m.updated_at,
        vector_clock=m.vector_clock or {},
        content_hash=m.content_hash,
        deleted_at=m.deleted_at,
        last_modified_by=m.last_modified_by,
        metadata=dict(m.extra_metadata) if m.extra_metadata else {},
        embedding=list(m.embedding)
        if hasattr(m, "embedding") and m.embedding is not None
        else None,
    )


def _session_model_to_synced(s: SyncedSessionModel) -> SyncedSession:
    """Convert database model to sync protocol model."""
    return SyncedSession(
        id=s.id,
        label=s.label,
        namespace_id=s.namespace_id,
        tool=s.tool,
        repo_url=s.repo_url,
        repo_name=s.repo_name,
        repo_path=s.repo_path,
        branch=s.branch,
        started_at=s.started_at,
        ended_at=s.ended_at,
        summary=s.summary,
        created_at=s.created_at,
        updated_at=s.updated_at,
        vector_clock=s.vector_clock or {},
        content_hash=s.content_hash,
        deleted_at=s.deleted_at,
        last_modified_by=s.last_modified_by,
        metadata=dict(s.extra_metadata) if s.extra_metadata else {},
    )


def _edge_model_to_synced(e: SyncedEdgeModel) -> SyncedEdge:
    """Convert database model to sync protocol model."""
    return SyncedEdge(
        id=e.id,
        from_id=e.from_id,
        to_id=e.to_id,
        relation=e.relation,
        weight=e.weight,
        created_by=e.created_by,
        created_at=e.created_at,
        updated_at=e.updated_at,
        vector_clock=e.vector_clock or {},
        deleted_at=e.deleted_at,
        last_modified_by=e.last_modified_by,
        metadata=dict(e.extra_metadata) if e.extra_metadata else {},
    )


# =============================================================================
# Helper Functions
# =============================================================================


async def _update_device_sync_state(
    session: AsyncSession,
    device_id: str,
    push_at: datetime | None = None,
    pull_at: datetime | None = None,
) -> None:
    """Update device sync state."""
    result = await session.execute(select(Device).where(Device.device_id == device_id))
    device = result.scalar_one_or_none()

    if device:
        now = datetime.now(timezone.utc)
        device.last_sync_at = now
        # Update sync_cursor on both push and pull
        # Push: device doesn't need to pull its own pushed data
        # Pull: device has received data up to this point
        if push_at:
            device.sync_cursor = push_at
        if pull_at:
            device.sync_cursor = pull_at

    # Also update sync_state table
    result = await session.execute(select(SyncState).where(SyncState.device_id == device_id))
    state = result.scalar_one_or_none()

    if state is None:
        state = SyncState(device_id=device_id)
        session.add(state)

    if push_at:
        state.last_push_at = push_at
        state.push_cursor = push_at
    if pull_at:
        state.last_pull_at = pull_at
        state.pull_cursor = pull_at
