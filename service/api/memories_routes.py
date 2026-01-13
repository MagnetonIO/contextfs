"""Memories API routes for web dashboard.

Provides read-only access to synced memories for the authenticated user.
"""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from contextfs.auth.api_keys import APIKey, User
from service.api.auth_middleware import require_auth
from service.db.models import SyncedMemoryModel as Memory
from service.db.session import get_session_dependency

router = APIRouter(prefix="/api/memories", tags=["memories"])


class MemoryResponse(BaseModel):
    """Memory data for frontend."""

    id: str
    content: str
    type: str
    tags: list[str]
    summary: str | None
    namespace_id: str
    repo_name: str | None
    source_tool: str | None
    project: str | None
    created_at: str
    updated_at: str


class MemorySearchResponse(BaseModel):
    """Search results."""

    memories: list[MemoryResponse]
    total: int
    limit: int
    offset: int


@router.get("/search", response_model=MemorySearchResponse)
async def search_memories(
    query: str = Query("*", description="Search query (* for all)"),
    type: str | None = Query(None, description="Filter by memory type"),
    namespace: str | None = Query(None, description="Filter by namespace"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
) -> MemorySearchResponse:
    """Search memories for the authenticated user.

    Filters by user_id for multi-tenant isolation.
    """
    user, _ = auth
    user_id = user.id

    # Build query - filter by user_id for multi-tenant isolation
    base_query = select(Memory).where(
        Memory.deleted_at.is_(None),
        Memory.user_id == user_id,
    )
    count_query = select(func.count(Memory.id)).where(
        Memory.deleted_at.is_(None),
        Memory.user_id == user_id,
    )

    # Apply filters
    if type:
        base_query = base_query.where(Memory.type == type)
        count_query = count_query.where(Memory.type == type)

    if namespace:
        base_query = base_query.where(Memory.namespace_id == namespace)
        count_query = count_query.where(Memory.namespace_id == namespace)

    # Text search if not wildcard
    if query and query != "*":
        # Use PostgreSQL full-text search
        search_filter = text(
            "to_tsvector('english', content) @@ plainto_tsquery('english', :query)"
        ).bindparams(query=query)
        base_query = base_query.where(search_filter)
        count_query = count_query.where(search_filter)

    # Get total count
    total_result = await session.execute(count_query)
    total = total_result.scalar() or 0

    # Get paginated results - order by created_at for consistent ordering
    base_query = base_query.order_by(Memory.created_at.desc()).offset(offset).limit(limit)
    result = await session.execute(base_query)
    memories = result.scalars().all()

    return MemorySearchResponse(
        memories=[
            MemoryResponse(
                id=m.id,
                content=m.content[:500] + "..." if len(m.content) > 500 else m.content,
                type=m.type,
                tags=m.tags or [],
                summary=m.summary,
                namespace_id=m.namespace_id,
                repo_name=m.repo_name,
                source_tool=m.source_tool,
                project=m.project,
                created_at=m.created_at.isoformat() if m.created_at else "",
                updated_at=m.updated_at.isoformat() if m.updated_at else "",
            )
            for m in memories
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/stats")
async def get_memory_stats(
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
) -> dict:
    """Get memory statistics for dashboard."""
    user, _ = auth
    user_id = user.id

    # Total count for this user
    total_result = await session.execute(
        select(func.count(Memory.id)).where(
            Memory.deleted_at.is_(None),
            Memory.user_id == user_id,
        )
    )
    total = total_result.scalar() or 0

    # Count by type
    type_result = await session.execute(
        select(Memory.type, func.count(Memory.id))
        .where(Memory.deleted_at.is_(None), Memory.user_id == user_id)
        .group_by(Memory.type)
    )
    by_type = {row[0]: row[1] for row in type_result.all()}

    # Count by namespace (top 10)
    ns_result = await session.execute(
        select(Memory.namespace_id, func.count(Memory.id))
        .where(Memory.deleted_at.is_(None), Memory.user_id == user_id)
        .group_by(Memory.namespace_id)
        .order_by(func.count(Memory.id).desc())
        .limit(10)
    )
    by_namespace = {row[0]: row[1] for row in ns_result.all()}

    return {
        "total": total,
        "by_type": by_type,
        "by_namespace": by_namespace,
    }


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
) -> MemoryResponse:
    """Get a specific memory by ID."""
    result = await session.execute(
        select(Memory).where(Memory.id == memory_id, Memory.deleted_at.is_(None))
    )
    memory = result.scalar_one_or_none()

    if not memory:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Memory not found")

    return MemoryResponse(
        id=memory.id,
        content=memory.content,
        type=memory.type,
        tags=memory.tags or [],
        summary=memory.summary,
        namespace_id=memory.namespace_id,
        repo_name=memory.repo_name,
        source_tool=memory.source_tool,
        project=memory.project,
        created_at=memory.created_at.isoformat() if memory.created_at else "",
        updated_at=memory.updated_at.isoformat() if memory.updated_at else "",
    )
