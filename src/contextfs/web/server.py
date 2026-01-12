"""
ContextFS Async Web Server (FastAPI)

Provides:
- REST API for memory operations
- WebSocket support for real-time updates
- MCP protocol endpoint
- Static file serving for web UI
- Database download for sql.js client
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from contextfs.core import ContextFS
from contextfs.fts import FTSBackend, HybridSearch
from contextfs.schemas import Memory, MemoryType, SearchResult, Session

# Optional asyncpg for PostgreSQL sync database
try:
    import asyncpg

    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

logger = logging.getLogger(__name__)


# ==================== Auth Models ====================


class SignupRequest(BaseModel):
    email: str
    name: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class VerifyEmailRequest(BaseModel):
    token: str


class ResendVerificationRequest(BaseModel):
    email: str


class CreateApiKeyRequest(BaseModel):
    name: str
    with_encryption: bool = True


class RevokeApiKeyRequest(BaseModel):
    key_id: str


# ==================== Pydantic Models ====================


class MemoryCreate(BaseModel):
    content: str
    type: str = "fact"
    tags: list[str] = Field(default_factory=list)
    summary: str | None = None
    namespace_id: str | None = None
    metadata: dict = Field(default_factory=dict)
    structured_data: dict | None = Field(
        default=None,
        description="Optional structured data validated against the type's JSON schema",
    )


class MemoryUpdate(BaseModel):
    content: str | None = None
    type: str | None = None
    tags: list[str] | None = None
    summary: str | None = None
    project: str | None = None


class SessionUpdate(BaseModel):
    label: str | None = None
    summary: str | None = None


class MemoryResponse(BaseModel):
    id: str
    content: str
    type: str
    tags: list[str]
    summary: str | None
    namespace_id: str
    source_file: str | None
    source_repo: str | None
    session_id: str | None
    created_at: str
    updated_at: str
    metadata: dict
    structured_data: dict | None = None


class SearchResultResponse(BaseModel):
    memory: MemoryResponse
    score: float
    highlights: list[str] = Field(default_factory=list)
    source: str | None = None  # "fts", "rag", or "hybrid"


class SessionResponse(BaseModel):
    id: str
    label: str | None
    namespace_id: str
    tool: str
    repo_path: str | None
    branch: str | None
    started_at: str
    ended_at: str | None
    summary: str | None
    message_count: int = 0


class SessionDetailResponse(SessionResponse):
    messages: list[dict] = Field(default_factory=list)


class StatsResponse(BaseModel):
    total_memories: int
    memories_by_type: dict
    total_sessions: int
    namespaces: list[str]
    fts_indexed: int
    rag_indexed: int


class APIResponse(BaseModel):
    success: bool
    data: Any | None = None
    error: str | None = None


class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: dict = Field(default_factory=dict)
    id: Any | None = None


class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Any | None = None
    error: dict | None = None
    id: Any | None = None


# ==================== WebSocket Manager ====================


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            with suppress(Exception):
                await connection.send_json(message)


# ==================== Application Factory ====================


def create_app(
    ctx: ContextFS | None = None,
    static_dir: Path | None = None,
    templates_dir: Path | None = None,
) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        ctx: ContextFS instance (created if not provided)
        static_dir: Static files directory
        templates_dir: Templates directory

    Returns:
        FastAPI application
    """
    # Initialize on first request if not provided
    app_state = {
        "ctx": ctx,
        "fts": None,
        "hybrid": None,
        "auth_storage": None,
        "sync_pool": None,  # PostgreSQL pool for sync database (memories)
    }

    ws_manager = ConnectionManager()

    static_dir = static_dir or Path(__file__).parent / "static"
    templates_dir = templates_dir or Path(__file__).parent / "templates"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        if app_state["ctx"] is None:
            app_state["ctx"] = ContextFS()

        ctx = app_state["ctx"]
        app_state["fts"] = FTSBackend(ctx._db_path)
        app_state["hybrid"] = HybridSearch(app_state["fts"], ctx.rag)

        # Initialize auth storage (uses config to determine backend)
        from contextfs.auth.storage import create_auth_storage

        app_state["auth_storage"] = create_auth_storage()

        # Initialize sync pool for PostgreSQL memories (if configured)
        postgres_url = os.environ.get("CONTEXTFS_POSTGRES_URL")
        if postgres_url and HAS_ASYNCPG:
            try:
                app_state["sync_pool"] = await asyncpg.create_pool(
                    postgres_url, min_size=2, max_size=10
                )
                logger.info("PostgreSQL sync pool created for memories")
            except Exception as e:
                logger.warning(f"Failed to create sync pool: {e}")

        logger.info("ContextFS web server started")
        yield
        # Shutdown
        if app_state["sync_pool"]:
            await app_state["sync_pool"].close()
        if app_state["auth_storage"]:
            await app_state["auth_storage"].close()
        if ctx:
            ctx.close()
        logger.info("ContextFS web server stopped")

    app = FastAPI(
        title="ContextFS",
        description="Universal AI Memory Layer",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_ctx() -> ContextFS:
        return app_state["ctx"]

    def get_fts() -> FTSBackend:
        return app_state["fts"]

    def get_hybrid() -> HybridSearch:
        return app_state["hybrid"]

    def get_auth_storage():
        """Get auth storage instance."""
        return app_state["auth_storage"]

    def get_sync_pool():
        """Get PostgreSQL sync pool for memories."""
        return app_state["sync_pool"]

    async def query_sync_memories(
        limit: int = 20,
        type_filter: str | None = None,
        query: str | None = None,
    ) -> list[dict]:
        """Query memories from PostgreSQL sync database."""
        pool = get_sync_pool()
        if not pool:
            # Fallback to local if no sync pool
            ctx = get_ctx()
            memories = ctx.list_recent(
                limit=limit, type=MemoryType(type_filter) if type_filter else None
            )
            return [serialize_memory(m) for m in memories]

        async with pool.acquire() as conn:
            if query and query != "*":
                # Full-text search
                rows = await conn.fetch(
                    """
                    SELECT id, content, type, tags, summary, namespace_id,
                           source_file, source_repo, session_id, created_at, updated_at,
                           metadata, structured_data
                    FROM memories
                    WHERE deleted_at IS NULL
                      AND ($1::text IS NULL OR type = $1)
                      AND to_tsvector('english', content) @@ plainto_tsquery('english', $2)
                    ORDER BY created_at DESC
                    LIMIT $3
                    """,
                    type_filter,
                    query,
                    limit,
                )
            else:
                # List recent
                rows = await conn.fetch(
                    """
                    SELECT id, content, type, tags, summary, namespace_id,
                           source_file, source_repo, session_id, created_at, updated_at,
                           metadata, structured_data
                    FROM memories
                    WHERE deleted_at IS NULL
                      AND ($1::text IS NULL OR type = $1)
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    type_filter,
                    limit,
                )

        def parse_json_field(value):
            """Parse JSON field - handles both dicts and JSON strings."""
            if value is None:
                return {}
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return {}
            return {}

        return [
            {
                "id": row["id"],
                "content": row["content"],
                "type": row["type"],
                "tags": row["tags"] or [],
                "summary": row["summary"],
                "namespace_id": row["namespace_id"],
                "source_file": row["source_file"],
                "source_repo": row["source_repo"],
                "session_id": row["session_id"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                "metadata": parse_json_field(row["metadata"]),
                "structured_data": parse_json_field(row["structured_data"]),
            }
            for row in rows
        ]

    async def get_sync_memory(memory_id: str) -> dict | None:
        """Get a specific memory from PostgreSQL sync database."""
        pool = get_sync_pool()
        if not pool:
            # Fallback to local if no sync pool
            ctx = get_ctx()
            memory = ctx.recall(memory_id)
            return serialize_memory(memory) if memory else None

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, content, type, tags, summary, namespace_id,
                       source_file, source_repo, session_id, created_at, updated_at,
                       metadata, structured_data
                FROM memories
                WHERE id = $1 AND deleted_at IS NULL
                """,
                memory_id,
            )

        if not row:
            return None

        def parse_json_field(value):
            """Parse JSON field - handles both dicts and JSON strings."""
            if value is None:
                return {}
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return {}
            return {}

        return {
            "id": row["id"],
            "content": row["content"],
            "type": row["type"],
            "tags": row["tags"] or [],
            "summary": row["summary"],
            "namespace_id": row["namespace_id"],
            "source_file": row["source_file"],
            "source_repo": row["source_repo"],
            "session_id": row["session_id"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
            "metadata": parse_json_field(row["metadata"]),
            "structured_data": parse_json_field(row["structured_data"]),
        }

    async def delete_sync_memory(memory_id: str) -> bool:
        """Soft delete a memory from PostgreSQL sync database."""
        pool = get_sync_pool()
        if not pool:
            # Fallback to local if no sync pool
            ctx = get_ctx()
            return ctx.delete(memory_id)

        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE memories SET deleted_at = NOW()
                WHERE id = $1 AND deleted_at IS NULL
                """,
                memory_id,
            )
        return result != "UPDATE 0"

    async def count_sync_memories() -> int:
        """Count total memories in PostgreSQL sync database."""
        pool = get_sync_pool()
        if not pool:
            # Fallback to local if no sync pool
            ctx = get_ctx()
            return len(ctx.list_recent(limit=100000))

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) as count FROM memories WHERE deleted_at IS NULL"
            )
        return row["count"] if row else 0

    # ==================== Search API ====================

    @app.get("/api/search", response_model=APIResponse)
    async def search(
        q: str = Query(..., description="Search query"),
        type: str | None = Query(None, description="Memory type filter"),
        namespace: str | None = Query(None, description="Namespace filter"),
        limit: int = Query(20, ge=1, le=100, description="Max results"),
        offset: int = Query(0, ge=0, description="Offset for pagination"),
        semantic: bool = Query(True, description="Use semantic search"),
        smart: bool = Query(False, description="Use smart routing based on memory type"),
    ):
        """Search memories using semantic or keyword search."""
        get_ctx()
        hybrid = get_hybrid()
        fts = get_fts()

        try:
            memory_type = MemoryType(type) if type else None
            # Fetch extra results to support offset pagination
            fetch_limit = limit + offset

            if smart:
                # Smart search routes to optimal backend based on memory type
                results = await asyncio.to_thread(
                    hybrid.smart_search,
                    query=q,
                    limit=fetch_limit,
                    type=memory_type,
                    namespace_id=namespace,
                )
            elif semantic:
                results = await asyncio.to_thread(
                    hybrid.search,
                    query=q,
                    limit=fetch_limit,
                    type=memory_type,
                    namespace_id=namespace,
                )
            else:
                results = await asyncio.to_thread(
                    fts.search,
                    query=q,
                    limit=fetch_limit,
                    type=memory_type,
                    namespace_id=namespace,
                )

            # Apply offset pagination
            paginated_results = results[offset : offset + limit]

            return APIResponse(
                success=True,
                data=[serialize_search_result(r) for r in paginated_results],
            )

        except Exception as e:
            logger.exception("Search error")
            return APIResponse(success=False, error=str(e))

    @app.get("/api/search/dual", response_model=APIResponse)
    async def search_dual(
        q: str = Query(..., description="Search query"),
        type: str | None = Query(None, description="Memory type filter"),
        namespace: str | None = Query(None, description="Namespace filter"),
        limit: int = Query(20, ge=1, le=100, description="Max results"),
    ):
        """Search both FTS and RAG backends separately and return results from each."""
        get_ctx()
        hybrid = get_hybrid()

        try:
            memory_type = MemoryType(type) if type else None

            results = await asyncio.to_thread(
                hybrid.search_both,
                query=q,
                limit=limit,
                type=memory_type,
                namespace_id=namespace,
            )

            return APIResponse(
                success=True,
                data={
                    "fts": [serialize_search_result(r) for r in results["fts"]],
                    "rag": [serialize_search_result(r) for r in results["rag"]],
                },
            )

        except Exception as e:
            logger.exception("Dual search error")
            return APIResponse(success=False, error=str(e))

    # ==================== Memory API ====================

    @app.get("/api/memories", response_model=APIResponse)
    async def list_memories(
        limit: int = Query(20, ge=1, le=100),
        type: str | None = None,
        namespace: str | None = None,
    ):
        """List recent memories from sync database."""
        try:
            memories = await query_sync_memories(limit=limit, type_filter=type)
            return APIResponse(success=True, data=memories)

        except Exception as e:
            logger.exception("List memories error")
            return APIResponse(success=False, error=str(e))

    @app.get("/api/memories/search")
    async def search_memories(
        query: str = Query("*", description="Search query"),
        type: str | None = Query(None, description="Memory type filter"),
        limit: int = Query(20, ge=1, le=100, description="Max results"),
    ):
        """Search memories from sync database - returns {memories: [...]} format for frontend."""
        try:
            memories = await query_sync_memories(
                limit=limit,
                type_filter=type,
                query=query if query != "*" else None,
            )
            return {"memories": memories}

        except Exception as e:
            logger.exception("Search memories error")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/memories/{memory_id}")
    async def get_memory(memory_id: str):
        """Get a specific memory from sync database - returns Memory object directly."""
        try:
            memory = await get_sync_memory(memory_id)

            if memory:
                return memory
            else:
                raise HTTPException(status_code=404, detail="Memory not found")

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Get memory error")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/memories", response_model=APIResponse)
    async def create_memory(data: MemoryCreate):
        """Create a new memory."""
        ctx = get_ctx()

        try:
            memory = await asyncio.to_thread(
                ctx.save,
                content=data.content,
                type=MemoryType(data.type),
                tags=data.tags,
                summary=data.summary,
                namespace_id=data.namespace_id,
                metadata=data.metadata,
                structured_data=data.structured_data,
            )

            # Notify WebSocket clients
            await ws_manager.broadcast(
                {
                    "type": "memory_created",
                    "memory": serialize_memory(memory),
                }
            )

            return APIResponse(success=True, data=serialize_memory(memory))

        except Exception as e:
            logger.exception("Create memory error")
            return APIResponse(success=False, error=str(e))

    @app.delete("/api/memories/{memory_id}", response_model=APIResponse)
    async def delete_memory(memory_id: str):
        """Delete a memory from sync database."""
        try:
            deleted = await delete_sync_memory(memory_id)

            if deleted:
                await ws_manager.broadcast(
                    {
                        "type": "memory_deleted",
                        "id": memory_id,
                    }
                )
                return APIResponse(success=True)
            else:
                raise HTTPException(status_code=404, detail="Memory not found")

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Delete memory error")
            return APIResponse(success=False, error=str(e))

    @app.put("/api/memories/{memory_id}", response_model=APIResponse)
    async def update_memory(memory_id: str, data: MemoryUpdate):
        """Update a memory."""
        ctx = get_ctx()

        try:
            memory_type = MemoryType(data.type) if data.type else None

            memory = await asyncio.to_thread(
                ctx.update,
                memory_id=memory_id,
                content=data.content,
                type=memory_type,
                tags=data.tags,
                summary=data.summary,
                project=data.project,
            )

            if memory:
                await ws_manager.broadcast(
                    {
                        "type": "memory_updated",
                        "memory": serialize_memory(memory),
                    }
                )
                return APIResponse(success=True, data=serialize_memory(memory))
            else:
                raise HTTPException(status_code=404, detail="Memory not found")

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Update memory error")
            return APIResponse(success=False, error=str(e))

    # ==================== Session API ====================

    @app.get("/api/sessions", response_model=APIResponse)
    async def list_sessions(
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
        tool: str | None = None,
    ):
        """List sessions."""
        ctx = get_ctx()

        try:
            sessions = await asyncio.to_thread(
                ctx.list_sessions,
                limit=limit,
                offset=offset,
                tool=tool,
            )

            return APIResponse(
                success=True,
                data=[serialize_session(s) for s in sessions],
            )

        except Exception as e:
            logger.exception("List sessions error")
            return APIResponse(success=False, error=str(e))

    @app.get("/api/sessions/{session_id}", response_model=APIResponse)
    async def get_session(session_id: str):
        """Get a specific session with messages."""
        ctx = get_ctx()

        try:
            session = await asyncio.to_thread(
                ctx.load_session,
                session_id=session_id,
            )

            if session:
                data = serialize_session(session)
                data["messages"] = [
                    {
                        "id": m.id,
                        "role": m.role,
                        "content": m.content,
                        "timestamp": m.timestamp.isoformat(),
                    }
                    for m in session.messages
                ]
                return APIResponse(success=True, data=data)
            else:
                raise HTTPException(status_code=404, detail="Session not found")

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Get session error")
            return APIResponse(success=False, error=str(e))

    @app.put("/api/sessions/{session_id}", response_model=APIResponse)
    async def update_session(session_id: str, data: SessionUpdate):
        """Update a session."""
        ctx = get_ctx()

        try:
            session = await asyncio.to_thread(
                ctx.update_session,
                session_id=session_id,
                label=data.label,
                summary=data.summary,
            )

            if session:
                await ws_manager.broadcast(
                    {
                        "type": "session_updated",
                        "session": serialize_session(session),
                    }
                )
                return APIResponse(success=True, data=serialize_session(session))
            else:
                raise HTTPException(status_code=404, detail="Session not found")

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Update session error")
            return APIResponse(success=False, error=str(e))

    @app.delete("/api/sessions/{session_id}", response_model=APIResponse)
    async def delete_session(session_id: str):
        """Delete a session."""
        ctx = get_ctx()

        try:
            deleted = await asyncio.to_thread(ctx.delete_session, session_id)

            if deleted:
                await ws_manager.broadcast(
                    {
                        "type": "session_deleted",
                        "id": session_id,
                    }
                )
                return APIResponse(success=True)
            else:
                raise HTTPException(status_code=404, detail="Session not found")

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Delete session error")
            return APIResponse(success=False, error=str(e))

    # ==================== Auth API ====================

    def _hash_password(password: str) -> str:
        """Hash password using bcrypt."""
        import bcrypt

        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def _verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash."""
        import bcrypt

        return bcrypt.checkpw(password.encode(), hashed.encode())

    def _generate_verification_token() -> str:
        """Generate a secure random token."""
        import secrets

        return secrets.token_urlsafe(32)

    def _log_verification_email(email: str, name: str, token: str) -> None:
        """Log verification link to terminal for local development."""
        import os

        base_url = os.environ.get("FRONTEND_URL", "http://localhost:3000")
        verify_url = f"{base_url}/verify-email?token={token}"
        print("\n")
        print("=" * 50)
        print("📧 VERIFICATION EMAIL")
        print("=" * 50)
        print(f"To: {email}")
        print(f"Name: {name}")
        print("")
        print("Click this link to verify your email:")
        print(f"🔗 {verify_url}")
        print("=" * 50)
        print("\n")

    @app.post("/api/auth/signup")
    async def signup(data: SignupRequest):
        """Register a new user with email/password."""
        from uuid import uuid4

        from contextfs.auth.api_keys import generate_api_key, generate_encryption_salt, hash_api_key

        try:
            if len(data.password) < 8:
                raise HTTPException(
                    status_code=400, detail="Password must be at least 8 characters"
                )

            storage = get_auth_storage()

            # Check if email exists
            existing = await storage.get_user_by_email(data.email)
            if existing:
                raise HTTPException(
                    status_code=400, detail="An account with this email already exists"
                )

            # Create user
            user_id = str(uuid4())
            password_hash = _hash_password(data.password)
            verification_token = _generate_verification_token()
            verification_expires = (datetime.now() + timedelta(hours=24)).isoformat()

            user = await storage.create_user(
                user_id=user_id,
                email=data.email,
                name=data.name,
                provider="email",
                password_hash=password_hash,
                email_verified=False,
                verification_token=verification_token,
                verification_token_expires=verification_expires,
            )

            # Create API key with E2EE salt (automatic for all users)
            raw_key, key_prefix = generate_api_key()
            key_hash = hash_api_key(raw_key)
            key_id = str(uuid4())
            encryption_salt = generate_encryption_salt()

            await storage.create_api_key(
                key_id=key_id,
                user_id=user_id,
                name="Default",
                key_hash=key_hash,
                key_prefix=key_prefix,
                encryption_salt=encryption_salt,
            )

            # Create free subscription
            sub_id = str(uuid4())
            await storage.create_subscription(
                sub_id=sub_id,
                user_id=user_id,
                tier="free",
                device_limit=3,
                memory_limit=10000,
            )

            # Log verification link for local development
            _log_verification_email(data.email, data.name, verification_token)

            return {
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "name": user.name,
                    "emailVerified": user.email_verified,
                    "createdAt": user.created_at,
                },
                "apiKey": raw_key,
                "message": "Account created! Check terminal for verification link.",
            }
        except HTTPException:
            raise
        except Exception:
            logger.exception("Signup error")
            raise HTTPException(status_code=500, detail="Something went wrong")

    @app.post("/api/auth/login")
    async def login(data: LoginRequest):
        """Authenticate user with email/password."""
        from uuid import uuid4

        from contextfs.auth.api_keys import generate_api_key, generate_encryption_salt, hash_api_key

        try:
            storage = get_auth_storage()

            # Get user
            user = await storage.get_user_by_email(data.email)
            if not user or user.provider != "email":
                raise HTTPException(status_code=401, detail="Invalid email or password")

            if not user.password_hash or not _verify_password(data.password, user.password_hash):
                raise HTTPException(status_code=401, detail="Invalid email or password")

            if not user.email_verified:
                raise HTTPException(status_code=403, detail="EmailNotVerified")

            # Get or create API key
            keys = await storage.list_api_keys(user.id)
            if keys:
                # Generate new key for this session (keep existing salt for E2EE continuity)
                raw_key, key_prefix = generate_api_key()
                key_hash = hash_api_key(raw_key)
                await storage.update_api_key(
                    keys[0].id,
                    key_hash=key_hash,
                    key_prefix=key_prefix,
                )
            else:
                # Create new API key with E2EE salt
                raw_key, key_prefix = generate_api_key()
                key_hash = hash_api_key(raw_key)
                key_id = str(uuid4())
                encryption_salt = generate_encryption_salt()
                await storage.create_api_key(
                    key_id=key_id,
                    user_id=user.id,
                    name="Default",
                    key_hash=key_hash,
                    key_prefix=key_prefix,
                    encryption_salt=encryption_salt,
                )

            # Update last login
            await storage.update_user(user.id, last_login=datetime.now().isoformat())

            return {
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "name": user.name,
                    "emailVerified": user.email_verified,
                    "createdAt": user.created_at,
                },
                "apiKey": raw_key,
            }
        except HTTPException:
            raise
        except Exception:
            logger.exception("Login error")
            raise HTTPException(status_code=500, detail="Something went wrong")

    @app.post("/api/auth/verify-email")
    async def verify_email(data: VerifyEmailRequest):
        """Verify email with token."""
        try:
            storage = get_auth_storage()
            result = await storage.get_user_by_verification_token(data.token)

            if not result:
                return {"success": False, "message": "Invalid verification token"}

            user, token_expires = result

            # Check if token expired
            if isinstance(token_expires, str):
                expires = datetime.fromisoformat(token_expires)
            else:
                expires = token_expires

            if datetime.now() > expires:
                return {"success": False, "message": "Verification token has expired"}

            # Mark email as verified
            await storage.update_user(
                user.id,
                email_verified=True,
                verification_token=None,
                verification_token_expires=None,
            )

            return {"success": True, "message": "Email verified successfully"}
        except Exception:
            logger.exception("Verify email error")
            raise HTTPException(status_code=500, detail="Something went wrong")

    @app.post("/api/auth/resend-verification")
    async def resend_verification(data: ResendVerificationRequest):
        """Resend verification email."""
        try:
            storage = get_auth_storage()
            user = await storage.get_user_by_email(data.email)

            if not user:
                return {"success": False, "message": "Email not found"}

            if user.email_verified:
                return {"success": False, "message": "Email is already verified"}

            # Generate new token
            new_token = _generate_verification_token()
            new_expires = (datetime.now() + timedelta(hours=24)).isoformat()

            await storage.update_user(
                user.id,
                verification_token=new_token,
                verification_token_expires=new_expires,
            )

            # Log verification link
            _log_verification_email(data.email, user.name or data.email, new_token)

            return {"success": True, "message": "Verification email sent"}
        except Exception:
            logger.exception("Resend verification error")
            raise HTTPException(status_code=500, detail="Something went wrong")

    @app.get("/api/auth/check-email")
    async def check_email(email: str = Query(...)):
        """Check if email is already registered."""
        try:
            storage = get_auth_storage()
            user = await storage.get_user_by_email(email)
            return {"exists": user is not None}
        except Exception:
            logger.exception("Check email error")
            raise HTTPException(status_code=500, detail="Something went wrong")

    @app.get("/api/auth/user")
    async def get_user_by_email(email: str = Query(...)):
        """Get user by email address."""
        try:
            storage = get_auth_storage()
            user = await storage.get_user_by_email(email)
            if user:
                return {
                    "id": user.id,
                    "email": user.email,
                    "name": user.name,
                    "emailVerified": user.email_verified,
                    "createdAt": user.created_at,
                }
            raise HTTPException(status_code=404, detail="User not found")
        except HTTPException:
            raise
        except Exception:
            logger.exception("Get user error")
            raise HTTPException(status_code=500, detail="Something went wrong")

    # ==================== API Key Management ====================

    async def get_user_from_api_key(api_key: str):
        """Get user ID from API key."""
        from contextfs.auth.api_keys import hash_api_key

        storage = get_auth_storage()
        key_hash = hash_api_key(api_key)
        return await storage.get_user_id_by_key_hash(key_hash)

    @app.get("/api/auth/me")
    async def get_current_user(request: Request):
        """Get current user from API key."""
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")

        user_id = await get_user_from_api_key(api_key)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid API key")

        storage = get_auth_storage()
        user = await storage.get_user_by_id(user_id)
        if user:
            return {
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "provider": user.provider,
            }
        raise HTTPException(status_code=404, detail="User not found")

    @app.get("/api/auth/api-keys")
    async def list_api_keys(request: Request):
        """List user's API keys."""
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")

        user_id = await get_user_from_api_key(api_key)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid API key")

        storage = get_auth_storage()
        keys = await storage.list_api_keys(user_id)
        return {
            "keys": [
                {
                    "id": key.id,
                    "name": key.name,
                    "key_prefix": key.key_prefix,
                    "is_active": key.is_active,
                    "created_at": key.created_at,
                    "last_used_at": key.last_used_at,
                }
                for key in keys
            ]
        }

    @app.get("/api/auth/encryption-salt")
    async def get_encryption_salt(request: Request):
        """Get the encryption salt for the current API key.

        Used by clients to derive the E2EE encryption key locally.
        The client derives: encryption_key = HKDF(api_key, salt)
        """
        from contextfs.auth.api_keys import hash_api_key

        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")

        storage = get_auth_storage()
        key_hash = hash_api_key(api_key)
        salt = await storage.get_encryption_salt_by_key_hash(key_hash)

        if salt is None:
            raise HTTPException(status_code=404, detail="No encryption salt found for this API key")

        return {"salt": salt}

    @app.post("/api/auth/api-keys")
    async def create_api_key(request: Request, data: CreateApiKeyRequest):
        """Create a new API key with automatic E2EE.

        E2EE is automatic for all users - the encryption key is derived
        client-side from the API key + salt. Server never sees the encryption key.
        """
        from uuid import uuid4

        from contextfs.auth.api_keys import generate_api_key, generate_encryption_salt, hash_api_key

        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")

        user_id = await get_user_from_api_key(api_key)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid API key")

        # Generate new key with E2EE salt (automatic for all users)
        raw_key, key_prefix = generate_api_key()
        key_hash = hash_api_key(raw_key)
        key_id = str(uuid4())
        encryption_salt = generate_encryption_salt()

        storage = get_auth_storage()
        await storage.create_api_key(
            key_id=key_id,
            user_id=user_id,
            name=data.name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            encryption_salt=encryption_salt,
        )

        config_snippet = f"""# ContextFS API Configuration
api_key: {raw_key}"""

        return {
            "id": key_id,
            "name": data.name,
            "api_key": raw_key,
            "key_prefix": key_prefix,
            "config_snippet": config_snippet,
            "e2ee": True,  # E2EE is automatic
        }

    @app.post("/api/auth/api-keys/revoke")
    async def revoke_api_key(request: Request, data: RevokeApiKeyRequest):
        """Revoke an API key."""
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")

        user_id = await get_user_from_api_key(api_key)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid API key")

        storage = get_auth_storage()
        revoked = await storage.revoke_api_key(data.key_id, user_id)
        if not revoked:
            raise HTTPException(status_code=404, detail="API key not found")
        return {"status": "revoked"}

    @app.delete("/api/auth/api-keys/{key_id}")
    async def delete_api_key(request: Request, key_id: str):
        """Delete an API key."""
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")

        user_id = await get_user_from_api_key(api_key)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid API key")

        storage = get_auth_storage()
        deleted = await storage.delete_api_key(key_id, user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="API key not found")
        return {"status": "deleted"}

    # ==================== Billing API ====================

    @app.get("/api/billing/subscription")
    async def get_subscription(request: Request):
        """Get user's subscription details."""
        from uuid import uuid4

        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")

        user_id = await get_user_from_api_key(api_key)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid API key")

        storage = get_auth_storage()
        sub = await storage.get_subscription(user_id)

        if sub:
            return {
                "tier": sub.tier,
                "status": sub.status,
                "device_limit": sub.device_limit,
                "memory_limit": sub.memory_limit,
                "current_period_end": sub.current_period_end,
            }

        # Create default free subscription if none exists
        sub_id = str(uuid4())
        sub = await storage.create_subscription(
            sub_id=sub_id,
            user_id=user_id,
            tier="free",
            device_limit=3,
            memory_limit=10000,
        )

        return {
            "tier": sub.tier,
            "status": sub.status,
            "device_limit": sub.device_limit,
            "memory_limit": sub.memory_limit,
            "current_period_end": sub.current_period_end,
        }

    @app.get("/api/billing/usage")
    async def get_usage(request: Request):
        """Get user's usage statistics from sync database."""
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")

        user_id = await get_user_from_api_key(api_key)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid API key")

        storage = get_auth_storage()

        # Get subscription limits
        sub = await storage.get_subscription(user_id)
        device_limit = sub.device_limit if sub else 3
        memory_limit = sub.memory_limit if sub else 10000

        # Count devices
        devices = await storage.list_devices(user_id)
        device_count = len(devices)

        # Count memories from sync database
        memory_count = await count_sync_memories()

        return {
            "device_count": device_count,
            "memory_count": memory_count,
            "device_limit": device_limit,
            "memory_limit": memory_limit,
            "device_usage_percent": round((device_count / device_limit) * 100, 1)
            if device_limit > 0
            else 0,
            "memory_usage_percent": round((memory_count / memory_limit) * 100, 1)
            if memory_limit > 0
            else 0,
        }

    @app.post("/api/billing/checkout")
    async def create_checkout(request: Request):
        """Create Stripe checkout session (stub)."""
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")

        user_id = await get_user_from_api_key(api_key)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid API key")

        # Stub - would integrate with Stripe in production
        return {"checkout_url": "https://checkout.stripe.com/stub"}

    @app.post("/api/billing/portal")
    async def create_portal(request: Request):
        """Create Stripe billing portal session (stub)."""
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")

        user_id = await get_user_from_api_key(api_key)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid API key")

        # Stub - would integrate with Stripe in production
        return {"portal_url": "https://billing.stripe.com/stub"}

    @app.post("/api/billing/cancel")
    async def cancel_subscription(request: Request):
        """Cancel subscription (stub)."""
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")

        user_id = await get_user_from_api_key(api_key)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid API key")

        # Stub - would integrate with Stripe in production
        return {
            "status": "cancelled",
            "message": "Subscription will be cancelled at end of billing period",
        }

    # ==================== Devices API ====================

    @app.get("/api/devices")
    async def list_devices(request: Request):
        """List user's registered devices."""
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")

        user_id = await get_user_from_api_key(api_key)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid API key")

        storage = get_auth_storage()
        devices = await storage.list_devices(user_id)

        return {
            "devices": [
                {
                    "id": device.id,
                    "name": device.name,
                    "device_type": device.device_type,
                    "os": device.os,
                    "os_version": device.os_version,
                    "is_current": False,  # Would check against current session
                    "last_sync_at": device.last_sync_at,
                    "created_at": device.created_at,
                }
                for device in devices
            ]
        }

    @app.delete("/api/devices/{device_id}")
    async def remove_device(request: Request, device_id: str):
        """Remove a device."""
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")

        user_id = await get_user_from_api_key(api_key)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid API key")

        storage = get_auth_storage()
        deleted = await storage.delete_device(device_id, user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Device not found")
        return {"status": "deleted"}

    # ==================== Stats & Utility API ====================

    @app.get("/api/stats", response_model=APIResponse)
    async def get_stats():
        """Get statistics."""
        ctx = get_ctx()
        fts = get_fts()

        try:
            memories = await asyncio.to_thread(ctx.list_recent, limit=10000)

            by_type: dict[str, int] = {}
            namespaces: set[str] = set()
            for m in memories:
                by_type[m.type.value] = by_type.get(m.type.value, 0) + 1
                namespaces.add(m.namespace_id)

            sessions = await asyncio.to_thread(ctx.list_sessions, limit=10000)
            fts_stats = await asyncio.to_thread(fts.get_stats)
            rag_stats = await asyncio.to_thread(ctx.rag.get_stats)

            return APIResponse(
                success=True,
                data={
                    "total_memories": len(memories),
                    "memories_by_type": by_type,
                    "total_sessions": len(sessions),
                    "namespaces": list(namespaces),
                    "fts_indexed": fts_stats.get("indexed_memories", 0),
                    "rag_indexed": rag_stats.get("total_memories", 0),
                },
            )

        except Exception as e:
            logger.exception("Stats error")
            return APIResponse(success=False, error=str(e))

    @app.get("/api/types", response_model=APIResponse)
    async def get_types():
        """Get all memory types with their configuration (colors, labels, etc).

        Use this endpoint to dynamically populate type dropdowns and styling.
        """
        from contextfs.schemas import get_memory_types

        try:
            types = get_memory_types()
            # Group by category for easier UI rendering
            core_types = [t for t in types if t.get("category") == "core"]
            extended_types = [t for t in types if t.get("category") == "extended"]

            return APIResponse(
                success=True,
                data={
                    "types": types,
                    "core": core_types,
                    "extended": extended_types,
                },
            )
        except Exception as e:
            logger.exception("Types error")
            return APIResponse(success=False, error=str(e))

    @app.get("/api/indexes", response_model=APIResponse)
    async def get_indexes():
        """Get full index status for all repositories.

        Returns list of all indexed repositories with:
        - namespace_id: Unique namespace identifier
        - repo_path: Full path to repository
        - repo_name: Repository directory name
        - files_indexed: Number of files indexed
        - commits_indexed: Number of git commits indexed
        - memories_created: Total memories created from indexing
        - indexed_at: Timestamp of last indexing
        """
        ctx = get_ctx()

        try:
            indexes = await asyncio.to_thread(ctx.list_indexes)

            total_files = 0
            total_commits = 0
            total_memories = 0

            index_list = []
            for idx in sorted(indexes, key=lambda x: x.memories_created, reverse=True):
                repo_name = idx.repo_path.split("/")[-1] if idx.repo_path else "unknown"
                index_list.append(
                    {
                        "namespace_id": idx.namespace_id,
                        "repo_path": idx.repo_path,
                        "repo_name": repo_name,
                        "files_indexed": idx.files_indexed,
                        "commits_indexed": idx.commits_indexed,
                        "memories_created": idx.memories_created,
                        "indexed_at": str(idx.indexed_at) if idx.indexed_at else None,
                    }
                )
                total_files += idx.files_indexed
                total_commits += idx.commits_indexed
                total_memories += idx.memories_created

            return APIResponse(
                success=True,
                data={
                    "indexes": index_list,
                    "totals": {
                        "repositories": len(indexes),
                        "files": total_files,
                        "commits": total_commits,
                        "memories": total_memories,
                    },
                },
            )
        except Exception as e:
            logger.exception("Indexes error")
            return APIResponse(success=False, error=str(e))

    @app.get("/api/namespaces", response_model=APIResponse)
    async def get_namespaces():
        """Get list of namespaces with human-readable display names."""
        ctx = get_ctx()

        try:
            memories = await asyncio.to_thread(ctx.list_recent, limit=10000)

            # Build namespace info - collect all source_repos for each namespace
            namespace_repos: dict[str, set[str]] = {}
            for m in memories:
                ns = m.namespace_id
                if ns not in namespace_repos:
                    namespace_repos[ns] = set()
                if m.source_repo:
                    namespace_repos[ns].add(m.source_repo)

            # Build final namespace info with best display name
            namespace_info = []
            for ns, repos in namespace_repos.items():
                # Filter out None-like values and get repo names
                valid_repos = [r for r in repos if r and r != "None"]

                if valid_repos:
                    # Use the first valid repo name
                    source_repo = valid_repos[0]
                    repo_name = source_repo.rstrip("/").split("/")[-1]
                    display_name = repo_name if repo_name else source_repo
                elif ns == "global":
                    display_name = "global"
                    source_repo = None
                else:
                    # No repo info - use namespace ID
                    display_name = ns
                    source_repo = None

                namespace_info.append(
                    {
                        "id": ns,
                        "display_name": display_name,
                        "source_repo": source_repo,
                    }
                )

            return APIResponse(success=True, data=namespace_info)

        except Exception as e:
            logger.exception("Namespaces error")
            return APIResponse(success=False, error=str(e))

    @app.get("/api/export")
    async def export_memories():
        """Export all memories as JSON."""
        ctx = get_ctx()

        try:
            memories = await asyncio.to_thread(ctx.list_recent, limit=100000)

            data = {
                "exported_at": datetime.now().isoformat(),
                "count": len(memories),
                "memories": [serialize_memory(m) for m in memories],
            }

            return JSONResponse(
                content=data,
                headers={"Content-Disposition": 'attachment; filename="contextfs-export.json"'},
            )

        except Exception as e:
            logger.exception("Export error")
            return APIResponse(success=False, error=str(e))

    @app.post("/api/sync", response_model=APIResponse)
    async def sync_to_postgres():
        """Sync to PostgreSQL."""
        ctx = get_ctx()

        try:
            from contextfs.sync import PostgresSync

            sync = PostgresSync()
            count = await sync.sync_all(ctx)
            return APIResponse(success=True, data={"synced": count})

        except ImportError:
            return APIResponse(success=False, error="PostgreSQL sync not available")
        except Exception as e:
            logger.exception("Sync error")
            return APIResponse(success=False, error=str(e))

    @app.get("/api/database")
    async def download_database():
        """Download SQLite database for sql.js."""
        ctx = get_ctx()

        db_path = ctx._db_path
        if db_path.exists():
            return FileResponse(
                path=db_path,
                filename="context.db",
                media_type="application/x-sqlite3",
            )
        else:
            raise HTTPException(status_code=404, detail="Database not found")

    # ==================== MCP Protocol ====================

    @app.post("/mcp", response_model=MCPResponse)
    async def handle_mcp(request: MCPRequest):
        """Handle MCP JSON-RPC requests."""
        ctx = get_ctx()
        hybrid = get_hybrid()
        fts = get_fts()

        try:
            if request.method == "tools/list":
                result = mcp_tools_list()
            elif request.method == "tools/call":
                result = await mcp_tools_call(request.params, ctx, hybrid, fts)
            else:
                return MCPResponse(
                    id=request.id,
                    error={"code": -32601, "message": f"Unknown method: {request.method}"},
                )

            return MCPResponse(id=request.id, result=result)

        except Exception as e:
            logger.exception("MCP error")
            return MCPResponse(
                id=request.id,
                error={"code": -32603, "message": str(e)},
            )

    @app.get("/mcp/sse")
    async def mcp_sse():
        """MCP Server-Sent Events endpoint."""

        async def event_stream():
            # Send initial tools list
            tools = mcp_tools_list()
            yield f"event: tools\ndata: {json.dumps(tools)}\n\n"

            # Keep connection alive
            while True:
                await asyncio.sleep(30)
                yield ": keepalive\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # ==================== WebSocket ====================

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        await ws_manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_json()
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)

    # ==================== Static Files & Index ====================

    @app.get("/")
    async def index():
        """Serve index.html."""
        index_path = templates_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return JSONResponse({"message": "ContextFS API", "docs": "/docs"})

    # Mount static files
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    return app


# ==================== Helper Functions ====================


def serialize_memory(memory: Memory) -> dict:
    """Serialize Memory to dict."""
    return {
        "id": memory.id,
        "content": memory.content,
        "type": memory.type.value,
        "tags": memory.tags,
        "summary": memory.summary,
        "namespace_id": memory.namespace_id,
        "source_file": memory.source_file,
        "source_repo": memory.source_repo,
        "session_id": memory.session_id,
        "created_at": memory.created_at.isoformat(),
        "updated_at": memory.updated_at.isoformat(),
        "metadata": memory.metadata,
        "structured_data": memory.structured_data,
    }


def serialize_session(session: Session) -> dict:
    """Serialize Session to dict."""
    return {
        "id": session.id,
        "label": session.label,
        "namespace_id": session.namespace_id,
        "tool": session.tool,
        "repo_path": session.repo_path,
        "branch": session.branch,
        "started_at": session.started_at.isoformat(),
        "ended_at": session.ended_at.isoformat() if session.ended_at else None,
        "summary": session.summary,
        "message_count": len(session.messages),
    }


def serialize_search_result(result: SearchResult) -> dict:
    """Serialize SearchResult to dict."""
    return {
        "memory": serialize_memory(result.memory),
        "score": result.score,
        "highlights": result.highlights,
        "source": result.source,
    }


def mcp_tools_list() -> dict:
    """Return list of available MCP tools."""
    return {
        "tools": [
            {
                "name": "contextfs_save",
                "description": "Save a memory to persistent storage",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to save"},
                        "type": {
                            "type": "string",
                            "enum": [
                                "fact",
                                "decision",
                                "procedural",
                                "episodic",
                                "user",
                                "code",
                                "error",
                                "commit",
                                "todo",
                                "issue",
                                "api",
                                "schema",
                                "test",
                                "review",
                                "release",
                                "config",
                                "dependency",
                                "doc",
                            ],
                            "default": "fact",
                        },
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "summary": {"type": "string"},
                        "structured_data": {
                            "type": "object",
                            "description": "Optional structured data validated against type's JSON schema",
                        },
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "contextfs_search",
                "description": "Search memories using semantic or keyword search",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "default": 10},
                        "type": {"type": "string"},
                        "semantic": {"type": "boolean", "default": True},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "contextfs_recall",
                "description": "Recall a specific memory by ID",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Memory ID (can be partial)"},
                    },
                    "required": ["id"],
                },
            },
            {
                "name": "contextfs_list",
                "description": "List recent memories",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 10},
                        "type": {"type": "string"},
                    },
                },
            },
            {
                "name": "contextfs_sessions",
                "description": "List recent sessions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "default": 10},
                    },
                },
            },
            {
                "name": "contextfs_load_session",
                "description": "Load a session by ID or label",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "label": {"type": "string"},
                    },
                },
            },
        ],
    }


async def mcp_tools_call(
    params: dict, ctx: ContextFS, hybrid: HybridSearch, fts: FTSBackend
) -> dict:
    """Execute an MCP tool call."""
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    if tool_name == "contextfs_save":
        memory = await asyncio.to_thread(
            ctx.save,
            content=arguments["content"],
            type=MemoryType(arguments.get("type", "fact")),
            tags=arguments.get("tags", []),
            summary=arguments.get("summary"),
            structured_data=arguments.get("structured_data"),
        )
        response_text = f"Saved memory {memory.id}"
        if memory.structured_data:
            response_text += f" with {len(memory.structured_data)} structured fields"
        return {"content": [{"type": "text", "text": response_text}]}

    elif tool_name == "contextfs_search":
        use_semantic = arguments.get("semantic", True)
        search_fn = hybrid.search if use_semantic else fts.search

        results = await asyncio.to_thread(
            search_fn,
            query=arguments["query"],
            limit=arguments.get("limit", 10),
            type=MemoryType(arguments["type"]) if arguments.get("type") else None,
        )
        text = "\n\n".join(
            [f"[{r.memory.type.value}] {r.memory.content[:200]}..." for r in results]
        )
        return {"content": [{"type": "text", "text": text or "No results found"}]}

    elif tool_name == "contextfs_recall":
        memory = await asyncio.to_thread(ctx.recall, arguments["id"])
        if memory:
            return {
                "content": [{"type": "text", "text": f"[{memory.type.value}] {memory.content}"}]
            }
        return {"content": [{"type": "text", "text": "Memory not found"}]}

    elif tool_name == "contextfs_list":
        memories = await asyncio.to_thread(
            ctx.list_recent,
            limit=arguments.get("limit", 10),
            type=MemoryType(arguments["type"]) if arguments.get("type") else None,
        )
        text = "\n".join([f"- [{m.type.value}] {m.id[:8]}: {m.content[:100]}..." for m in memories])
        return {"content": [{"type": "text", "text": text or "No memories"}]}

    elif tool_name == "contextfs_sessions":
        sessions = await asyncio.to_thread(
            ctx.list_sessions,
            limit=arguments.get("limit", 10),
        )
        text = "\n".join([f"- {s.id[:8]}: {s.tool} ({s.started_at.isoformat()})" for s in sessions])
        return {"content": [{"type": "text", "text": text or "No sessions"}]}

    elif tool_name == "contextfs_load_session":
        session = await asyncio.to_thread(
            ctx.load_session,
            session_id=arguments.get("session_id"),
            label=arguments.get("label"),
        )
        if session:
            messages = "\n".join([f"{m.role}: {m.content[:200]}" for m in session.messages[-10:]])
            return {"content": [{"type": "text", "text": messages or "Empty session"}]}
        return {"content": [{"type": "text", "text": "Session not found"}]}

    else:
        return {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}]}


# ==================== Server Runner ====================


def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    ctx: ContextFS | None = None,
    reload: bool = False,
) -> None:
    """
    Run the web server.

    Args:
        host: Server host
        port: Server port
        ctx: ContextFS instance
        reload: Enable auto-reload (dev mode)
    """
    app = create_app(ctx)
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ContextFS Web Server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_server(host=args.host, port=args.port, reload=args.reload)
