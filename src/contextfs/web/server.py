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
from pathlib import Path
from typing import Optional, Any, List
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from contextfs.core import ContextFS
from contextfs.schemas import MemoryType, Memory, Session, SearchResult
from contextfs.fts import FTSBackend, HybridSearch

logger = logging.getLogger(__name__)


# ==================== Pydantic Models ====================

class MemoryCreate(BaseModel):
    content: str
    type: str = "fact"
    tags: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    namespace_id: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class MemoryResponse(BaseModel):
    id: str
    content: str
    type: str
    tags: List[str]
    summary: Optional[str]
    namespace_id: str
    source_file: Optional[str]
    source_repo: Optional[str]
    session_id: Optional[str]
    created_at: str
    updated_at: str
    metadata: dict


class SearchResultResponse(BaseModel):
    memory: MemoryResponse
    score: float
    highlights: List[str] = Field(default_factory=list)


class SessionResponse(BaseModel):
    id: str
    label: Optional[str]
    namespace_id: str
    tool: str
    repo_path: Optional[str]
    branch: Optional[str]
    started_at: str
    ended_at: Optional[str]
    summary: Optional[str]
    message_count: int = 0


class SessionDetailResponse(SessionResponse):
    messages: List[dict] = Field(default_factory=list)


class StatsResponse(BaseModel):
    total_memories: int
    memories_by_type: dict
    total_sessions: int
    namespaces: List[str]
    fts_indexed: int
    rag_indexed: int


class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None


class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: dict = Field(default_factory=dict)
    id: Optional[Any] = None


class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[dict] = None
    id: Optional[Any] = None


# ==================== WebSocket Manager ====================

class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


# ==================== Application Factory ====================

def create_app(
    ctx: Optional[ContextFS] = None,
    static_dir: Optional[Path] = None,
    templates_dir: Optional[Path] = None,
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

        logger.info("ContextFS web server started")
        yield
        # Shutdown
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

    # ==================== Search API ====================

    @app.get("/api/search", response_model=APIResponse)
    async def search(
        q: str = Query(..., description="Search query"),
        type: Optional[str] = Query(None, description="Memory type filter"),
        namespace: Optional[str] = Query(None, description="Namespace filter"),
        limit: int = Query(20, ge=1, le=100, description="Max results"),
        semantic: bool = Query(True, description="Use semantic search"),
    ):
        """Search memories using semantic or keyword search."""
        ctx = get_ctx()
        hybrid = get_hybrid()
        fts = get_fts()

        try:
            memory_type = MemoryType(type) if type else None

            if semantic:
                results = await asyncio.to_thread(
                    hybrid.search,
                    query=q,
                    limit=limit,
                    type=memory_type,
                    namespace_id=namespace,
                )
            else:
                results = await asyncio.to_thread(
                    fts.search,
                    query=q,
                    limit=limit,
                    type=memory_type,
                    namespace_id=namespace,
                )

            return APIResponse(
                success=True,
                data=[serialize_search_result(r) for r in results],
            )

        except Exception as e:
            logger.exception("Search error")
            return APIResponse(success=False, error=str(e))

    # ==================== Memory API ====================

    @app.get("/api/memories", response_model=APIResponse)
    async def list_memories(
        limit: int = Query(20, ge=1, le=100),
        type: Optional[str] = None,
        namespace: Optional[str] = None,
    ):
        """List recent memories."""
        ctx = get_ctx()

        try:
            memories = await asyncio.to_thread(
                ctx.list_recent,
                limit=limit,
                type=MemoryType(type) if type else None,
                namespace_id=namespace,
            )

            return APIResponse(
                success=True,
                data=[serialize_memory(m) for m in memories],
            )

        except Exception as e:
            logger.exception("List memories error")
            return APIResponse(success=False, error=str(e))

    @app.get("/api/memories/{memory_id}", response_model=APIResponse)
    async def get_memory(memory_id: str):
        """Get a specific memory."""
        ctx = get_ctx()

        try:
            memory = await asyncio.to_thread(ctx.recall, memory_id)

            if memory:
                return APIResponse(success=True, data=serialize_memory(memory))
            else:
                raise HTTPException(status_code=404, detail="Memory not found")

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Get memory error")
            return APIResponse(success=False, error=str(e))

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
            )

            # Notify WebSocket clients
            await ws_manager.broadcast({
                "type": "memory_created",
                "memory": serialize_memory(memory),
            })

            return APIResponse(success=True, data=serialize_memory(memory))

        except Exception as e:
            logger.exception("Create memory error")
            return APIResponse(success=False, error=str(e))

    @app.delete("/api/memories/{memory_id}", response_model=APIResponse)
    async def delete_memory(memory_id: str):
        """Delete a memory."""
        ctx = get_ctx()

        try:
            deleted = await asyncio.to_thread(ctx.delete, memory_id)

            if deleted:
                await ws_manager.broadcast({
                    "type": "memory_deleted",
                    "id": memory_id,
                })
                return APIResponse(success=True)
            else:
                raise HTTPException(status_code=404, detail="Memory not found")

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Delete memory error")
            return APIResponse(success=False, error=str(e))

    # ==================== Session API ====================

    @app.get("/api/sessions", response_model=APIResponse)
    async def list_sessions(
        limit: int = Query(20, ge=1, le=100),
        tool: Optional[str] = None,
    ):
        """List sessions."""
        ctx = get_ctx()

        try:
            sessions = await asyncio.to_thread(
                ctx.list_sessions,
                limit=limit,
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

    @app.get("/api/namespaces", response_model=APIResponse)
    async def get_namespaces():
        """Get list of namespaces."""
        ctx = get_ctx()

        try:
            memories = await asyncio.to_thread(ctx.list_recent, limit=10000)
            namespaces = list(set(m.namespace_id for m in memories))
            return APIResponse(success=True, data=namespaces)

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
                headers={
                    "Content-Disposition": 'attachment; filename="contextfs-export.json"'
                },
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
                            "enum": ["fact", "decision", "procedural", "episodic", "user", "code", "error"],
                            "default": "fact",
                        },
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "summary": {"type": "string"},
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


async def mcp_tools_call(params: dict, ctx: ContextFS, hybrid: HybridSearch, fts: FTSBackend) -> dict:
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
        )
        return {"content": [{"type": "text", "text": f"Saved memory {memory.id}"}]}

    elif tool_name == "contextfs_search":
        use_semantic = arguments.get("semantic", True)
        search_fn = hybrid.search if use_semantic else fts.search

        results = await asyncio.to_thread(
            search_fn,
            query=arguments["query"],
            limit=arguments.get("limit", 10),
            type=MemoryType(arguments["type"]) if arguments.get("type") else None,
        )
        text = "\n\n".join([
            f"[{r.memory.type.value}] {r.memory.content[:200]}..."
            for r in results
        ])
        return {"content": [{"type": "text", "text": text or "No results found"}]}

    elif tool_name == "contextfs_recall":
        memory = await asyncio.to_thread(ctx.recall, arguments["id"])
        if memory:
            return {"content": [{"type": "text", "text": f"[{memory.type.value}] {memory.content}"}]}
        return {"content": [{"type": "text", "text": "Memory not found"}]}

    elif tool_name == "contextfs_list":
        memories = await asyncio.to_thread(
            ctx.list_recent,
            limit=arguments.get("limit", 10),
            type=MemoryType(arguments["type"]) if arguments.get("type") else None,
        )
        text = "\n".join([
            f"- [{m.type.value}] {m.id[:8]}: {m.content[:100]}..."
            for m in memories
        ])
        return {"content": [{"type": "text", "text": text or "No memories"}]}

    elif tool_name == "contextfs_sessions":
        sessions = await asyncio.to_thread(
            ctx.list_sessions,
            limit=arguments.get("limit", 10),
        )
        text = "\n".join([
            f"- {s.id[:8]}: {s.tool} ({s.started_at.isoformat()})"
            for s in sessions
        ])
        return {"content": [{"type": "text", "text": text or "No sessions"}]}

    elif tool_name == "contextfs_load_session":
        session = await asyncio.to_thread(
            ctx.load_session,
            session_id=arguments.get("session_id"),
            label=arguments.get("label"),
        )
        if session:
            messages = "\n".join([
                f"{m.role}: {m.content[:200]}"
                for m in session.messages[-10:]
            ])
            return {"content": [{"type": "text", "text": messages or "Empty session"}]}
        return {"content": [{"type": "text", "text": "Session not found"}]}

    else:
        return {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}]}


# ==================== Server Runner ====================

def run_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    ctx: Optional[ContextFS] = None,
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
