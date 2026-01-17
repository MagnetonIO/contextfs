"""WebSocket routes for real-time memory updates.

Provides WebSocket connections for clients to receive real-time
notifications when memories are created, updated, or deleted.
"""

import asyncio
import json
import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from service.api.auth_middleware import validate_api_key
from service.db.session import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

# Store active WebSocket connections by user_id
# In production, use Redis pub/sub for multi-instance support
_connections: dict[str, set[WebSocket]] = {}
_lock = asyncio.Lock()


async def add_connection(user_id: str, websocket: WebSocket) -> None:
    """Add a WebSocket connection for a user."""
    async with _lock:
        if user_id not in _connections:
            _connections[user_id] = set()
        _connections[user_id].add(websocket)
        logger.info(
            f"WebSocket connected for user {user_id[:8]}... (total: {len(_connections[user_id])})"
        )


async def remove_connection(user_id: str, websocket: WebSocket) -> None:
    """Remove a WebSocket connection for a user."""
    async with _lock:
        if user_id in _connections:
            _connections[user_id].discard(websocket)
            if not _connections[user_id]:
                del _connections[user_id]
            logger.info(f"WebSocket disconnected for user {user_id[:8]}...")


async def notify_user(user_id: str, event_type: str, data: dict) -> None:
    """Send a notification to all connected clients for a user.

    Args:
        user_id: The user ID to notify
        event_type: Event type (memory_created, memory_updated, memory_deleted)
        data: Event data (e.g., memory_id, count)
    """
    async with _lock:
        connections = _connections.get(user_id, set()).copy()

    if not connections:
        return

    message = json.dumps(
        {
            "type": event_type,
            "data": data,
        }
    )

    disconnected = []
    for ws in connections:
        try:
            await ws.send_text(message)
        except Exception as e:
            logger.warning(f"Failed to send WebSocket message: {e}")
            disconnected.append(ws)

    # Clean up disconnected clients
    for ws in disconnected:
        await remove_connection(user_id, ws)


@router.websocket("/ws/memories")
async def memories_websocket(
    websocket: WebSocket,
    api_key: str = Query(..., description="API key for authentication"),
) -> None:
    """WebSocket endpoint for real-time memory updates.

    Connect with: ws://host:port/ws/memories?api_key=ctxfs_...

    Messages sent to client:
    - {"type": "memory_created", "data": {"count": 5}}
    - {"type": "memory_updated", "data": {"memory_id": "..."}}
    - {"type": "memory_deleted", "data": {"memory_id": "..."}}
    - {"type": "ping"}

    Client can send:
    - {"type": "pong"} - Response to ping
    """
    # Validate API key before accepting connection
    async with get_session() as session:
        auth = await validate_api_key(api_key, session)

    if not auth:
        await websocket.close(code=4001, reason="Invalid API key")
        return

    user, _ = auth
    user_id = user.id

    # Accept the connection
    await websocket.accept()
    await add_connection(user_id, websocket)

    try:
        # Send initial connection confirmation
        await websocket.send_json(
            {
                "type": "connected",
                "data": {"user_id": user_id[:8] + "..."},
            }
        )

        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages with timeout for ping
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,  # Send ping every 30s
                )

                # Handle client messages
                try:
                    data = json.loads(message)
                    if data.get("type") == "pong":
                        continue  # Acknowledge pong
                except json.JSONDecodeError:
                    pass

            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await remove_connection(user_id, websocket)


# Helper function to be called from sync_routes when memories are pushed
async def broadcast_memory_update(user_id: str, count: int) -> None:
    """Broadcast memory update to connected clients.

    Called from sync_routes.py after memories are pushed.
    """
    await notify_user(user_id, "memory_created", {"count": count})
