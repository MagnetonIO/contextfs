"""Sync service FastAPI application.

Main entry point for the ContextFS sync server.
Run with: uvicorn service.api.main:app --host 0.0.0.0 --port 8766
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from contextfs.auth import init_auth_middleware
from service.api.auth_routes import router as auth_router
from service.api.billing_routes import router as billing_router
from service.api.sync_routes import router as sync_router
from service.db.session import close_db, create_tables, init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    logger.info("Starting ContextFS Sync Service...")

    # Initialize database
    await init_db()
    await create_tables()
    logger.info("Database initialized")

    # Initialize auth middleware
    db_path = os.environ.get("CONTEXTFS_DB_PATH", "contextfs.db")
    init_auth_middleware(db_path)
    logger.info("Auth middleware initialized")

    yield

    # Cleanup
    logger.info("Shutting down ContextFS Sync Service...")
    await close_db()


app = FastAPI(
    title="ContextFS Sync Service",
    description="Multi-device memory synchronization service with vector clock conflict resolution",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(sync_router)
app.include_router(auth_router)
app.include_router(billing_router)


# =============================================================================
# Health Check
# =============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "contextfs-sync",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "ContextFS Sync Service",
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "sync": {
                "register": "POST /api/sync/register",
                "push": "POST /api/sync/push",
                "pull": "POST /api/sync/pull",
                "status": "POST /api/sync/status",
            },
            "auth": {
                "me": "GET /api/auth/me",
                "api_keys": "GET /api/auth/api-keys",
                "create_key": "POST /api/auth/api-keys",
                "oauth_init": "POST /api/auth/oauth/init",
                "oauth_callback": "POST /api/auth/oauth/callback",
            },
            "billing": {
                "checkout": "POST /api/billing/checkout",
                "portal": "POST /api/billing/portal",
                "subscription": "GET /api/billing/subscription",
                "usage": "GET /api/billing/usage",
                "webhook": "POST /api/billing/webhook",
            },
        },
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("CONTEXTFS_SYNC_PORT", "8766"))
    host = os.environ.get("CONTEXTFS_SYNC_HOST", "0.0.0.0")

    uvicorn.run(
        "service.api.main:app",
        host=host,
        port=port,
        reload=os.environ.get("CONTEXTFS_DEV", "").lower() == "true",
    )
