"""Sync service FastAPI application.

Main entry point for the ContextFS sync server.
Run with: uvicorn service.api.main:app --host 0.0.0.0 --port 8766

All data stored in Postgres (users, api_keys, subscriptions, sync data).
Admin user created on startup if ADMIN_EMAIL and ADMIN_PASSWORD env vars are set.
"""

import hashlib
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from contextfs.auth import generate_api_key, hash_api_key
from service.api.admin_routes import router as admin_router
from service.api.auth_routes import router as auth_router
from service.api.billing_routes import router as billing_router
from service.api.devices_routes import router as devices_router
from service.api.memories_routes import router as memories_router
from service.api.sync_routes import router as sync_router
from service.api.team_routes import router as team_router
from service.db.models import APIKeyModel, SubscriptionModel, UsageModel, UserModel
from service.db.session import close_db, get_session, init_db
from service.migrations.runner import run_migrations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ADMIN_USER_ID = "admin-00000000-0000-0000-0000-000000000001"


def _hash_password(password: str) -> str:
    """Hash password with SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


async def ensure_admin_user() -> str | None:
    """Create admin user from env vars if not exists. Returns API key if created."""
    admin_email = os.environ.get("ADMIN_EMAIL")
    admin_password = os.environ.get("ADMIN_PASSWORD")

    if not admin_email or not admin_password:
        logger.info("Admin not configured (set ADMIN_EMAIL + ADMIN_PASSWORD)")
        return None

    async with get_session() as session:
        # Check if admin exists with active key
        result = await session.execute(
            select(UserModel, APIKeyModel)
            .outerjoin(
                APIKeyModel,
                (UserModel.id == APIKeyModel.user_id) & (APIKeyModel.is_active.is_(True)),
            )
            .where(UserModel.id == ADMIN_USER_ID)
        )
        row = result.first()

        if row and row[1]:  # User exists with active key
            # Ensure admin flag is set
            user = row[0]
            if user and not user.is_admin:
                user.is_admin = True
                await session.commit()
                logger.info(f"Updated admin flag for: {admin_email}")
            logger.info(f"Admin exists with key prefix: {row[1].key_prefix}...")
            return None

        user = row[0] if row else None

        # Create user if not exists
        if not user:
            password_hash = _hash_password(admin_password)
            user = UserModel(
                id=ADMIN_USER_ID,
                email=admin_email,
                name="Admin",
                provider="system",
                password_hash=password_hash,
                email_verified=True,
                is_admin=True,
            )
            session.add(user)
            # Commit user first to satisfy foreign key constraints
            await session.commit()

            # Initialize subscription
            subscription = SubscriptionModel(
                id=str(uuid4()),
                user_id=ADMIN_USER_ID,
                tier="team",  # Admin gets team tier
                device_limit=-1,  # Unlimited
                memory_limit=-1,  # Unlimited
                status="active",
            )
            session.add(subscription)

            # Initialize usage
            usage = UsageModel(
                user_id=ADMIN_USER_ID,
                device_count=0,
                memory_count=0,
            )
            session.add(usage)
            await session.commit()

            logger.info(f"Created admin: {admin_email}")

        # Create API key
        env_key = os.environ.get("ADMIN_API_KEY")
        if env_key and env_key.startswith("ctxfs_"):
            full_key = env_key
            key_prefix = env_key[6:14]
        else:
            full_key, key_prefix = generate_api_key()

        api_key = APIKeyModel(
            id=str(uuid4()),
            user_id=ADMIN_USER_ID,
            name="Admin Key",
            key_hash=hash_api_key(full_key),
            key_prefix=key_prefix,
            is_active=True,
        )
        session.add(api_key)
        await session.commit()

        return full_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    logger.info("Starting ContextFS Sync Service...")

    # Run Alembic migrations (creates tables and applies schema changes)
    logger.info("Running Alembic migrations...")
    try:
        migrated = run_migrations()
        if migrated:
            logger.info("Alembic migrations applied successfully")
        else:
            logger.info("Database already at latest revision")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

    # Initialize async database engine (after migrations complete)
    await init_db()
    logger.info("Database initialized (Postgres)")

    # Create admin if configured
    admin_key = await ensure_admin_user()
    if admin_key:
        logger.info("=" * 50)
        logger.info(f"ADMIN API KEY: {admin_key}")
        logger.info("=" * 50)

    logger.info("Auth initialized")

    yield

    logger.info("Shutting down...")
    await close_db()


app = FastAPI(
    title="ContextFS Sync Service",
    description="Multi-device memory synchronization service with vector clock conflict resolution",
    version="0.2.0",
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
app.include_router(devices_router)
app.include_router(memories_router)
app.include_router(admin_router)
app.include_router(team_router)


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
                "oauth_token": "POST /api/auth/oauth/token",
            },
            "billing": {
                "checkout": "POST /api/billing/checkout",
                "portal": "POST /api/billing/portal",
                "subscription": "GET /api/billing/subscription",
                "usage": "GET /api/billing/usage",
                "webhook": "POST /api/billing/webhook",
            },
            "devices": {
                "list": "GET /api/devices",
                "remove": "DELETE /api/devices/{device_id}",
            },
            "teams": {
                "create": "POST /api/teams",
                "list": "GET /api/teams",
                "get": "GET /api/teams/{team_id}",
                "members": "GET /api/teams/{team_id}/members",
                "invite": "POST /api/teams/{team_id}/invite",
                "accept": "POST /api/teams/invitations/accept",
                "update_role": "PUT /api/teams/{team_id}/members/{user_id}/role",
                "remove_member": "DELETE /api/teams/{team_id}/members/{user_id}",
                "delete": "DELETE /api/teams/{team_id}",
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
