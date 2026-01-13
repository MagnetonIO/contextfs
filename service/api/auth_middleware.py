"""Postgres-based authentication middleware for ContextFS sync service.

Uses SQLAlchemy async session for user/API key validation.
"""

from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from contextfs.auth import hash_api_key
from contextfs.auth.api_keys import APIKey, User
from service.db.models import APIKeyModel, UserModel
from service.db.session import get_session_dependency

# Header for API key authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def validate_api_key(
    api_key: str,
    session: AsyncSession,
) -> tuple[User, APIKey] | None:
    """Validate an API key and return user info.

    Args:
        api_key: The full API key (ctxfs_...)
        session: Database session

    Returns:
        Tuple of (User, APIKey) if valid, None otherwise
    """
    if not api_key or not api_key.startswith("ctxfs_"):
        return None

    key_hash = hash_api_key(api_key)

    # Find the API key
    result = await session.execute(
        select(APIKeyModel).where(
            APIKeyModel.key_hash == key_hash,
            APIKeyModel.is_active.is_(True),
        )
    )
    key_model = result.scalar_one_or_none()

    if not key_model:
        return None

    # Get the user
    user_result = await session.execute(select(UserModel).where(UserModel.id == key_model.user_id))
    user_model = user_result.scalar_one_or_none()

    if not user_model:
        return None

    # Update last_used_at
    await session.execute(
        update(APIKeyModel)
        .where(APIKeyModel.id == key_model.id)
        .values(last_used_at=datetime.now(timezone.utc))
    )
    await session.commit()

    # Convert to auth types
    user = User(
        id=user_model.id,
        email=user_model.email,
        name=user_model.name,
        provider=user_model.provider,
        provider_id=user_model.provider_id,
        created_at=user_model.created_at,
    )

    api_key_obj = APIKey(
        id=key_model.id,
        user_id=key_model.user_id,
        name=key_model.name,
        key_prefix=key_model.key_prefix,
        encryption_salt=key_model.encryption_salt,
        is_active=key_model.is_active,
        created_at=key_model.created_at,
        last_used_at=key_model.last_used_at,
    )

    return user, api_key_obj


async def get_current_user(
    api_key: str | None = Depends(API_KEY_HEADER),
    session: AsyncSession = Depends(get_session_dependency),
) -> tuple[User, APIKey] | None:
    """FastAPI dependency to get the current authenticated user.

    This is an optional dependency - returns None if not authenticated.
    """
    if not api_key:
        return None

    return await validate_api_key(api_key, session)


async def require_auth(
    auth: tuple[User, APIKey] | None = Depends(get_current_user),
) -> tuple[User, APIKey]:
    """FastAPI dependency that requires authentication."""
    if auth is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return auth
