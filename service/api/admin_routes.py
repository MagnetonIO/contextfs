"""Admin API routes for ContextFS.

Provides admin-only endpoints for managing users, viewing usage stats,
and other administrative tasks.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from contextfs.auth.api_keys import APIKey, User
from service.api.auth_middleware import require_auth
from service.db.models import (
    APIKeyModel,
    Device,
    SubscriptionModel,
    SyncedMemoryModel,
    TeamMemberModel,
    UserModel,
)
from service.db.session import get_session_dependency

router = APIRouter(prefix="/api/admin", tags=["admin"])


# =============================================================================
# Pydantic Models
# =============================================================================


class UserListItem(BaseModel):
    """User summary for list view."""

    id: str
    email: str
    name: str | None
    provider: str
    tier: str
    device_count: int
    memory_count: int
    created_at: str
    last_login: str | None


class UserListResponse(BaseModel):
    """List of users."""

    users: list[UserListItem]
    total: int


class UserDetailResponse(BaseModel):
    """Detailed user info for admin view."""

    id: str
    email: str
    name: str | None
    provider: str
    created_at: str
    last_login: str | None
    subscription: dict
    usage: dict
    devices: list[dict]
    api_keys: list[dict]


class AdminStatsResponse(BaseModel):
    """Platform-wide statistics."""

    total_users: int
    total_memories: int
    total_devices: int
    users_by_tier: dict
    users_by_provider: dict


# =============================================================================
# Helpers
# =============================================================================


async def _is_system_admin(session: AsyncSession, user_id: str) -> bool:
    """Check if user is a system admin via database flag."""
    result = await session.execute(select(UserModel).where(UserModel.id == user_id))
    db_user = result.scalar_one_or_none()
    return db_user is not None and getattr(db_user, "is_admin", False)


async def _is_team_admin(session: AsyncSession, user_id: str) -> bool:
    """Check if user is an admin/owner of any team."""
    result = await session.execute(
        select(TeamMemberModel).where(
            TeamMemberModel.user_id == user_id,
            TeamMemberModel.role.in_(["owner", "admin"]),
        )
    )
    return result.scalar_one_or_none() is not None


async def _get_user_team_ids(session: AsyncSession, user_id: str) -> list[str]:
    """Get all team IDs where user is admin/owner."""
    result = await session.execute(
        select(TeamMemberModel.team_id).where(
            TeamMemberModel.user_id == user_id,
            TeamMemberModel.role.in_(["owner", "admin"]),
        )
    )
    return [row[0] for row in result.all()]


async def _get_team_member_user_ids(session: AsyncSession, team_ids: list[str]) -> set[str]:
    """Get all user IDs that belong to the given teams."""
    if not team_ids:
        return set()
    result = await session.execute(
        select(TeamMemberModel.user_id).where(TeamMemberModel.team_id.in_(team_ids))
    )
    return {row[0] for row in result.all()}


async def _require_admin(auth: tuple[User, APIKey], session: AsyncSession) -> User:
    """Require system admin privileges."""
    user, _ = auth
    if not await _is_system_admin(session, user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


async def _require_admin_or_team_admin(
    auth: tuple[User, APIKey], session: AsyncSession
) -> tuple[User, bool, list[str]]:
    """Require system admin or team admin privileges.

    Returns (user, is_system_admin, team_ids_if_team_admin)
    """
    user, _ = auth
    if await _is_system_admin(session, user.id):
        return user, True, []

    team_ids = await _get_user_team_ids(session, user.id)
    if team_ids:
        return user, False, team_ids

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Admin access required",
    )


def _format_dt(dt) -> str:
    """Format datetime for response."""
    if dt is None:
        return ""
    if isinstance(dt, str):
        return dt
    return dt.isoformat()


def _generate_password(length: int = 12) -> str:
    """Generate a secure random password.

    Password will contain:
    - At least 1 uppercase letter
    - At least 1 lowercase letter
    - At least 1 digit
    - At least 1 special character
    """
    import secrets
    import string

    # Ensure minimum requirements
    uppercase = secrets.choice(string.ascii_uppercase)
    lowercase = secrets.choice(string.ascii_lowercase)
    digit = secrets.choice(string.digits)
    special = secrets.choice("!@#$%^&*")

    # Fill remaining with random characters
    remaining_length = length - 4
    all_chars = string.ascii_letters + string.digits + "!@#$%^&*"
    remaining = "".join(secrets.choice(all_chars) for _ in range(remaining_length))

    # Combine and shuffle
    password_list = list(uppercase + lowercase + digit + special + remaining)
    secrets.SystemRandom().shuffle(password_list)

    return "".join(password_list)


def _validate_password(password: str) -> tuple[bool, str]:
    """Validate password meets requirements.

    Requirements:
    - Minimum 8 characters
    - At least 1 uppercase letter
    - At least 1 lowercase letter
    - At least 1 digit

    Returns:
        (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters"

    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"

    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"

    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit"

    return True, ""


# =============================================================================
# Admin Routes
# =============================================================================


@router.get("/users", response_model=UserListResponse)
async def list_users(
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
    tier: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """List users. System admins see all users, team admins see only their team members."""
    current_user, is_system_admin, team_ids = await _require_admin_or_team_admin(auth, session)

    # For team admins, get the user IDs in their teams
    allowed_user_ids: set[str] | None = None
    if not is_system_admin:
        allowed_user_ids = await _get_team_member_user_ids(session, team_ids)
        if not allowed_user_ids:
            return UserListResponse(users=[], total=0)

    # Base query
    query = select(UserModel)
    count_query = select(func.count(UserModel.id))

    # Filter by allowed users (for team admins)
    if allowed_user_ids is not None:
        query = query.where(UserModel.id.in_(allowed_user_ids))
        count_query = count_query.where(UserModel.id.in_(allowed_user_ids))

    # Filter by tier if specified
    if tier:
        query = query.join(SubscriptionModel, UserModel.id == SubscriptionModel.user_id).where(
            SubscriptionModel.tier == tier
        )
        count_query = count_query.join(
            SubscriptionModel, UserModel.id == SubscriptionModel.user_id
        ).where(SubscriptionModel.tier == tier)

    # Get total count
    total_result = await session.execute(count_query)
    total = total_result.scalar() or 0

    # Get users with pagination
    query = query.order_by(UserModel.created_at.desc()).offset(offset).limit(limit)
    result = await session.execute(query)
    users = result.scalars().all()

    # Get subscription and usage for each user
    user_list = []
    for u in users:
        # Get subscription
        sub_result = await session.execute(
            select(SubscriptionModel).where(SubscriptionModel.user_id == u.id)
        )
        sub = sub_result.scalar_one_or_none()

        # Get device count
        device_result = await session.execute(
            select(func.count(Device.device_id)).where(Device.user_id == u.id)
        )
        device_count = device_result.scalar() or 0

        # Get memory count
        memory_result = await session.execute(
            select(func.count(SyncedMemoryModel.id)).where(
                SyncedMemoryModel.user_id == u.id,
                SyncedMemoryModel.deleted_at.is_(None),
            )
        )
        memory_count = memory_result.scalar() or 0

        user_list.append(
            UserListItem(
                id=u.id,
                email=u.email,
                name=u.name,
                provider=u.provider,
                tier=sub.tier if sub else "free",
                device_count=device_count,
                memory_count=memory_count,
                created_at=_format_dt(u.created_at),
                last_login=_format_dt(u.last_login),
            )
        )

    return UserListResponse(users=user_list, total=total)


@router.get("/users/{user_id}", response_model=UserDetailResponse)
async def get_user_detail(
    user_id: str,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Get detailed user info (admin only)."""
    await _require_admin(auth, session)

    # Get user
    result = await session.execute(select(UserModel).where(UserModel.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get subscription
    sub_result = await session.execute(
        select(SubscriptionModel).where(SubscriptionModel.user_id == user_id)
    )
    sub = sub_result.scalar_one_or_none()

    # Get devices
    device_result = await session.execute(select(Device).where(Device.user_id == user_id))
    devices = device_result.scalars().all()

    # Get API keys (excluding session keys)
    key_result = await session.execute(
        select(APIKeyModel).where(
            APIKeyModel.user_id == user_id,
            APIKeyModel.name.notin_(["Login Session", "OAuth Session"]),
        )
    )
    api_keys = key_result.scalars().all()

    # Get memory count
    memory_result = await session.execute(
        select(func.count(SyncedMemoryModel.id)).where(
            SyncedMemoryModel.user_id == user_id,
            SyncedMemoryModel.deleted_at.is_(None),
        )
    )
    memory_count = memory_result.scalar() or 0

    return UserDetailResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        provider=user.provider,
        created_at=_format_dt(user.created_at),
        last_login=_format_dt(user.last_login),
        subscription={
            "tier": sub.tier if sub else "free",
            "status": sub.status if sub else "active",
            "device_limit": sub.device_limit if sub else 2,
            "memory_limit": sub.memory_limit if sub else 5000,
            "stripe_customer_id": sub.stripe_customer_id if sub else None,
            "current_period_end": _format_dt(sub.current_period_end) if sub else None,
        },
        usage={
            "device_count": len(devices),
            "memory_count": memory_count,
        },
        devices=[
            {
                "device_id": d.device_id,
                "device_name": d.device_name,
                "platform": d.platform,
                "last_sync_at": _format_dt(d.last_sync_at),
                "registered_at": _format_dt(d.registered_at),
            }
            for d in devices
        ],
        api_keys=[
            {
                "id": k.id,
                "name": k.name,
                "key_prefix": k.key_prefix,
                "is_active": k.is_active,
                "created_at": _format_dt(k.created_at),
                "last_used_at": _format_dt(k.last_used_at),
            }
            for k in api_keys
        ],
    )


@router.get("/stats", response_model=AdminStatsResponse)
async def get_admin_stats(
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Get platform-wide statistics (admin only)."""
    await _require_admin(auth, session)

    # Total users
    user_result = await session.execute(select(func.count(UserModel.id)))
    total_users = user_result.scalar() or 0

    # Total memories
    memory_result = await session.execute(
        select(func.count(SyncedMemoryModel.id)).where(SyncedMemoryModel.deleted_at.is_(None))
    )
    total_memories = memory_result.scalar() or 0

    # Total devices
    device_result = await session.execute(select(func.count(Device.device_id)))
    total_devices = device_result.scalar() or 0

    # Users by tier
    tier_result = await session.execute(
        select(SubscriptionModel.tier, func.count(SubscriptionModel.user_id)).group_by(
            SubscriptionModel.tier
        )
    )
    users_by_tier = {row[0]: row[1] for row in tier_result.all()}

    # Users by provider
    provider_result = await session.execute(
        select(UserModel.provider, func.count(UserModel.id)).group_by(UserModel.provider)
    )
    users_by_provider = {row[0]: row[1] for row in provider_result.all()}

    return AdminStatsResponse(
        total_users=total_users,
        total_memories=total_memories,
        total_devices=total_devices,
        users_by_tier=users_by_tier,
        users_by_provider=users_by_provider,
    )


@router.post("/users/{user_id}/upgrade")
async def upgrade_user(
    user_id: str,
    tier: str,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Manually upgrade a user's tier (admin only)."""
    await _require_admin(auth, session)

    from service.api.billing_routes import TIER_LIMITS

    if tier not in TIER_LIMITS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tier. Valid tiers: {list(TIER_LIMITS.keys())}",
        )

    # Get or create subscription
    result = await session.execute(
        select(SubscriptionModel).where(SubscriptionModel.user_id == user_id)
    )
    sub = result.scalar_one_or_none()

    limits = TIER_LIMITS[tier]

    if sub:
        sub.tier = tier
        sub.device_limit = limits["device_limit"]
        sub.memory_limit = limits["memory_limit"]
        sub.updated_at = datetime.now(timezone.utc)
    else:
        from uuid import uuid4

        sub = SubscriptionModel(
            id=str(uuid4()),
            user_id=user_id,
            tier=tier,
            device_limit=limits["device_limit"],
            memory_limit=limits["memory_limit"],
            status="active",
        )
        session.add(sub)

    await session.commit()

    return {"status": "success", "tier": tier}


@router.post("/users/{user_id}/make-admin")
async def make_user_admin(
    user_id: str,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Make a user an admin (admin only)."""
    await _require_admin(auth, session)

    # Get user
    result = await session.execute(select(UserModel).where(UserModel.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Update subscription to admin tier
    sub_result = await session.execute(
        select(SubscriptionModel).where(SubscriptionModel.user_id == user_id)
    )
    sub = sub_result.scalar_one_or_none()

    if sub:
        sub.tier = "admin"
        sub.device_limit = -1  # Unlimited
        sub.memory_limit = -1  # Unlimited
        sub.updated_at = datetime.now(timezone.utc)
    else:
        from uuid import uuid4

        sub = SubscriptionModel(
            id=str(uuid4()),
            user_id=user_id,
            tier="admin",
            device_limit=-1,
            memory_limit=-1,
            status="active",
        )
        session.add(sub)

    await session.commit()

    return {"status": "success", "message": f"User {user.email} is now an admin"}


class CreateUserRequest(BaseModel):
    """Request to create a new user."""

    email: str
    name: str | None = None
    password: str | None = None
    tier: str = "free"
    send_welcome_email: bool = True


@router.post("/users")
async def create_user(
    request: CreateUserRequest,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Create a new user (admin only)."""
    import hashlib
    from uuid import uuid4

    await _require_admin(auth, session)

    # Check if email already exists
    result = await session.execute(select(UserModel).where(UserModel.email == request.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already exists")

    # Handle password - validate if provided, generate if not
    password = request.password
    if password:
        # Validate provided password
        is_valid, error_msg = _validate_password(password)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
    else:
        # Generate a secure password (user will set their own via email link)
        password = _generate_password(12)

    password_hash = hashlib.sha256(password.encode()).hexdigest()

    # Create user
    user_id = str(uuid4())
    new_user = UserModel(
        id=user_id,
        email=request.email,
        name=request.name,
        provider="admin_created",
        provider_id=user_id,
        password_hash=password_hash,
        email_verified=True,
    )
    session.add(new_user)

    # Flush user first to satisfy foreign key constraint
    await session.flush()

    # Create subscription
    from service.api.billing_routes import TIER_LIMITS

    limits = TIER_LIMITS.get(request.tier, TIER_LIMITS["free"])
    subscription = SubscriptionModel(
        id=str(uuid4()),
        user_id=user_id,
        tier=request.tier,
        device_limit=limits["device_limit"],
        memory_limit=limits["memory_limit"],
        status="active",
    )
    session.add(subscription)

    # Create password reset token and send welcome email
    email_sent = False
    if request.send_welcome_email:
        try:
            from service.email_service import (
                create_password_reset_token,
                send_welcome_email,
            )

            reset_token = await create_password_reset_token(session, user_id)
            await session.commit()

            email_sent = await send_welcome_email(
                to_email=request.email,
                user_name=request.name,
                reset_token=reset_token,
            )
        except Exception as e:
            # Log but don't fail user creation if email fails
            print(f"Failed to send welcome email: {e}")
            await session.commit()
    else:
        await session.commit()

    return {
        "status": "success",
        "user_id": user_id,
        "email": request.email,
        "tier": request.tier,
        "welcome_email_sent": email_sent,
    }


@router.post("/users/{user_id}/deactivate")
async def deactivate_user(
    user_id: str,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Deactivate a user (admin only). Revokes all API keys."""
    from sqlalchemy import update

    await _require_admin(auth, session)
    admin_user, _ = auth

    # Prevent self-deactivation
    if user_id == admin_user.id:
        raise HTTPException(status_code=400, detail="Cannot deactivate yourself")

    # Get user
    result = await session.execute(select(UserModel).where(UserModel.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Revoke all API keys
    await session.execute(
        update(APIKeyModel).where(APIKeyModel.user_id == user_id).values(is_active=False)
    )

    # Update subscription status
    await session.execute(
        update(SubscriptionModel)
        .where(SubscriptionModel.user_id == user_id)
        .values(status="deactivated")
    )

    await session.commit()

    return {"status": "success", "message": f"User {user.email} has been deactivated"}


@router.post("/users/{user_id}/reactivate")
async def reactivate_user(
    user_id: str,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Reactivate a deactivated user (admin only)."""
    from sqlalchemy import update

    await _require_admin(auth, session)

    # Get user
    result = await session.execute(select(UserModel).where(UserModel.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Reactivate subscription
    await session.execute(
        update(SubscriptionModel)
        .where(SubscriptionModel.user_id == user_id)
        .values(status="active")
    )

    await session.commit()

    return {"status": "success", "message": f"User {user.email} has been reactivated"}


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Permanently delete a user and all their data (admin only)."""
    from sqlalchemy import delete

    await _require_admin(auth, session)
    admin_user, _ = auth

    # Prevent self-deletion
    if user_id == admin_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    # Get user
    result = await session.execute(select(UserModel).where(UserModel.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    email = user.email

    # Delete in order to respect foreign keys:
    # 1. Delete API keys
    await session.execute(delete(APIKeyModel).where(APIKeyModel.user_id == user_id))

    # 2. Delete devices
    await session.execute(delete(Device).where(Device.user_id == user_id))

    # 3. Delete memories (soft delete - set deleted_at)
    from sqlalchemy import update

    await session.execute(
        update(SyncedMemoryModel)
        .where(SyncedMemoryModel.user_id == user_id)
        .values(deleted_at=datetime.now(timezone.utc))
    )

    # 4. Delete subscription
    await session.execute(delete(SubscriptionModel).where(SubscriptionModel.user_id == user_id))

    # 5. Delete password reset tokens if they exist
    try:
        from service.db.models import PasswordResetToken

        await session.execute(
            delete(PasswordResetToken).where(PasswordResetToken.user_id == user_id)
        )
    except Exception:
        pass  # Table might not exist yet

    # 6. Delete user
    await session.execute(delete(UserModel).where(UserModel.id == user_id))

    await session.commit()

    return {"status": "success", "message": f"User {email} has been permanently deleted"}


class AdminResetPasswordRequest(BaseModel):
    """Request to reset user password by admin."""

    password: str | None = None  # If None, sends reset email instead
    send_email: bool = False  # If True, sends reset email


@router.post("/users/{user_id}/reset-password")
async def admin_reset_password(
    user_id: str,
    request: AdminResetPasswordRequest,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Reset a user's password (admin only).

    Either sets a new password directly, or sends a password reset email.
    """
    await _require_admin(auth, session)

    # Get user
    result = await session.execute(select(UserModel).where(UserModel.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if user is email-based (has password)
    if user.provider not in ("email", "admin_created"):
        raise HTTPException(
            status_code=400, detail=f"Cannot reset password for {user.provider} OAuth user"
        )

    if request.send_email:
        # Send password reset email
        from service.email_service import create_password_reset_token, send_password_reset_email

        reset_token = await create_password_reset_token(session, user_id)
        await session.commit()

        email_sent = await send_password_reset_email(
            to_email=user.email,
            user_name=user.name,
            reset_token=reset_token,
        )

        if not email_sent:
            raise HTTPException(status_code=500, detail="Failed to send reset email")

        return {"status": "success", "message": f"Password reset email sent to {user.email}"}

    elif request.password:
        # Validate and set password directly
        is_valid, error_msg = _validate_password(request.password)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        # Hash and update password
        import hashlib

        password_hash = hashlib.sha256(request.password.encode()).hexdigest()

        from sqlalchemy import update

        await session.execute(
            update(UserModel).where(UserModel.id == user_id).values(password_hash=password_hash)
        )
        await session.commit()

        return {"status": "success", "message": f"Password updated for {user.email}"}

    else:
        raise HTTPException(
            status_code=400, detail="Either password or send_email must be provided"
        )
