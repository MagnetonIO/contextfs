"""Authentication API routes for ContextFS.

Handles user registration, API key management, and OAuth callbacks.
All data stored in Postgres.
"""

import hashlib
import os
import secrets
from datetime import datetime, timezone
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from contextfs.auth import generate_api_key, hash_api_key
from contextfs.auth.api_keys import APIKey, User
from contextfs.encryption import derive_encryption_key_base64
from service.api.auth_middleware import require_auth
from service.db.models import APIKeyModel, SubscriptionModel, UsageModel, UserModel
from service.db.session import get_session_dependency

router = APIRouter(prefix="/api/auth", tags=["auth"])

# API server URL - used in config snippets
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.contextfs.ai")


# =============================================================================
# Pydantic Models
# =============================================================================


class UserResponse(BaseModel):
    """User profile response."""

    id: str
    email: str
    name: str | None
    provider: str
    is_admin: bool = False


class CreateAPIKeyRequest(BaseModel):
    """Request to create a new API key."""

    name: str
    with_encryption: bool = True


class CreateAPIKeyResponse(BaseModel):
    """Response with new API key (shown only once!)."""

    id: str
    name: str
    api_key: str
    encryption_key: str | None
    key_prefix: str
    config_snippet: str


class APIKeyListItem(BaseModel):
    """API key list item (no secret values)."""

    id: str
    name: str
    key_prefix: str
    is_active: bool
    created_at: str
    last_used_at: str | None


class APIKeyListResponse(BaseModel):
    """List of API keys."""

    keys: list[APIKeyListItem]


class RevokeKeyRequest(BaseModel):
    """Request to revoke an API key."""

    key_id: str


class OAuthInitRequest(BaseModel):
    """Request to initiate OAuth flow."""

    provider: str
    redirect_uri: str


class OAuthInitResponse(BaseModel):
    """Response with OAuth authorization URL."""

    auth_url: str
    state: str


class LoginRequest(BaseModel):
    """Request for email/password login."""

    email: str
    password: str
    session_type: str = "Web Session"  # "Web Session" for frontend, "CLI Session" for CLI


class LoginUserResponse(BaseModel):
    """User info in login response."""

    id: str
    email: str
    name: str | None
    emailVerified: bool = True
    createdAt: str | None = None
    is_admin: bool = False


class LoginResponse(BaseModel):
    """Response with API key after successful login."""

    user: LoginUserResponse
    apiKey: str
    encryptionKey: str | None = None


class OAuthCallbackRequest(BaseModel):
    """OAuth callback data."""

    provider: str
    code: str
    state: str


class OAuthTokenRequest(BaseModel):
    """OAuth token exchange - for NextAuth which already has the access_token."""

    provider: str
    access_token: str


class OAuthCallbackResponse(BaseModel):
    """Response after OAuth callback."""

    user: UserResponse
    api_key: str
    encryption_key: str | None


# =============================================================================
# Helper Functions
# =============================================================================


def _hash_password(password: str) -> str:
    """Hash password with SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


async def _create_api_key(
    session: AsyncSession,
    user_id: str,
    name: str,
    with_encryption: bool = True,
) -> tuple[str, str | None]:
    """Create a new API key for a user.

    Returns: (full_api_key, encryption_salt)
    """
    full_key, key_prefix = generate_api_key()
    key_hash = hash_api_key(full_key)
    key_id = str(uuid4())

    encryption_salt = None
    if with_encryption:
        encryption_salt = secrets.token_urlsafe(32)

    key_model = APIKeyModel(
        id=key_id,
        user_id=user_id,
        name=name,
        key_hash=key_hash,
        key_prefix=key_prefix,
        encryption_salt=encryption_salt,
        is_active=True,
    )
    session.add(key_model)
    await session.commit()

    return full_key, encryption_salt


async def _get_or_create_user(
    session: AsyncSession,
    email: str,
    name: str | None,
    provider: str,
    provider_id: str,
) -> tuple[str, bool]:
    """Get or create user by email.

    Returns: (user_id, is_new_user)
    """
    result = await session.execute(select(UserModel).where(UserModel.email == email))
    user = result.scalar_one_or_none()

    if user:
        # Update existing user
        user.name = name
        user.provider_id = provider_id
        user.last_login = datetime.now(timezone.utc)
        await session.commit()
        return user.id, False

    # Create new user
    user_id = str(uuid4())
    new_user = UserModel(
        id=user_id,
        email=email,
        name=name,
        provider=provider,
        provider_id=provider_id,
        email_verified=True,
    )
    session.add(new_user)
    # Commit user first to satisfy foreign key constraints
    await session.commit()

    # Initialize subscription
    subscription = SubscriptionModel(
        id=str(uuid4()),
        user_id=user_id,
        tier="free",
        device_limit=2,
        memory_limit=5000,
        status="active",
    )
    session.add(subscription)

    # Initialize usage
    usage = UsageModel(
        user_id=user_id,
        device_count=0,
        memory_count=0,
    )
    session.add(usage)

    await session.commit()
    return user_id, True


# =============================================================================
# Auth Routes
# =============================================================================


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    session: AsyncSession = Depends(get_session_dependency),
):
    """Login with email and password, returns an API key."""
    password_hash = _hash_password(request.password)

    result = await session.execute(
        select(UserModel).where(
            UserModel.email == request.email,
            UserModel.password_hash == password_hash,
        )
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Delete old session keys of the same type
    await session.execute(
        delete(APIKeyModel).where(
            APIKeyModel.user_id == user.id,
            APIKeyModel.name == request.session_type,
        )
    )

    # Create new session key (E2EE controlled by env var)
    e2ee_enabled = os.environ.get("CONTEXTFS_E2EE_ENABLED", "false").lower() == "true"
    full_key, encryption_salt = await _create_api_key(
        session, user.id, request.session_type, with_encryption=e2ee_enabled
    )

    # Derive encryption key for client if E2EE enabled
    encryption_key = None
    if encryption_salt:
        encryption_key = derive_encryption_key_base64(full_key, encryption_salt)

    return LoginResponse(
        user=LoginUserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            is_admin=getattr(user, "is_admin", False),
        ),
        apiKey=full_key,
        encryptionKey=encryption_key,
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Get current authenticated user's profile."""
    user, _ = auth

    # Fetch is_admin flag from database
    result = await session.execute(select(UserModel).where(UserModel.id == user.id))
    db_user = result.scalar_one_or_none()
    is_admin = db_user.is_admin if db_user and hasattr(db_user, "is_admin") else False

    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        provider=user.provider,
        is_admin=is_admin,
    )


@router.get("/encryption-salt")
async def get_encryption_salt(
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Get encryption salt for the current API key.

    Used by sync client to derive encryption key for E2EE.
    The encryption key = derive_key(api_key + salt).
    """
    _, api_key = auth

    # Fetch the full API key record to get the salt
    result = await session.execute(select(APIKeyModel).where(APIKeyModel.id == api_key.id))
    key_record = result.scalar_one_or_none()

    if not key_record or not key_record.encryption_salt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No encryption salt configured for this API key",
        )

    return {"salt": key_record.encryption_salt}


@router.post("/api-keys", response_model=CreateAPIKeyResponse)
async def create_api_key_endpoint(
    request: CreateAPIKeyRequest,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Create a new API key."""
    user, _ = auth

    full_key, encryption_salt = await _create_api_key(
        session, user.id, request.name, request.with_encryption
    )

    encryption_key = None
    if encryption_salt:
        encryption_key = derive_encryption_key_base64(full_key, encryption_salt)

    key_prefix = full_key.split("_")[1][:8] if "_" in full_key else full_key[:8]

    config_lines = [
        "cloud:",
        "  enabled: true",
        f"  api_key: {full_key}",
    ]
    if encryption_key:
        config_lines.append(f"  encryption_key: {encryption_key}")
    config_lines.append(f"  server_url: {API_BASE_URL}")

    return CreateAPIKeyResponse(
        id=str(uuid4()),
        name=request.name,
        api_key=full_key,
        encryption_key=encryption_key,
        key_prefix=key_prefix,
        config_snippet="\n".join(config_lines),
    )


@router.get("/api-keys", response_model=APIKeyListResponse)
async def list_api_keys(
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """List all API keys for the current user."""
    user, _ = auth

    result = await session.execute(
        select(APIKeyModel)
        .where(
            APIKeyModel.user_id == user.id,
            APIKeyModel.name.notin_(["Login Session", "OAuth Session"]),
        )
        .order_by(APIKeyModel.created_at.desc())
    )
    keys = result.scalars().all()

    def format_datetime(dt) -> str:
        if dt is None:
            return ""
        if isinstance(dt, str):
            return dt
        return dt.isoformat()

    return APIKeyListResponse(
        keys=[
            APIKeyListItem(
                id=k.id,
                name=k.name,
                key_prefix=k.key_prefix,
                is_active=k.is_active,
                created_at=format_datetime(k.created_at),
                last_used_at=format_datetime(k.last_used_at) or None,
            )
            for k in keys
        ]
    )


@router.post("/api-keys/revoke")
async def revoke_api_key(
    request: RevokeKeyRequest,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Revoke an API key."""
    user, current_key = auth

    if request.key_id == current_key.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot revoke the API key currently in use",
        )

    result = await session.execute(
        update(APIKeyModel)
        .where(APIKeyModel.id == request.key_id, APIKeyModel.user_id == user.id)
        .values(is_active=False)
    )

    if result.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")

    await session.commit()
    return {"status": "revoked"}


@router.delete("/api-keys/{key_id}")
async def delete_api_key(
    key_id: str,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Permanently delete an API key."""
    user, current_key = auth

    if key_id == current_key.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete the API key currently in use",
        )

    result = await session.execute(
        delete(APIKeyModel).where(APIKeyModel.id == key_id, APIKeyModel.user_id == user.id)
    )

    if result.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")

    await session.commit()
    return {"status": "deleted"}


# =============================================================================
# OAuth Routes
# =============================================================================


@router.post("/oauth/init", response_model=OAuthInitResponse)
async def init_oauth(request: OAuthInitRequest):
    """Initialize OAuth flow."""
    import urllib.parse

    state = secrets.token_urlsafe(32)

    if request.provider == "google":
        client_id = os.environ.get("GOOGLE_CLIENT_ID")
        if not client_id:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Google OAuth not configured",
            )

        params = {
            "client_id": client_id,
            "redirect_uri": request.redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "access_type": "offline",
        }
        auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"

    elif request.provider == "github":
        client_id = os.environ.get("GITHUB_CLIENT_ID")
        if not client_id:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GitHub OAuth not configured",
            )

        params = {
            "client_id": client_id,
            "redirect_uri": request.redirect_uri,
            "scope": "user:email",
            "state": state,
        }
        auth_url = f"https://github.com/login/oauth/authorize?{urllib.parse.urlencode(params)}"

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown provider: {request.provider}",
        )

    return OAuthInitResponse(auth_url=auth_url, state=state)


@router.post("/oauth/callback", response_model=OAuthCallbackResponse)
async def oauth_callback(
    request: OAuthCallbackRequest,
    session: AsyncSession = Depends(get_session_dependency),
):
    """Handle OAuth callback - exchange code for tokens."""
    if request.provider == "google":
        client_id = os.environ.get("GOOGLE_CLIENT_ID")
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")

        async with httpx.AsyncClient() as client:
            token_resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": request.code,
                    "grant_type": "authorization_code",
                    "redirect_uri": os.environ.get("GOOGLE_REDIRECT_URI"),
                },
            )
            tokens = token_resp.json()

            userinfo_resp = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            )
            userinfo = userinfo_resp.json()

        email = userinfo["email"]
        name = userinfo.get("name")
        provider_id = userinfo["id"]

    elif request.provider == "github":
        client_id = os.environ.get("GITHUB_CLIENT_ID")
        client_secret = os.environ.get("GITHUB_CLIENT_SECRET")

        async with httpx.AsyncClient() as client:
            token_resp = await client.post(
                "https://github.com/login/oauth/access_token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": request.code,
                },
                headers={"Accept": "application/json"},
            )
            tokens = token_resp.json()

            user_resp = await client.get(
                "https://api.github.com/user",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            )
            userinfo = user_resp.json()

            emails_resp = await client.get(
                "https://api.github.com/user/emails",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            )
            emails = emails_resp.json()
            primary_email = next((e["email"] for e in emails if e["primary"]), None)

        email = primary_email or userinfo.get("email")
        name = userinfo.get("name") or userinfo.get("login")
        provider_id = str(userinfo["id"])

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown provider: {request.provider}",
        )

    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not retrieve email from OAuth provider",
        )

    user_id, _ = await _get_or_create_user(session, email, name, request.provider, provider_id)

    full_key, encryption_salt = await _create_api_key(
        session, user_id, "Default Key", with_encryption=True
    )

    encryption_key = None
    if encryption_salt:
        encryption_key = derive_encryption_key_base64(full_key, encryption_salt)

    # Fetch is_admin from database
    user_result = await session.execute(select(UserModel).where(UserModel.id == user_id))
    db_user = user_result.scalar_one_or_none()
    is_admin = getattr(db_user, "is_admin", False) if db_user else False

    return OAuthCallbackResponse(
        user=UserResponse(
            id=user_id, email=email, name=name, provider=request.provider, is_admin=is_admin
        ),
        api_key=full_key,
        encryption_key=encryption_key,
    )


@router.post("/oauth/token", response_model=OAuthCallbackResponse)
async def oauth_token_exchange(
    request: OAuthTokenRequest,
    session: AsyncSession = Depends(get_session_dependency),
):
    """Exchange OAuth access_token for ContextFS API key (for NextAuth)."""
    if request.provider == "google":
        async with httpx.AsyncClient() as client:
            userinfo_resp = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {request.access_token}"},
            )
            if userinfo_resp.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid Google access token",
                )
            userinfo = userinfo_resp.json()

        email = userinfo["email"]
        name = userinfo.get("name")
        provider_id = userinfo["id"]

    elif request.provider == "github":
        async with httpx.AsyncClient() as client:
            user_resp = await client.get(
                "https://api.github.com/user",
                headers={"Authorization": f"Bearer {request.access_token}"},
            )
            if user_resp.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid GitHub access token",
                )
            userinfo = user_resp.json()

            emails_resp = await client.get(
                "https://api.github.com/user/emails",
                headers={"Authorization": f"Bearer {request.access_token}"},
            )
            emails = emails_resp.json() if emails_resp.status_code == 200 else []
            primary_email = next((e["email"] for e in emails if e.get("primary")), None)

        email = primary_email or userinfo.get("email")
        name = userinfo.get("name") or userinfo.get("login")
        provider_id = str(userinfo["id"])

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown provider: {request.provider}",
        )

    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not retrieve email from OAuth provider",
        )

    user_id, _ = await _get_or_create_user(session, email, name, request.provider, provider_id)

    # Delete old OAuth session keys for this user
    await session.execute(
        delete(APIKeyModel).where(
            APIKeyModel.user_id == user_id,
            APIKeyModel.name == "OAuth Session",
        )
    )

    full_key, encryption_salt = await _create_api_key(
        session, user_id, "OAuth Session", with_encryption=True
    )

    encryption_key = None
    if encryption_salt:
        encryption_key = derive_encryption_key_base64(full_key, encryption_salt)

    # Fetch is_admin from database
    user_result = await session.execute(select(UserModel).where(UserModel.id == user_id))
    db_user = user_result.scalar_one_or_none()
    is_admin = getattr(db_user, "is_admin", False) if db_user else False

    return OAuthCallbackResponse(
        user=UserResponse(
            id=user_id, email=email, name=name, provider=request.provider, is_admin=is_admin
        ),
        api_key=full_key,
        encryption_key=encryption_key,
    )


# =============================================================================
# Password Reset Routes
# =============================================================================


class PasswordResetRequest(BaseModel):
    """Request to initiate password reset."""

    email: str


class PasswordResetResponse(BaseModel):
    """Response to password reset request."""

    message: str


class ResetPasswordRequest(BaseModel):
    """Request to reset password with token."""

    token: str
    new_password: str


@router.post("/forgot-password", response_model=PasswordResetResponse)
async def forgot_password(
    request: PasswordResetRequest,
    session: AsyncSession = Depends(get_session_dependency),
):
    """Request a password reset email."""
    # Find user by email
    result = await session.execute(select(UserModel).where(UserModel.email == request.email))
    user = result.scalar_one_or_none()

    # Always return success to prevent email enumeration
    if not user:
        return PasswordResetResponse(
            message="If an account exists with this email, you will receive a password reset link."
        )

    try:
        from service.email_service import (
            create_password_reset_token,
            send_password_reset_email,
        )

        reset_token = await create_password_reset_token(session, user.id)
        await session.commit()

        await send_password_reset_email(
            to_email=user.email,
            user_name=user.name,
            reset_token=reset_token,
        )
    except Exception as e:
        print(f"Failed to send password reset email: {e}")

    return PasswordResetResponse(
        message="If an account exists with this email, you will receive a password reset link."
    )


def _validate_password(password: str) -> tuple[bool, str]:
    """Validate password meets requirements.

    Requirements:
    - Minimum 8 characters
    - At least 1 uppercase letter
    - At least 1 lowercase letter
    - At least 1 digit
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


@router.post("/reset-password", response_model=PasswordResetResponse)
async def reset_password(
    request: ResetPasswordRequest,
    session: AsyncSession = Depends(get_session_dependency),
):
    """Reset password using a token."""
    from service.db.models import PasswordResetToken

    # Validate password
    is_valid, error_msg = _validate_password(request.new_password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        )

    # Hash the token to compare with stored hash
    token_hash = hashlib.sha256(request.token.encode()).hexdigest()

    # Find valid token
    result = await session.execute(
        select(PasswordResetToken).where(
            PasswordResetToken.token_hash == token_hash,
            PasswordResetToken.used_at.is_(None),
            PasswordResetToken.expires_at > datetime.now(timezone.utc),
        )
    )
    reset_token = result.scalar_one_or_none()

    if not reset_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    # Update user's password
    password_hash = _hash_password(request.new_password)
    await session.execute(
        update(UserModel)
        .where(UserModel.id == reset_token.user_id)
        .values(password_hash=password_hash)
    )

    # Mark token as used
    reset_token.used_at = datetime.now(timezone.utc)

    await session.commit()

    return PasswordResetResponse(message="Password has been reset successfully")


@router.get("/verify-reset-token")
async def verify_reset_token(
    token: str,
    session: AsyncSession = Depends(get_session_dependency),
):
    """Verify if a reset token is valid."""
    from service.db.models import PasswordResetToken

    token_hash = hashlib.sha256(token.encode()).hexdigest()

    result = await session.execute(
        select(PasswordResetToken).where(
            PasswordResetToken.token_hash == token_hash,
            PasswordResetToken.used_at.is_(None),
            PasswordResetToken.expires_at > datetime.now(timezone.utc),
        )
    )
    reset_token = result.scalar_one_or_none()

    return {"valid": reset_token is not None}
