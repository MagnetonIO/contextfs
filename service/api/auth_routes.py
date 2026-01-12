"""Authentication API routes for ContextFS.

Handles user registration, API key management, and OAuth callbacks.
"""

import os
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from contextfs.auth import APIKey, APIKeyService, User, require_auth
from contextfs.encryption import derive_encryption_key_base64

router = APIRouter(prefix="/api/auth", tags=["auth"])


# Pydantic models
class UserResponse(BaseModel):
    """User profile response."""

    id: str
    email: str
    name: str | None
    provider: str


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


# OAuth models (for frontend to call)
class OAuthInitRequest(BaseModel):
    """Request to initiate OAuth flow."""

    provider: str  # google, github
    redirect_uri: str


class OAuthInitResponse(BaseModel):
    """Response with OAuth authorization URL."""

    auth_url: str
    state: str


class OAuthCallbackRequest(BaseModel):
    """OAuth callback data."""

    provider: str
    code: str
    state: str


class OAuthCallbackResponse(BaseModel):
    """Response after OAuth callback."""

    user: UserResponse
    api_key: str
    encryption_key: str | None


# Dependency to get APIKeyService
def get_api_key_service() -> APIKeyService:
    """Get APIKeyService instance."""
    db_path = os.environ.get("CONTEXTFS_DB_PATH", "contextfs.db")
    return APIKeyService(db_path)


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    auth: tuple[User, APIKey] = Depends(require_auth),
):
    """Get current authenticated user's profile."""
    user, api_key = auth
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        provider=user.provider,
    )


@router.post("/api-keys", response_model=CreateAPIKeyResponse)
async def create_api_key(
    request: CreateAPIKeyRequest,
    auth: tuple[User, APIKey] = Depends(require_auth),
    api_key_service: APIKeyService = Depends(get_api_key_service),
):
    """Create a new API key.

    IMPORTANT: The returned api_key and encryption_key are shown only once!
    """
    user, current_key = auth

    api_key, encryption_salt = await api_key_service.create_key(
        user_id=user.id,
        name=request.name,
        with_encryption=request.with_encryption,
    )

    # Derive encryption key if salt was generated
    encryption_key = None
    if encryption_salt:
        encryption_key = derive_encryption_key_base64(api_key, encryption_salt)

    # Get key prefix for the response
    key_prefix = api_key.split("_")[1][:8] if "_" in api_key else api_key[:8]

    # Generate config snippet
    config_lines = [
        "cloud:",
        "  enabled: true",
        f"  api_key: {api_key}",
    ]
    if encryption_key:
        config_lines.append(f"  encryption_key: {encryption_key}")
    config_lines.append("  server_url: https://api.contextfs.ai")

    config_snippet = "\n".join(config_lines)

    return CreateAPIKeyResponse(
        id=str(uuid4()),  # Generate ID for the key
        name=request.name,
        api_key=api_key,
        encryption_key=encryption_key,
        key_prefix=key_prefix,
        config_snippet=config_snippet,
    )


@router.get("/api-keys", response_model=APIKeyListResponse)
async def list_api_keys(
    auth: tuple[User, APIKey] = Depends(require_auth),
    api_key_service: APIKeyService = Depends(get_api_key_service),
):
    """List all API keys for the current user."""
    user, current_key = auth

    keys = await api_key_service.list_keys(user.id)

    return APIKeyListResponse(
        keys=[
            APIKeyListItem(
                id=key.id,
                name=key.name,
                key_prefix=key.key_prefix,
                is_active=key.is_active,
                created_at=key.created_at.isoformat(),
                last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
            )
            for key in keys
        ]
    )


@router.post("/api-keys/revoke")
async def revoke_api_key(
    request: RevokeKeyRequest,
    auth: tuple[User, APIKey] = Depends(require_auth),
    api_key_service: APIKeyService = Depends(get_api_key_service),
):
    """Revoke an API key (deactivate it)."""
    user, current_key = auth

    # Prevent revoking the key currently in use
    if request.key_id == current_key.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot revoke the API key currently in use",
        )

    success = await api_key_service.revoke_key(request.key_id, user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    return {"status": "revoked"}


@router.delete("/api-keys/{key_id}")
async def delete_api_key(
    key_id: str,
    auth: tuple[User, APIKey] = Depends(require_auth),
    api_key_service: APIKeyService = Depends(get_api_key_service),
):
    """Permanently delete an API key."""
    user, current_key = auth

    # Prevent deleting the key currently in use
    if key_id == current_key.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete the API key currently in use",
        )

    success = await api_key_service.delete_key(key_id, user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    return {"status": "deleted"}


# =============================================================================
# OAuth Routes (for first-time registration)
# These are called by the frontend to handle OAuth flows
# =============================================================================


@router.post("/oauth/init", response_model=OAuthInitResponse)
async def init_oauth(request: OAuthInitRequest):
    """Initialize OAuth flow.

    Returns the authorization URL to redirect the user to.
    """
    import secrets
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
    api_key_service: APIKeyService = Depends(get_api_key_service),
):
    """Handle OAuth callback.

    Exchanges code for tokens, creates/updates user, and generates API key.
    """
    import aiosqlite
    import httpx

    db_path = os.environ.get("CONTEXTFS_DB_PATH", "contextfs.db")

    if request.provider == "google":
        # Exchange code for tokens
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

            # Get user info
            userinfo_resp = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            )
            userinfo = userinfo_resp.json()

        email = userinfo["email"]
        name = userinfo.get("name")
        provider_id = userinfo["id"]

    elif request.provider == "github":
        # Exchange code for tokens
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

            # Get user info
            user_resp = await client.get(
                "https://api.github.com/user",
                headers={"Authorization": f"Bearer {tokens['access_token']}"},
            )
            userinfo = user_resp.json()

            # Get email (may require separate call)
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

    # Create or update user
    async with aiosqlite.connect(db_path) as db:
        # Check if user exists
        cursor = await db.execute(
            "SELECT id FROM users WHERE email = ?",
            (email,),
        )
        row = await cursor.fetchone()

        if row:
            user_id = row[0]
            # Update existing user
            await db.execute(
                "UPDATE users SET name = ?, provider_id = ? WHERE id = ?",
                (name, provider_id, user_id),
            )
        else:
            # Create new user
            user_id = str(uuid4())
            await db.execute(
                "INSERT INTO users (id, email, name, provider, provider_id) VALUES (?, ?, ?, ?, ?)",
                (user_id, email, name, request.provider, provider_id),
            )

            # Initialize free subscription
            from contextfs.billing import StripeService

            stripe_service = StripeService(db_path)
            await stripe_service.initialize_free_subscription(user_id)

            # Initialize usage tracking
            await db.execute(
                "INSERT INTO usage (user_id, device_count, memory_count) VALUES (?, 0, 0)",
                (user_id,),
            )

        await db.commit()

    # Create initial API key
    api_key, encryption_salt = await api_key_service.create_key(
        user_id=user_id,
        name="Default Key",
        with_encryption=True,
    )

    encryption_key = None
    if encryption_salt:
        encryption_key = derive_encryption_key_base64(api_key, encryption_salt)

    return OAuthCallbackResponse(
        user=UserResponse(
            id=user_id,
            email=email,
            name=name,
            provider=request.provider,
        ),
        api_key=api_key,
        encryption_key=encryption_key,
    )
