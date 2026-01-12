"""FastAPI authentication middleware for ContextFS.

Provides middleware and dependencies for authenticating API requests
using API keys passed in the X-API-Key header.
"""

from collections.abc import Callable

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from contextfs.auth.api_keys import APIKey, APIKeyService, User

# Header for API key authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthMiddleware:
    """Middleware for API key authentication."""

    def __init__(self, db_path: str):
        """Initialize the auth middleware.

        Args:
            db_path: Path to the SQLite database
        """
        self.api_key_service = APIKeyService(db_path)

    async def authenticate(self, api_key: str | None) -> tuple[User, APIKey] | None:
        """Authenticate a request using an API key.

        Args:
            api_key: The API key from the request header

        Returns:
            Tuple of (User, APIKey) if authenticated, None otherwise
        """
        if not api_key:
            return None

        return await self.api_key_service.validate_key(api_key)


# Global middleware instance (set by app startup)
_auth_middleware: AuthMiddleware | None = None


def init_auth_middleware(db_path: str) -> AuthMiddleware:
    """Initialize the global auth middleware.

    Args:
        db_path: Path to the SQLite database

    Returns:
        The initialized AuthMiddleware instance
    """
    global _auth_middleware
    _auth_middleware = AuthMiddleware(db_path)
    return _auth_middleware


def get_auth_middleware() -> AuthMiddleware:
    """Get the global auth middleware instance.

    Returns:
        The AuthMiddleware instance

    Raises:
        RuntimeError: If auth middleware hasn't been initialized
    """
    if _auth_middleware is None:
        raise RuntimeError("Auth middleware not initialized. Call init_auth_middleware first.")
    return _auth_middleware


async def get_current_user(
    api_key: str | None = Depends(API_KEY_HEADER),
) -> tuple[User, APIKey] | None:
    """FastAPI dependency to get the current authenticated user.

    This is an optional dependency - returns None if not authenticated.

    Args:
        api_key: The API key from the X-API-Key header

    Returns:
        Tuple of (User, APIKey) if authenticated, None otherwise
    """
    middleware = get_auth_middleware()
    return await middleware.authenticate(api_key)


async def require_auth(
    auth: tuple[User, APIKey] | None = Depends(get_current_user),
) -> tuple[User, APIKey]:
    """FastAPI dependency that requires authentication.

    Use this for routes that require a valid API key.

    Args:
        auth: The authentication result from get_current_user

    Returns:
        Tuple of (User, APIKey)

    Raises:
        HTTPException: 401 if not authenticated
    """
    if auth is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return auth


def require_active_subscription(
    allowed_tiers: list[str] | None = None,
) -> Callable:
    """Create a dependency that requires an active subscription.

    Args:
        allowed_tiers: List of allowed tier names (None = any active subscription)

    Returns:
        FastAPI dependency function
    """

    async def dependency(
        request: Request,
        auth: tuple[User, APIKey] = Depends(require_auth),
    ) -> tuple[User, APIKey]:
        """Check subscription status."""
        user, api_key = auth

        # Get subscription from database
        # This would need to be implemented with a SubscriptionService
        # For now, we'll just return the auth tuple
        # In production, this would check the subscriptions table
        return auth

    return dependency


class RateLimiter:
    """Simple in-memory rate limiter.

    For production, use Redis or a distributed rate limiting solution.
    """

    def __init__(self, requests_per_minute: int = 100):
        """Initialize the rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute
        """
        self.requests_per_minute = requests_per_minute
        self._requests: dict[str, list[float]] = {}

    async def check_rate_limit(self, user_id: str) -> bool:
        """Check if a user is within their rate limit.

        Args:
            user_id: The user's ID

        Returns:
            True if within limit, False if exceeded
        """
        import time

        now = time.time()
        window_start = now - 60  # 1 minute window

        if user_id not in self._requests:
            self._requests[user_id] = []

        # Clean old requests
        self._requests[user_id] = [t for t in self._requests[user_id] if t > window_start]

        # Check limit
        if len(self._requests[user_id]) >= self.requests_per_minute:
            return False

        # Record this request
        self._requests[user_id].append(now)
        return True


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter(requests_per_minute: int = 100) -> RateLimiter:
    """Get or create the global rate limiter.

    Args:
        requests_per_minute: Maximum requests per minute

    Returns:
        The RateLimiter instance
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(requests_per_minute)
    return _rate_limiter


async def check_rate_limit(
    auth: tuple[User, APIKey] = Depends(require_auth),
) -> tuple[User, APIKey]:
    """FastAPI dependency that enforces rate limiting.

    Args:
        auth: The authenticated user and API key

    Returns:
        The auth tuple if within rate limit

    Raises:
        HTTPException: 429 if rate limit exceeded
    """
    user, api_key = auth
    rate_limiter = get_rate_limiter()

    if not await rate_limiter.check_rate_limit(user.id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": "60"},
        )

    return auth
