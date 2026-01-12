"""ContextFS Authentication Module.

Provides API key generation, validation, and middleware for authenticating
requests to the sync server.
"""

from contextfs.auth.api_keys import (
    APIKey,
    APIKeyService,
    User,
    generate_api_key,
    hash_api_key,
    verify_api_key,
)
from contextfs.auth.middleware import (
    AuthMiddleware,
    get_current_user,
    init_auth_middleware,
    require_auth,
)

__all__ = [
    "APIKey",
    "APIKeyService",
    "AuthMiddleware",
    "User",
    "generate_api_key",
    "get_current_user",
    "hash_api_key",
    "init_auth_middleware",
    "require_auth",
    "verify_api_key",
]
