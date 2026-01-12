"""Factory for creating auth storage backends.

Selects the appropriate backend based on CONTEXTFS_BACKEND environment variable:
- sqlite (default): SQLite for local CLI/MCP
- postgres: PostgreSQL for production web API
"""

from pathlib import Path

from .base import AuthStorage


def create_auth_storage(config=None) -> AuthStorage:
    """Create auth storage based on configuration.

    Uses the CONTEXTFS_BACKEND environment variable to select backend:
    - sqlite (default): Uses SQLite in ~/.contextfs/context.db
    - postgres: Uses PostgreSQL specified by CONTEXTFS_POSTGRES_URL

    Args:
        config: Optional Config object. Uses global config if not provided.

    Returns:
        AuthStorage instance (SQLiteAuthStorage or PostgresAuthStorage)

    Raises:
        ImportError: If required dependencies not installed
    """
    if config is None:
        from contextfs.config import get_config

        config = get_config()

    from contextfs.config import BackendType

    if config.backend in (BackendType.POSTGRES, BackendType.POSTGRES_FALKORDB):
        from .postgres import PostgresAuthStorage

        return PostgresAuthStorage(config.postgres_url)
    else:
        from .sqlite import SQLiteAuthStorage

        db_path = Path(config.data_dir).expanduser() / config.sqlite_filename
        return SQLiteAuthStorage(db_path)


# Admin API key for testing (bypasses email verification)
ADMIN_API_KEY_PREFIX = "ctxfs_admin_"
ADMIN_USER_ID = "admin-00000000-0000-0000-0000-000000000000"


async def create_admin_user(storage: AuthStorage, admin_key: str | None = None) -> tuple[str, str]:
    """Create or get admin user for testing.

    Admin user bypasses email verification and billing restrictions.
    Use for local development and testing only.
    E2EE is automatic - salt is generated with the API key.

    Args:
        storage: Auth storage backend
        admin_key: Optional pre-generated admin API key

    Returns:
        Tuple of (user_id, api_key)
    """
    from uuid import uuid4

    from contextfs.auth.api_keys import generate_api_key, generate_encryption_salt, hash_api_key

    # Check if admin user exists
    admin_email = "admin@contextfs.local"
    existing = await storage.get_user_by_email(admin_email)

    if existing:
        # Get or create API key for admin
        keys = await storage.list_api_keys(existing.id)
        if keys:
            # Generate new key for this session with E2EE salt
            raw_key, key_prefix = generate_api_key()
            key_hash = hash_api_key(raw_key)
            encryption_salt = generate_encryption_salt()
            await storage.update_api_key(
                keys[0].id,
                key_hash=key_hash,
                key_prefix=key_prefix,
                encryption_salt=encryption_salt,
            )
            return (existing.id, raw_key)

    # Create admin user
    user = await storage.create_user(
        user_id=ADMIN_USER_ID,
        email=admin_email,
        name="Admin",
        provider="admin",
        password_hash=None,
        email_verified=True,  # Admin is always verified
    )

    # Create API key with E2EE salt
    raw_key, key_prefix = generate_api_key()
    key_hash = hash_api_key(raw_key)
    key_id = str(uuid4())
    encryption_salt = generate_encryption_salt()

    await storage.create_api_key(
        key_id=key_id,
        user_id=user.id,
        name="Admin Key",
        key_hash=key_hash,
        key_prefix=key_prefix,
        encryption_salt=encryption_salt,
    )

    # Create unlimited subscription
    sub_id = str(uuid4())
    await storage.create_subscription(
        sub_id=sub_id,
        user_id=user.id,
        tier="admin",
        device_limit=-1,  # Unlimited
        memory_limit=-1,  # Unlimited
    )

    return (user.id, raw_key)
