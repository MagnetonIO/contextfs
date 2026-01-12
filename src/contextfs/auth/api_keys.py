"""API Key generation and validation for ContextFS.

Key format: ctxfs_ + 32 random bytes (base64url encoded)
Storage: SHA-256 hash of the key, with prefix for identification
"""

import base64
import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

import aiosqlite

# Key prefix for all ContextFS API keys
KEY_PREFIX = "ctxfs_"


@dataclass
class APIKey:
    """Represents an API key record."""

    id: str
    user_id: str
    name: str
    key_prefix: str
    encryption_salt: str | None
    is_active: bool
    created_at: datetime
    last_used_at: datetime | None


@dataclass
class User:
    """Represents a user record."""

    id: str
    email: str
    name: str | None
    provider: str
    provider_id: str | None
    created_at: datetime


def generate_api_key() -> tuple[str, str]:
    """Generate a new API key.

    Returns:
        Tuple of (full_key, key_prefix) where:
        - full_key: The complete API key to give to the user (ctxfs_...)
        - key_prefix: First 8 chars of the key part for identification
    """
    # Generate 32 random bytes and encode as base64url (no padding)
    random_bytes = secrets.token_bytes(32)
    key_part = base64.urlsafe_b64encode(random_bytes).decode("ascii").rstrip("=")

    full_key = f"{KEY_PREFIX}{key_part}"
    key_prefix = key_part[:8]

    return full_key, key_prefix


def generate_encryption_salt() -> str:
    """Generate a random salt for E2EE key derivation.

    Returns:
        Base64-encoded 32-byte salt
    """
    salt_bytes = secrets.token_bytes(32)
    return base64.urlsafe_b64encode(salt_bytes).decode("ascii")


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage.

    Args:
        api_key: The full API key (ctxfs_...)

    Returns:
        SHA-256 hash of the key as hex string
    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def verify_api_key(api_key: str, stored_hash: str) -> bool:
    """Verify an API key against its stored hash.

    Args:
        api_key: The full API key to verify
        stored_hash: The stored SHA-256 hash

    Returns:
        True if the key matches, False otherwise
    """
    return secrets.compare_digest(hash_api_key(api_key), stored_hash)


class APIKeyService:
    """Service for managing API keys in the database."""

    def __init__(self, db_path: str):
        """Initialize the API key service.

        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path

    async def create_key(
        self,
        user_id: str,
        name: str,
        with_encryption: bool = True,
    ) -> tuple[str, str | None]:
        """Create a new API key for a user.

        Args:
            user_id: The user's ID
            name: A descriptive name for the key
            with_encryption: Whether to generate an encryption salt

        Returns:
            Tuple of (api_key, encryption_salt) - api_key is shown once only
        """
        key_id = str(uuid4())
        full_key, key_prefix = generate_api_key()
        key_hash = hash_api_key(full_key)
        encryption_salt = generate_encryption_salt() if with_encryption else None

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO api_keys (id, user_id, name, key_hash, key_prefix, encryption_salt)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (key_id, user_id, name, key_hash, key_prefix, encryption_salt),
            )
            await db.commit()

        return full_key, encryption_salt

    async def validate_key(self, api_key: str) -> tuple[User, APIKey] | None:
        """Validate an API key and return the associated user.

        Args:
            api_key: The full API key to validate

        Returns:
            Tuple of (User, APIKey) if valid, None otherwise
        """
        if not api_key.startswith(KEY_PREFIX):
            return None

        key_hash = hash_api_key(api_key)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Find the API key
            cursor = await db.execute(
                """
                SELECT k.*, u.email, u.name as user_name, u.provider, u.provider_id,
                       u.created_at as user_created_at
                FROM api_keys k
                JOIN users u ON k.user_id = u.id
                WHERE k.key_hash = ? AND k.is_active = 1
                """,
                (key_hash,),
            )
            row = await cursor.fetchone()

            if not row:
                return None

            # Update last used timestamp
            await db.execute(
                "UPDATE api_keys SET last_used_at = datetime('now') WHERE id = ?",
                (row["id"],),
            )
            await db.commit()

            user = User(
                id=row["user_id"],
                email=row["email"],
                name=row["user_name"],
                provider=row["provider"],
                provider_id=row["provider_id"],
                created_at=datetime.fromisoformat(row["user_created_at"]),
            )

            api_key_obj = APIKey(
                id=row["id"],
                user_id=row["user_id"],
                name=row["name"],
                key_prefix=row["key_prefix"],
                encryption_salt=row["encryption_salt"],
                is_active=bool(row["is_active"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                last_used_at=(
                    datetime.fromisoformat(row["last_used_at"]) if row["last_used_at"] else None
                ),
            )

            return user, api_key_obj

    async def list_keys(self, user_id: str) -> list[APIKey]:
        """List all API keys for a user (without the actual key values).

        Args:
            user_id: The user's ID

        Returns:
            List of APIKey objects (key values not included)
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute(
                """
                SELECT id, user_id, name, key_prefix, encryption_salt,
                       is_active, created_at, last_used_at
                FROM api_keys
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                (user_id,),
            )
            rows = await cursor.fetchall()

            return [
                APIKey(
                    id=row["id"],
                    user_id=row["user_id"],
                    name=row["name"],
                    key_prefix=row["key_prefix"],
                    encryption_salt=row["encryption_salt"],
                    is_active=bool(row["is_active"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                    last_used_at=(
                        datetime.fromisoformat(row["last_used_at"]) if row["last_used_at"] else None
                    ),
                )
                for row in rows
            ]

    async def revoke_key(self, key_id: str, user_id: str) -> bool:
        """Revoke an API key.

        Args:
            key_id: The key's ID
            user_id: The user's ID (for authorization)

        Returns:
            True if the key was revoked, False if not found
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                UPDATE api_keys
                SET is_active = 0
                WHERE id = ? AND user_id = ?
                """,
                (key_id, user_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def delete_key(self, key_id: str, user_id: str) -> bool:
        """Permanently delete an API key.

        Args:
            key_id: The key's ID
            user_id: The user's ID (for authorization)

        Returns:
            True if the key was deleted, False if not found
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM api_keys WHERE id = ? AND user_id = ?",
                (key_id, user_id),
            )
            await db.commit()
            return cursor.rowcount > 0
