"""SQLite implementation of AuthStorage.

Used for local CLI/MCP development. Stores auth data in the same
database as local memories (~/.contextfs/context.db).
"""

from datetime import datetime
from pathlib import Path

import aiosqlite

from .base import ApiKey, AuthStorage, Device, Subscription, User


class SQLiteAuthStorage(AuthStorage):
    """SQLite-based auth storage for local development."""

    def __init__(self, db_path: Path):
        """Initialize with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def _get_connection(self) -> aiosqlite.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = await aiosqlite.connect(self.db_path)
            self._conn.row_factory = aiosqlite.Row
        return self._conn

    def _parse_datetime(self, value: str | None) -> datetime | None:
        """Parse datetime from string."""
        if value is None:
            return None
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None

    # ==================== User Operations ====================

    async def create_user(
        self,
        user_id: str,
        email: str,
        name: str,
        provider: str,
        password_hash: str | None = None,
        email_verified: bool = False,
        verification_token: str | None = None,
        verification_token_expires: datetime | None = None,
    ) -> User:
        """Create a new user."""
        conn = await self._get_connection()
        created_at = datetime.utcnow()

        await conn.execute(
            """
            INSERT INTO users (id, email, name, provider, password_hash,
                               email_verified, verification_token,
                               verification_token_expires, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                email,
                name,
                provider,
                password_hash,
                1 if email_verified else 0,
                verification_token,
                verification_token_expires.isoformat() if verification_token_expires else None,
                created_at.isoformat(),
            ),
        )
        await conn.commit()

        return User(
            id=user_id,
            email=email,
            name=name,
            provider=provider,
            password_hash=password_hash,
            email_verified=email_verified,
            created_at=created_at,
        )

    async def get_user_by_id(self, user_id: str) -> User | None:
        """Get user by ID."""
        conn = await self._get_connection()
        cursor = await conn.execute(
            """
            SELECT id, email, name, provider, password_hash, email_verified,
                   created_at, last_login
            FROM users WHERE id = ?
            """,
            (user_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return User(
            id=row["id"],
            email=row["email"],
            name=row["name"] or "",
            provider=row["provider"],
            password_hash=row["password_hash"],
            email_verified=bool(row["email_verified"]),
            created_at=self._parse_datetime(row["created_at"]) or datetime.utcnow(),
            last_login=self._parse_datetime(row["last_login"]),
        )

    async def get_user_by_email(self, email: str) -> User | None:
        """Get user by email address."""
        conn = await self._get_connection()
        cursor = await conn.execute(
            """
            SELECT id, email, name, provider, password_hash, email_verified,
                   created_at, last_login
            FROM users WHERE email = ?
            """,
            (email,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return User(
            id=row["id"],
            email=row["email"],
            name=row["name"] or "",
            provider=row["provider"],
            password_hash=row["password_hash"],
            email_verified=bool(row["email_verified"]),
            created_at=self._parse_datetime(row["created_at"]) or datetime.utcnow(),
            last_login=self._parse_datetime(row["last_login"]),
        )

    async def update_user(
        self,
        user_id: str,
        email_verified: bool | None = None,
        verification_token: str | None = None,
        verification_token_expires: datetime | None = None,
        last_login: datetime | None = None,
        name: str | None = None,
    ) -> User | None:
        """Update user fields."""
        conn = await self._get_connection()

        updates = []
        params = []

        if email_verified is not None:
            updates.append("email_verified = ?")
            params.append(1 if email_verified else 0)

        if verification_token is not None:
            updates.append("verification_token = ?")
            params.append(verification_token if verification_token else None)

        if verification_token_expires is not None:
            updates.append("verification_token_expires = ?")
            params.append(
                verification_token_expires.isoformat() if verification_token_expires else None
            )

        if last_login is not None:
            updates.append("last_login = ?")
            params.append(last_login.isoformat())

        if name is not None:
            updates.append("name = ?")
            params.append(name)

        if not updates:
            return await self.get_user_by_id(user_id)

        params.append(user_id)
        await conn.execute(
            f"UPDATE users SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        await conn.commit()

        return await self.get_user_by_id(user_id)

    async def get_user_by_verification_token(self, token: str) -> tuple[User, datetime] | None:
        """Get user by verification token."""
        conn = await self._get_connection()
        cursor = await conn.execute(
            """
            SELECT id, email, name, provider, password_hash, email_verified,
                   created_at, last_login, verification_token_expires
            FROM users WHERE verification_token = ?
            """,
            (token,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        user = User(
            id=row["id"],
            email=row["email"],
            name=row["name"] or "",
            provider=row["provider"],
            password_hash=row["password_hash"],
            email_verified=bool(row["email_verified"]),
            created_at=self._parse_datetime(row["created_at"]) or datetime.utcnow(),
            last_login=self._parse_datetime(row["last_login"]),
        )

        expires = self._parse_datetime(row["verification_token_expires"])
        if expires is None:
            expires = datetime.utcnow()

        return (user, expires)

    # ==================== API Key Operations ====================

    async def create_api_key(
        self,
        key_id: str,
        user_id: str,
        name: str,
        key_hash: str,
        key_prefix: str,
        encryption_salt: str | None = None,
    ) -> ApiKey:
        """Create a new API key."""
        conn = await self._get_connection()
        created_at = datetime.utcnow()

        await conn.execute(
            """
            INSERT INTO api_keys (id, user_id, name, key_hash, key_prefix,
                                  encryption_salt, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (key_id, user_id, name, key_hash, key_prefix, encryption_salt, created_at.isoformat()),
        )
        await conn.commit()

        return ApiKey(
            id=key_id,
            user_id=user_id,
            name=name,
            key_prefix=key_prefix,
            is_active=True,
            created_at=created_at,
            encryption_salt=encryption_salt,
        )

    async def get_user_id_by_key_hash(self, key_hash: str) -> str | None:
        """Get user ID by API key hash."""
        conn = await self._get_connection()
        cursor = await conn.execute(
            "SELECT user_id FROM api_keys WHERE key_hash = ? AND is_active = 1",
            (key_hash,),
        )
        row = await cursor.fetchone()
        return row["user_id"] if row else None

    async def get_encryption_salt_by_key_hash(self, key_hash: str) -> str | None:
        """Get encryption salt for an API key by its hash."""
        conn = await self._get_connection()
        cursor = await conn.execute(
            "SELECT encryption_salt FROM api_keys WHERE key_hash = ? AND is_active = 1",
            (key_hash,),
        )
        row = await cursor.fetchone()
        return row["encryption_salt"] if row else None

    async def list_api_keys(self, user_id: str) -> list[ApiKey]:
        """List all API keys for a user."""
        conn = await self._get_connection()
        cursor = await conn.execute(
            """
            SELECT id, user_id, name, key_prefix, is_active, created_at,
                   last_used_at, encryption_salt
            FROM api_keys WHERE user_id = ? ORDER BY created_at DESC
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()

        return [
            ApiKey(
                id=row["id"],
                user_id=row["user_id"],
                name=row["name"],
                key_prefix=row["key_prefix"],
                is_active=bool(row["is_active"]),
                created_at=self._parse_datetime(row["created_at"]) or datetime.utcnow(),
                last_used_at=self._parse_datetime(row["last_used_at"]),
                encryption_salt=row["encryption_salt"],
            )
            for row in rows
        ]

    async def update_api_key(
        self,
        key_id: str,
        key_hash: str | None = None,
        key_prefix: str | None = None,
        last_used_at: datetime | None = None,
        encryption_salt: str | None = None,
    ) -> bool:
        """Update API key fields."""
        conn = await self._get_connection()

        updates = []
        params = []

        if key_hash is not None:
            updates.append("key_hash = ?")
            params.append(key_hash)

        if key_prefix is not None:
            updates.append("key_prefix = ?")
            params.append(key_prefix)

        if last_used_at is not None:
            updates.append("last_used_at = ?")
            params.append(last_used_at.isoformat())

        if encryption_salt is not None:
            updates.append("encryption_salt = ?")
            params.append(encryption_salt)

        if not updates:
            return True

        params.append(key_id)
        cursor = await conn.execute(
            f"UPDATE api_keys SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        await conn.commit()
        return cursor.rowcount > 0

    async def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        """Revoke an API key."""
        conn = await self._get_connection()
        cursor = await conn.execute(
            "UPDATE api_keys SET is_active = 0 WHERE id = ? AND user_id = ?",
            (key_id, user_id),
        )
        await conn.commit()
        return cursor.rowcount > 0

    async def delete_api_key(self, key_id: str, user_id: str) -> bool:
        """Permanently delete an API key."""
        conn = await self._get_connection()
        cursor = await conn.execute(
            "DELETE FROM api_keys WHERE id = ? AND user_id = ?",
            (key_id, user_id),
        )
        await conn.commit()
        return cursor.rowcount > 0

    # ==================== Subscription Operations ====================

    async def get_subscription(self, user_id: str) -> Subscription | None:
        """Get user's subscription."""
        conn = await self._get_connection()
        cursor = await conn.execute(
            """
            SELECT id, user_id, tier, status, device_limit, memory_limit,
                   current_period_end, stripe_customer_id, stripe_subscription_id
            FROM subscriptions WHERE user_id = ?
            """,
            (user_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return Subscription(
            id=row["id"],
            user_id=row["user_id"],
            tier=row["tier"],
            status=row["status"],
            device_limit=row["device_limit"] or 3,
            memory_limit=row["memory_limit"] or 10000,
            current_period_end=self._parse_datetime(row["current_period_end"]),
            stripe_customer_id=row["stripe_customer_id"],
            stripe_subscription_id=row["stripe_subscription_id"],
        )

    async def create_subscription(
        self,
        sub_id: str,
        user_id: str,
        tier: str = "free",
        device_limit: int = 3,
        memory_limit: int = 10000,
    ) -> Subscription:
        """Create a subscription for user."""
        conn = await self._get_connection()
        created_at = datetime.utcnow()

        await conn.execute(
            """
            INSERT INTO subscriptions (id, user_id, tier, status, device_limit,
                                        memory_limit, created_at)
            VALUES (?, ?, ?, 'active', ?, ?, ?)
            """,
            (sub_id, user_id, tier, device_limit, memory_limit, created_at.isoformat()),
        )
        await conn.commit()

        return Subscription(
            id=sub_id,
            user_id=user_id,
            tier=tier,
            status="active",
            device_limit=device_limit,
            memory_limit=memory_limit,
        )

    async def update_subscription(
        self,
        user_id: str,
        tier: str | None = None,
        status: str | None = None,
        device_limit: int | None = None,
        memory_limit: int | None = None,
        stripe_customer_id: str | None = None,
        stripe_subscription_id: str | None = None,
        current_period_end: datetime | None = None,
    ) -> Subscription | None:
        """Update subscription fields."""
        conn = await self._get_connection()

        updates = ["updated_at = ?"]
        params = [datetime.utcnow().isoformat()]

        if tier is not None:
            updates.append("tier = ?")
            params.append(tier)

        if status is not None:
            updates.append("status = ?")
            params.append(status)

        if device_limit is not None:
            updates.append("device_limit = ?")
            params.append(device_limit)

        if memory_limit is not None:
            updates.append("memory_limit = ?")
            params.append(memory_limit)

        if stripe_customer_id is not None:
            updates.append("stripe_customer_id = ?")
            params.append(stripe_customer_id)

        if stripe_subscription_id is not None:
            updates.append("stripe_subscription_id = ?")
            params.append(stripe_subscription_id)

        if current_period_end is not None:
            updates.append("current_period_end = ?")
            params.append(current_period_end.isoformat())

        params.append(user_id)
        await conn.execute(
            f"UPDATE subscriptions SET {', '.join(updates)} WHERE user_id = ?",
            params,
        )
        await conn.commit()

        return await self.get_subscription(user_id)

    # ==================== Device Operations ====================

    async def list_devices(self, user_id: str) -> list[Device]:
        """List all devices for a user."""
        conn = await self._get_connection()

        cursor = await conn.execute(
            """
            SELECT id, user_id, name, device_type, os, os_version,
                   last_sync_at, created_at
            FROM devices WHERE user_id = ? ORDER BY created_at DESC
            """,
            (user_id,),
        )
        rows = await cursor.fetchall()

        return [
            Device(
                id=row["id"],
                user_id=row["user_id"],
                name=row["name"],
                device_type=row["device_type"],
                os=row["os"],
                os_version=row["os_version"],
                last_sync_at=self._parse_datetime(row["last_sync_at"]),
                created_at=self._parse_datetime(row["created_at"]),
            )
            for row in rows
        ]

    async def delete_device(self, device_id: str, user_id: str) -> bool:
        """Delete a device."""
        conn = await self._get_connection()
        cursor = await conn.execute(
            "DELETE FROM devices WHERE id = ? AND user_id = ?",
            (device_id, user_id),
        )
        await conn.commit()
        return cursor.rowcount > 0

    async def get_device_count(self, user_id: str) -> int:
        """Get count of devices for user."""
        conn = await self._get_connection()
        cursor = await conn.execute(
            "SELECT COUNT(*) as count FROM devices WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        return row["count"] if row else 0

    # ==================== Connection Management ====================

    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
