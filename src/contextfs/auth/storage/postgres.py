"""PostgreSQL implementation of AuthStorage.

Used for production web API (contextfs.ai). Stores auth data in
PostgreSQL (Supabase or local).
"""

import logging
from datetime import datetime

from .base import ApiKey, AuthStorage, Device, Subscription, User

logger = logging.getLogger(__name__)


class PostgresAuthStorage(AuthStorage):
    """PostgreSQL-based auth storage for production."""

    def __init__(self, connection_string: str):
        """Initialize with connection string.

        Args:
            connection_string: PostgreSQL connection URL
                e.g., postgresql://user:pass@host:5432/dbname
        """
        self.connection_string = connection_string
        self._pool = None

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            try:
                import asyncpg
            except ImportError:
                raise ImportError(
                    "asyncpg is required for PostgreSQL auth storage. "
                    "Install with: pip install contextfs[postgres]"
                )

            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=1,
                max_size=10,
            )
            logger.info("PostgreSQL auth storage pool created")

        return self._pool

    def _parse_datetime(self, value) -> datetime | None:
        """Parse datetime from database value."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except (ValueError, TypeError):
            return None

    # ==================== User Operations ====================

    def _format_datetime(self, value) -> str | None:
        """Format datetime as ISO string for TEXT columns."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

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
        pool = await self._get_pool()
        created_at = datetime.utcnow()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO users (id, email, name, provider, password_hash,
                                   email_verified, verification_token,
                                   verification_token_expires, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                user_id,
                email,
                name,
                provider,
                password_hash,
                email_verified,
                verification_token,
                self._format_datetime(verification_token_expires),
                self._format_datetime(created_at),
            )

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
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, email, name, provider, password_hash, email_verified,
                       created_at, last_login
                FROM users WHERE id = $1
                """,
                user_id,
            )

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
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, email, name, provider, password_hash, email_verified,
                       created_at, last_login
                FROM users WHERE email = $1
                """,
                email,
            )

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
        pool = await self._get_pool()

        updates = []
        params = []
        param_count = 0

        if email_verified is not None:
            param_count += 1
            updates.append(f"email_verified = ${param_count}")
            params.append(email_verified)

        if verification_token is not None:
            param_count += 1
            updates.append(f"verification_token = ${param_count}")
            params.append(verification_token if verification_token else None)

        if verification_token_expires is not None:
            param_count += 1
            updates.append(f"verification_token_expires = ${param_count}")
            params.append(self._format_datetime(verification_token_expires))

        if last_login is not None:
            param_count += 1
            updates.append(f"last_login = ${param_count}")
            params.append(self._format_datetime(last_login))

        if name is not None:
            param_count += 1
            updates.append(f"name = ${param_count}")
            params.append(name)

        if not updates:
            return await self.get_user_by_id(user_id)

        param_count += 1
        params.append(user_id)

        async with pool.acquire() as conn:
            await conn.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE id = ${param_count}",
                *params,
            )

        return await self.get_user_by_id(user_id)

    async def get_user_by_verification_token(self, token: str) -> tuple[User, datetime] | None:
        """Get user by verification token."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, email, name, provider, password_hash, email_verified,
                       created_at, last_login, verification_token_expires
                FROM users WHERE verification_token = $1
                """,
                token,
            )

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
        pool = await self._get_pool()
        created_at = datetime.utcnow()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO api_keys (id, user_id, name, key_hash, key_prefix,
                                      encryption_salt, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                key_id,
                user_id,
                name,
                key_hash,
                key_prefix,
                encryption_salt,
                self._format_datetime(created_at),
            )

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
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT user_id FROM api_keys WHERE key_hash = $1 AND is_active = true",
                key_hash,
            )

        return row["user_id"] if row else None

    async def get_encryption_salt_by_key_hash(self, key_hash: str) -> str | None:
        """Get encryption salt for an API key by its hash."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT encryption_salt FROM api_keys WHERE key_hash = $1 AND is_active = true",
                key_hash,
            )

        return row["encryption_salt"] if row else None

    async def list_api_keys(self, user_id: str) -> list[ApiKey]:
        """List all API keys for a user."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, user_id, name, key_prefix, is_active, created_at,
                       last_used_at, encryption_salt
                FROM api_keys WHERE user_id = $1 ORDER BY created_at DESC
                """,
                user_id,
            )

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
        pool = await self._get_pool()

        updates = []
        params = []
        param_count = 0

        if key_hash is not None:
            param_count += 1
            updates.append(f"key_hash = ${param_count}")
            params.append(key_hash)

        if key_prefix is not None:
            param_count += 1
            updates.append(f"key_prefix = ${param_count}")
            params.append(key_prefix)

        if last_used_at is not None:
            param_count += 1
            updates.append(f"last_used_at = ${param_count}")
            params.append(self._format_datetime(last_used_at))

        if encryption_salt is not None:
            param_count += 1
            updates.append(f"encryption_salt = ${param_count}")
            params.append(encryption_salt)

        if not updates:
            return True

        param_count += 1
        params.append(key_id)

        async with pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE api_keys SET {', '.join(updates)} WHERE id = ${param_count}",
                *params,
            )

        return result != "UPDATE 0"

    async def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        """Revoke an API key."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE api_keys SET is_active = false WHERE id = $1 AND user_id = $2",
                key_id,
                user_id,
            )

        return result != "UPDATE 0"

    async def delete_api_key(self, key_id: str, user_id: str) -> bool:
        """Permanently delete an API key."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM api_keys WHERE id = $1 AND user_id = $2",
                key_id,
                user_id,
            )

        return result != "DELETE 0"

    # ==================== Subscription Operations ====================

    async def get_subscription(self, user_id: str) -> Subscription | None:
        """Get user's subscription."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, user_id, tier, status, device_limit, memory_limit,
                       current_period_end, stripe_customer_id, stripe_subscription_id
                FROM subscriptions WHERE user_id = $1
                """,
                user_id,
            )

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
        pool = await self._get_pool()
        created_at = datetime.utcnow()

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO subscriptions (id, user_id, tier, status, device_limit,
                                            memory_limit, created_at)
                VALUES ($1, $2, $3, 'active', $4, $5, $6)
                """,
                sub_id,
                user_id,
                tier,
                device_limit,
                memory_limit,
                self._format_datetime(created_at),
            )

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
        pool = await self._get_pool()

        updates = ["updated_at = NOW()"]
        params = []
        param_count = 0

        if tier is not None:
            param_count += 1
            updates.append(f"tier = ${param_count}")
            params.append(tier)

        if status is not None:
            param_count += 1
            updates.append(f"status = ${param_count}")
            params.append(status)

        if device_limit is not None:
            param_count += 1
            updates.append(f"device_limit = ${param_count}")
            params.append(device_limit)

        if memory_limit is not None:
            param_count += 1
            updates.append(f"memory_limit = ${param_count}")
            params.append(memory_limit)

        if stripe_customer_id is not None:
            param_count += 1
            updates.append(f"stripe_customer_id = ${param_count}")
            params.append(stripe_customer_id)

        if stripe_subscription_id is not None:
            param_count += 1
            updates.append(f"stripe_subscription_id = ${param_count}")
            params.append(stripe_subscription_id)

        if current_period_end is not None:
            param_count += 1
            updates.append(f"current_period_end = ${param_count}")
            params.append(self._format_datetime(current_period_end))

        param_count += 1
        params.append(user_id)

        async with pool.acquire() as conn:
            await conn.execute(
                f"UPDATE subscriptions SET {', '.join(updates)} WHERE user_id = ${param_count}",
                *params,
            )

        return await self.get_subscription(user_id)

    # ==================== Device Operations ====================

    async def list_devices(self, user_id: str) -> list[Device]:
        """List all devices for a user."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Query maps sync-style columns to Device model
            # device_id -> id, device_name -> name, platform -> device_type/os
            rows = await conn.fetch(
                """
                SELECT
                    device_id as id,
                    user_id,
                    device_name as name,
                    platform as device_type,
                    platform as os,
                    client_version as os_version,
                    last_sync_at,
                    registered_at as created_at
                FROM devices WHERE user_id = $1 ORDER BY registered_at DESC
                """,
                user_id,
            )

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
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM devices WHERE device_id = $1 AND user_id = $2",
                device_id,
                user_id,
            )

        return result != "DELETE 0"

    async def get_device_count(self, user_id: str) -> int:
        """Get count of devices for user."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) as count FROM devices WHERE user_id = $1",
                user_id,
            )

        return row["count"] if row else 0

    # ==================== Connection Management ====================

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL auth storage pool closed")
