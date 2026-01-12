"""Abstract base class for auth storage backends.

Defines the interface for user, API key, subscription, and device operations.
Implementations: SQLiteAuthStorage (local), PostgresAuthStorage (production).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class User:
    """User data model."""

    id: str
    email: str
    name: str
    provider: str
    password_hash: str | None
    email_verified: bool
    created_at: datetime
    last_login: datetime | None = None


@dataclass
class ApiKey:
    """API key data model."""

    id: str
    user_id: str
    name: str
    key_prefix: str
    is_active: bool
    created_at: datetime
    last_used_at: datetime | None = None
    encryption_salt: str | None = None


@dataclass
class Subscription:
    """Subscription data model."""

    id: str
    user_id: str
    tier: str
    status: str
    device_limit: int
    memory_limit: int
    current_period_end: datetime | None = None
    stripe_customer_id: str | None = None
    stripe_subscription_id: str | None = None


@dataclass
class Device:
    """Device data model."""

    id: str
    user_id: str
    name: str
    device_type: str | None = None
    os: str | None = None
    os_version: str | None = None
    last_sync_at: datetime | None = None
    created_at: datetime | None = None


class AuthStorage(ABC):
    """Abstract base class for auth storage backends.

    Provides a consistent interface for authentication data storage,
    supporting both SQLite (local) and PostgreSQL (production) backends.
    """

    # ==================== User Operations ====================

    @abstractmethod
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
        """Create a new user.

        Args:
            user_id: Unique user ID
            email: User's email address
            name: User's display name
            provider: Auth provider ('email', 'google', 'github')
            password_hash: Hashed password (for email provider)
            email_verified: Whether email is verified
            verification_token: Email verification token
            verification_token_expires: Token expiration time

        Returns:
            Created user object
        """
        ...

    @abstractmethod
    async def get_user_by_id(self, user_id: str) -> User | None:
        """Get user by ID."""
        ...

    @abstractmethod
    async def get_user_by_email(self, email: str) -> User | None:
        """Get user by email address."""
        ...

    @abstractmethod
    async def update_user(
        self,
        user_id: str,
        email_verified: bool | None = None,
        verification_token: str | None = None,
        verification_token_expires: datetime | None = None,
        last_login: datetime | None = None,
        name: str | None = None,
    ) -> User | None:
        """Update user fields.

        Args:
            user_id: User ID to update
            **kwargs: Fields to update

        Returns:
            Updated user or None if not found
        """
        ...

    @abstractmethod
    async def get_user_by_verification_token(self, token: str) -> tuple[User, datetime] | None:
        """Get user by verification token.

        Returns:
            Tuple of (user, token_expires) or None
        """
        ...

    # ==================== API Key Operations ====================

    @abstractmethod
    async def create_api_key(
        self,
        key_id: str,
        user_id: str,
        name: str,
        key_hash: str,
        key_prefix: str,
        encryption_salt: str | None = None,
    ) -> ApiKey:
        """Create a new API key.

        Args:
            key_id: Unique key ID
            user_id: Owner user ID
            name: Key name/description
            key_hash: SHA-256 hash of the key
            key_prefix: First 8 chars of key for identification
            encryption_salt: Optional E2EE salt

        Returns:
            Created API key object (without the raw key)
        """
        ...

    @abstractmethod
    async def get_user_id_by_key_hash(self, key_hash: str) -> str | None:
        """Get user ID by API key hash.

        Args:
            key_hash: SHA-256 hash of the API key

        Returns:
            User ID or None if key not found/inactive
        """
        ...

    @abstractmethod
    async def get_encryption_salt_by_key_hash(self, key_hash: str) -> str | None:
        """Get encryption salt for an API key by its hash.

        Used for automatic E2EE - client derives encryption key from API key + salt.

        Args:
            key_hash: SHA-256 hash of the API key

        Returns:
            Base64-encoded encryption salt or None if not found
        """
        ...

    @abstractmethod
    async def list_api_keys(self, user_id: str) -> list[ApiKey]:
        """List all API keys for a user.

        Args:
            user_id: User ID

        Returns:
            List of API keys (without raw key values)
        """
        ...

    @abstractmethod
    async def update_api_key(
        self,
        key_id: str,
        key_hash: str | None = None,
        key_prefix: str | None = None,
        last_used_at: datetime | None = None,
        encryption_salt: str | None = None,
    ) -> bool:
        """Update API key fields.

        Returns:
            True if updated, False if not found
        """
        ...

    @abstractmethod
    async def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        """Revoke (soft delete) an API key.

        Args:
            key_id: Key ID to revoke
            user_id: Owner user ID (for authorization check)

        Returns:
            True if revoked, False if not found
        """
        ...

    @abstractmethod
    async def delete_api_key(self, key_id: str, user_id: str) -> bool:
        """Permanently delete an API key.

        Args:
            key_id: Key ID to delete
            user_id: Owner user ID (for authorization check)

        Returns:
            True if deleted, False if not found
        """
        ...

    # ==================== Subscription Operations ====================

    @abstractmethod
    async def get_subscription(self, user_id: str) -> Subscription | None:
        """Get user's subscription.

        Args:
            user_id: User ID

        Returns:
            Subscription or None
        """
        ...

    @abstractmethod
    async def create_subscription(
        self,
        sub_id: str,
        user_id: str,
        tier: str = "free",
        device_limit: int = 3,
        memory_limit: int = 10000,
    ) -> Subscription:
        """Create a subscription for user.

        Args:
            sub_id: Subscription ID
            user_id: User ID
            tier: Subscription tier (free, pro, team)
            device_limit: Max devices allowed
            memory_limit: Max memories allowed

        Returns:
            Created subscription
        """
        ...

    @abstractmethod
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
        """Update subscription fields.

        Returns:
            Updated subscription or None if not found
        """
        ...

    # ==================== Device Operations ====================

    @abstractmethod
    async def list_devices(self, user_id: str) -> list[Device]:
        """List all devices for a user.

        Args:
            user_id: User ID

        Returns:
            List of devices
        """
        ...

    @abstractmethod
    async def delete_device(self, device_id: str, user_id: str) -> bool:
        """Delete a device.

        Args:
            device_id: Device ID to delete
            user_id: Owner user ID (for authorization check)

        Returns:
            True if deleted, False if not found
        """
        ...

    @abstractmethod
    async def get_device_count(self, user_id: str) -> int:
        """Get count of devices for user.

        Returns:
            Number of devices
        """
        ...

    # ==================== Connection Management ====================

    @abstractmethod
    async def close(self) -> None:
        """Close database connection."""
        ...
