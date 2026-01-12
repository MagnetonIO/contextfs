"""Stripe service for ContextFS billing.

Handles checkout sessions, customer portal, and subscription management.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

import aiosqlite
import stripe


@dataclass
class Subscription:
    """Represents a user's subscription."""

    id: str
    user_id: str
    tier: str
    stripe_customer_id: str | None
    stripe_subscription_id: str | None
    device_limit: int
    memory_limit: int
    status: str
    current_period_end: datetime | None


class StripeService:
    """Service for Stripe operations."""

    def __init__(self, db_path: str, api_key: str | None = None):
        """Initialize the Stripe service.

        Args:
            db_path: Path to SQLite database
            api_key: Stripe API key (defaults to STRIPE_SECRET_KEY env var)
        """
        self.db_path = db_path
        stripe.api_key = api_key or os.environ.get("STRIPE_SECRET_KEY")

        # Price IDs from environment
        self.price_ids = {
            "pro": os.environ.get("STRIPE_PRICE_PRO"),
            "team": os.environ.get("STRIPE_PRICE_TEAM"),
        }

    async def get_subscription(self, user_id: str) -> Subscription | None:
        """Get a user's current subscription.

        Args:
            user_id: The user's ID

        Returns:
            Subscription object or None if not found
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM subscriptions WHERE user_id = ?",
                (user_id,),
            )
            row = await cursor.fetchone()

            if not row:
                return None

            return Subscription(
                id=row["id"],
                user_id=row["user_id"],
                tier=row["tier"],
                stripe_customer_id=row["stripe_customer_id"],
                stripe_subscription_id=row["stripe_subscription_id"],
                device_limit=row["device_limit"],
                memory_limit=row["memory_limit"],
                status=row["status"],
                current_period_end=(
                    datetime.fromisoformat(row["current_period_end"])
                    if row["current_period_end"]
                    else None
                ),
            )

    async def create_or_get_customer(self, user_id: str, email: str) -> str:
        """Create or get a Stripe customer for a user.

        Args:
            user_id: The user's ID
            email: The user's email

        Returns:
            Stripe customer ID
        """
        # Check if user already has a customer ID
        sub = await self.get_subscription(user_id)
        if sub and sub.stripe_customer_id:
            return sub.stripe_customer_id

        # Create new Stripe customer
        customer = stripe.Customer.create(
            email=email,
            metadata={"user_id": user_id},
        )

        # Update subscription record with customer ID
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO subscriptions (id, user_id, stripe_customer_id)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET stripe_customer_id = ?
                """,
                (str(uuid4()), user_id, customer.id, customer.id),
            )
            await db.commit()

        return customer.id

    async def create_checkout_session(
        self,
        user_id: str,
        email: str,
        tier: str,
        success_url: str,
        cancel_url: str,
    ) -> str:
        """Create a Stripe checkout session.

        Args:
            user_id: The user's ID
            email: The user's email
            tier: The tier to subscribe to (pro or team)
            success_url: URL to redirect on success
            cancel_url: URL to redirect on cancel

        Returns:
            Checkout session URL

        Raises:
            ValueError: If tier is invalid or price not configured
        """
        if tier not in self.price_ids:
            raise ValueError(f"Invalid tier: {tier}")

        price_id = self.price_ids[tier]
        if not price_id:
            raise ValueError(f"Price ID not configured for tier: {tier}")

        customer_id = await self.create_or_get_customer(user_id, email)

        session = stripe.checkout.Session.create(
            customer=customer_id,
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={"user_id": user_id, "tier": tier},
        )

        return session.url

    async def create_portal_session(
        self,
        user_id: str,
        return_url: str,
    ) -> str | None:
        """Create a Stripe customer portal session.

        Args:
            user_id: The user's ID
            return_url: URL to return to after portal

        Returns:
            Portal session URL, or None if no customer
        """
        sub = await self.get_subscription(user_id)
        if not sub or not sub.stripe_customer_id:
            return None

        session = stripe.billing_portal.Session.create(
            customer=sub.stripe_customer_id,
            return_url=return_url,
        )

        return session.url

    async def update_subscription(
        self,
        user_id: str,
        tier: str,
        stripe_subscription_id: str | None = None,
        status: str = "active",
        current_period_end: datetime | None = None,
    ) -> None:
        """Update a user's subscription.

        Args:
            user_id: The user's ID
            tier: The new tier
            stripe_subscription_id: Stripe subscription ID
            status: Subscription status
            current_period_end: When the current period ends
        """
        from contextfs.billing.tiers import get_tier

        tier_obj = get_tier(tier)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                UPDATE subscriptions
                SET tier = ?,
                    stripe_subscription_id = COALESCE(?, stripe_subscription_id),
                    device_limit = ?,
                    memory_limit = ?,
                    status = ?,
                    current_period_end = ?,
                    updated_at = datetime('now')
                WHERE user_id = ?
                """,
                (
                    tier,
                    stripe_subscription_id,
                    tier_obj.device_limit,
                    tier_obj.memory_limit,
                    status,
                    current_period_end.isoformat() if current_period_end else None,
                    user_id,
                ),
            )
            await db.commit()

    async def cancel_subscription(self, user_id: str) -> bool:
        """Cancel a user's subscription (downgrade to free).

        Args:
            user_id: The user's ID

        Returns:
            True if subscription was cancelled
        """
        sub = await self.get_subscription(user_id)
        if not sub or not sub.stripe_subscription_id:
            return False

        # Cancel in Stripe (at period end)
        stripe.Subscription.modify(
            sub.stripe_subscription_id,
            cancel_at_period_end=True,
        )

        # Update status
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE subscriptions SET status = 'canceling' WHERE user_id = ?",
                (user_id,),
            )
            await db.commit()

        return True

    async def initialize_free_subscription(self, user_id: str) -> None:
        """Initialize a free subscription for a new user.

        Args:
            user_id: The user's ID
        """
        from contextfs.billing.tiers import get_tier

        tier = get_tier("free")

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR IGNORE INTO subscriptions
                (id, user_id, tier, device_limit, memory_limit, status)
                VALUES (?, ?, 'free', ?, ?, 'active')
                """,
                (str(uuid4()), user_id, tier.device_limit, tier.memory_limit),
            )
            await db.commit()

    async def check_usage_limits(
        self,
        user_id: str,
        check_devices: bool = False,
        check_memories: bool = False,
    ) -> tuple[bool, str]:
        """Check if user is within their usage limits.

        Args:
            user_id: The user's ID
            check_devices: Check device limit
            check_memories: Check memory limit

        Returns:
            Tuple of (is_allowed, message)
        """
        from contextfs.billing.tiers import can_add_device, can_add_memory

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Get subscription and usage
            cursor = await db.execute(
                """
                SELECT s.tier, u.device_count, u.memory_count
                FROM subscriptions s
                LEFT JOIN usage u ON s.user_id = u.user_id
                WHERE s.user_id = ?
                """,
                (user_id,),
            )
            row = await cursor.fetchone()

            if not row:
                return False, "No subscription found"

            tier = row["tier"]
            device_count = row["device_count"] or 0
            memory_count = row["memory_count"] or 0

            if check_devices and not can_add_device(device_count, tier):
                return False, f"Device limit reached for {tier} tier"

            if check_memories and not can_add_memory(memory_count, tier):
                return False, f"Memory limit reached for {tier} tier"

            return True, "OK"
