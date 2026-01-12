"""Stripe webhook handlers for ContextFS.

Handles subscription lifecycle events from Stripe.
"""

import os
from datetime import datetime
from typing import TYPE_CHECKING

import stripe

if TYPE_CHECKING:
    from contextfs.billing.stripe_service import StripeService


class WebhookHandler:
    """Handler for Stripe webhooks."""

    def __init__(self, stripe_service: "StripeService"):
        """Initialize the webhook handler.

        Args:
            stripe_service: The StripeService instance for database operations
        """
        self.stripe_service = stripe_service
        self.webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")

    def verify_signature(self, payload: bytes, sig_header: str) -> dict | None:
        """Verify webhook signature and parse event.

        Args:
            payload: Raw request body
            sig_header: Stripe-Signature header value

        Returns:
            Parsed event dict, or None if verification fails
        """
        if not self.webhook_secret:
            # In development, skip signature verification
            import json

            return json.loads(payload)

        try:
            event = stripe.Webhook.construct_event(
                payload,
                sig_header,
                self.webhook_secret,
            )
            return event
        except (ValueError, stripe.error.SignatureVerificationError):
            return None

    async def handle_event(self, event: dict) -> bool:
        """Route and handle a webhook event.

        Args:
            event: Parsed Stripe event

        Returns:
            True if handled successfully, False otherwise
        """
        event_type = event.get("type")
        handlers = {
            "checkout.session.completed": self._handle_checkout_completed,
            "customer.subscription.created": self._handle_subscription_created,
            "customer.subscription.updated": self._handle_subscription_updated,
            "customer.subscription.deleted": self._handle_subscription_deleted,
            "invoice.payment_failed": self._handle_payment_failed,
            "invoice.payment_succeeded": self._handle_payment_succeeded,
        }

        handler = handlers.get(event_type)
        if handler:
            return await handler(event["data"]["object"])

        # Unknown event type - log but don't fail
        return True

    async def _handle_checkout_completed(self, session: dict) -> bool:
        """Handle successful checkout.

        Activates the subscription after checkout completes.
        """
        user_id = session.get("metadata", {}).get("user_id")
        tier = session.get("metadata", {}).get("tier")

        if not user_id or not tier:
            return False

        subscription_id = session.get("subscription")

        await self.stripe_service.update_subscription(
            user_id=user_id,
            tier=tier,
            stripe_subscription_id=subscription_id,
            status="active",
        )

        return True

    async def _handle_subscription_created(self, subscription: dict) -> bool:
        """Handle new subscription creation."""
        _customer_id = subscription.get("customer")  # noqa: F841
        _subscription_id = subscription.get("id")  # noqa: F841
        _status = subscription.get("status")  # noqa: F841

        # Look up user by customer ID and update
        # This is handled by checkout.session.completed in most cases
        return True

    async def _handle_subscription_updated(self, subscription: dict) -> bool:
        """Handle subscription updates (upgrades, downgrades, renewals)."""
        customer_id = subscription.get("customer")
        subscription_id = subscription.get("id")
        status = subscription.get("status")
        current_period_end = subscription.get("current_period_end")

        # Get tier from price
        items = subscription.get("items", {}).get("data", [])
        if items:
            price_id = items[0].get("price", {}).get("id")
            tier = self._tier_from_price(price_id)
        else:
            tier = "free"

        # Find user by customer ID
        user_id = await self._get_user_by_customer(customer_id)
        if not user_id:
            return False

        period_end = None
        if current_period_end:
            period_end = datetime.fromtimestamp(current_period_end)

        await self.stripe_service.update_subscription(
            user_id=user_id,
            tier=tier,
            stripe_subscription_id=subscription_id,
            status=status,
            current_period_end=period_end,
        )

        return True

    async def _handle_subscription_deleted(self, subscription: dict) -> bool:
        """Handle subscription cancellation (downgrade to free)."""
        customer_id = subscription.get("customer")

        user_id = await self._get_user_by_customer(customer_id)
        if not user_id:
            return False

        await self.stripe_service.update_subscription(
            user_id=user_id,
            tier="free",
            status="active",  # Active on free tier
        )

        return True

    async def _handle_payment_failed(self, invoice: dict) -> bool:
        """Handle failed payment (mark subscription as past_due)."""
        customer_id = invoice.get("customer")

        user_id = await self._get_user_by_customer(customer_id)
        if not user_id:
            return False

        # Get current subscription
        sub = await self.stripe_service.get_subscription(user_id)
        if sub:
            await self.stripe_service.update_subscription(
                user_id=user_id,
                tier=sub.tier,
                status="past_due",
            )

        return True

    async def _handle_payment_succeeded(self, invoice: dict) -> bool:
        """Handle successful payment (reactivate if was past_due)."""
        customer_id = invoice.get("customer")

        user_id = await self._get_user_by_customer(customer_id)
        if not user_id:
            return False

        sub = await self.stripe_service.get_subscription(user_id)
        if sub and sub.status == "past_due":
            await self.stripe_service.update_subscription(
                user_id=user_id,
                tier=sub.tier,
                status="active",
            )

        return True

    def _tier_from_price(self, price_id: str) -> str:
        """Map a Stripe price ID to a tier name."""
        price_map = {
            os.environ.get("STRIPE_PRICE_PRO"): "pro",
            os.environ.get("STRIPE_PRICE_TEAM"): "team",
        }
        return price_map.get(price_id, "free")

    async def _get_user_by_customer(self, customer_id: str) -> str | None:
        """Look up user ID by Stripe customer ID."""
        import aiosqlite

        async with aiosqlite.connect(self.stripe_service.db_path) as db:
            cursor = await db.execute(
                "SELECT user_id FROM subscriptions WHERE stripe_customer_id = ?",
                (customer_id,),
            )
            row = await cursor.fetchone()
            return row[0] if row else None
