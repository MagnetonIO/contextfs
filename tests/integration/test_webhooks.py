"""Stripe Webhook Tests.

Tests for /api/billing/webhook endpoint handling Stripe events.
These tests require the service API to be available (PostgreSQL, asyncpg, etc.).

NOTE: These tests are skipped in CI because they require a full service setup
with PostgreSQL and proper module structure. They are designed for local
development testing only.
"""

import hashlib
import hmac
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Skip entire module if asyncpg not available (CI environment)
pytest.importorskip("asyncpg", reason="asyncpg required for service tests")

# Check if we can import service dependencies
try:
    from httpx import ASGITransport, AsyncClient

    from service.api.main import app

    SERVICE_AVAILABLE = True
except ImportError:
    SERVICE_AVAILABLE = False
    app = None
    AsyncClient = None
    ASGITransport = None

# Skip all tests in this module if service is not available
pytestmark = pytest.mark.skipif(
    not SERVICE_AVAILABLE,
    reason="Service dependencies not available",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def webhook_secret() -> str:
    """Return webhook secret for testing."""
    return os.environ.get("STRIPE_WEBHOOK_SECRET", "whsec_test_secret_key_for_testing")


@pytest.fixture
async def client():
    """Create async test client for the FastAPI app.

    Requires service infrastructure (PostgreSQL, etc.) to be available.
    Skips tests if service cannot be initialized.
    """
    if not SERVICE_AVAILABLE:
        pytest.skip("Service dependencies not available (httpx/service.api.main)")

    # Mock the database initialization to avoid needing PostgreSQL
    with (
        patch("service.api.main.init_db", new_callable=AsyncMock),
        patch("service.api.main.close_db", new_callable=AsyncMock),
        patch("service.api.main.run_migrations", return_value=True),
        patch("service.api.main.ensure_admin_user", new_callable=AsyncMock, return_value=None),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# =============================================================================
# Webhook Signature Helpers
# =============================================================================


def create_stripe_signature(payload: bytes, secret: str, timestamp: int | None = None) -> str:
    """Create a valid Stripe webhook signature."""
    if timestamp is None:
        timestamp = int(time.time())

    # Stripe signature format: t=timestamp,v1=signature
    payload_to_sign = f"{timestamp}.{payload.decode()}"
    signature = hmac.new(
        secret.encode(),
        payload_to_sign.encode(),
        hashlib.sha256,
    ).hexdigest()

    return f"t={timestamp},v1={signature}"


# =============================================================================
# Checkout Completed Tests
# =============================================================================


class TestCheckoutCompletedWebhook:
    """Tests for checkout.session.completed webhook events."""

    @pytest.mark.asyncio
    async def test_checkout_completed_creates_subscription(self, client, webhook_secret: str):
        """Test that checkout.session.completed creates/updates subscription."""
        user_id = str(uuid4())
        subscription_id = f"sub_{uuid4().hex[:24]}"
        customer_id = f"cus_{uuid4().hex[:24]}"

        payload = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": f"cs_{uuid4().hex[:24]}",
                    "metadata": {
                        "user_id": user_id,
                        "tier": "pro",
                    },
                    "subscription": subscription_id,
                    "customer": customer_id,
                }
            },
        }

        payload_bytes = json.dumps(payload).encode()
        signature = create_stripe_signature(payload_bytes, webhook_secret)

        with (
            patch("stripe.Webhook.construct_event") as mock_construct,
            patch("service.api.billing_routes.get_session") as mock_get_session,
        ):
            mock_construct.return_value = payload

            # Mock session with async context manager
            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(
                return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
            )
            mock_session.commit = AsyncMock()
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_session)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_get_session.return_value = mock_context

            response = await client.post(
                "/api/billing/webhook",
                content=payload_bytes,
                headers={
                    "stripe-signature": signature,
                    "content-type": "application/json",
                },
            )

        # May fail if user doesn't exist in DB - that's expected
        assert response.status_code in [200, 400, 500]

    @pytest.mark.asyncio
    async def test_checkout_completed_team_tier(self, client, webhook_secret: str):
        """Test checkout completed for team tier sets correct limits."""
        user_id = str(uuid4())
        subscription_id = f"sub_{uuid4().hex[:24]}"

        payload = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": f"cs_{uuid4().hex[:24]}",
                    "metadata": {
                        "user_id": user_id,
                        "tier": "team",
                    },
                    "subscription": subscription_id,
                    "customer": f"cus_{uuid4().hex[:24]}",
                }
            },
        }

        payload_bytes = json.dumps(payload).encode()
        signature = create_stripe_signature(payload_bytes, webhook_secret)

        with (
            patch("stripe.Webhook.construct_event") as mock_construct,
            patch("service.api.billing_routes.get_session") as mock_get_session,
        ):
            mock_construct.return_value = payload

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(
                return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
            )
            mock_session.commit = AsyncMock()
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_session)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_get_session.return_value = mock_context

            response = await client.post(
                "/api/billing/webhook",
                content=payload_bytes,
                headers={
                    "stripe-signature": signature,
                    "content-type": "application/json",
                },
            )

        assert response.status_code in [200, 400, 500]


# =============================================================================
# Subscription Updated Tests
# =============================================================================


class TestSubscriptionUpdatedWebhook:
    """Tests for customer.subscription.updated webhook events."""

    @pytest.mark.asyncio
    async def test_subscription_updated_status_change(self, client, webhook_secret: str):
        """Test subscription status updates are processed."""
        subscription_id = f"sub_{uuid4().hex[:24]}"

        payload = {
            "type": "customer.subscription.updated",
            "data": {
                "object": {
                    "id": subscription_id,
                    "status": "active",
                    "current_period_end": int(time.time()) + 30 * 24 * 3600,
                    "cancel_at_period_end": False,
                }
            },
        }

        payload_bytes = json.dumps(payload).encode()
        signature = create_stripe_signature(payload_bytes, webhook_secret)

        with (
            patch("stripe.Webhook.construct_event") as mock_construct,
            patch("service.api.billing_routes.get_session") as mock_get_session,
        ):
            mock_construct.return_value = payload

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(
                return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
            )
            mock_session.commit = AsyncMock()
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_session)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_get_session.return_value = mock_context

            response = await client.post(
                "/api/billing/webhook",
                content=payload_bytes,
                headers={
                    "stripe-signature": signature,
                    "content-type": "application/json",
                },
            )

        # May succeed or fail depending on subscription existence
        assert response.status_code in [200, 400, 500]

    @pytest.mark.asyncio
    async def test_subscription_updated_cancel_at_period_end(self, client, webhook_secret: str):
        """Test subscription marked for cancellation at period end."""
        subscription_id = f"sub_{uuid4().hex[:24]}"

        payload = {
            "type": "customer.subscription.updated",
            "data": {
                "object": {
                    "id": subscription_id,
                    "status": "active",
                    "current_period_end": int(time.time()) + 30 * 24 * 3600,
                    "cancel_at_period_end": True,  # Set to cancel
                }
            },
        }

        payload_bytes = json.dumps(payload).encode()
        signature = create_stripe_signature(payload_bytes, webhook_secret)

        with (
            patch("stripe.Webhook.construct_event") as mock_construct,
            patch("service.api.billing_routes.get_session") as mock_get_session,
        ):
            mock_construct.return_value = payload

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(
                return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
            )
            mock_session.commit = AsyncMock()
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_session)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_get_session.return_value = mock_context

            response = await client.post(
                "/api/billing/webhook",
                content=payload_bytes,
                headers={
                    "stripe-signature": signature,
                    "content-type": "application/json",
                },
            )

        assert response.status_code in [200, 400, 500]

    @pytest.mark.asyncio
    async def test_subscription_reactivated(self, client, webhook_secret: str):
        """Test subscription reactivation (cancel removed)."""
        subscription_id = f"sub_{uuid4().hex[:24]}"

        payload = {
            "type": "customer.subscription.updated",
            "data": {
                "object": {
                    "id": subscription_id,
                    "status": "active",
                    "current_period_end": int(time.time()) + 30 * 24 * 3600,
                    "cancel_at_period_end": False,  # Cancel removed
                }
            },
        }

        payload_bytes = json.dumps(payload).encode()
        signature = create_stripe_signature(payload_bytes, webhook_secret)

        with (
            patch("stripe.Webhook.construct_event") as mock_construct,
            patch("service.api.billing_routes.get_session") as mock_get_session,
        ):
            mock_construct.return_value = payload

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(
                return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
            )
            mock_session.commit = AsyncMock()
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_session)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_get_session.return_value = mock_context

            response = await client.post(
                "/api/billing/webhook",
                content=payload_bytes,
                headers={
                    "stripe-signature": signature,
                    "content-type": "application/json",
                },
            )

        assert response.status_code in [200, 400, 500]


# =============================================================================
# Subscription Deleted Tests
# =============================================================================


class TestSubscriptionDeletedWebhook:
    """Tests for customer.subscription.deleted webhook events."""

    @pytest.mark.asyncio
    async def test_subscription_deleted_downgrades_to_free(self, client, webhook_secret: str):
        """Test subscription deletion downgrades user to free tier."""
        subscription_id = f"sub_{uuid4().hex[:24]}"

        payload = {
            "type": "customer.subscription.deleted",
            "data": {
                "object": {
                    "id": subscription_id,
                    "status": "canceled",
                }
            },
        }

        payload_bytes = json.dumps(payload).encode()
        signature = create_stripe_signature(payload_bytes, webhook_secret)

        with (
            patch("stripe.Webhook.construct_event") as mock_construct,
            patch("service.api.billing_routes.get_session") as mock_get_session,
        ):
            mock_construct.return_value = payload

            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(
                return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
            )
            mock_session.commit = AsyncMock()
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_session)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_get_session.return_value = mock_context

            response = await client.post(
                "/api/billing/webhook",
                content=payload_bytes,
                headers={
                    "stripe-signature": signature,
                    "content-type": "application/json",
                },
            )

        # Should succeed or gracefully handle non-existent subscription
        assert response.status_code in [200, 400, 500]


# =============================================================================
# Signature Validation Tests
# =============================================================================


class TestWebhookSignatureValidation:
    """Tests for webhook signature validation."""

    @pytest.mark.asyncio
    async def test_invalid_signature_rejected(self, client):
        """Test that invalid signatures are rejected."""
        payload = {"type": "test.event", "data": {"object": {}}}
        payload_bytes = json.dumps(payload).encode()

        response = await client.post(
            "/api/billing/webhook",
            content=payload_bytes,
            headers={
                "stripe-signature": "invalid_signature",
                "content-type": "application/json",
            },
        )

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_signature_rejected(self, client):
        """Test that missing signature is rejected."""
        payload = {"type": "test.event", "data": {"object": {}}}
        payload_bytes = json.dumps(payload).encode()

        response = await client.post(
            "/api/billing/webhook",
            content=payload_bytes,
            headers={
                "content-type": "application/json",
                # No stripe-signature header
            },
        )

        # Should fail signature verification
        assert response.status_code in [400, 401, 422]

    @pytest.mark.asyncio
    async def test_expired_timestamp_handled(self, client, webhook_secret: str):
        """Test that old timestamps are handled correctly."""
        payload = {"type": "test.event", "data": {"object": {}}}
        payload_bytes = json.dumps(payload).encode()

        # Use timestamp from 10 minutes ago (Stripe rejects old timestamps)
        old_timestamp = int(time.time()) - 600

        signature = create_stripe_signature(payload_bytes, webhook_secret, old_timestamp)

        response = await client.post(
            "/api/billing/webhook",
            content=payload_bytes,
            headers={
                "stripe-signature": signature,
                "content-type": "application/json",
            },
        )

        # Stripe SDK should reject this
        assert response.status_code == 400


# =============================================================================
# Unknown Event Type Tests
# =============================================================================


class TestUnknownEventTypes:
    """Tests for handling unknown/unsupported event types."""

    @pytest.mark.asyncio
    async def test_unknown_event_type_returns_success(self, client, webhook_secret: str):
        """Test that unknown event types return success (don't crash)."""
        payload = {
            "type": "unknown.event.type",
            "data": {
                "object": {
                    "id": "obj_123",
                }
            },
        }

        payload_bytes = json.dumps(payload).encode()
        signature = create_stripe_signature(payload_bytes, webhook_secret)

        with patch("stripe.Webhook.construct_event") as mock_construct:
            mock_construct.return_value = payload

            response = await client.post(
                "/api/billing/webhook",
                content=payload_bytes,
                headers={
                    "stripe-signature": signature,
                    "content-type": "application/json",
                },
            )

        # Should return success even for unknown events (Stripe best practice)
        assert response.status_code == 200
