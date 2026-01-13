"""Billing API routes for ContextFS.

Handles Stripe checkout, portal, webhooks, and subscription queries.
All data stored in Postgres.
"""

import os
from datetime import datetime, timezone
from uuid import uuid4

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from contextfs.auth.api_keys import APIKey, User
from service.api.auth_middleware import require_auth
from service.db.models import Device, SubscriptionModel, SyncedMemoryModel
from service.db.session import get_session_dependency

router = APIRouter(prefix="/api/billing", tags=["billing"])


# =============================================================================
# Pydantic Models
# =============================================================================


class CheckoutRequest(BaseModel):
    """Request to create a checkout session."""

    tier: str
    success_url: str
    cancel_url: str


class CheckoutResponse(BaseModel):
    """Response with checkout session URL."""

    checkout_url: str


class PortalRequest(BaseModel):
    """Request to create a portal session."""

    return_url: str


class PortalResponse(BaseModel):
    """Response with portal session URL."""

    portal_url: str


class SubscriptionResponse(BaseModel):
    """Current subscription details."""

    tier: str
    status: str
    device_limit: int
    memory_limit: int
    current_period_end: str | None


class UsageResponse(BaseModel):
    """Current usage statistics."""

    device_count: int
    memory_count: int
    device_limit: int
    memory_limit: int
    device_usage_percent: float
    memory_usage_percent: float


# =============================================================================
# Stripe Configuration
# =============================================================================


TIER_PRICE_IDS = {
    "pro": os.environ.get("STRIPE_PRO_PRICE_ID"),
    "team": os.environ.get("STRIPE_TEAM_PRICE_ID"),
}

TIER_LIMITS = {
    "free": {"device_limit": 3, "memory_limit": 10000},
    "pro": {"device_limit": 10, "memory_limit": 100000},
    "team": {"device_limit": -1, "memory_limit": -1},  # Unlimited
    "admin": {"device_limit": -1, "memory_limit": -1},  # Unlimited
}


def init_stripe():
    """Initialize Stripe with API key."""
    stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")


# =============================================================================
# Routes
# =============================================================================


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout(
    request: CheckoutRequest,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Create a Stripe checkout session for subscription upgrade."""
    init_stripe()
    user, _ = auth

    if request.tier not in ("pro", "team"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid tier. Must be 'pro' or 'team'",
        )

    price_id = TIER_PRICE_IDS.get(request.tier)
    if not price_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Stripe not configured for this tier",
        )

    # Get or create Stripe customer
    result = await session.execute(
        select(SubscriptionModel).where(SubscriptionModel.user_id == user.id)
    )
    sub = result.scalar_one_or_none()

    if sub and sub.stripe_customer_id:
        customer_id = sub.stripe_customer_id
    else:
        # Create Stripe customer
        customer = stripe.Customer.create(email=user.email)
        customer_id = customer.id

        if sub:
            sub.stripe_customer_id = customer_id
        else:
            sub = SubscriptionModel(
                id=str(uuid4()),
                user_id=user.id,
                tier="free",
                stripe_customer_id=customer_id,
            )
            session.add(sub)
        await session.commit()

    # Create checkout session
    checkout_session = stripe.checkout.Session.create(
        customer=customer_id,
        payment_method_types=["card"],
        line_items=[{"price": price_id, "quantity": 1}],
        mode="subscription",
        success_url=request.success_url,
        cancel_url=request.cancel_url,
        metadata={"user_id": user.id, "tier": request.tier},
    )

    return CheckoutResponse(checkout_url=checkout_session.url)


@router.post("/portal", response_model=PortalResponse)
async def create_portal(
    request: PortalRequest,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Create a Stripe customer portal session."""
    init_stripe()
    user, _ = auth

    result = await session.execute(
        select(SubscriptionModel).where(SubscriptionModel.user_id == user.id)
    )
    sub = result.scalar_one_or_none()

    if not sub or not sub.stripe_customer_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No billing account found. Subscribe first.",
        )

    portal_session = stripe.billing_portal.Session.create(
        customer=sub.stripe_customer_id,
        return_url=request.return_url,
    )

    return PortalResponse(portal_url=portal_session.url)


@router.get("/subscription", response_model=SubscriptionResponse)
async def get_subscription(
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Get current subscription details."""
    user, _ = auth

    result = await session.execute(
        select(SubscriptionModel).where(SubscriptionModel.user_id == user.id)
    )
    sub = result.scalar_one_or_none()

    if not sub:
        return SubscriptionResponse(
            tier="free",
            status="active",
            device_limit=3,
            memory_limit=10000,
            current_period_end=None,
        )

    return SubscriptionResponse(
        tier=sub.tier,
        status=sub.status,
        device_limit=sub.device_limit,
        memory_limit=sub.memory_limit,
        current_period_end=sub.current_period_end.isoformat() if sub.current_period_end else None,
    )


@router.get("/usage", response_model=UsageResponse)
async def get_usage(
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Get current usage statistics."""
    from sqlalchemy import func

    user, _ = auth

    # Get subscription for limits
    sub_result = await session.execute(
        select(SubscriptionModel).where(SubscriptionModel.user_id == user.id)
    )
    sub = sub_result.scalar_one_or_none()
    device_limit = sub.device_limit if sub else 3
    memory_limit = sub.memory_limit if sub else 10000

    # Count devices for this user
    device_result = await session.execute(
        select(func.count(Device.device_id)).where(Device.user_id == user.id)
    )
    device_count = device_result.scalar() or 0

    # Count memories for this user (excluding deleted)
    memory_result = await session.execute(
        select(func.count(SyncedMemoryModel.id)).where(
            SyncedMemoryModel.user_id == user.id,
            SyncedMemoryModel.deleted_at.is_(None),
        )
    )
    memory_count = memory_result.scalar() or 0

    # Calculate percentages (handle unlimited = -1)
    device_percent = (device_count / device_limit * 100) if device_limit > 0 else 0
    memory_percent = (memory_count / memory_limit * 100) if memory_limit > 0 else 0

    return UsageResponse(
        device_count=device_count,
        memory_count=memory_count,
        device_limit=device_limit,
        memory_limit=memory_limit,
        device_usage_percent=round(device_percent, 1),
        memory_usage_percent=round(memory_percent, 1),
    )


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    session: AsyncSession = Depends(get_session_dependency),
):
    """Handle Stripe webhook events."""
    init_stripe()

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except stripe.error.SignatureVerificationError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid webhook signature",
        )

    # Handle subscription events
    if event["type"] == "checkout.session.completed":
        checkout = event["data"]["object"]
        user_id = checkout["metadata"].get("user_id")
        tier = checkout["metadata"].get("tier", "pro")
        subscription_id = checkout.get("subscription")
        customer_id = checkout.get("customer")

        if user_id and subscription_id:
            limits = TIER_LIMITS.get(tier, TIER_LIMITS["pro"])

            result = await session.execute(
                select(SubscriptionModel).where(SubscriptionModel.user_id == user_id)
            )
            sub = result.scalar_one_or_none()

            if sub:
                sub.tier = tier
                sub.status = "active"
                sub.stripe_subscription_id = subscription_id
                sub.stripe_customer_id = customer_id
                sub.device_limit = limits["device_limit"]
                sub.memory_limit = limits["memory_limit"]
                sub.updated_at = datetime.now(timezone.utc)
            else:
                sub = SubscriptionModel(
                    id=str(uuid4()),
                    user_id=user_id,
                    tier=tier,
                    status="active",
                    stripe_subscription_id=subscription_id,
                    stripe_customer_id=customer_id,
                    device_limit=limits["device_limit"],
                    memory_limit=limits["memory_limit"],
                )
                session.add(sub)

            await session.commit()

    elif event["type"] == "customer.subscription.deleted":
        subscription = event["data"]["object"]
        subscription_id = subscription["id"]

        result = await session.execute(
            select(SubscriptionModel).where(
                SubscriptionModel.stripe_subscription_id == subscription_id
            )
        )
        sub = result.scalar_one_or_none()

        if sub:
            sub.tier = "free"
            sub.status = "canceled"
            sub.device_limit = 3
            sub.memory_limit = 10000
            sub.stripe_subscription_id = None
            sub.updated_at = datetime.now(timezone.utc)
            await session.commit()

    elif event["type"] == "customer.subscription.updated":
        subscription = event["data"]["object"]
        subscription_id = subscription["id"]
        status_val = subscription["status"]
        period_end = subscription.get("current_period_end")

        result = await session.execute(
            select(SubscriptionModel).where(
                SubscriptionModel.stripe_subscription_id == subscription_id
            )
        )
        sub = result.scalar_one_or_none()

        if sub:
            sub.status = status_val
            if period_end:
                sub.current_period_end = datetime.fromtimestamp(period_end, tz=timezone.utc)
            sub.updated_at = datetime.now(timezone.utc)
            await session.commit()

    return {"status": "success"}


@router.post("/cancel")
async def cancel_subscription(
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
):
    """Cancel subscription (will downgrade to free at period end)."""
    init_stripe()
    user, _ = auth

    result = await session.execute(
        select(SubscriptionModel).where(SubscriptionModel.user_id == user.id)
    )
    sub = result.scalar_one_or_none()

    if not sub or not sub.stripe_subscription_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active subscription to cancel",
        )

    # Cancel at period end
    stripe.Subscription.modify(
        sub.stripe_subscription_id,
        cancel_at_period_end=True,
    )

    sub.status = "canceling"
    sub.updated_at = datetime.now(timezone.utc)
    await session.commit()

    return {"status": "canceling", "message": "Subscription will cancel at period end"}
