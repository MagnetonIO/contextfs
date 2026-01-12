"""Billing API routes for ContextFS.

Handles Stripe checkout, portal, webhooks, and subscription queries.
"""

import os

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from contextfs.auth import APIKey, User, require_auth
from contextfs.billing import StripeService, WebhookHandler

router = APIRouter(prefix="/api/billing", tags=["billing"])


# Pydantic models for request/response
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


# Dependency to get StripeService
def get_stripe_service() -> StripeService:
    """Get StripeService instance."""
    db_path = os.environ.get("CONTEXTFS_DB_PATH", "contextfs.db")
    return StripeService(db_path)


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout(
    request: CheckoutRequest,
    auth: tuple[User, APIKey] = Depends(require_auth),
    stripe_service: StripeService = Depends(get_stripe_service),
):
    """Create a Stripe checkout session for subscription upgrade."""
    user, api_key = auth

    if request.tier not in ("pro", "team"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid tier. Must be 'pro' or 'team'",
        )

    try:
        checkout_url = await stripe_service.create_checkout_session(
            user_id=user.id,
            email=user.email,
            tier=request.tier,
            success_url=request.success_url,
            cancel_url=request.cancel_url,
        )
        return CheckoutResponse(checkout_url=checkout_url)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/portal", response_model=PortalResponse)
async def create_portal(
    request: PortalRequest,
    auth: tuple[User, APIKey] = Depends(require_auth),
    stripe_service: StripeService = Depends(get_stripe_service),
):
    """Create a Stripe customer portal session."""
    user, api_key = auth

    portal_url = await stripe_service.create_portal_session(
        user_id=user.id,
        return_url=request.return_url,
    )

    if not portal_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No billing account found. Subscribe first.",
        )

    return PortalResponse(portal_url=portal_url)


@router.get("/subscription", response_model=SubscriptionResponse)
async def get_subscription(
    auth: tuple[User, APIKey] = Depends(require_auth),
    stripe_service: StripeService = Depends(get_stripe_service),
):
    """Get current subscription details."""
    user, api_key = auth

    sub = await stripe_service.get_subscription(user.id)

    if not sub:
        # Return free tier defaults
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
    stripe_service: StripeService = Depends(get_stripe_service),
):
    """Get current usage statistics."""
    import aiosqlite

    user, api_key = auth

    sub = await stripe_service.get_subscription(user.id)
    device_limit = sub.device_limit if sub else 3
    memory_limit = sub.memory_limit if sub else 10000

    # Get usage from database
    db_path = os.environ.get("CONTEXTFS_DB_PATH", "contextfs.db")
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT device_count, memory_count FROM usage WHERE user_id = ?",
            (user.id,),
        )
        row = await cursor.fetchone()

    device_count = row[0] if row else 0
    memory_count = row[1] if row else 0

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
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    # Create services
    db_path = os.environ.get("CONTEXTFS_DB_PATH", "contextfs.db")
    stripe_service = StripeService(db_path)
    webhook_handler = WebhookHandler(stripe_service)

    # Verify and parse event
    event = webhook_handler.verify_signature(payload, sig_header)
    if not event:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid webhook signature",
        )

    # Handle the event
    success = await webhook_handler.handle_event(event)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to handle webhook event",
        )

    return {"status": "success"}


@router.post("/cancel")
async def cancel_subscription(
    auth: tuple[User, APIKey] = Depends(require_auth),
    stripe_service: StripeService = Depends(get_stripe_service),
):
    """Cancel subscription (will downgrade to free at period end)."""
    user, api_key = auth

    success = await stripe_service.cancel_subscription(user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active subscription to cancel",
        )

    return {"status": "canceling", "message": "Subscription will cancel at period end"}
