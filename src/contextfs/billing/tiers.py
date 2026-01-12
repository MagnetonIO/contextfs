"""Subscription tier definitions for ContextFS.

Defines the pricing tiers and their associated limits.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Tier:
    """Represents a subscription tier."""

    name: str
    display_name: str
    price_monthly: int  # In cents
    device_limit: int
    memory_limit: int
    stripe_price_id: str | None = None  # Set via environment

    @property
    def price_display(self) -> str:
        """Format price for display."""
        if self.price_monthly == 0:
            return "Free"
        return f"${self.price_monthly / 100:.0f}/mo"


# Tier definitions matching the plan
TIERS = {
    "free": Tier(
        name="free",
        display_name="Free",
        price_monthly=0,
        device_limit=3,
        memory_limit=10_000,
    ),
    "pro": Tier(
        name="pro",
        display_name="Pro",
        price_monthly=900,  # $9/mo
        device_limit=10,
        memory_limit=100_000,
    ),
    "team": Tier(
        name="team",
        display_name="Team",
        price_monthly=2900,  # $29/mo
        device_limit=-1,  # Unlimited
        memory_limit=-1,  # Unlimited
    ),
}


def get_tier(tier_name: str) -> Tier:
    """Get a tier by name.

    Args:
        tier_name: The tier name (free, pro, team)

    Returns:
        The Tier object

    Raises:
        ValueError: If tier name is invalid
    """
    tier = TIERS.get(tier_name.lower())
    if tier is None:
        raise ValueError(f"Unknown tier: {tier_name}")
    return tier


def get_tier_limits(tier_name: str) -> tuple[int, int]:
    """Get the limits for a tier.

    Args:
        tier_name: The tier name

    Returns:
        Tuple of (device_limit, memory_limit)
        -1 means unlimited
    """
    tier = get_tier(tier_name)
    return tier.device_limit, tier.memory_limit


def check_limit(current: int, limit: int) -> bool:
    """Check if a value is within limit.

    Args:
        current: Current usage count
        limit: The limit (-1 for unlimited)

    Returns:
        True if within limit, False if exceeded
    """
    if limit == -1:  # Unlimited
        return True
    return current < limit


def can_add_device(current_devices: int, tier_name: str) -> bool:
    """Check if a user can add another device.

    Args:
        current_devices: Current device count
        tier_name: User's tier name

    Returns:
        True if can add device, False otherwise
    """
    tier = get_tier(tier_name)
    return check_limit(current_devices, tier.device_limit)


def can_add_memory(current_memories: int, tier_name: str) -> bool:
    """Check if a user can add another memory.

    Args:
        current_memories: Current memory count
        tier_name: User's tier name

    Returns:
        True if can add memory, False otherwise
    """
    tier = get_tier(tier_name)
    return check_limit(current_memories, tier.memory_limit)
