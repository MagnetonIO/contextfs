"""Vector clock implementation for sync conflict resolution.

Vector clocks track causality across devices, enabling detection of:
- Stale updates (happens-before relationship)
- Concurrent modifications (conflicts requiring resolution)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, Field


def _ensure_tz_aware(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware (assumes UTC if naive)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


class VectorClock(BaseModel):
    """
    Vector clock for tracking causality across devices.

    Format: {device_id: counter}
    Example: {"laptop-abc123": 5, "desktop-def456": 3}

    Scalability:
    - Prune inactive devices (>30 days) to limit growth
    - Maximum 50 active devices by default
    """

    clock: dict[str, int] = Field(default_factory=dict)

    def increment(self, device_id: str) -> VectorClock:
        """
        Increment counter for a device.

        Args:
            device_id: Unique identifier for the device

        Returns:
            New VectorClock with incremented counter
        """
        new_clock = self.clock.copy()
        new_clock[device_id] = new_clock.get(device_id, 0) + 1
        return VectorClock(clock=new_clock)

    def merge(self, other: VectorClock) -> VectorClock:
        """
        Merge two vector clocks (take max of each component).

        Used after resolving conflicts to combine both versions.

        Args:
            other: Vector clock to merge with

        Returns:
            New VectorClock with merged counters
        """
        merged = self.clock.copy()
        for device_id, counter in other.clock.items():
            merged[device_id] = max(merged.get(device_id, 0), counter)
        return VectorClock(clock=merged)

    def happens_before(self, other: VectorClock) -> bool:
        """
        Check if self happens-before other (self < other).

        Returns True if all components of self are <= other,
        and at least one component is strictly less.

        Args:
            other: Vector clock to compare against

        Returns:
            True if self causally precedes other
        """
        all_keys = set(self.clock) | set(other.clock)

        all_leq = all(self.clock.get(k, 0) <= other.clock.get(k, 0) for k in all_keys)
        any_less = any(self.clock.get(k, 0) < other.clock.get(k, 0) for k in all_keys)

        return all_leq and any_less

    def concurrent_with(self, other: VectorClock) -> bool:
        """
        Check if clocks are concurrent (neither happens-before and not equal).

        Concurrent clocks indicate independent modifications
        that may require conflict resolution.

        Args:
            other: Vector clock to compare against

        Returns:
            True if clocks are concurrent (conflict)
        """
        # Equal clocks are not concurrent
        if self.equal_to(other):
            return False
        return not self.happens_before(other) and not other.happens_before(self)

    def dominates(self, other: VectorClock) -> bool:
        """
        Check if self dominates other (other happens-before self).

        Args:
            other: Vector clock to compare against

        Returns:
            True if self causally follows other
        """
        return other.happens_before(self)

    def equal_to(self, other: VectorClock) -> bool:
        """
        Check if two vector clocks are identical.

        Args:
            other: Vector clock to compare against

        Returns:
            True if clocks are equal
        """
        all_keys = set(self.clock) | set(other.clock)
        return all(self.clock.get(k, 0) == other.clock.get(k, 0) for k in all_keys)

    def prune(
        self,
        active_devices: set[str] | None = None,
        max_devices: int = 50,
    ) -> VectorClock:
        """
        Prune vector clock to limit growth.

        Keeps only specified active devices or limits to top N by counter.

        Args:
            active_devices: Set of device IDs to keep (if provided)
            max_devices: Maximum number of devices to keep (if active_devices not provided)

        Returns:
            New VectorClock with pruned entries
        """
        if active_devices is not None:
            # Keep only specified active devices
            pruned = {k: v for k, v in self.clock.items() if k in active_devices}
        elif len(self.clock) > max_devices:
            # Keep top N devices by counter value
            sorted_devices = sorted(self.clock.items(), key=lambda x: x[1], reverse=True)
            pruned = dict(sorted_devices[:max_devices])
        else:
            pruned = self.clock.copy()

        return VectorClock(clock=pruned)

    def to_dict(self) -> dict[str, int]:
        """Export as dict for JSON storage."""
        return self.clock.copy()

    def to_json(self) -> str:
        """Export as JSON string for storage."""
        import json

        return json.dumps(self.clock, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, int] | None) -> VectorClock:
        """Create from dict."""
        return cls(clock=data or {})

    @classmethod
    def from_json(cls, json_str: str | None) -> VectorClock:
        """Create from JSON string."""
        import json

        if not json_str:
            return cls(clock={})
        return cls(clock=json.loads(json_str))

    def __repr__(self) -> str:
        """String representation."""
        return f"VectorClock({self.clock})"

    def __bool__(self) -> bool:
        """Return True if clock has any entries."""
        return bool(self.clock)


class DeviceTracker(BaseModel):
    """
    Track device activity for vector clock pruning.

    Maintains last-seen timestamps for devices to enable
    pruning of inactive devices from vector clocks.
    """

    devices: dict[str, datetime] = Field(default_factory=dict)
    prune_after_days: int = Field(default=30)

    def update(self, device_id: str, timestamp: datetime | None = None) -> None:
        """Update last-seen timestamp for a device."""
        self.devices[device_id] = timestamp or datetime.now(timezone.utc)

    def get_active_devices(self, as_of: datetime | None = None) -> set[str]:
        """
        Get set of devices active within the prune window.

        Args:
            as_of: Reference timestamp (defaults to now)

        Returns:
            Set of active device IDs
        """
        reference = _ensure_tz_aware(as_of or datetime.now(timezone.utc))
        cutoff = reference - timedelta(days=self.prune_after_days)

        return {
            device_id
            for device_id, last_seen in self.devices.items()
            if _ensure_tz_aware(last_seen) >= cutoff
        }

    def prune_clock(self, clock: VectorClock) -> VectorClock:
        """
        Prune a vector clock to only include active devices.

        Args:
            clock: Vector clock to prune

        Returns:
            Pruned vector clock
        """
        active = self.get_active_devices()
        return clock.prune(active_devices=active)

    def to_dict(self) -> dict[str, Any]:
        """Export as dict for storage."""
        return {
            "devices": {k: v.isoformat() for k, v in self.devices.items()},
            "prune_after_days": self.prune_after_days,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DeviceTracker:
        """Create from dict."""
        if not data:
            return cls()

        devices = {}
        for device_id, timestamp_str in data.get("devices", {}).items():
            if isinstance(timestamp_str, str):
                devices[device_id] = datetime.fromisoformat(timestamp_str)
            else:
                devices[device_id] = timestamp_str

        return cls(
            devices=devices,
            prune_after_days=data.get("prune_after_days", 30),
        )
