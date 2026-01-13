"""Device management routes."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from contextfs.auth.api_keys import APIKey, User
from service.api.auth_middleware import require_auth
from service.db.models import Device
from service.db.session import get_session_dependency

router = APIRouter(prefix="/api/devices", tags=["devices"])


class DeviceResponse(BaseModel):
    """Device info for frontend."""

    id: str
    name: str
    device_type: str | None
    os: str | None
    os_version: str | None
    is_current: bool
    last_sync_at: str | None
    created_at: str


class DeviceListResponse(BaseModel):
    """List of devices."""

    devices: list[DeviceResponse]


@router.get("", response_model=DeviceListResponse)
async def list_devices(
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
) -> DeviceListResponse:
    """List all devices for the current user."""
    user, _ = auth
    user_id = user.id

    result = await session.execute(
        select(Device).where(Device.user_id == user_id).order_by(Device.registered_at.desc())
    )
    devices = result.scalars().all()

    return DeviceListResponse(
        devices=[
            DeviceResponse(
                id=d.device_id,
                name=d.device_name,
                device_type=d.platform,
                os=d.extra_metadata.get("os") if d.extra_metadata else None,
                os_version=d.extra_metadata.get("os_version") if d.extra_metadata else None,
                is_current=False,  # Could track current device via session
                last_sync_at=d.last_sync_at.isoformat() if d.last_sync_at else None,
                created_at=d.registered_at.isoformat()
                if d.registered_at
                else datetime.now(timezone.utc).isoformat(),
            )
            for d in devices
        ]
    )


@router.delete("/{device_id}")
async def remove_device(
    device_id: str,
    auth: tuple[User, APIKey] = Depends(require_auth),
    session: AsyncSession = Depends(get_session_dependency),
) -> dict:
    """Remove a device."""
    user, _ = auth
    user_id = user.id

    # Verify device belongs to user
    result = await session.execute(
        select(Device).where(Device.device_id == device_id, Device.user_id == user_id)
    )
    device = result.scalar_one_or_none()

    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    await session.execute(delete(Device).where(Device.device_id == device_id))
    await session.commit()

    return {"status": "deleted"}
