"""API module for sync service."""

from service.api.main import app
from service.api.sync_routes import router

__all__ = ["app", "router"]
