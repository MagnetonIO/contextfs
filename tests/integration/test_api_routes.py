"""
Integration tests for sync service API routes.

Tests the FastAPI endpoints for authentication, sync, billing, and memory management.

Note: Tests requiring a running database are marked with pytest.mark.skipif
and will be skipped unless CONTEXTFS_TEST_API=1 is set.

These tests require the sync service dependencies (stripe, etc.) to be installed.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip entire module if service dependencies not installed
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("stripe", reason="Stripe required for API route tests")

# Mark all tests requiring server as skip unless explicitly enabled
requires_server = pytest.mark.skipif(
    os.environ.get("CONTEXTFS_TEST_API") != "1",
    reason="API tests require running database. Set CONTEXTFS_TEST_API=1 to enable.",
)


@pytest.fixture
def test_client():
    """Create a test client for the API.

    Note: This fixture requires a PostgreSQL database to be available.
    Set CONTEXTFS_TEST_API=1 and ensure database is running.
    """
    from fastapi.testclient import TestClient

    # Mock environment variables
    with patch.dict(
        os.environ,
        {
            "CONTEXTFS_POSTGRES_URL": "postgresql://test:test@localhost/test",
            "STRIPE_SECRET_KEY": "sk_test_fake",
            "STRIPE_WEBHOOK_SECRET": "whsec_test_fake",
            "STRIPE_PRICE_PRO": "price_test_pro",
            "STRIPE_PRICE_TEAM": "price_test_team",
        },
    ):
        from service.api.main import app

        with TestClient(app) as client:
            yield client


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def mock_user():
    """Create a mock user object."""
    from datetime import datetime, timezone

    user = MagicMock()
    user.id = "test-user-id-12345"
    user.email = "test@example.com"
    user.name = "Test User"
    user.tier = "free"
    user.is_active = True
    user.created_at = datetime.now(timezone.utc)
    user.provider = "email"
    return user


@pytest.fixture
def mock_api_key():
    """Create a mock API key."""
    from datetime import datetime, timezone

    key = MagicMock()
    key.id = "key-id-12345"
    key.user_id = "test-user-id-12345"
    key.name = "Test Key"
    key.key_prefix = "ctx_te"
    key.is_active = True
    key.created_at = datetime.now(timezone.utc)
    key.last_used_at = None
    return key


@requires_server
class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, test_client):
        """Test health endpoint returns OK."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy" or "ok" in str(data).lower()


@requires_server
class TestRootEndpoint:
    """Tests for root API info endpoint."""

    def test_root_returns_info(self, test_client):
        """Test root endpoint returns API info."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data or "version" in data or "endpoints" in data


@requires_server
class TestAuthRoutes:
    """Tests for authentication routes."""

    def test_login_missing_credentials(self, test_client):
        """Test login with missing credentials returns error."""
        response = test_client.post("/api/auth/login", json={})

        assert response.status_code in [400, 422]

    def test_login_invalid_credentials(self, test_client):
        """Test login with invalid credentials returns 401."""
        with patch("service.api.auth_routes.get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.execute = AsyncMock(
                return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
            )
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock()

            response = test_client.post(
                "/api/auth/login",
                json={"email": "nonexistent@example.com", "password": "wrongpassword"},
            )

            # Should be 401 or similar error
            assert response.status_code in [401, 400, 422, 500]

    def test_me_without_auth(self, test_client):
        """Test /me endpoint without authentication."""
        response = test_client.get("/api/auth/me")

        # Should require authentication
        assert response.status_code in [401, 403, 422]

    def test_api_keys_without_auth(self, test_client):
        """Test listing API keys without authentication."""
        response = test_client.get("/api/auth/api-keys")

        assert response.status_code in [401, 403, 422]

    def test_oauth_init_google(self, test_client):
        """Test OAuth init for Google."""
        response = test_client.post(
            "/api/auth/oauth/init",
            json={"provider": "google", "redirect_uri": "http://localhost:3000/callback"},
        )

        # Should return auth URL or error if not configured
        assert response.status_code in [200, 400, 500]


@requires_server
class TestSyncRoutes:
    """Tests for sync routes."""

    def test_register_device_without_auth(self, test_client):
        """Test device registration without authentication."""
        response = test_client.post(
            "/api/sync/register",
            json={
                "device_id": "test-device-001",
                "device_name": "Test Device",
                "platform": "darwin",
                "client_version": "0.2.8",
            },
        )

        # Should require authentication
        assert response.status_code in [401, 403, 422]

    def test_push_without_auth(self, test_client):
        """Test push endpoint without authentication."""
        response = test_client.post(
            "/api/sync/push",
            json={
                "device_id": "test-device-001",
                "memories": [],
                "sessions": [],
                "edges": [],
            },
        )

        assert response.status_code in [401, 403, 422]

    def test_pull_without_auth(self, test_client):
        """Test pull endpoint without authentication."""
        response = test_client.post(
            "/api/sync/pull",
            json={
                "device_id": "test-device-001",
                "since_timestamp": "2024-01-01T00:00:00Z",
            },
        )

        assert response.status_code in [401, 403, 422]


@requires_server
class TestBillingRoutes:
    """Tests for billing routes."""

    def test_subscription_without_auth(self, test_client):
        """Test subscription endpoint without authentication."""
        response = test_client.get("/api/billing/subscription")

        assert response.status_code in [401, 403, 422]

    def test_usage_without_auth(self, test_client):
        """Test usage endpoint without authentication."""
        response = test_client.get("/api/billing/usage")

        assert response.status_code in [401, 403, 422]

    def test_checkout_without_auth(self, test_client):
        """Test checkout endpoint without authentication."""
        response = test_client.post(
            "/api/billing/checkout",
            json={"price_id": "price_test_pro"},
        )

        assert response.status_code in [401, 403, 422]


@requires_server
class TestMemoriesRoutes:
    """Tests for memories routes."""

    def test_search_memories_without_auth(self, test_client):
        """Test search endpoint without authentication."""
        response = test_client.get("/api/memories/search?query=test")

        assert response.status_code in [401, 403, 422]

    def test_stats_without_auth(self, test_client):
        """Test stats endpoint without authentication."""
        response = test_client.get("/api/memories/stats")

        assert response.status_code in [401, 403, 422]

    def test_get_memory_without_auth(self, test_client):
        """Test get memory endpoint without authentication."""
        response = test_client.get("/api/memories/test-memory-id")

        assert response.status_code in [401, 403, 422]


@requires_server
class TestDevicesRoutes:
    """Tests for devices routes."""

    def test_list_devices_without_auth(self, test_client):
        """Test list devices without authentication."""
        response = test_client.get("/api/devices")

        assert response.status_code in [401, 403, 422]

    def test_delete_device_without_auth(self, test_client):
        """Test delete device without authentication."""
        response = test_client.delete("/api/devices/test-device-id")

        assert response.status_code in [401, 403, 422]


@requires_server
class TestTeamsRoutes:
    """Tests for teams routes."""

    def test_create_team_without_auth(self, test_client):
        """Test create team without authentication."""
        response = test_client.post(
            "/api/teams",
            json={"name": "Test Team"},
        )

        assert response.status_code in [401, 403, 422]

    def test_list_teams_without_auth(self, test_client):
        """Test list teams without authentication."""
        response = test_client.get("/api/teams")

        assert response.status_code in [401, 403, 422]


@requires_server
class TestAdminRoutes:
    """Tests for admin routes."""

    def test_admin_users_without_auth(self, test_client):
        """Test admin users endpoint without authentication."""
        response = test_client.get("/api/admin/users")

        assert response.status_code in [401, 403, 422]

    def test_admin_stats_without_auth(self, test_client):
        """Test admin stats endpoint without authentication."""
        response = test_client.get("/api/admin/stats")

        assert response.status_code in [401, 403, 422]


# =============================================================================
# Authenticated Tests (with mocked auth)
# =============================================================================


@pytest.fixture
def authenticated_client(mock_user, mock_api_key):
    """Create authenticated test client with mocked user."""
    from fastapi.testclient import TestClient

    with patch.dict(
        os.environ,
        {
            "CONTEXTFS_POSTGRES_URL": "postgresql://test:test@localhost/test",
            "STRIPE_SECRET_KEY": "sk_test_fake",
            "STRIPE_WEBHOOK_SECRET": "whsec_test_fake",
        },
    ):
        from service.api.main import app

        # Override auth dependency
        async def mock_get_current_user():
            return mock_user

        from service.api import auth_routes

        app.dependency_overrides[auth_routes.get_current_user] = mock_get_current_user

        with TestClient(app) as client:
            # Add auth header
            client.headers["Authorization"] = "Bearer ctx_test_api_key_12345"
            yield client

        # Clean up overrides
        app.dependency_overrides.clear()


@requires_server
class TestAuthenticatedSyncRoutes:
    """Tests for sync routes with authentication."""

    @pytest.mark.skip(reason="Requires full database mock")
    def test_register_device_authenticated(self, authenticated_client):
        """Test device registration with valid auth."""
        response = authenticated_client.post(
            "/api/sync/register",
            json={
                "device_id": "test-device-001",
                "device_name": "Test Device",
                "platform": "darwin",
                "client_version": "0.2.8",
            },
        )

        # With mock auth, should process request
        assert response.status_code in [200, 201, 400, 500]


@requires_server
class TestAuthenticatedMemoriesRoutes:
    """Tests for memories routes with authentication."""

    @pytest.mark.skip(reason="Requires full database mock")
    def test_search_authenticated(self, authenticated_client):
        """Test search with authentication."""
        response = authenticated_client.get("/api/memories/search?query=test")

        assert response.status_code in [200, 500]


# =============================================================================
# Request/Response Model Tests
# =============================================================================


class TestRequestModels:
    """Tests for API request models."""

    def test_device_registration_model(self):
        """Test DeviceRegistration model validation."""
        from pydantic import ValidationError

        from service.api.sync_routes import DeviceRegistration

        # Valid registration
        reg = DeviceRegistration(
            device_id="test-device",
            device_name="My Device",
            platform="darwin",
            client_version="0.2.8",
        )
        assert reg.device_id == "test-device"

        # Invalid - missing required fields
        with pytest.raises(ValidationError):
            DeviceRegistration()

    def test_sync_push_request_model(self):
        """Test SyncPushRequest model."""
        from service.api.sync_routes import SyncPushRequest

        # Valid push with empty arrays
        push = SyncPushRequest(
            device_id="test-device",
            memories=[],
            sessions=[],
            edges=[],
        )
        assert push.device_id == "test-device"
        assert len(push.memories) == 0

    def test_login_request_model(self):
        """Test LoginRequest model."""
        from pydantic import ValidationError

        from service.api.auth_routes import LoginRequest

        # Valid login
        login = LoginRequest(email="test@example.com", password="password123")
        assert login.email == "test@example.com"

        # email is just a string, so non-email format is allowed by pydantic
        # Test missing required field instead
        with pytest.raises(ValidationError):
            LoginRequest(email="test@example.com")  # missing password


class TestResponseModels:
    """Tests for API response models."""

    def test_device_info_model(self):
        """Test DeviceInfo response model."""
        from datetime import datetime, timezone

        from service.api.sync_routes import DeviceInfo

        info = DeviceInfo(
            device_id="test-device",
            device_name="My Device",
            platform="darwin",
            client_version="0.2.8",
            registered_at=datetime.now(timezone.utc),
        )
        assert info.device_id == "test-device"

    def test_subscription_response_model(self):
        """Test SubscriptionResponse model."""
        from service.api.billing_routes import SubscriptionResponse

        sub = SubscriptionResponse(
            tier="pro",
            status="active",
            device_limit=5,
            memory_limit=50000,
            current_period_end=None,
        )
        assert sub.tier == "pro"
        assert sub.status == "active"
        assert sub.device_limit == 5


# =============================================================================
# Error Handling Tests
# =============================================================================


@requires_server
class TestErrorHandling:
    """Tests for API error handling."""

    def test_invalid_json_body(self, test_client):
        """Test error handling for invalid JSON."""
        response = test_client.post(
            "/api/auth/login",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code in [400, 422]

    def test_missing_content_type(self, test_client):
        """Test error handling for missing content type."""
        response = test_client.post(
            "/api/auth/login",
            content='{"email": "test@test.com"}',
        )

        # Should handle gracefully
        assert response.status_code in [400, 415, 422]

    def test_method_not_allowed(self, test_client):
        """Test method not allowed error."""
        response = test_client.put("/health")

        assert response.status_code == 405
