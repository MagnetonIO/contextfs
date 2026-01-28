"""Tests for authentication race condition fixes.

Tests the atomic upsert for _get_or_create_user and atomic session
key creation with limits in _create_session_key.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

# Add service directory to path for imports
service_dir = Path(__file__).parent.parent.parent.parent / "service"
if str(service_dir) not in sys.path:
    sys.path.insert(0, str(service_dir))


class TestGetOrCreateUserAtomic:
    """Tests for atomic user creation."""

    @pytest.mark.asyncio
    async def test_atomic_upsert_creates_new_user(self):
        """Test that _get_or_create_user creates a new user atomically."""
        # Import here to avoid circular imports during test collection
        from service.api.auth_routes import _get_or_create_user

        # Create a mock session
        mock_session = AsyncMock()

        # Mock the execute result for INSERT ... ON CONFLICT
        mock_row = MagicMock()
        mock_row.__getitem__ = lambda _, idx: (
            str(uuid4()) if idx == 0 else datetime.now(timezone.utc)
        )
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_result.scalar_one_or_none.return_value = None  # No existing subscription

        # First call returns the upsert result, second returns no subscription
        mock_session.execute = AsyncMock(side_effect=[mock_result, mock_result])

        email = "test@example.com"
        name = "Test User"

        user_id, is_new = await _get_or_create_user(
            mock_session, email, name, "google", "google_123"
        )

        assert user_id is not None
        assert mock_session.execute.called
        assert mock_session.commit.called

    @pytest.mark.asyncio
    async def test_atomic_upsert_updates_existing_user(self):
        """Test that _get_or_create_user updates an existing user."""
        from service.api.auth_routes import _get_or_create_user

        mock_session = AsyncMock()

        # Existing user - created_at is in the past
        existing_user_id = str(uuid4())
        past_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

        mock_row = MagicMock()
        mock_row.__getitem__ = lambda _, idx: existing_user_id if idx == 0 else past_time
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_row

        mock_session.execute = AsyncMock(return_value=mock_result)

        email = "existing@example.com"
        user_id, is_new = await _get_or_create_user(
            mock_session, email, "Updated Name", "google", "google_456"
        )

        assert user_id == existing_user_id
        # For existing users, is_new should be False (created_at far in past)
        assert not is_new or (datetime.now(timezone.utc) - past_time).total_seconds() < 1


class TestCreateSessionKey:
    """Tests for atomic session key creation with limits."""

    @pytest.mark.asyncio
    async def test_create_session_key_uses_for_update(self):
        """Test that _create_session_key uses SELECT FOR UPDATE for locking."""
        from service.api.auth_routes import _create_session_key

        mock_session = AsyncMock()
        user_id = str(uuid4())

        # Mock count result (0 existing sessions)
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0

        # Mock execute to succeed - returns count result for first call
        mock_session.execute = AsyncMock(return_value=mock_count_result)
        mock_session.add = MagicMock()

        full_key, salt = await _create_session_key(
            mock_session, user_id, "OAuth Session", with_encryption=True
        )

        # Verify key was generated
        assert full_key is not None
        assert full_key.startswith("ctxfs_")
        assert salt is not None

        # Verify session operations
        assert mock_session.execute.call_count >= 2  # SELECT FOR UPDATE + COUNT
        assert mock_session.add.called
        assert mock_session.commit.called

    @pytest.mark.asyncio
    async def test_create_session_key_without_encryption(self):
        """Test session key creation without encryption."""
        from service.api.auth_routes import _create_session_key

        mock_session = AsyncMock()
        user_id = str(uuid4())

        # Mock count result (0 existing sessions)
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0

        mock_session.execute = AsyncMock(return_value=mock_count_result)

        full_key, salt = await _create_session_key(
            mock_session, user_id, "CLI Session", with_encryption=False
        )

        assert full_key is not None
        assert salt is None  # No encryption salt when disabled

    @pytest.mark.asyncio
    async def test_create_session_key_enforces_limit(self):
        """Test that _create_session_key enforces session limits."""
        from service.api.auth_routes import _create_session_key

        mock_session = AsyncMock()
        user_id = str(uuid4())

        # Mock count result showing 3 existing sessions (at limit)
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 3

        # Mock oldest keys query
        mock_oldest_result = MagicMock()
        mock_oldest_result.fetchall.return_value = [(str(uuid4()),)]

        # Return different results for different queries
        call_count = 0

        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: SELECT FOR UPDATE
                return MagicMock()
            elif call_count == 2:
                # Second call: COUNT
                return mock_count_result
            elif call_count == 3:
                # Third call: SELECT oldest keys
                return mock_oldest_result
            else:
                # Fourth call: DELETE
                return MagicMock()

        mock_session.execute = mock_execute
        mock_session.add = MagicMock()

        full_key, salt = await _create_session_key(
            mock_session, user_id, "OAuth Session", with_encryption=True, max_sessions=3
        )

        # Verify key was generated
        assert full_key is not None
        assert full_key.startswith("ctxfs_")

        # Verify delete was called (limit enforcement)
        assert call_count >= 4  # FOR UPDATE, COUNT, SELECT oldest, DELETE

    @pytest.mark.asyncio
    async def test_create_session_key_no_delete_under_limit(self):
        """Test that _create_session_key doesn't delete when under limit."""
        from service.api.auth_routes import _create_session_key

        mock_session = AsyncMock()
        user_id = str(uuid4())

        # Mock count result showing 2 existing sessions (under limit of 10)
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 2

        call_count = 0

        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return mock_count_result
            return MagicMock()

        mock_session.execute = mock_execute
        mock_session.add = MagicMock()

        full_key, salt = await _create_session_key(
            mock_session, user_id, "OAuth Session", with_encryption=True
        )

        assert full_key is not None
        # Should only be 2 calls: FOR UPDATE and COUNT (no DELETE since under limit)
        assert call_count == 2


class TestRetryDecorator:
    """Tests for the retry decorator."""

    @pytest.mark.asyncio
    async def test_retry_on_integrity_error(self):
        """Test that the retry decorator retries on IntegrityError."""
        from sqlalchemy.exc import IntegrityError

        from service.api.auth_routes import with_retry

        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        async def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise IntegrityError("test", {}, Exception("duplicate key"))
            return "success"

        result = await flaky_function()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self):
        """Test that retry decorator raises after max retries."""
        from sqlalchemy.exc import IntegrityError

        from service.api.auth_routes import with_retry

        @with_retry(max_retries=2, base_delay=0.01)
        async def always_fails():
            raise IntegrityError("test", {}, Exception("duplicate key"))

        with pytest.raises(IntegrityError):
            await always_fails()

    @pytest.mark.asyncio
    async def test_no_retry_on_other_errors(self):
        """Test that retry decorator doesn't retry on non-IntegrityError."""
        from service.api.auth_routes import with_retry

        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        async def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("not an integrity error")

        with pytest.raises(ValueError):
            await raises_value_error()

        # Should not retry on ValueError
        assert call_count == 1
