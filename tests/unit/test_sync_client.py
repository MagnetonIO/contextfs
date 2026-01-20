"""Sync Client Tests.

Tests for the SyncClient class that handles syncing local memories
with the remote sync server.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from contextfs.schemas import Memory, MemoryType
from contextfs.sync.client import SyncClient, _ensure_tz_aware


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Set up temporary home directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)
    return home


@pytest.fixture
def mock_ctx(temp_home):
    """Create a mock ContextFS instance."""
    ctx = MagicMock()
    db_path = temp_home / ".contextfs" / "context.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    ctx._db_path = db_path

    # Create SQLite database with memories table (matching actual schema)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            type TEXT NOT NULL,
            summary TEXT,
            tags TEXT,
            source_tool TEXT,
            source_repo TEXT,
            project TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            metadata TEXT,
            namespace_id TEXT,
            synced_at TIMESTAMP,
            authoritative INTEGER DEFAULT 0,
            structured_data TEXT,
            vector_clock TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            label TEXT,
            tool TEXT,
            summary TEXT,
            messages TEXT,
            created_at TIMESTAMP,
            ended_at TIMESTAMP,
            started_at TIMESTAMP,
            namespace_id TEXT
        )
    """)
    conn.commit()
    conn.close()

    return ctx


@pytest.fixture
def sample_memory():
    """Create a sample memory for testing."""
    return Memory(
        id="test-memory-123",
        content="Test content for sync",
        type=MemoryType.FACT,
        tags=["test", "sync"],
        summary="Test summary",
        source_tool="claude-code",
        source_repo="test-repo",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestEnsureTzAware:
    """Tests for _ensure_tz_aware helper function."""

    def test_none_input(self):
        """Test with None input."""
        result = _ensure_tz_aware(None)
        assert result is None

    def test_naive_datetime(self):
        """Test converting naive datetime to UTC."""
        naive = datetime(2024, 1, 15, 12, 0, 0)
        result = _ensure_tz_aware(naive)
        assert result is not None
        assert result.tzinfo is not None

    def test_aware_datetime(self):
        """Test with already timezone-aware datetime."""
        aware = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = _ensure_tz_aware(aware)
        assert result == aware


# =============================================================================
# SyncClient Initialization Tests
# =============================================================================


class TestSyncClientInit:
    """Tests for SyncClient initialization."""

    def test_init_basic(self, mock_ctx, temp_home):
        """Test basic initialization."""
        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        )
        assert client.server_url == "http://localhost:8766"
        assert client.device_id == "test-device"
        assert client._ctx == mock_ctx

    def test_init_strips_trailing_slash(self, mock_ctx, temp_home):
        """Test URL trailing slash is stripped."""
        client = SyncClient(
            server_url="http://localhost:8766/",
            ctx=mock_ctx,
            device_id="test-device",
        )
        assert client.server_url == "http://localhost:8766"

    def test_init_with_api_key(self, mock_ctx, temp_home):
        """Test initialization with API key."""
        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
            api_key="test-api-key",
        )
        assert client._api_key == "test-api-key"
        assert "X-API-Key" in client._client.headers

    def test_init_auto_device_id(self, mock_ctx, temp_home):
        """Test device ID is auto-generated."""
        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
        )
        assert client.device_id is not None
        assert len(client.device_id) > 0

        # Should be persisted
        device_id_path = temp_home / ".contextfs" / "device_id"
        assert device_id_path.exists()
        assert device_id_path.read_text().strip() == client.device_id

    def test_init_loads_existing_device_id(self, mock_ctx, temp_home):
        """Test loading existing device ID."""
        device_id_path = temp_home / ".contextfs" / "device_id"
        device_id_path.parent.mkdir(parents=True, exist_ok=True)
        device_id_path.write_text("existing-device-id")

        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
        )
        assert client.device_id == "existing-device-id"


# =============================================================================
# Sync State Tests
# =============================================================================


class TestSyncState:
    """Tests for sync state persistence."""

    def test_save_and_load_sync_state(self, mock_ctx, temp_home):
        """Test saving and loading sync state."""
        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        )

        # Set some state
        client._last_sync = datetime.now(timezone.utc)
        client._last_push = datetime.now(timezone.utc)
        client._last_pull = datetime.now(timezone.utc)

        # Save state
        client._save_sync_state()

        # Create new client and verify state loaded
        client2 = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        )

        assert client2._last_sync is not None
        assert client2._last_push is not None
        assert client2._last_pull is not None

    def test_ensure_sync_state_table(self, mock_ctx, temp_home):
        """Test sync_state table is created."""
        _client = SyncClient(  # noqa: F841  # Creating client has side effect
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        )

        # Verify table exists
        conn = sqlite3.connect(mock_ctx._db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sync_state'"
        )
        assert cursor.fetchone() is not None
        conn.close()


# =============================================================================
# Content Hash Tests
# =============================================================================


class TestContentHash:
    """Tests for content hashing."""

    def test_compute_content_hash(self):
        """Test computing content hash."""
        hash1 = SyncClient.compute_content_hash("test content")
        hash2 = SyncClient.compute_content_hash("test content")
        hash3 = SyncClient.compute_content_hash("different content")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16

    def test_hash_empty_content(self):
        """Test hashing empty content."""
        hash_val = SyncClient.compute_content_hash("")
        assert len(hash_val) == 16


# =============================================================================
# Rejected IDs Tests
# =============================================================================


class TestRejectedIds:
    """Tests for rejected IDs management."""

    def test_save_and_load_rejected_ids(self, mock_ctx, temp_home):
        """Test saving and loading rejected IDs."""
        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        )

        ids = ["id1", "id2", "id3"]
        client._save_rejected_ids(ids)

        loaded = client._load_rejected_ids()
        assert loaded == ids

    def test_load_rejected_ids_no_file(self, mock_ctx, temp_home):
        """Test loading rejected IDs when file doesn't exist."""
        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        )

        loaded = client._load_rejected_ids()
        assert loaded == []

    def test_clear_rejected_ids(self, mock_ctx, temp_home):
        """Test clearing rejected IDs."""
        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        )

        client._save_rejected_ids(["id1"])
        assert client._load_rejected_ids() == ["id1"]

        client._clear_rejected_ids()
        assert client._load_rejected_ids() == []


# =============================================================================
# Device Registration Tests
# =============================================================================


class TestDeviceRegistration:
    """Tests for device registration."""

    @pytest.mark.asyncio
    async def test_register_device(self, mock_ctx, temp_home):
        """Test device registration."""
        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "device_id": "test-device",
            "device_name": "Test Device",
            "platform": "darwin",
            "client_version": "0.1.0",
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        mock_response.raise_for_status = MagicMock()

        client._client.post = AsyncMock(return_value=mock_response)

        result = await client.register_device(
            device_name="Test Device",
            device_platform="darwin",
        )

        assert result.device_id == "test-device"
        client._client.post.assert_called_once()


# =============================================================================
# Push Tests
# =============================================================================


class TestPush:
    """Tests for push operations."""

    @pytest.mark.asyncio
    async def test_push_memories(self, mock_ctx, temp_home, sample_memory):
        """Test pushing memories to server."""
        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        )

        # Mock HTTP response (must include 'success' field)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "accepted": 1,
            "rejected": 0,
            "conflicts": [],
            "server_timestamp": datetime.now(timezone.utc).isoformat(),
            "pushed_items": [
                {
                    "id": sample_memory.id,
                    "type": "memory",
                    "summary": sample_memory.summary,
                }
            ],
        }
        mock_response.raise_for_status = MagicMock()

        client._client.post = AsyncMock(return_value=mock_response)
        client._client.get = AsyncMock(
            return_value=MagicMock(status_code=404)
        )  # No encryption salt

        # Mock getting embeddings
        client._get_embeddings_from_chroma = MagicMock(return_value={})

        result = await client.push(memories=[sample_memory])

        assert result.accepted == 1
        assert result.rejected == 0

    @pytest.mark.asyncio
    async def test_push_empty_list(self, mock_ctx, temp_home):
        """Test pushing empty list."""
        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        )

        # Mock HTTP response (must include 'success' field)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "accepted": 0,
            "rejected": 0,
            "conflicts": [],
            "server_timestamp": datetime.now(timezone.utc).isoformat(),
            "pushed_items": [],
        }
        mock_response.raise_for_status = MagicMock()

        client._client.post = AsyncMock(return_value=mock_response)
        client._client.get = AsyncMock(return_value=MagicMock(status_code=404))
        client._get_embeddings_from_chroma = MagicMock(return_value={})

        result = await client.push(memories=[])

        assert result.accepted == 0


# =============================================================================
# Pull Tests
# =============================================================================


class TestPull:
    """Tests for pull operations."""

    @pytest.mark.asyncio
    async def test_pull_memories(self, mock_ctx, temp_home):
        """Test pulling memories from server."""
        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        )

        # Mock HTTP response (must include 'success' field)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "memories": [
                {
                    "id": "remote-memory-1",
                    "content": "Remote content",
                    "type": "fact",
                    "summary": "Remote summary",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "vector_clock": {},
                    "content_hash": "abc123",
                }
            ],
            "sessions": [],
            "edges": [],
            "server_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        mock_response.raise_for_status = MagicMock()

        client._client.post = AsyncMock(return_value=mock_response)
        client._client.get = AsyncMock(return_value=MagicMock(status_code=404))

        # Mock saving memories
        client._save_pulled_memories = MagicMock()
        client._save_pulled_sessions = MagicMock()
        client._process_deleted_ids = MagicMock()

        result = await client.pull()

        assert len(result.memories) == 1
        assert result.memories[0].id == "remote-memory-1"

    @pytest.mark.asyncio
    async def test_pull_no_new_data(self, mock_ctx, temp_home):
        """Test pulling when no new data."""
        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "memories": [],
            "sessions": [],
            "edges": [],
            "server_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        mock_response.raise_for_status = MagicMock()

        client._client.post = AsyncMock(return_value=mock_response)
        client._client.get = AsyncMock(return_value=MagicMock(status_code=404))
        client._save_pulled_memories = MagicMock()
        client._save_pulled_sessions = MagicMock()
        client._process_deleted_ids = MagicMock()

        result = await client.pull()

        assert len(result.memories) == 0
        assert len(result.sessions) == 0


# =============================================================================
# Encryption Tests
# =============================================================================


class TestEncryption:
    """Tests for E2EE encryption."""

    def test_is_encrypted_false_by_default(self, mock_ctx, temp_home):
        """Test encryption is off by default."""
        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        )
        assert client.is_encrypted is False

    def test_encrypt_content_no_crypto(self, mock_ctx, temp_home):
        """Test encryption passthrough when no crypto configured."""
        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        )
        content = "test content"
        result = client._encrypt_content(content)
        assert result == content

    def test_decrypt_content_no_crypto(self, mock_ctx, temp_home):
        """Test decryption passthrough when no crypto configured."""
        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        )
        content = "test content"
        result = client._decrypt_content(content, encrypted=False)
        assert result == content

    @pytest.mark.asyncio
    async def test_ensure_e2ee_no_api_key(self, mock_ctx, temp_home):
        """Test E2EE init with no API key."""
        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        )

        await client._ensure_e2ee_initialized()

        assert client._e2ee_initialized is True
        assert client._crypto is None


# =============================================================================
# Path Normalization Tests
# =============================================================================


class TestPathNormalization:
    """Tests for path normalization."""

    def test_normalize_memory_paths_no_source(self, mock_ctx, temp_home, sample_memory):
        """Test normalizing memory without source file."""
        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        )

        result = client._normalize_memory_paths(sample_memory)

        assert result["repo_url"] is None
        assert result["repo_name"] is None
        assert result["relative_path"] is None


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_ctx, temp_home):
        """Test using client as async context manager."""
        async with SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        ) as client:
            assert client.server_url == "http://localhost:8766"


# =============================================================================
# Vector Clock Tests
# =============================================================================


class TestVectorClockIntegration:
    """Tests for vector clock integration."""

    def test_device_tracker_update(self, mock_ctx, temp_home):
        """Test device tracker updates."""
        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        )

        # Update device tracker
        client._device_tracker.update("test-device")

        # Device should now be tracked
        assert "test-device" in client._device_tracker.devices
        assert client._device_tracker.devices["test-device"] is not None


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_register_device_http_error(self, mock_ctx, temp_home):
        """Test handling HTTP error during registration."""
        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        )

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = Exception("Unauthorized")

        client._client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(Exception, match="Unauthorized"):
            await client.register_device()

    def test_load_sync_state_corrupted_db(self, mock_ctx, temp_home):
        """Test loading sync state from corrupted database."""
        # Write invalid data to db
        conn = sqlite3.connect(mock_ctx._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sync_state (
                id INTEGER PRIMARY KEY,
                device_id TEXT NOT NULL UNIQUE,
                device_tracker TEXT DEFAULT '{}'
            )
        """)
        conn.execute(
            "INSERT INTO sync_state (device_id, device_tracker) VALUES (?, ?)",
            ("test-device", "invalid json{{{"),
        )
        conn.commit()
        conn.close()

        # Should not raise - just logs warning
        client = SyncClient(
            server_url="http://localhost:8766",
            ctx=mock_ctx,
            device_id="test-device",
        )
        assert client is not None
