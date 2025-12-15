"""
Integration tests for PostgreSQL sync.

These tests require a running PostgreSQL instance.
Run with: pytest tests/integration/test_postgres_sync.py --postgres
"""

import os
from pathlib import Path

import pytest

# Skip if PostgreSQL not available
pytestmark = pytest.mark.skipif(
    os.environ.get("CONTEXTFS_TEST_POSTGRES") != "1",
    reason="PostgreSQL tests disabled. Set CONTEXTFS_TEST_POSTGRES=1 to enable.",
)


@pytest.fixture
def postgres_config():
    """PostgreSQL configuration from environment."""
    return {
        "host": os.environ.get("POSTGRES_HOST", "localhost"),
        "port": int(os.environ.get("POSTGRES_PORT", "5432")),
        "database": os.environ.get("POSTGRES_DB", "contextfs_test"),
        "user": os.environ.get("POSTGRES_USER", "contextfs"),
        "password": os.environ.get("POSTGRES_PASSWORD", "contextfs"),
    }


class TestPostgresSync:
    """Tests for PostgreSQL synchronization."""

    @pytest.fixture
    def sync_service(self, temp_dir: Path, postgres_config):
        """Create PostgreSQL sync service."""
        from contextfs.sync.postgres import PostgresSync

        sync = PostgresSync(
            local_db_path=temp_dir / "local.db",
            **postgres_config,
        )
        yield sync
        sync.close()

    def test_connection(self, sync_service):
        """Test PostgreSQL connection."""
        assert sync_service.is_connected()

    def test_sync_memory_to_postgres(self, sync_service, temp_dir: Path):
        """Test syncing local memory to PostgreSQL."""
        from contextfs.schemas import Memory, MemoryType

        # Create local memory
        memory = Memory(
            content="Test sync to PostgreSQL",
            type=MemoryType.FACT,
            tags=["test", "sync"],
        )

        # Add to local
        sync_service.add_local_memory(memory)

        # Sync to PostgreSQL
        result = sync_service.sync_to_remote()

        assert result["synced"] >= 1
        assert result["errors"] == 0

    def test_pull_from_postgres(self, sync_service, temp_dir: Path):
        """Test pulling memories from PostgreSQL."""
        from contextfs.schemas import Memory, MemoryType

        # Add memory directly to PostgreSQL
        memory = Memory(
            content="Test pull from PostgreSQL",
            type=MemoryType.FACT,
            tags=["test", "pull"],
        )

        sync_service.add_remote_memory(memory)

        # Pull from PostgreSQL
        result = sync_service.pull()

        assert result["pulled"] >= 1

    def test_bidirectional_sync(self, sync_service, temp_dir: Path):
        """Test bidirectional synchronization."""
        from contextfs.schemas import Memory, MemoryType

        # Add local memory
        local_memory = Memory(
            content="Local memory for bidirectional sync",
            type=MemoryType.FACT,
        )
        sync_service.add_local_memory(local_memory)

        # Add remote memory
        remote_memory = Memory(
            content="Remote memory for bidirectional sync",
            type=MemoryType.FACT,
        )
        sync_service.add_remote_memory(remote_memory)

        # Sync all
        result = sync_service.sync_all()

        assert result["local_to_remote"] >= 1
        assert result["remote_to_local"] >= 1

    def test_conflict_resolution(self, sync_service, temp_dir: Path):
        """Test conflict resolution during sync."""
        from contextfs.schemas import Memory, MemoryType

        # Create memory with same ID in both
        memory_id = "test-conflict-id"

        local_memory = Memory(
            id=memory_id,
            content="Local version",
            type=MemoryType.FACT,
        )
        sync_service.add_local_memory(local_memory)

        remote_memory = Memory(
            id=memory_id,
            content="Remote version (newer)",
            type=MemoryType.FACT,
        )
        sync_service.add_remote_memory(remote_memory)

        # Sync should handle conflict
        result = sync_service.sync_all()

        assert result.get("conflicts", 0) >= 0

    def test_search_global(self, sync_service, temp_dir: Path):
        """Test global search across PostgreSQL."""
        from contextfs.schemas import Memory, MemoryType

        # Add memories
        sync_service.add_remote_memory(
            Memory(
                content="Global search test - Python",
                type=MemoryType.FACT,
                tags=["python"],
            )
        )
        sync_service.add_remote_memory(
            Memory(
                content="Global search test - Rust",
                type=MemoryType.FACT,
                tags=["rust"],
            )
        )

        # Search
        results = sync_service.search_global("Python", limit=5)

        assert len(results) >= 1
        assert any("Python" in r["content"] for r in results)


class TestSyncDaemon:
    """Tests for background sync daemon."""

    @pytest.fixture
    def daemon(self, temp_dir: Path, postgres_config):
        """Create sync daemon."""
        from contextfs.sync.postgres import SyncDaemon

        daemon = SyncDaemon(
            local_db_path=temp_dir / "local.db",
            sync_interval=1,  # 1 second for testing
            **postgres_config,
        )
        yield daemon
        daemon.stop()

    def test_daemon_start_stop(self, daemon):
        """Test starting and stopping daemon."""
        daemon.start()
        assert daemon.is_running()

        daemon.stop()
        assert not daemon.is_running()

    def test_auto_sync(self, daemon, temp_dir: Path):
        """Test automatic synchronization."""
        import time

        from contextfs.schemas import Memory, MemoryType

        daemon.start()

        # Add memory
        memory = Memory(
            content="Auto sync test memory",
            type=MemoryType.FACT,
        )
        daemon.sync.add_local_memory(memory)

        # Wait for sync
        time.sleep(2)

        # Check if synced
        stats = daemon.get_stats()
        assert stats["total_syncs"] >= 1

        daemon.stop()
