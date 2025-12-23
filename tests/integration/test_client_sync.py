"""Integration tests for client sync.

Tests for vector clock, path resolver, and sync client functionality.
Server-side tests require docker-compose.sync.yml to be running.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from contextfs.sync.path_resolver import PathResolver, PortablePath, RepoRegistry
from contextfs.sync.vector_clock import DeviceTracker, VectorClock

# =============================================================================
# Vector Clock Tests
# =============================================================================


class TestVectorClock:
    """Unit tests for vector clock."""

    def test_empty_clock(self):
        """Test empty vector clock."""
        clock = VectorClock()
        assert clock.clock == {}
        assert not clock  # Should be falsy

    def test_increment(self):
        """Test incrementing counter for a device."""
        clock = VectorClock()
        clock = clock.increment("device-a")
        assert clock.clock == {"device-a": 1}
        clock = clock.increment("device-a")
        assert clock.clock == {"device-a": 2}
        clock = clock.increment("device-b")
        assert clock.clock == {"device-a": 2, "device-b": 1}

    def test_happens_before(self):
        """Test happens-before relationship."""
        clock1 = VectorClock(clock={"a": 1, "b": 1})
        clock2 = VectorClock(clock={"a": 2, "b": 1})

        assert clock1.happens_before(clock2)
        assert not clock2.happens_before(clock1)

    def test_happens_before_with_subset(self):
        """Test happens-before with different device sets."""
        clock1 = VectorClock(clock={"a": 1})
        clock2 = VectorClock(clock={"a": 1, "b": 1})

        assert clock1.happens_before(clock2)
        assert not clock2.happens_before(clock1)

    def test_concurrent(self):
        """Test concurrent (conflicting) clocks."""
        clock1 = VectorClock(clock={"a": 2, "b": 1})
        clock2 = VectorClock(clock={"a": 1, "b": 2})

        assert clock1.concurrent_with(clock2)
        assert clock2.concurrent_with(clock1)
        assert not clock1.happens_before(clock2)
        assert not clock2.happens_before(clock1)

    def test_equal(self):
        """Test equal clocks."""
        clock1 = VectorClock(clock={"a": 1, "b": 2})
        clock2 = VectorClock(clock={"a": 1, "b": 2})

        assert clock1.equal_to(clock2)
        assert not clock1.happens_before(clock2)
        assert not clock2.happens_before(clock1)
        assert not clock1.concurrent_with(clock2)

    def test_merge(self):
        """Test merging two clocks."""
        clock1 = VectorClock(clock={"a": 2, "b": 1})
        clock2 = VectorClock(clock={"a": 1, "b": 3, "c": 1})

        merged = clock1.merge(clock2)
        assert merged.clock == {"a": 2, "b": 3, "c": 1}

    def test_prune_by_active_devices(self):
        """Test pruning to active devices only."""
        clock = VectorClock(clock={"a": 1, "b": 2, "c": 3, "d": 4})
        active = {"a", "c"}

        pruned = clock.prune(active_devices=active)
        assert pruned.clock == {"a": 1, "c": 3}

    def test_prune_by_max_devices(self):
        """Test pruning to max devices."""
        clock = VectorClock(clock={"a": 1, "b": 5, "c": 3, "d": 2})

        pruned = clock.prune(max_devices=2)
        # Should keep highest counter devices: b=5, c=3
        assert len(pruned.clock) == 2
        assert "b" in pruned.clock
        assert "c" in pruned.clock

    def test_serialization(self):
        """Test JSON serialization."""
        clock = VectorClock(clock={"device-1": 5, "device-2": 3})

        # To dict
        as_dict = clock.to_dict()
        assert as_dict == {"device-1": 5, "device-2": 3}

        # To JSON
        as_json = clock.to_json()
        assert json.loads(as_json) == as_dict

        # From dict
        restored = VectorClock.from_dict(as_dict)
        assert restored.clock == clock.clock

        # From JSON
        restored = VectorClock.from_json(as_json)
        assert restored.clock == clock.clock

    def test_from_empty(self):
        """Test creating from empty/None values."""
        assert VectorClock.from_dict(None).clock == {}
        assert VectorClock.from_dict({}).clock == {}
        assert VectorClock.from_json(None).clock == {}
        assert VectorClock.from_json("").clock == {}


class TestDeviceTracker:
    """Tests for device tracker."""

    def test_update_device(self):
        """Test updating device last-seen time."""
        tracker = DeviceTracker()
        now = datetime.now()

        tracker.update("device-1", now)
        assert "device-1" in tracker.devices
        assert tracker.devices["device-1"] == now

    def test_get_active_devices(self):
        """Test getting active devices within window."""
        tracker = DeviceTracker(prune_after_days=30)
        now = datetime.now()

        # Active device
        tracker.update("active", now)

        # Inactive device (40 days ago)
        tracker.devices["inactive"] = now - timedelta(days=40)

        active = tracker.get_active_devices(as_of=now)
        assert "active" in active
        assert "inactive" not in active

    def test_prune_clock(self):
        """Test pruning a clock based on device activity."""
        tracker = DeviceTracker(prune_after_days=30)
        now = datetime.now()

        tracker.update("active", now)
        tracker.devices["inactive"] = now - timedelta(days=40)

        clock = VectorClock(clock={"active": 5, "inactive": 10})
        pruned = tracker.prune_clock(clock)

        assert pruned.clock == {"active": 5}


# =============================================================================
# Path Resolver Tests
# =============================================================================


class TestRepoRegistry:
    """Tests for repo registry."""

    def test_register_and_lookup(self):
        """Test registering and looking up repos."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.json"
            registry = RepoRegistry()
            registry._registry_path = registry_path

            # Register a repo
            registry.register("git@github.com:user/repo.git", "/path/to/repo")

            # Lookup
            assert registry.get_repo_url("/path/to/repo") == "git@github.com:user/repo.git"

    def test_url_normalization(self):
        """Test URL normalization."""
        # HTTPS to SSH
        assert (
            RepoRegistry._normalize_url("https://github.com/user/repo")
            == "git@github.com:user/repo.git"
        )
        assert (
            RepoRegistry._normalize_url("https://github.com/user/repo.git")
            == "git@github.com:user/repo.git"
        )

        # Already SSH
        assert (
            RepoRegistry._normalize_url("git@github.com:user/repo")
            == "git@github.com:user/repo.git"
        )
        assert (
            RepoRegistry._normalize_url("git@github.com:user/repo.git")
            == "git@github.com:user/repo.git"
        )

    def test_find_containing_repo(self):
        """Test finding repo containing a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RepoRegistry()
            registry._registry_path = Path(tmpdir) / "registry.json"

            # Create a fake repo structure (resolve to handle macOS /var -> /private/var)
            repo_path = (Path(tmpdir) / "myrepo").resolve()
            repo_path.mkdir()
            (repo_path / "src").mkdir()
            (repo_path / "src" / "main.py").touch()

            # Register it
            registry.register("git@github.com:user/myrepo.git", str(repo_path))

            # Find containing repo
            result = registry.find_containing_repo(repo_path / "src" / "main.py")
            assert result is not None
            repo_url, local_path = result
            assert repo_url == "git@github.com:user/myrepo.git"
            assert local_path.resolve() == repo_path.resolve()


class TestPortablePath:
    """Tests for portable path."""

    def test_valid_path(self):
        """Test valid portable path."""
        path = PortablePath(
            repo_url="git@github.com:user/repo.git",
            repo_name="repo",
            relative_path="src/main.py",
        )
        assert path.is_valid()

    def test_invalid_path(self):
        """Test invalid portable paths."""
        # Missing repo_url
        path = PortablePath(relative_path="src/main.py")
        assert not path.is_valid()

        # Missing relative_path
        path = PortablePath(repo_url="git@github.com:user/repo.git")
        assert not path.is_valid()

        # Empty
        path = PortablePath()
        assert not path.is_valid()


class TestPathResolver:
    """Tests for path resolver."""

    def test_normalize_with_registry(self):
        """Test normalizing path using registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup registry
            registry = RepoRegistry()
            registry._registry_path = Path(tmpdir) / "registry.json"

            # Create repo structure
            repo_path = Path(tmpdir) / "myrepo"
            repo_path.mkdir()
            (repo_path / "src").mkdir()
            file_path = repo_path / "src" / "main.py"
            file_path.touch()

            # Register repo
            registry.register("git@github.com:user/myrepo.git", str(repo_path))

            # Create resolver with this registry
            resolver = PathResolver(registry=registry)

            # Normalize
            portable = resolver.normalize(file_path)

            # Since there's no .git, it falls back to registry
            result = registry.find_containing_repo(file_path)
            if result:
                assert portable.repo_url == "git@github.com:user/myrepo.git"
                assert portable.relative_path == "src/main.py"

    def test_resolve_portable_path(self):
        """Test resolving portable path to local path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup registry
            registry = RepoRegistry()
            registry._registry_path = Path(tmpdir) / "registry.json"

            # Create repo
            repo_path = Path(tmpdir) / "myrepo"
            repo_path.mkdir()
            (repo_path / "src").mkdir()

            # Register
            registry.register("git@github.com:user/myrepo.git", str(repo_path))

            # Resolve
            resolver = PathResolver(registry=registry)
            portable = PortablePath(
                repo_url="git@github.com:user/myrepo.git",
                relative_path="src/main.py",
            )

            local_path = resolver.resolve(portable)
            assert local_path == repo_path / "src" / "main.py"

    def test_resolve_unknown_repo(self):
        """Test resolving path for unknown repo."""
        resolver = PathResolver(registry=RepoRegistry())
        portable = PortablePath(
            repo_url="git@github.com:unknown/repo.git",
            relative_path="src/main.py",
        )

        local_path = resolver.resolve(portable)
        assert local_path is None


# =============================================================================
# Sync Client Tests (require running server)
# =============================================================================


@pytest.mark.skipif(
    os.environ.get("CONTEXTFS_TEST_SYNC") != "1",
    reason="Sync tests disabled. Set CONTEXTFS_TEST_SYNC=1 and run docker-compose.sync.yml",
)
class TestSyncClient:
    """Integration tests for sync client (require running server)."""

    @pytest.fixture
    def server_url(self):
        """Get sync server URL."""
        return os.environ.get("CONTEXTFS_SYNC_URL", "http://localhost:8766")

    @pytest.fixture
    async def client(self, server_url):
        """Create sync client."""
        from contextfs.sync import SyncClient

        client = SyncClient(
            server_url=server_url,
            device_id=f"test-device-{datetime.now().timestamp()}",
        )
        yield client
        await client.close()

    async def test_register_device(self, client):
        """Test device registration."""
        info = await client.register_device(
            device_name="Test Device",
            device_platform="test",
        )

        assert info.device_id == client.device_id
        assert info.device_name == "Test Device"
        assert info.platform == "test"

    async def test_push_empty(self, client):
        """Test pushing with no changes."""
        await client.register_device("Test Device", "test")

        response = await client.push(memories=[])

        assert response.success
        assert response.accepted == 0

    async def test_sync_status(self, client):
        """Test getting sync status."""
        await client.register_device("Test Device", "test")

        status = await client.status()

        assert status.device_id == client.device_id
        assert status.server_timestamp is not None
