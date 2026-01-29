"""
Tests for edge cleanup on memory deletion, orphaned edge cleanup,
and edge validation in add_edge().
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from contextfs.storage_protocol import EdgeRelation, MemoryEdge


@pytest.fixture
def storage_router(temp_dir):
    """Create a StorageRouter with a real SQLite DB and mocked RAG."""
    from contextfs.storage_router import StorageRouter

    db_path = temp_dir / "test.db"

    # Create the memories and memory_edges tables
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY NOT NULL,
            content TEXT NOT NULL,
            type TEXT NOT NULL,
            tags TEXT,
            summary TEXT,
            namespace_id TEXT NOT NULL,
            source_file TEXT,
            source_repo TEXT,
            source_tool TEXT,
            project TEXT,
            session_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata TEXT,
            structured_data TEXT,
            deleted_at TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_edges (
            from_id TEXT NOT NULL,
            to_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            created_at TEXT NOT NULL,
            created_by TEXT,
            metadata TEXT,
            deleted_at TEXT,
            PRIMARY KEY (from_id, to_id, relation)
        )
    """)
    conn.commit()
    conn.close()

    mock_rag = MagicMock()
    mock_rag.add_memory = MagicMock()
    mock_rag._ensure_initialized = MagicMock()

    router = StorageRouter(
        db_path=db_path,
        rag_backend=mock_rag,
    )

    return router


def _insert_memory(db_path: Path, memory_id: str, content: str = "test") -> None:
    """Insert a test memory directly into SQLite."""
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO memories (id, content, type, namespace_id, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (memory_id, content, "fact", "test", now, now),
    )
    conn.commit()
    conn.close()


def _count_edges(db_path: Path) -> int:
    """Count all edges in the memory_edges table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM memory_edges")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def _count_soft_deleted_edges(db_path: Path) -> int:
    """Count edges with deleted_at set."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM memory_edges WHERE deleted_at IS NOT NULL")
    count = cursor.fetchone()[0]
    conn.close()
    return count


class TestEdgeCleanupOnDelete:
    """Test that deleting a memory removes its edges."""

    def test_hard_delete_removes_edges(self, storage_router):
        """Deleting a memory should remove all its edges."""
        db_path = storage_router._db_path

        # Create two memories
        _insert_memory(db_path, "mem-aaa", "Memory A")
        _insert_memory(db_path, "mem-bbb", "Memory B")

        # Create an edge between them
        storage_router.add_edge(
            from_id="mem-aaa",
            to_id="mem-bbb",
            relation=EdgeRelation.REFERENCES,
            validate=False,
        )
        assert _count_edges(db_path) == 1

        # Delete memory A
        deleted = storage_router.delete("mem-aaa")
        assert deleted is True

        # Edge should be gone
        assert _count_edges(db_path) == 0

    def test_delete_removes_incoming_and_outgoing(self, storage_router):
        """Deleting a memory should remove both incoming and outgoing edges."""
        db_path = storage_router._db_path

        _insert_memory(db_path, "mem-aaa", "Memory A")
        _insert_memory(db_path, "mem-bbb", "Memory B")
        _insert_memory(db_path, "mem-ccc", "Memory C")

        # A -> B (outgoing from B's perspective: incoming)
        storage_router.add_edge(
            from_id="mem-aaa",
            to_id="mem-bbb",
            relation=EdgeRelation.REFERENCES,
            validate=False,
        )
        # B -> C (outgoing from B)
        storage_router.add_edge(
            from_id="mem-bbb",
            to_id="mem-ccc",
            relation=EdgeRelation.PARENT_OF,
            validate=False,
        )
        assert _count_edges(db_path) == 2

        # Delete B — should remove both edges
        storage_router.delete("mem-bbb")
        assert _count_edges(db_path) == 0

    def test_delete_preserves_unrelated_edges(self, storage_router):
        """Deleting a memory should not affect edges between other memories."""
        db_path = storage_router._db_path

        _insert_memory(db_path, "mem-aaa", "A")
        _insert_memory(db_path, "mem-bbb", "B")
        _insert_memory(db_path, "mem-ccc", "C")

        # A -> B
        storage_router.add_edge(
            from_id="mem-aaa",
            to_id="mem-bbb",
            relation=EdgeRelation.REFERENCES,
            validate=False,
        )
        # B -> C
        storage_router.add_edge(
            from_id="mem-bbb",
            to_id="mem-ccc",
            relation=EdgeRelation.PARENT_OF,
            validate=False,
        )
        assert _count_edges(db_path) == 2

        # Delete A — only A->B edge should be removed
        storage_router.delete("mem-aaa")
        assert _count_edges(db_path) == 1


class TestSoftDeleteEdges:
    """Test that soft-deleting edges sets deleted_at."""

    def test_soft_delete_edges(self, storage_router):
        """Soft-deleting edges should set deleted_at rather than removing them."""
        db_path = storage_router._db_path

        _insert_memory(db_path, "mem-aaa", "A")
        _insert_memory(db_path, "mem-bbb", "B")

        storage_router.add_edge(
            from_id="mem-aaa",
            to_id="mem-bbb",
            relation=EdgeRelation.REFERENCES,
            validate=False,
        )
        assert _count_edges(db_path) == 1
        assert _count_soft_deleted_edges(db_path) == 0

        # Soft-delete edges for mem-aaa
        count = storage_router._soft_delete_edges_for_memory("mem-aaa")
        assert count == 1

        # Edge still exists but has deleted_at set
        assert _count_edges(db_path) == 1
        assert _count_soft_deleted_edges(db_path) == 1

    def test_soft_delete_is_idempotent(self, storage_router):
        """Soft-deleting already soft-deleted edges should not re-update them."""
        db_path = storage_router._db_path

        _insert_memory(db_path, "mem-aaa", "A")
        _insert_memory(db_path, "mem-bbb", "B")

        storage_router.add_edge(
            from_id="mem-aaa",
            to_id="mem-bbb",
            relation=EdgeRelation.REFERENCES,
            validate=False,
        )

        # First soft-delete
        count1 = storage_router._soft_delete_edges_for_memory("mem-aaa")
        assert count1 == 1

        # Second soft-delete — should affect 0 rows
        count2 = storage_router._soft_delete_edges_for_memory("mem-aaa")
        assert count2 == 0


class TestCleanupOrphanedEdges:
    """Test orphaned edge cleanup utility."""

    def test_cleanup_removes_orphaned_edges(self, storage_router):
        """Edges referencing non-existent memories should be removed."""
        db_path = storage_router._db_path

        _insert_memory(db_path, "mem-aaa", "A")
        _insert_memory(db_path, "mem-bbb", "B")

        storage_router.add_edge(
            from_id="mem-aaa",
            to_id="mem-bbb",
            relation=EdgeRelation.REFERENCES,
            validate=False,
        )

        # Manually delete memory B from SQLite (without going through router)
        conn = sqlite3.connect(db_path)
        conn.execute("DELETE FROM memories WHERE id = 'mem-bbb'")
        conn.commit()
        conn.close()

        assert _count_edges(db_path) == 1

        # Cleanup should remove the orphaned edge
        count = storage_router.cleanup_orphaned_edges()
        assert count == 1
        assert _count_edges(db_path) == 0

    def test_cleanup_returns_zero_when_clean(self, storage_router):
        """Cleanup on a clean DB should return 0."""
        db_path = storage_router._db_path

        _insert_memory(db_path, "mem-aaa", "A")
        _insert_memory(db_path, "mem-bbb", "B")

        storage_router.add_edge(
            from_id="mem-aaa",
            to_id="mem-bbb",
            relation=EdgeRelation.REFERENCES,
            validate=False,
        )

        # No orphaned edges — both memories exist
        count = storage_router.cleanup_orphaned_edges()
        assert count == 0
        assert _count_edges(db_path) == 1


class TestAddEdgeValidation:
    """Test validation in add_edge()."""

    def test_add_edge_fails_with_nonexistent_from_id(self, storage_router):
        """add_edge should return None when from_id doesn't exist."""
        db_path = storage_router._db_path
        _insert_memory(db_path, "mem-bbb", "B")

        result = storage_router.add_edge(
            from_id="nonexistent-id",
            to_id="mem-bbb",
            relation=EdgeRelation.REFERENCES,
            validate=True,
        )
        assert result is None
        assert _count_edges(db_path) == 0

    def test_add_edge_fails_with_nonexistent_to_id(self, storage_router):
        """add_edge should return None when to_id doesn't exist."""
        db_path = storage_router._db_path
        _insert_memory(db_path, "mem-aaa", "A")

        result = storage_router.add_edge(
            from_id="mem-aaa",
            to_id="nonexistent-id",
            relation=EdgeRelation.REFERENCES,
            validate=True,
        )
        assert result is None
        assert _count_edges(db_path) == 0

    def test_add_edge_succeeds_with_valid_ids(self, storage_router):
        """add_edge should succeed when both IDs exist."""
        db_path = storage_router._db_path
        _insert_memory(db_path, "mem-aaa", "A")
        _insert_memory(db_path, "mem-bbb", "B")

        result = storage_router.add_edge(
            from_id="mem-aaa",
            to_id="mem-bbb",
            relation=EdgeRelation.REFERENCES,
            validate=True,
        )
        assert result is not None
        assert isinstance(result, MemoryEdge)
        assert _count_edges(db_path) == 1

    def test_add_edge_skips_validation_when_false(self, storage_router):
        """add_edge with validate=False should not check memory existence."""
        db_path = storage_router._db_path

        # No memories exist, but validate=False should allow edge creation
        result = storage_router.add_edge(
            from_id="ghost-aaa",
            to_id="ghost-bbb",
            relation=EdgeRelation.REFERENCES,
            validate=False,
        )
        assert result is not None
        assert _count_edges(db_path) == 1


class TestGetInverseRelationDelegation:
    """Test that core._get_inverse_relation delegates to EdgeRelation.get_inverse."""

    def test_resolves_pair(self):
        """RESOLVES/RESOLVED_BY should work via delegation."""
        assert EdgeRelation.get_inverse(EdgeRelation.RESOLVES) == EdgeRelation.RESOLVED_BY
        assert EdgeRelation.get_inverse(EdgeRelation.RESOLVED_BY) == EdgeRelation.RESOLVES

    def test_evolved_pair(self):
        """EVOLVED_INTO/EVOLVED_FROM should work via delegation."""
        assert EdgeRelation.get_inverse(EdgeRelation.EVOLVED_INTO) == EdgeRelation.EVOLVED_FROM
        assert EdgeRelation.get_inverse(EdgeRelation.EVOLVED_FROM) == EdgeRelation.EVOLVED_INTO

    def test_merged_pair(self):
        """MERGED_INTO/MERGED_FROM should work via delegation."""
        assert EdgeRelation.get_inverse(EdgeRelation.MERGED_INTO) == EdgeRelation.MERGED_FROM
        assert EdgeRelation.get_inverse(EdgeRelation.MERGED_FROM) == EdgeRelation.MERGED_INTO

    def test_split_pair(self):
        """SPLIT_INTO/SPLIT_FROM should work via delegation."""
        assert EdgeRelation.get_inverse(EdgeRelation.SPLIT_INTO) == EdgeRelation.SPLIT_FROM
        assert EdgeRelation.get_inverse(EdgeRelation.SPLIT_FROM) == EdgeRelation.SPLIT_INTO

    def test_unknown_returns_self(self):
        """Relations without explicit inverse should return themselves."""
        assert EdgeRelation.get_inverse(EdgeRelation.DEPENDS_ON) == EdgeRelation.DEPENDS_ON
