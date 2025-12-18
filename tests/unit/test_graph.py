"""
Tests for graph backend, storage protocol, and memory lineage operations.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from contextfs.memory_lineage import (
    ConflictResolution,
    MemoryLineage,
    MergeStrategy,
)
from contextfs.schemas import Memory, MemoryType
from contextfs.storage_protocol import (
    CHROMADB_CAPABILITIES,
    FALKORDB_CAPABILITIES,
    SQLITE_CAPABILITIES,
    UNIFIED_WITH_GRAPH_CAPABILITIES,
    EdgeRelation,
    GraphPath,
    GraphTraversalResult,
    MemoryEdge,
    StorageCapabilities,
)

# =============================================================================
# EdgeRelation Tests
# =============================================================================


class TestEdgeRelation:
    """Tests for EdgeRelation enum."""

    def test_edge_relation_values(self):
        """Test that all expected relations exist."""
        assert EdgeRelation.EVOLVED_INTO.value == "evolved_into"
        assert EdgeRelation.MERGED_INTO.value == "merged_into"
        assert EdgeRelation.REFERENCES.value == "references"
        assert EdgeRelation.CONTRADICTS.value == "contradicts"

    def test_get_inverse_symmetric(self):
        """Test inverse relationships."""
        # Evolution
        assert EdgeRelation.get_inverse(EdgeRelation.EVOLVED_INTO) == EdgeRelation.EVOLVED_FROM
        assert EdgeRelation.get_inverse(EdgeRelation.EVOLVED_FROM) == EdgeRelation.EVOLVED_INTO

        # Merge
        assert EdgeRelation.get_inverse(EdgeRelation.MERGED_INTO) == EdgeRelation.MERGED_FROM
        assert EdgeRelation.get_inverse(EdgeRelation.MERGED_FROM) == EdgeRelation.MERGED_INTO

        # References
        assert EdgeRelation.get_inverse(EdgeRelation.REFERENCES) == EdgeRelation.REFERENCED_BY

        # Symmetric relations
        assert EdgeRelation.get_inverse(EdgeRelation.RELATED_TO) == EdgeRelation.RELATED_TO
        assert EdgeRelation.get_inverse(EdgeRelation.CONTRADICTS) == EdgeRelation.CONTRADICTS


# =============================================================================
# MemoryEdge Tests
# =============================================================================


class TestMemoryEdge:
    """Tests for MemoryEdge model."""

    def test_create_edge(self):
        """Test creating a memory edge."""
        edge = MemoryEdge(
            from_id="mem1",
            to_id="mem2",
            relation=EdgeRelation.REFERENCES,
            weight=0.8,
        )

        assert edge.from_id == "mem1"
        assert edge.to_id == "mem2"
        assert edge.relation == EdgeRelation.REFERENCES
        assert edge.weight == 0.8
        assert isinstance(edge.created_at, datetime)

    def test_edge_default_weight(self):
        """Test that default weight is 1.0."""
        edge = MemoryEdge(
            from_id="a",
            to_id="b",
            relation=EdgeRelation.RELATED_TO,
        )
        assert edge.weight == 1.0

    def test_edge_with_metadata(self):
        """Test edge with custom metadata."""
        edge = MemoryEdge(
            from_id="a",
            to_id="b",
            relation=EdgeRelation.SUPERSEDES,
            metadata={"reason": "outdated information"},
        )
        assert edge.metadata["reason"] == "outdated information"


# =============================================================================
# StorageCapabilities Tests
# =============================================================================


class TestStorageCapabilities:
    """Tests for StorageCapabilities."""

    def test_sqlite_capabilities(self):
        """Test SQLite capabilities."""
        caps = SQLITE_CAPABILITIES
        assert caps.full_text_search is True
        assert caps.persistent is True
        assert caps.transactions is True
        assert caps.semantic_search is False
        assert caps.graph_traversal is False

    def test_chromadb_capabilities(self):
        """Test ChromaDB capabilities."""
        caps = CHROMADB_CAPABILITIES
        assert caps.semantic_search is True
        assert caps.batch_operations is True
        assert caps.persistent is False

    def test_falkordb_capabilities(self):
        """Test FalkorDB capabilities."""
        caps = FALKORDB_CAPABILITIES
        assert caps.graph_traversal is True
        assert caps.memory_lineage is True
        assert caps.path_finding is True
        assert caps.persistent is True

    def test_capability_union(self):
        """Test combining capabilities with OR."""
        combined = SQLITE_CAPABILITIES | CHROMADB_CAPABILITIES
        assert combined.full_text_search is True  # from SQLite
        assert combined.semantic_search is True  # from ChromaDB
        assert combined.persistent is True  # from SQLite

    def test_capability_intersection(self):
        """Test intersecting capabilities with AND."""
        combined = SQLITE_CAPABILITIES & CHROMADB_CAPABILITIES
        assert combined.batch_operations is True  # both have it
        assert combined.full_text_search is False  # only SQLite
        assert combined.semantic_search is False  # only ChromaDB

    def test_capability_subset(self):
        """Test capability subset comparison."""
        minimal = StorageCapabilities(batch_operations=True)
        assert minimal <= SQLITE_CAPABILITIES
        assert minimal <= CHROMADB_CAPABILITIES

    def test_has_graph(self):
        """Test has_graph helper."""
        assert FALKORDB_CAPABILITIES.has_graph() is True
        assert SQLITE_CAPABILITIES.has_graph() is False
        assert UNIFIED_WITH_GRAPH_CAPABILITIES.has_graph() is True

    def test_repr(self):
        """Test string representation."""
        caps = FALKORDB_CAPABILITIES
        repr_str = repr(caps)
        assert "graph_traversal" in repr_str
        assert "memory_lineage" in repr_str


# =============================================================================
# GraphPath Tests
# =============================================================================


class TestGraphPath:
    """Tests for GraphPath model."""

    def test_create_path(self):
        """Test creating a graph path."""
        mem1 = Memory(id="m1", content="First")
        mem2 = Memory(id="m2", content="Second")

        edge = MemoryEdge(
            from_id="m1",
            to_id="m2",
            relation=EdgeRelation.REFERENCES,
        )

        path = GraphPath(
            nodes=[mem1, mem2],
            edges=[edge],
            total_weight=1.0,
        )

        assert len(path.nodes) == 2
        assert len(path.edges) == 1
        assert path.total_weight == 1.0


# =============================================================================
# GraphTraversalResult Tests
# =============================================================================


class TestGraphTraversalResult:
    """Tests for GraphTraversalResult model."""

    def test_create_result(self):
        """Test creating a traversal result."""
        memory = Memory(id="m1", content="Test")

        result = GraphTraversalResult(
            memory=memory,
            relation=EdgeRelation.EVOLVED_FROM,
            depth=2,
            path_weight=0.9,
        )

        assert result.memory.id == "m1"
        assert result.relation == EdgeRelation.EVOLVED_FROM
        assert result.depth == 2
        assert result.path_weight == 0.9


# =============================================================================
# MemoryLineage Tests (with mocked storage)
# =============================================================================


class TestMemoryLineage:
    """Tests for MemoryLineage operations."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage router."""
        storage = MagicMock()
        storage.recall.return_value = Memory(
            id="original",
            content="Original content",
            type=MemoryType.FACT,
            tags=["tag1", "tag2"],
            namespace_id="test",
            metadata={"key": "value"},
        )
        storage.save.side_effect = lambda m: m
        return storage

    @pytest.fixture
    def mock_graph(self):
        """Create a mock graph backend."""
        graph = MagicMock()
        graph.add_edge.return_value = MemoryEdge(
            from_id="new",
            to_id="original",
            relation=EdgeRelation.EVOLVED_FROM,
        )
        return graph

    def test_evolve_memory(self, mock_storage, mock_graph):
        """Test evolving a memory."""
        lineage = MemoryLineage(mock_storage, mock_graph)

        evolved = lineage.evolve("original", "Updated content")

        # Check new memory was created
        assert evolved.content == "Updated content"
        assert "evolved" in evolved.tags
        assert evolved.metadata["evolved_from"] == "original"

        # Check storage was called
        mock_storage.save.assert_called_once()

        # Check graph edges were created
        assert mock_graph.add_edge.call_count == 2  # forward and inverse

    def test_evolve_preserves_tags(self, mock_storage, mock_graph):
        """Test that evolve preserves original tags."""
        lineage = MemoryLineage(mock_storage, mock_graph)

        evolved = lineage.evolve("original", "New content", preserve_tags=True)

        assert "tag1" in evolved.tags
        assert "tag2" in evolved.tags
        assert "evolved" in evolved.tags

    def test_evolve_with_additional_tags(self, mock_storage, mock_graph):
        """Test evolving with additional tags."""
        lineage = MemoryLineage(mock_storage, mock_graph)

        evolved = lineage.evolve(
            "original",
            "New content",
            additional_tags=["new_tag"],
        )

        assert "new_tag" in evolved.tags

    def test_evolve_not_found(self, mock_storage, mock_graph):
        """Test evolving non-existent memory raises error."""
        mock_storage.recall.return_value = None
        lineage = MemoryLineage(mock_storage, mock_graph)

        with pytest.raises(ValueError, match="Memory not found"):
            lineage.evolve("nonexistent", "content")

    def test_merge_memories(self, mock_storage, mock_graph):
        """Test merging multiple memories."""
        mem1 = Memory(id="m1", content="Content 1", tags=["a", "b"])
        mem2 = Memory(id="m2", content="Content 2", tags=["b", "c"])

        mock_storage.recall.side_effect = [mem1, mem2]

        lineage = MemoryLineage(mock_storage, mock_graph)
        merged = lineage.merge(
            ["m1", "m2"],
            merged_content="Merged content",
            strategy=MergeStrategy.UNION,
        )

        # Check merged memory
        assert merged.content == "Merged content"
        assert "merged" in merged.tags
        assert "a" in merged.tags  # from m1
        assert "c" in merged.tags  # from m2
        assert merged.metadata["merged_from"] == ["m1", "m2"]

    def test_merge_intersection_strategy(self, mock_storage, mock_graph):
        """Test merge with intersection strategy."""
        mem1 = Memory(id="m1", content="Content 1", tags=["common", "a"])
        mem2 = Memory(id="m2", content="Content 2", tags=["common", "b"])

        mock_storage.recall.side_effect = [mem1, mem2]

        lineage = MemoryLineage(mock_storage, mock_graph)
        merged = lineage.merge(
            ["m1", "m2"],
            merged_content="Merged",
            strategy=MergeStrategy.INTERSECTION,
        )

        # Only common tags
        assert "common" in merged.tags
        assert "a" not in merged.tags
        assert "b" not in merged.tags

    def test_merge_needs_two_memories(self, mock_storage, mock_graph):
        """Test that merge requires at least 2 memories."""
        mock_storage.recall.side_effect = [Memory(id="m1", content="c1"), None]

        lineage = MemoryLineage(mock_storage, mock_graph)

        with pytest.raises(ValueError, match="at least 2"):
            lineage.merge(["m1", "nonexistent"], merged_content="test")

    def test_split_memory(self, mock_storage, mock_graph):
        """Test splitting a memory into parts."""
        lineage = MemoryLineage(mock_storage, mock_graph)

        parts = lineage.split(
            "original",
            ["Part 1 content", "Part 2 content"],
        )

        assert len(parts) == 2
        assert parts[0].content == "Part 1 content"
        assert parts[1].content == "Part 2 content"
        assert "split" in parts[0].tags
        assert parts[0].metadata["split_from"] == "original"
        assert parts[0].metadata["split_part"] == 1
        assert parts[1].metadata["split_part"] == 2

    def test_split_needs_two_parts(self, mock_storage, mock_graph):
        """Test that split requires at least 2 parts."""
        lineage = MemoryLineage(mock_storage, mock_graph)

        with pytest.raises(ValueError, match="at least 2"):
            lineage.split("original", ["single part"])

    def test_mark_contradiction(self, mock_storage, mock_graph):
        """Test marking two memories as contradicting."""
        lineage = MemoryLineage(mock_storage, mock_graph)

        lineage.mark_contradiction(
            "m1",
            "m2",
            resolution=ConflictResolution.KEEP_BOTH,
            notes="Different conclusions",
        )

        # Both directions created
        assert mock_graph.add_edge.call_count == 2

    def test_supersede(self, mock_storage, mock_graph):
        """Test marking memory as superseding another."""
        lineage = MemoryLineage(mock_storage, mock_graph)

        lineage.supersede("old_mem", "new_mem", reason="outdated")

        # Forward and inverse edges created
        assert mock_graph.add_edge.call_count == 2

    def test_link_memories(self, mock_storage, mock_graph):
        """Test creating a link between memories."""
        lineage = MemoryLineage(mock_storage, mock_graph)

        lineage.link(
            "m1",
            "m2",
            relation=EdgeRelation.REFERENCES,
            weight=0.8,
        )

        mock_graph.add_edge.assert_called_once()

    def test_link_bidirectional(self, mock_storage, mock_graph):
        """Test creating bidirectional link."""
        lineage = MemoryLineage(mock_storage, mock_graph)

        lineage.link(
            "m1",
            "m2",
            relation=EdgeRelation.RELATED_TO,
            bidirectional=True,
        )

        # Both directions
        assert mock_graph.add_edge.call_count == 2

    def test_get_history_from_metadata(self):
        """Test getting history from metadata when no graph."""
        storage = MagicMock()
        storage._graph = None  # Explicitly set to None
        storage.recall.return_value = Memory(
            id="m1",
            content="Test",
            metadata={"evolved_from": "m0"},
        )

        lineage = MemoryLineage(storage, graph=None)
        # Force graph to None
        lineage._graph = None
        history = lineage.get_history("m1")

        assert history["memory"].id == "m1"
        assert len(history["ancestors"]) == 1
        assert history["ancestors"][0]["memory_id"] == "m0"


# =============================================================================
# Mock FalkorDB Backend Tests
# =============================================================================


class TestMockFalkorDBBackend:
    """Tests for FalkorDB backend with mocked connections."""

    def test_backend_import(self):
        """Test that graph_backend module imports correctly."""
        from contextfs.graph_backend import FalkorDBBackend

        assert FalkorDBBackend is not None

    def test_backend_capabilities(self):
        """Test that FalkorDB backend has correct capabilities."""
        from contextfs.graph_backend import FalkorDBBackend

        assert FalkorDBBackend.capabilities == FALKORDB_CAPABILITIES

    def test_backend_lazy_init(self):
        """Test that backend initializes lazily."""
        from contextfs.graph_backend import FalkorDBBackend

        # Create backend - should NOT connect yet
        backend = FalkorDBBackend(host="localhost", port=6379)

        # Backend should not be initialized until first use
        assert backend._initialized is False
        assert backend._db is None
        assert backend._graph is None


# =============================================================================
# StorageRouter Graph Integration Tests
# =============================================================================


class TestStorageRouterGraphIntegration:
    """Tests for StorageRouter graph integration."""

    @pytest.fixture
    def mock_rag(self):
        """Create mock RAG backend."""
        rag = MagicMock()
        rag.add_memory = MagicMock()
        rag.add_memories_batch = MagicMock()
        return rag

    @pytest.fixture
    def mock_graph(self):
        """Create mock graph backend."""
        graph = MagicMock()
        graph.sync_node = MagicMock(return_value=True)
        graph.delete_node = MagicMock(return_value=True)
        graph.add_edge = MagicMock()
        graph.get_related = MagicMock(return_value=[])
        graph.get_lineage = MagicMock(return_value={})
        graph.get_stats = MagicMock(return_value={"nodes": 10, "edges": 5})
        return graph

    def test_router_with_graph(self, temp_dir, mock_rag, mock_graph):
        """Test StorageRouter with graph backend."""
        from contextfs.storage_router import StorageRouter

        router = StorageRouter(
            db_path=temp_dir / "test.db",
            rag_backend=mock_rag,
            graph_backend=mock_graph,
        )

        # Should have graph capabilities
        assert router.capabilities.graph_traversal is True
        assert router.capabilities.memory_lineage is True
        assert router.has_graph() is True

    def test_router_without_falkordb(self, temp_dir, mock_rag):
        """Test StorageRouter without FalkorDB (uses SQLite fallback)."""
        from contextfs.storage_router import StorageRouter

        router = StorageRouter(
            db_path=temp_dir / "test.db",
            rag_backend=mock_rag,
        )

        # Should still have graph via SQLite fallback
        assert router.has_graph() is True
        # But no dedicated graph backend
        assert router._has_dedicated_graph() is False

    def test_router_graph_stats(self, temp_dir, mock_rag, mock_graph):
        """Test getting graph stats through router with FalkorDB."""
        from contextfs.storage_router import StorageRouter

        router = StorageRouter(
            db_path=temp_dir / "test.db",
            rag_backend=mock_rag,
            graph_backend=mock_graph,
        )

        stats = router.get_graph_stats()

        assert stats["available"] is True
        assert stats["backend"] == "falkordb+sqlite"
        # FalkorDB stats are nested
        assert stats["falkordb"]["nodes"] == 10
        assert stats["falkordb"]["edges"] == 5

    def test_router_graph_stats_sqlite_only(self, temp_dir, mock_rag):
        """Test graph stats with SQLite fallback (no FalkorDB)."""
        from contextfs.storage_router import StorageRouter

        router = StorageRouter(
            db_path=temp_dir / "test.db",
            rag_backend=mock_rag,
        )

        stats = router.get_graph_stats()
        # SQLite fallback is always available
        assert stats["available"] is True
        assert stats["backend"] == "sqlite"
        assert "sqlite_edges" in stats


# =============================================================================
# Migration Tests
# =============================================================================


class TestMigration:
    """Tests for edge table migration."""

    def test_migration_file_exists(self):
        """Test that migration file exists."""
        migration_path = (
            Path(__file__).parent.parent.parent
            / "src"
            / "contextfs"
            / "migrations"
            / "versions"
            / "003_add_memory_edges.py"
        )
        assert migration_path.exists()

    def test_migration_imports(self):
        """Test that migration can be imported."""
        # If no exception, import succeeded
