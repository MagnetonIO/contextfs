"""
Storage Backend Protocol for ContextFS.

Defines type-safe interfaces for pluggable storage backends.
Enables modular architecture where SQLite, ChromaDB, Postgres,
FalkorDB, or other backends can be swapped or combined.

Usage:
    class MyCustomBackend:
        def save(self, memory: Memory) -> Memory: ...
        def recall(self, memory_id: str) -> Memory | None: ...
        # ... implement all protocol methods

    # Type checker validates implementation
    backend: StorageBackend = MyCustomBackend()
"""

from abc import abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from contextfs.schemas import Memory, MemoryType, SearchResult

# =============================================================================
# Graph Relationship Types
# =============================================================================


class EdgeRelation(str, Enum):
    """
    Types of relationships between memories.

    Defines the semantic meaning of edges in the memory graph.
    """

    # Evolution relationships
    EVOLVED_INTO = "evolved_into"  # Memory was updated/refined
    EVOLVED_FROM = "evolved_from"  # Inverse of evolved_into

    # Merge relationships
    MERGED_INTO = "merged_into"  # Multiple memories combined
    MERGED_FROM = "merged_from"  # Inverse: this memory came from merge

    # Split relationships
    SPLIT_INTO = "split_into"  # Memory was split into parts
    SPLIT_FROM = "split_from"  # Inverse: this came from a split

    # Reference relationships
    REFERENCES = "references"  # Memory references another
    REFERENCED_BY = "referenced_by"  # Inverse of references

    # Semantic relationships
    RELATED_TO = "related_to"  # General semantic similarity
    CONTRADICTS = "contradicts"  # Conflicting information
    SUPERSEDES = "supersedes"  # Replaces outdated memory
    SUPERSEDED_BY = "superseded_by"  # Inverse of supersedes

    # Hierarchical relationships
    PARENT_OF = "parent_of"  # Parent concept/topic
    CHILD_OF = "child_of"  # Child concept/topic
    PART_OF = "part_of"  # Component relationship
    CONTAINS = "contains"  # Inverse of part_of

    # Causal relationships
    CAUSED_BY = "caused_by"  # Causal dependency
    CAUSES = "causes"  # Inverse of caused_by

    @classmethod
    def get_inverse(cls, relation: "EdgeRelation") -> "EdgeRelation":
        """Get the inverse relationship."""
        inverses = {
            cls.EVOLVED_INTO: cls.EVOLVED_FROM,
            cls.EVOLVED_FROM: cls.EVOLVED_INTO,
            cls.MERGED_INTO: cls.MERGED_FROM,
            cls.MERGED_FROM: cls.MERGED_INTO,
            cls.SPLIT_INTO: cls.SPLIT_FROM,
            cls.SPLIT_FROM: cls.SPLIT_INTO,
            cls.REFERENCES: cls.REFERENCED_BY,
            cls.REFERENCED_BY: cls.REFERENCES,
            cls.SUPERSEDES: cls.SUPERSEDED_BY,
            cls.SUPERSEDED_BY: cls.SUPERSEDES,
            cls.PARENT_OF: cls.CHILD_OF,
            cls.CHILD_OF: cls.PARENT_OF,
            cls.PART_OF: cls.CONTAINS,
            cls.CONTAINS: cls.PART_OF,
            cls.CAUSED_BY: cls.CAUSES,
            cls.CAUSES: cls.CAUSED_BY,
            cls.RELATED_TO: cls.RELATED_TO,  # Symmetric
            cls.CONTRADICTS: cls.CONTRADICTS,  # Symmetric
        }
        return inverses.get(relation, relation)


class MemoryEdge(BaseModel):
    """
    An edge (relationship) between two memories.

    Represents a typed, directed relationship in the memory graph.
    """

    from_id: str = Field(..., description="Source memory ID")
    to_id: str = Field(..., description="Target memory ID")
    relation: EdgeRelation = Field(..., description="Type of relationship")

    # Edge metadata
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Edge strength")
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str | None = Field(default=None, description="Tool that created edge")

    # Additional properties
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True  # Edges are immutable once created


class GraphPath(BaseModel):
    """A path through the memory graph."""

    nodes: list[Memory] = Field(..., description="Memories in path order")
    edges: list[MemoryEdge] = Field(..., description="Edges connecting nodes")
    total_weight: float = Field(default=0.0, description="Sum of edge weights")


class GraphTraversalResult(BaseModel):
    """Result of a graph traversal operation."""

    memory: Memory
    relation: EdgeRelation
    depth: int = Field(ge=0, description="Distance from origin")
    path_weight: float = Field(default=1.0, description="Product of edge weights")


@runtime_checkable
class StorageBackend(Protocol):
    """
    Protocol for storage backends.

    Any class implementing these methods can be used as a storage backend.
    Use @runtime_checkable to allow isinstance() checks.
    """

    @abstractmethod
    def save(self, memory: Memory) -> Memory:
        """
        Save a memory to storage.

        Args:
            memory: Memory object to save

        Returns:
            Saved Memory object (may have updated fields)
        """
        ...

    @abstractmethod
    def save_batch(self, memories: list[Memory]) -> int:
        """
        Save multiple memories in batch.

        Args:
            memories: List of Memory objects to save

        Returns:
            Number of memories successfully saved
        """
        ...

    @abstractmethod
    def recall(self, memory_id: str) -> Memory | None:
        """
        Recall a specific memory by ID.

        Args:
            memory_id: Memory ID (can be partial, at least 8 chars)

        Returns:
            Memory if found, None otherwise
        """
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 10,
        type: MemoryType | None = None,
        tags: list[str] | None = None,
        namespace_id: str | None = None,
        project: str | None = None,
        min_score: float = 0.3,
    ) -> list[SearchResult]:
        """
        Search memories.

        Args:
            query: Search query
            limit: Maximum results
            type: Filter by memory type
            tags: Filter by tags
            namespace_id: Filter by namespace
            project: Filter by project
            min_score: Minimum similarity score

        Returns:
            List of SearchResult objects
        """
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: Memory ID (can be partial)

        Returns:
            True if deleted, False if not found
        """
        ...

    @abstractmethod
    def delete_by_namespace(self, namespace_id: str) -> int:
        """
        Delete all memories in a namespace.

        Args:
            namespace_id: Namespace to clear

        Returns:
            Number of memories deleted
        """
        ...


@runtime_checkable
class SearchableBackend(Protocol):
    """
    Protocol for backends that support semantic search.

    Extends basic storage with vector similarity search.
    """

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 10,
        type: MemoryType | None = None,
        namespace_id: str | None = None,
        min_score: float = 0.3,
    ) -> list[SearchResult]:
        """Semantic search for similar memories."""
        ...

    @abstractmethod
    def get_embedding(self, text: str) -> list[float]:
        """Generate embedding vector for text."""
        ...


@runtime_checkable
class PersistentBackend(Protocol):
    """
    Protocol for backends with SQL-like persistent storage.

    Supports structured queries and transactions.
    """

    @abstractmethod
    def save(self, memory: Memory) -> Memory:
        """Save memory to persistent storage."""
        ...

    @abstractmethod
    def recall(self, memory_id: str) -> Memory | None:
        """Recall by exact or partial ID."""
        ...

    @abstractmethod
    def list_recent(
        self,
        limit: int = 10,
        type: MemoryType | None = None,
        namespace_id: str | None = None,
    ) -> list[Memory]:
        """List recent memories with filters."""
        ...

    @abstractmethod
    def update(
        self,
        memory_id: str,
        content: str | None = None,
        type: MemoryType | None = None,
        tags: list[str] | None = None,
        summary: str | None = None,
    ) -> Memory | None:
        """Update an existing memory."""
        ...


@runtime_checkable
class SyncableBackend(Protocol):
    """
    Protocol for backends that support synchronization.

    Used for multi-device sync, backup, and replication.
    """

    @abstractmethod
    def get_changes_since(self, timestamp: str) -> list[Memory]:
        """Get all changes since a timestamp."""
        ...

    @abstractmethod
    def apply_changes(self, memories: list[Memory]) -> int:
        """Apply changes from another source."""
        ...

    @abstractmethod
    def get_sync_status(self) -> dict:
        """Get synchronization status."""
        ...


# =============================================================================
# Graph Backend Protocol
# =============================================================================


@runtime_checkable
class GraphBackend(Protocol):
    """
    Protocol for backends supporting graph operations.

    Enables memory lineage tracking, relationship modeling,
    and graph traversal queries.

    Implementations: FalkorDB, Neo4j, SQLite (fallback via edge table)
    """

    @abstractmethod
    def add_edge(
        self,
        from_id: str,
        to_id: str,
        relation: EdgeRelation,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEdge:
        """
        Create a directed edge between two memories.

        Args:
            from_id: Source memory ID
            to_id: Target memory ID
            relation: Type of relationship
            weight: Edge strength (0.0-1.0)
            metadata: Additional edge properties

        Returns:
            Created MemoryEdge object

        Raises:
            ValueError: If either memory doesn't exist
        """
        ...

    @abstractmethod
    def remove_edge(
        self,
        from_id: str,
        to_id: str,
        relation: EdgeRelation | None = None,
    ) -> bool:
        """
        Remove edge(s) between two memories.

        Args:
            from_id: Source memory ID
            to_id: Target memory ID
            relation: Specific relation to remove (None = all relations)

        Returns:
            True if any edges were removed
        """
        ...

    @abstractmethod
    def get_edges(
        self,
        memory_id: str,
        direction: Literal["outgoing", "incoming", "both"] = "both",
        relation: EdgeRelation | None = None,
    ) -> list[MemoryEdge]:
        """
        Get all edges connected to a memory.

        Args:
            memory_id: Memory ID to query
            direction: Edge direction filter
            relation: Filter by relation type

        Returns:
            List of MemoryEdge objects
        """
        ...

    @abstractmethod
    def get_related(
        self,
        memory_id: str,
        relation: EdgeRelation | None = None,
        direction: Literal["outgoing", "incoming", "both"] = "outgoing",
        max_depth: int = 1,
        min_weight: float = 0.0,
    ) -> list[GraphTraversalResult]:
        """
        Get memories related to a given memory via graph traversal.

        Args:
            memory_id: Starting memory ID
            relation: Filter by relation type (None = all)
            direction: Traversal direction
            max_depth: Maximum hops from origin
            min_weight: Minimum edge weight to traverse

        Returns:
            List of (Memory, relation, depth) tuples
        """
        ...

    @abstractmethod
    def find_path(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 5,
        relation: EdgeRelation | None = None,
    ) -> GraphPath | None:
        """
        Find shortest path between two memories.

        Args:
            from_id: Starting memory ID
            to_id: Target memory ID
            max_depth: Maximum path length
            relation: Restrict to specific relation type

        Returns:
            GraphPath if found, None otherwise
        """
        ...

    @abstractmethod
    def get_subgraph(
        self,
        root_id: str,
        max_depth: int = 2,
        relation: EdgeRelation | None = None,
    ) -> dict[str, Any]:
        """
        Extract a subgraph rooted at a memory.

        Args:
            root_id: Root memory ID
            max_depth: Maximum depth to traverse
            relation: Filter by relation type

        Returns:
            Dict with 'nodes' (list[Memory]) and 'edges' (list[MemoryEdge])
        """
        ...

    @abstractmethod
    def get_lineage(
        self,
        memory_id: str,
        direction: Literal["ancestors", "descendants", "both"] = "both",
    ) -> dict[str, Any]:
        """
        Get the evolution lineage of a memory.

        Traces EVOLVED_FROM, MERGED_FROM, SPLIT_FROM relationships.

        Args:
            memory_id: Memory to trace
            direction: Which direction to trace

        Returns:
            Dict with 'root', 'ancestors', 'descendants', 'history'
        """
        ...

    @abstractmethod
    def sync_node(self, memory: Memory) -> bool:
        """
        Sync a memory node to the graph backend.

        Called by StorageRouter to keep graph in sync with primary storage.

        Args:
            memory: Memory to sync as a node

        Returns:
            True if sync successful
        """
        ...

    @abstractmethod
    def delete_node(self, memory_id: str) -> bool:
        """
        Delete a memory node and all its edges from the graph.

        Args:
            memory_id: Memory ID to delete

        Returns:
            True if deleted, False if not found
        """
        ...


class StorageCapabilities:
    """
    Describes what a storage backend supports.

    Used for feature detection at runtime.
    Capabilities form a lattice under subset ordering.
    """

    def __init__(
        self,
        semantic_search: bool = False,
        full_text_search: bool = False,
        persistent: bool = False,
        syncable: bool = False,
        batch_operations: bool = False,
        transactions: bool = False,
        # Graph capabilities
        graph_traversal: bool = False,
        memory_lineage: bool = False,
        path_finding: bool = False,
    ):
        self.semantic_search = semantic_search
        self.full_text_search = full_text_search
        self.persistent = persistent
        self.syncable = syncable
        self.batch_operations = batch_operations
        self.transactions = transactions
        # Graph capabilities
        self.graph_traversal = graph_traversal
        self.memory_lineage = memory_lineage
        self.path_finding = path_finding

    def __repr__(self) -> str:
        caps = []
        if self.semantic_search:
            caps.append("semantic_search")
        if self.full_text_search:
            caps.append("fts")
        if self.persistent:
            caps.append("persistent")
        if self.syncable:
            caps.append("syncable")
        if self.batch_operations:
            caps.append("batch")
        if self.transactions:
            caps.append("transactions")
        if self.graph_traversal:
            caps.append("graph_traversal")
        if self.memory_lineage:
            caps.append("memory_lineage")
        if self.path_finding:
            caps.append("path_finding")
        return f"StorageCapabilities({', '.join(caps)})"

    def __le__(self, other: "StorageCapabilities") -> bool:
        """Check if self's capabilities are subset of other's."""
        return all(
            [
                (not self.semantic_search) or other.semantic_search,
                (not self.full_text_search) or other.full_text_search,
                (not self.persistent) or other.persistent,
                (not self.syncable) or other.syncable,
                (not self.batch_operations) or other.batch_operations,
                (not self.transactions) or other.transactions,
                (not self.graph_traversal) or other.graph_traversal,
                (not self.memory_lineage) or other.memory_lineage,
                (not self.path_finding) or other.path_finding,
            ]
        )

    def __or__(self, other: "StorageCapabilities") -> "StorageCapabilities":
        """Combine capabilities (join in lattice)."""
        return StorageCapabilities(
            semantic_search=self.semantic_search or other.semantic_search,
            full_text_search=self.full_text_search or other.full_text_search,
            persistent=self.persistent or other.persistent,
            syncable=self.syncable or other.syncable,
            batch_operations=self.batch_operations or other.batch_operations,
            transactions=self.transactions or other.transactions,
            graph_traversal=self.graph_traversal or other.graph_traversal,
            memory_lineage=self.memory_lineage or other.memory_lineage,
            path_finding=self.path_finding or other.path_finding,
        )

    def __and__(self, other: "StorageCapabilities") -> "StorageCapabilities":
        """Intersect capabilities (meet in lattice)."""
        return StorageCapabilities(
            semantic_search=self.semantic_search and other.semantic_search,
            full_text_search=self.full_text_search and other.full_text_search,
            persistent=self.persistent and other.persistent,
            syncable=self.syncable and other.syncable,
            batch_operations=self.batch_operations and other.batch_operations,
            transactions=self.transactions and other.transactions,
            graph_traversal=self.graph_traversal and other.graph_traversal,
            memory_lineage=self.memory_lineage and other.memory_lineage,
            path_finding=self.path_finding and other.path_finding,
        )

    def has_graph(self) -> bool:
        """Check if any graph capabilities are present."""
        return self.graph_traversal or self.memory_lineage or self.path_finding


# =============================================================================
# Common Capability Configurations
# =============================================================================

SQLITE_CAPABILITIES = StorageCapabilities(
    full_text_search=True,
    persistent=True,
    batch_operations=True,
    transactions=True,
)

CHROMADB_CAPABILITIES = StorageCapabilities(
    semantic_search=True,
    batch_operations=True,
)

FALKORDB_CAPABILITIES = StorageCapabilities(
    persistent=True,
    graph_traversal=True,
    memory_lineage=True,
    path_finding=True,
    transactions=True,
)

# SQLite edge table fallback (limited graph support)
SQLITE_GRAPH_CAPABILITIES = StorageCapabilities(
    full_text_search=True,
    persistent=True,
    batch_operations=True,
    transactions=True,
    graph_traversal=True,  # Basic traversal via joins
    memory_lineage=True,  # Lineage queries possible
    # path_finding=False - Complex path finding not efficient in SQLite
)

UNIFIED_CAPABILITIES = StorageCapabilities(
    semantic_search=True,
    full_text_search=True,
    persistent=True,
    batch_operations=True,
    transactions=True,
)

# Full capabilities with FalkorDB
UNIFIED_WITH_GRAPH_CAPABILITIES = StorageCapabilities(
    semantic_search=True,
    full_text_search=True,
    persistent=True,
    batch_operations=True,
    transactions=True,
    graph_traversal=True,
    memory_lineage=True,
    path_finding=True,
)
