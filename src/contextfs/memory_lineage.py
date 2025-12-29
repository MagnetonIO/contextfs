"""
Memory Lineage Operations for ContextFS.

Provides high-level operations for memory evolution, merging, and splitting.
Works with any GraphBackend implementation (FalkorDB, SQLite fallback).

Usage:
    from contextfs.memory_lineage import MemoryLineage

    lineage = MemoryLineage(storage_router)

    # Evolve a memory (update with history preservation)
    evolved = lineage.evolve("mem123", "Updated content...")

    # Merge multiple memories
    merged = lineage.merge(["mem1", "mem2", "mem3"], strategy="union")

    # Split a memory into parts
    parts = lineage.split("mem123", ["Part 1", "Part 2"])
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

from contextfs.schemas import Memory, MemoryType
from contextfs.storage_protocol import (
    EdgeRelation,
    GraphBackend,
    MemoryEdge,
)

if TYPE_CHECKING:
    from contextfs.storage_router import StorageRouter

logger = logging.getLogger(__name__)


class MergeStrategy(str, Enum):
    """Strategies for merging multiple memories."""

    UNION = "union"  # Combine all tags and metadata
    INTERSECTION = "intersection"  # Keep only common elements
    LATEST = "latest"  # Use most recent memory's attributes
    OLDEST = "oldest"  # Use oldest memory's attributes
    WEIGHTED = "weighted"  # Weight by edge weights or scores


class ConflictResolution(str, Enum):
    """How to handle conflicting information."""

    KEEP_BOTH = "keep_both"  # Mark as contradicting, keep both
    SUPERSEDE = "supersede"  # New memory supersedes old
    MANUAL = "manual"  # Require manual resolution


class MemoryLineage:
    """
    High-level operations for memory lineage management.

    Provides evolve, merge, split, and conflict resolution operations
    that maintain proper graph relationships between memories.

    Attributes:
        storage: StorageRouter for memory persistence
        graph: GraphBackend for relationship management
    """

    def __init__(
        self,
        storage: StorageRouter,
        graph: GraphBackend | None = None,
    ) -> None:
        """
        Initialize MemoryLineage.

        Args:
            storage: StorageRouter instance
            graph: Optional GraphBackend (uses storage._graph if available)
        """
        self._storage = storage
        self._graph = graph or getattr(storage, "_graph", None)

    # =========================================================================
    # Evolution Operations
    # =========================================================================

    def evolve(
        self,
        memory_id: str,
        new_content: str,
        summary: str | None = None,
        preserve_tags: bool = True,
        additional_tags: list[str] | None = None,
    ) -> Memory:
        """
        Evolve a memory by creating an updated version while preserving history.

        The original memory remains unchanged. A new memory is created with
        an EVOLVED_FROM relationship to the original.

        Args:
            memory_id: ID of memory to evolve
            new_content: Updated content for new memory
            summary: Optional new summary
            preserve_tags: Whether to copy tags from original
            additional_tags: Additional tags for new memory

        Returns:
            New evolved Memory object

        Raises:
            ValueError: If original memory not found

        Example:
            >>> evolved = lineage.evolve("abc123", "Updated documentation...")
            >>> print(evolved.metadata["evolved_from"])
            'abc123'
        """
        # Get original memory
        original = self._storage.recall(memory_id)
        if not original:
            raise ValueError(f"Memory not found: {memory_id}")

        # Build tags
        tags = []
        if preserve_tags:
            tags.extend(original.tags)
        if additional_tags:
            tags.extend(additional_tags)
        if "evolved" not in tags:
            tags.append("evolved")
        tags = list(set(tags))  # Deduplicate

        # Create evolved memory
        evolved = Memory(
            content=new_content,
            type=original.type,
            tags=tags,
            summary=summary or original.summary,
            namespace_id=original.namespace_id,
            source_repo=original.source_repo,
            project=original.project,
            source_tool=original.source_tool,
            metadata={
                **original.metadata,
                "evolved_from": memory_id,
                "evolution_timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        # Save evolved memory
        self._storage.save(evolved)

        # Create graph relationships
        if self._graph:
            try:
                # Evolved memory -> original
                self._graph.add_edge(
                    from_id=evolved.id,
                    to_id=memory_id,
                    relation=EdgeRelation.EVOLVED_FROM,
                )
                # Original -> evolved memory (inverse)
                self._graph.add_edge(
                    from_id=memory_id,
                    to_id=evolved.id,
                    relation=EdgeRelation.EVOLVED_INTO,
                )
            except Exception as e:
                logger.warning(f"Failed to create evolution edges: {e}")

        logger.info(f"Evolved memory {memory_id} -> {evolved.id}")
        return evolved

    # =========================================================================
    # Merge Operations
    # =========================================================================

    def merge(
        self,
        memory_ids: list[str],
        merged_content: str | None = None,
        summary: str | None = None,
        strategy: MergeStrategy = MergeStrategy.UNION,
        memory_type: MemoryType | None = None,
    ) -> Memory:
        """
        Merge multiple memories into a single memory.

        Creates a new memory with MERGED_FROM relationships to all originals.
        Original memories are not modified.

        Args:
            memory_ids: List of memory IDs to merge (minimum 2)
            merged_content: Content for merged memory (auto-generated if None)
            summary: Summary for merged memory
            strategy: How to combine tags and metadata
            memory_type: Type for merged memory (uses most common if None)

        Returns:
            New merged Memory object

        Raises:
            ValueError: If fewer than 2 valid memories provided

        Example:
            >>> merged = lineage.merge(["mem1", "mem2"], strategy=MergeStrategy.UNION)
            >>> print(merged.metadata["merged_from"])
            ['mem1', 'mem2']
        """
        # Fetch all memories
        memories = []
        for mid in memory_ids:
            memory = self._storage.recall(mid)
            if memory:
                memories.append(memory)
            else:
                logger.warning(f"Memory not found for merge: {mid}")

        if len(memories) < 2:
            raise ValueError(f"Need at least 2 valid memories to merge, got {len(memories)}")

        # Determine merged attributes based on strategy
        tags, metadata, final_type = self._apply_merge_strategy(memories, strategy)

        # Add merge-specific tags
        if "merged" not in tags:
            tags.append("merged")

        # Auto-generate content if not provided
        if merged_content is None:
            merged_content = self._generate_merged_content(memories)

        # Determine memory type
        if memory_type:
            final_type = memory_type

        # Create merged memory
        merged = Memory(
            content=merged_content,
            type=final_type,
            tags=list(set(tags)),
            summary=summary or f"Merged from {len(memories)} memories",
            namespace_id=memories[0].namespace_id,
            source_repo=memories[0].source_repo,
            project=memories[0].project,
            source_tool=memories[0].source_tool,
            metadata={
                **metadata,
                "merged_from": [m.id for m in memories],
                "merge_strategy": strategy.value,
                "merge_timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        # Save merged memory
        self._storage.save(merged)

        # Create graph relationships
        if self._graph:
            for original in memories:
                try:
                    # Merged memory -> original
                    self._graph.add_edge(
                        from_id=merged.id,
                        to_id=original.id,
                        relation=EdgeRelation.MERGED_FROM,
                    )
                    # Original -> merged memory (inverse)
                    self._graph.add_edge(
                        from_id=original.id,
                        to_id=merged.id,
                        relation=EdgeRelation.MERGED_INTO,
                    )
                except Exception as e:
                    logger.warning(f"Failed to create merge edge: {e}")

        logger.info(f"Merged {len(memories)} memories -> {merged.id}")
        return merged

    def _apply_merge_strategy(
        self,
        memories: list[Memory],
        strategy: MergeStrategy,
    ) -> tuple[list[str], dict[str, Any], MemoryType]:
        """Apply merge strategy to determine combined attributes."""
        if strategy == MergeStrategy.UNION:
            # Combine all tags
            all_tags = [tag for m in memories for tag in m.tags]
            # Combine metadata (later memories override)
            combined_meta = {}
            for m in memories:
                combined_meta.update(m.metadata)
            # Most common type
            types = [m.type for m in memories]
            final_type = max(set(types), key=types.count)

        elif strategy == MergeStrategy.INTERSECTION:
            # Keep only common tags
            tag_sets = [set(m.tags) for m in memories]
            all_tags = list(set.intersection(*tag_sets)) if tag_sets else []
            # Keep only common metadata keys
            if memories:
                common_keys = set(memories[0].metadata.keys())
                for m in memories[1:]:
                    common_keys &= set(m.metadata.keys())
                combined_meta = {k: memories[0].metadata[k] for k in common_keys}
            else:
                combined_meta = {}
            # Most common type
            types = [m.type for m in memories]
            final_type = max(set(types), key=types.count)

        elif strategy == MergeStrategy.LATEST:
            # Use most recent memory
            latest = max(memories, key=lambda m: m.created_at)
            all_tags = latest.tags.copy()
            combined_meta = latest.metadata.copy()
            final_type = latest.type

        elif strategy == MergeStrategy.OLDEST:
            # Use oldest memory
            oldest = min(memories, key=lambda m: m.created_at)
            all_tags = oldest.tags.copy()
            combined_meta = oldest.metadata.copy()
            final_type = oldest.type

        else:  # WEIGHTED - default to union
            all_tags = [tag for m in memories for tag in m.tags]
            combined_meta = {}
            for m in memories:
                combined_meta.update(m.metadata)
            types = [m.type for m in memories]
            final_type = max(set(types), key=types.count)

        return all_tags, combined_meta, final_type

    def _generate_merged_content(self, memories: list[Memory]) -> str:
        """Generate content for merged memory."""
        sections = []
        for i, m in enumerate(memories, 1):
            sections.append(f"[Source {i}]: {m.content[:500]}")
        return "\n\n".join(sections)

    # =========================================================================
    # Split Operations
    # =========================================================================

    def split(
        self,
        memory_id: str,
        parts: list[str],
        summaries: list[str] | None = None,
        preserve_tags: bool = True,
    ) -> list[Memory]:
        """
        Split a memory into multiple parts.

        Creates new memories with SPLIT_FROM relationships to the original.
        Original memory is not modified.

        Args:
            memory_id: ID of memory to split
            parts: List of content strings for each part
            summaries: Optional summaries for each part
            preserve_tags: Whether to copy tags from original

        Returns:
            List of new Memory objects

        Raises:
            ValueError: If original memory not found or fewer than 2 parts

        Example:
            >>> parts = lineage.split("mem123", [
            ...     "First concept...",
            ...     "Second concept...",
            ... ])
            >>> print(parts[0].metadata["split_from"])
            'mem123'
        """
        if len(parts) < 2:
            raise ValueError("Need at least 2 parts to split a memory")

        # Get original memory
        original = self._storage.recall(memory_id)
        if not original:
            raise ValueError(f"Memory not found: {memory_id}")

        # Ensure summaries list matches parts
        if summaries and len(summaries) != len(parts):
            raise ValueError("Number of summaries must match number of parts")

        created_memories = []

        for i, content in enumerate(parts):
            # Build tags
            tags = []
            if preserve_tags:
                tags.extend(original.tags)
            tags.append("split")
            tags.append(f"split_part_{i + 1}")
            tags = list(set(tags))

            # Create split memory
            split_memory = Memory(
                content=content,
                type=original.type,
                tags=tags,
                summary=summaries[i] if summaries else f"Part {i + 1} of split memory",
                namespace_id=original.namespace_id,
                source_repo=original.source_repo,
                project=original.project,
                source_tool=original.source_tool,
                metadata={
                    **original.metadata,
                    "split_from": memory_id,
                    "split_part": i + 1,
                    "split_total": len(parts),
                    "split_timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Save split memory
            self._storage.save(split_memory)
            created_memories.append(split_memory)

            # Create graph relationships
            if self._graph:
                try:
                    # Split memory -> original
                    self._graph.add_edge(
                        from_id=split_memory.id,
                        to_id=memory_id,
                        relation=EdgeRelation.SPLIT_FROM,
                    )
                    # Original -> split memory (inverse)
                    self._graph.add_edge(
                        from_id=memory_id,
                        to_id=split_memory.id,
                        relation=EdgeRelation.SPLIT_INTO,
                    )
                except Exception as e:
                    logger.warning(f"Failed to create split edge: {e}")

        logger.info(f"Split memory {memory_id} into {len(parts)} parts")
        return created_memories

    # =========================================================================
    # Conflict Resolution
    # =========================================================================

    def mark_contradiction(
        self,
        memory_id_1: str,
        memory_id_2: str,
        resolution: ConflictResolution = ConflictResolution.KEEP_BOTH,
        notes: str | None = None,
    ) -> tuple[MemoryEdge | None, MemoryEdge | None]:
        """
        Mark two memories as contradicting each other.

        Args:
            memory_id_1: First memory ID
            memory_id_2: Second memory ID
            resolution: How the conflict was/should be resolved
            notes: Optional notes about the contradiction

        Returns:
            Tuple of created edges (both directions)
        """
        edges: list[MemoryEdge | None] = [None, None]

        if self._graph:
            metadata = {
                "resolution": resolution.value,
                "notes": notes,
                "detected_at": datetime.now(timezone.utc).isoformat(),
            }

            try:
                edges[0] = self._graph.add_edge(
                    from_id=memory_id_1,
                    to_id=memory_id_2,
                    relation=EdgeRelation.CONTRADICTS,
                    metadata=metadata,
                )
                edges[1] = self._graph.add_edge(
                    from_id=memory_id_2,
                    to_id=memory_id_1,
                    relation=EdgeRelation.CONTRADICTS,
                    metadata=metadata,
                )
                logger.info(f"Marked contradiction: {memory_id_1} <-> {memory_id_2}")
            except Exception as e:
                logger.warning(f"Failed to create contradiction edges: {e}")

        return edges[0], edges[1]

    def supersede(
        self,
        old_memory_id: str,
        new_memory_id: str,
        reason: str | None = None,
    ) -> MemoryEdge | None:
        """
        Mark a memory as superseding another (making it obsolete).

        Args:
            old_memory_id: ID of outdated memory
            new_memory_id: ID of newer, correct memory
            reason: Reason for superseding

        Returns:
            Created edge or None
        """
        if not self._graph:
            return None

        try:
            edge = self._graph.add_edge(
                from_id=new_memory_id,
                to_id=old_memory_id,
                relation=EdgeRelation.SUPERSEDES,
                metadata={
                    "reason": reason,
                    "superseded_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            # Also create inverse
            self._graph.add_edge(
                from_id=old_memory_id,
                to_id=new_memory_id,
                relation=EdgeRelation.SUPERSEDED_BY,
            )
            logger.info(f"Marked {new_memory_id} supersedes {old_memory_id}")
            return edge
        except Exception as e:
            logger.warning(f"Failed to create supersede edge: {e}")
            return None

    # =========================================================================
    # Reference Operations
    # =========================================================================

    def link(
        self,
        from_memory_id: str,
        to_memory_id: str,
        relation: EdgeRelation = EdgeRelation.REFERENCES,
        weight: float = 1.0,
        bidirectional: bool = False,
    ) -> MemoryEdge | None:
        """
        Create a relationship link between two memories.

        Args:
            from_memory_id: Source memory ID
            to_memory_id: Target memory ID
            relation: Type of relationship
            weight: Relationship strength (0.0-1.0)
            bidirectional: Whether to create inverse edge

        Returns:
            Created edge or None
        """
        if not self._graph:
            return None

        try:
            edge = self._graph.add_edge(
                from_id=from_memory_id,
                to_id=to_memory_id,
                relation=relation,
                weight=weight,
            )

            if bidirectional:
                inverse = EdgeRelation.get_inverse(relation)
                self._graph.add_edge(
                    from_id=to_memory_id,
                    to_id=from_memory_id,
                    relation=inverse,
                    weight=weight,
                )

            logger.debug(f"Linked {from_memory_id} -{relation.value}-> {to_memory_id}")
            return edge
        except Exception as e:
            logger.warning(f"Failed to create link: {e}")
            return None

    # =========================================================================
    # Lineage Queries
    # =========================================================================

    def get_history(self, memory_id: str) -> dict[str, Any]:
        """
        Get the complete evolution history of a memory.

        Traces all ancestors and descendants through evolution,
        merge, and split relationships.

        Args:
            memory_id: Memory ID to trace

        Returns:
            Dict with:
                - root: ID of original ancestor
                - memory: Current memory object
                - ancestors: List of ancestor memories with depths
                - descendants: List of descendant memories with depths
                - timeline: Chronologically ordered history
        """
        memory = self._storage.recall(memory_id)
        if not memory:
            return {"error": f"Memory not found: {memory_id}"}

        result: dict[str, Any] = {
            "root": memory_id,
            "memory": memory,
            "ancestors": [],
            "descendants": [],
            "timeline": [],
        }

        if self._graph:
            try:
                lineage = self._graph.get_lineage(memory_id, direction="both")
                result["root"] = lineage.get("root", memory_id)
                result["ancestors"] = lineage.get("ancestors", [])
                result["descendants"] = lineage.get("descendants", [])
                result["timeline"] = lineage.get("history", [])
            except Exception as e:
                logger.warning(f"Failed to get lineage from graph: {e}")

        # Fallback to metadata
        if not result["ancestors"] and memory.metadata.get("evolved_from"):
            result["ancestors"].append(
                {
                    "memory_id": memory.metadata["evolved_from"],
                    "relation": "evolved_from",
                }
            )

        if not result["ancestors"] and memory.metadata.get("merged_from"):
            for mid in memory.metadata["merged_from"]:
                result["ancestors"].append(
                    {
                        "memory_id": mid,
                        "relation": "merged_from",
                    }
                )

        if not result["ancestors"] and memory.metadata.get("split_from"):
            result["ancestors"].append(
                {
                    "memory_id": memory.metadata["split_from"],
                    "relation": "split_from",
                }
            )

        return result

    def find_related(
        self,
        memory_id: str,
        relation: EdgeRelation | None = None,
        max_depth: int = 2,
    ) -> list[dict[str, Any]]:
        """
        Find memories related to a given memory.

        Args:
            memory_id: Starting memory ID
            relation: Filter by relation type (None = all)
            max_depth: Maximum traversal depth

        Returns:
            List of related memories with relationship info
        """
        if not self._graph:
            return []

        try:
            results = self._graph.get_related(
                memory_id=memory_id,
                relation=relation,
                direction="both",
                max_depth=max_depth,
            )
            return [
                {
                    "memory": r.memory,
                    "relation": r.relation.value,
                    "depth": r.depth,
                    "weight": r.path_weight,
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Failed to find related: {e}")
            return []
