"""
Graph Backend Implementation for ContextFS.

Provides FalkorDB-based graph storage for memory relationships,
lineage tracking, and graph traversal operations.

Implements the GraphBackend protocol with full type safety.

Usage:
    from contextfs.graph_backend import FalkorDBBackend

    # Connect to FalkorDB
    graph = FalkorDBBackend(host="localhost", port=6379)

    # Add relationship
    edge = graph.add_edge("mem1", "mem2", EdgeRelation.EVOLVED_INTO)

    # Traverse relationships
    related = graph.get_related("mem1", max_depth=2)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from contextfs.schemas import Memory, MemoryType
from contextfs.storage_protocol import (
    FALKORDB_CAPABILITIES,
    EdgeRelation,
    GraphPath,
    GraphTraversalResult,
    MemoryEdge,
    StorageCapabilities,
)

if TYPE_CHECKING:
    from falkordb import FalkorDB, Graph

logger = logging.getLogger(__name__)


class FalkorDBBackend:
    """
    FalkorDB implementation of the GraphBackend protocol.

    Provides graph storage for memory relationships using Cypher queries.
    FalkorDB is a high-performance graph database optimized for GraphRAG.

    Attributes:
        capabilities: StorageCapabilities descriptor for this backend
    """

    capabilities: StorageCapabilities = FALKORDB_CAPABILITIES

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str | None = None,
        graph_name: str = "contextfs_memory",
    ) -> None:
        """
        Initialize FalkorDB connection.

        Args:
            host: FalkorDB server host
            port: FalkorDB server port
            password: Optional authentication password
            graph_name: Name of the graph to use
        """
        self._host = host
        self._port = port
        self._password = password
        self._graph_name = graph_name
        self._db: FalkorDB | None = None
        self._graph: Graph | None = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily initialize FalkorDB connection."""
        if self._initialized:
            return

        try:
            from falkordb import FalkorDB

            # Connect to FalkorDB
            if self._password:
                self._db = FalkorDB(
                    host=self._host,
                    port=self._port,
                    password=self._password,
                )
            else:
                self._db = FalkorDB(host=self._host, port=self._port)

            # Select or create graph
            self._graph = self._db.select_graph(self._graph_name)

            # Create indexes for efficient lookups
            self._create_indexes()

            self._initialized = True
            logger.info(f"Connected to FalkorDB at {self._host}:{self._port}")

        except ImportError:
            raise ImportError("FalkorDB package not installed. Install with: pip install falkordb")
        except Exception as e:
            logger.error(f"Failed to connect to FalkorDB: {e}")
            raise

    def _create_indexes(self) -> None:
        """Create indexes for efficient queries."""
        if not self._graph:
            return

        try:
            # Index on Memory.id for fast lookups
            self._graph.query("CREATE INDEX FOR (m:Memory) ON (m.id)")
        except Exception:
            # Index may already exist
            pass

        try:
            # Index on Memory.type for type-based queries
            self._graph.query("CREATE INDEX FOR (m:Memory) ON (m.type)")
        except Exception:
            pass

        try:
            # Index on Memory.namespace_id for namespace queries
            self._graph.query("CREATE INDEX FOR (m:Memory) ON (m.namespace_id)")
        except Exception:
            pass

    def _memory_to_props(self, memory: Memory) -> dict[str, Any]:
        """Convert Memory to FalkorDB node properties."""
        return {
            "id": memory.id,
            "content": memory.content[:1000],  # Truncate for graph storage
            "type": memory.type.value,
            "tags": json.dumps(memory.tags),
            "summary": memory.summary or "",
            "namespace_id": memory.namespace_id,
            "source_repo": memory.source_repo or "",
            "project": memory.project or "",
            "created_at": memory.created_at.isoformat(),
        }

    def _props_to_memory(self, props: dict[str, Any]) -> Memory:
        """Convert FalkorDB node properties to Memory."""
        return Memory(
            id=props["id"],
            content=props.get("content", ""),
            type=MemoryType(props.get("type", "fact")),
            tags=json.loads(props.get("tags", "[]")),
            summary=props.get("summary") or None,
            namespace_id=props.get("namespace_id", "global"),
            source_repo=props.get("source_repo") or None,
            project=props.get("project") or None,
            created_at=datetime.fromisoformat(props.get("created_at", datetime.now().isoformat())),
        )

    # =========================================================================
    # GraphBackend Protocol Implementation
    # =========================================================================

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        relation: EdgeRelation,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEdge:
        """Create a directed edge between two memories."""
        self._ensure_initialized()

        if not self._graph:
            raise RuntimeError("Graph not initialized")

        # Create edge with properties
        edge_props = {
            "weight": weight,
            "created_at": datetime.now().isoformat(),
            "metadata": json.dumps(metadata or {}),
        }

        # Use Cypher MERGE to create edge (idempotent)
        query = f"""
        MATCH (a:Memory {{id: $from_id}})
        MATCH (b:Memory {{id: $to_id}})
        MERGE (a)-[r:{relation.value.upper()} {{
            weight: $weight,
            created_at: $created_at,
            metadata: $metadata
        }}]->(b)
        RETURN r
        """

        try:
            result = self._graph.query(
                query,
                {
                    "from_id": from_id,
                    "to_id": to_id,
                    "weight": weight,
                    "created_at": edge_props["created_at"],
                    "metadata": edge_props["metadata"],
                },
            )

            if not result.result_set:
                raise ValueError(
                    f"Could not create edge: one or both memories not found "
                    f"(from={from_id}, to={to_id})"
                )

            return MemoryEdge(
                from_id=from_id,
                to_id=to_id,
                relation=relation,
                weight=weight,
                created_at=datetime.fromisoformat(edge_props["created_at"]),
                metadata=metadata or {},
            )

        except Exception as e:
            logger.error(f"Failed to create edge: {e}")
            raise

    def remove_edge(
        self,
        from_id: str,
        to_id: str,
        relation: EdgeRelation | None = None,
    ) -> bool:
        """Remove edge(s) between two memories."""
        self._ensure_initialized()

        if not self._graph:
            return False

        if relation:
            # Remove specific relation
            query = f"""
            MATCH (a:Memory {{id: $from_id}})-[r:{relation.value.upper()}]->(b:Memory {{id: $to_id}})
            DELETE r
            RETURN count(r) as deleted
            """
        else:
            # Remove all relations
            query = """
            MATCH (a:Memory {id: $from_id})-[r]->(b:Memory {id: $to_id})
            DELETE r
            RETURN count(r) as deleted
            """

        try:
            result = self._graph.query(query, {"from_id": from_id, "to_id": to_id})
            return bool(result.result_set and result.result_set[0][0] > 0)
        except Exception as e:
            logger.error(f"Failed to remove edge: {e}")
            return False

    def get_edges(
        self,
        memory_id: str,
        direction: Literal["outgoing", "incoming", "both"] = "both",
        relation: EdgeRelation | None = None,
    ) -> list[MemoryEdge]:
        """Get all edges connected to a memory."""
        self._ensure_initialized()

        if not self._graph:
            return []

        # Build direction-aware pattern
        if direction == "outgoing":
            pattern = "(a:Memory {id: $memory_id})-[r]->(b:Memory)"
        elif direction == "incoming":
            pattern = "(a:Memory {id: $memory_id})<-[r]-(b:Memory)"
        else:
            pattern = "(a:Memory {id: $memory_id})-[r]-(b:Memory)"

        # Add relation filter if specified
        if relation:
            pattern = pattern.replace("-[r]-", f"-[r:{relation.value.upper()}]-")

        query = f"""
        MATCH {pattern}
        RETURN a.id as from_id, b.id as to_id, type(r) as relation,
               r.weight as weight, r.created_at as created_at, r.metadata as metadata
        """

        try:
            result = self._graph.query(query, {"memory_id": memory_id})
            edges = []

            for row in result.result_set:
                # Determine direction for edge construction
                from_id = row[0] if direction != "incoming" else row[1]
                to_id = row[1] if direction != "incoming" else row[0]

                edges.append(
                    MemoryEdge(
                        from_id=from_id,
                        to_id=to_id,
                        relation=EdgeRelation(row[2].lower()),
                        weight=row[3] or 1.0,
                        created_at=datetime.fromisoformat(row[4]) if row[4] else datetime.now(),
                        metadata=json.loads(row[5]) if row[5] else {},
                    )
                )

            return edges

        except Exception as e:
            logger.error(f"Failed to get edges: {e}")
            return []

    def get_related(
        self,
        memory_id: str,
        relation: EdgeRelation | None = None,
        direction: Literal["outgoing", "incoming", "both"] = "outgoing",
        max_depth: int = 1,
        min_weight: float = 0.0,
    ) -> list[GraphTraversalResult]:
        """Get memories related to a given memory via graph traversal."""
        self._ensure_initialized()

        if not self._graph:
            return []

        # Build direction-aware traversal pattern
        if direction == "outgoing":
            arrow = "->"
        elif direction == "incoming":
            arrow = "<-"
        else:
            arrow = "-"

        # Build relation filter
        rel_filter = f":{relation.value.upper()}" if relation else ""

        query = f"""
        MATCH path = (a:Memory {{id: $memory_id}})-[r{rel_filter}*1..{max_depth}]{arrow}(b:Memory)
        WHERE all(edge IN relationships(path) WHERE edge.weight >= $min_weight)
        WITH b, relationships(path) as rels, length(path) as depth
        RETURN b, type(last(rels)) as relation, depth,
               reduce(w = 1.0, r IN rels | w * r.weight) as path_weight
        ORDER BY depth, path_weight DESC
        """

        try:
            result = self._graph.query(query, {"memory_id": memory_id, "min_weight": min_weight})
            results = []

            for row in result.result_set:
                node_props = row[0].properties
                results.append(
                    GraphTraversalResult(
                        memory=self._props_to_memory(node_props),
                        relation=EdgeRelation(row[1].lower()),
                        depth=row[2],
                        path_weight=row[3],
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Failed to get related memories: {e}")
            return []

    def find_path(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 5,
        relation: EdgeRelation | None = None,
    ) -> GraphPath | None:
        """Find shortest path between two memories."""
        self._ensure_initialized()

        if not self._graph:
            return None

        rel_filter = f":{relation.value.upper()}" if relation else ""

        query = f"""
        MATCH path = shortestPath(
            (a:Memory {{id: $from_id}})-[r{rel_filter}*1..{max_depth}]-(b:Memory {{id: $to_id}})
        )
        RETURN nodes(path) as nodes, relationships(path) as rels,
               reduce(w = 0.0, r IN relationships(path) | w + r.weight) as total_weight
        """

        try:
            result = self._graph.query(query, {"from_id": from_id, "to_id": to_id})

            if not result.result_set:
                return None

            row = result.result_set[0]
            nodes = [self._props_to_memory(n.properties) for n in row[0]]
            edges = []

            for i, rel in enumerate(row[1]):
                edges.append(
                    MemoryEdge(
                        from_id=nodes[i].id,
                        to_id=nodes[i + 1].id,
                        relation=EdgeRelation(rel.relation.lower()),
                        weight=rel.properties.get("weight", 1.0),
                    )
                )

            return GraphPath(
                nodes=nodes,
                edges=edges,
                total_weight=row[2],
            )

        except Exception as e:
            logger.error(f"Failed to find path: {e}")
            return None

    def get_subgraph(
        self,
        root_id: str,
        max_depth: int = 2,
        relation: EdgeRelation | None = None,
    ) -> dict[str, Any]:
        """Extract a subgraph rooted at a memory."""
        self._ensure_initialized()

        if not self._graph:
            return {"nodes": [], "edges": []}

        rel_filter = f":{relation.value.upper()}" if relation else ""

        query = f"""
        MATCH path = (root:Memory {{id: $root_id}})-[r{rel_filter}*0..{max_depth}]-(connected:Memory)
        WITH collect(distinct connected) as nodes, collect(distinct r) as all_rels
        UNWIND all_rels as rel_list
        UNWIND rel_list as rel
        RETURN nodes, collect(distinct rel) as edges
        """

        try:
            result = self._graph.query(query, {"root_id": root_id})

            if not result.result_set:
                # Return just the root node
                root_query = "MATCH (m:Memory {id: $root_id}) RETURN m"
                root_result = self._graph.query(root_query, {"root_id": root_id})
                if root_result.result_set:
                    return {
                        "nodes": [self._props_to_memory(root_result.result_set[0][0].properties)],
                        "edges": [],
                    }
                return {"nodes": [], "edges": []}

            row = result.result_set[0]
            nodes = [self._props_to_memory(n.properties) for n in row[0]]
            edges = []

            for rel in row[1]:
                edges.append(
                    MemoryEdge(
                        from_id=rel.src_node,
                        to_id=rel.dest_node,
                        relation=EdgeRelation(rel.relation.lower()),
                        weight=rel.properties.get("weight", 1.0),
                    )
                )

            return {"nodes": nodes, "edges": edges}

        except Exception as e:
            logger.error(f"Failed to get subgraph: {e}")
            return {"nodes": [], "edges": []}

    def get_lineage(
        self,
        memory_id: str,
        direction: Literal["ancestors", "descendants", "both"] = "both",
    ) -> dict[str, Any]:
        """Get the evolution lineage of a memory."""
        self._ensure_initialized()

        if not self._graph:
            return {"root": memory_id, "ancestors": [], "descendants": [], "history": []}

        # Lineage relations
        lineage_rels = "EVOLVED_FROM|MERGED_FROM|SPLIT_FROM"

        result_data: dict[str, Any] = {
            "root": memory_id,
            "ancestors": [],
            "descendants": [],
            "history": [],
        }

        try:
            # Get ancestors (follow *_FROM relations backward)
            if direction in ("ancestors", "both"):
                ancestor_query = f"""
                MATCH path = (m:Memory {{id: $memory_id}})-[:{lineage_rels}*1..10]->(ancestor:Memory)
                RETURN ancestor, length(path) as depth
                ORDER BY depth
                """
                ancestor_result = self._graph.query(ancestor_query, {"memory_id": memory_id})
                for row in ancestor_result.result_set:
                    result_data["ancestors"].append(
                        {
                            "memory": self._props_to_memory(row[0].properties),
                            "depth": row[1],
                        }
                    )

            # Get descendants (follow inverse relations forward)
            if direction in ("descendants", "both"):
                descendant_query = """
                MATCH path = (m:Memory {id: $memory_id})<-[:EVOLVED_FROM|MERGED_FROM|SPLIT_FROM*1..10]-(descendant:Memory)
                RETURN descendant, length(path) as depth
                ORDER BY depth
                """
                descendant_result = self._graph.query(descendant_query, {"memory_id": memory_id})
                for row in descendant_result.result_set:
                    result_data["descendants"].append(
                        {
                            "memory": self._props_to_memory(row[0].properties),
                            "depth": row[1],
                        }
                    )

            # Find root of lineage
            if result_data["ancestors"]:
                result_data["root"] = result_data["ancestors"][-1]["memory"].id

            # Build chronological history
            all_memories = (
                [{"memory_id": memory_id, "depth": 0}]
                + [
                    {"memory_id": a["memory"].id, "depth": -a["depth"]}
                    for a in result_data["ancestors"]
                ]
                + [
                    {"memory_id": d["memory"].id, "depth": d["depth"]}
                    for d in result_data["descendants"]
                ]
            )
            result_data["history"] = sorted(all_memories, key=lambda x: x["depth"])

            return result_data

        except Exception as e:
            logger.error(f"Failed to get lineage: {e}")
            return result_data

    def sync_node(self, memory: Memory) -> bool:
        """Sync a memory node to the graph backend."""
        self._ensure_initialized()

        if not self._graph:
            return False

        props = self._memory_to_props(memory)

        query = """
        MERGE (m:Memory {id: $id})
        SET m.content = $content,
            m.type = $type,
            m.tags = $tags,
            m.summary = $summary,
            m.namespace_id = $namespace_id,
            m.source_repo = $source_repo,
            m.project = $project,
            m.created_at = $created_at
        RETURN m
        """

        try:
            result = self._graph.query(query, props)
            return bool(result.result_set)
        except Exception as e:
            logger.error(f"Failed to sync node: {e}")
            return False

    def delete_node(self, memory_id: str) -> bool:
        """Delete a memory node and all its edges from the graph."""
        self._ensure_initialized()

        if not self._graph:
            return False

        query = """
        MATCH (m:Memory {id: $memory_id})
        DETACH DELETE m
        RETURN count(m) as deleted
        """

        try:
            result = self._graph.query(query, {"memory_id": memory_id})
            return bool(result.result_set and result.result_set[0][0] > 0)
        except Exception as e:
            logger.error(f"Failed to delete node: {e}")
            return False

    # =========================================================================
    # Additional FalkorDB-Specific Methods
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        self._ensure_initialized()

        if not self._graph:
            return {"nodes": 0, "edges": 0, "connected": False}

        try:
            node_result = self._graph.query("MATCH (n:Memory) RETURN count(n)")
            edge_result = self._graph.query("MATCH ()-[r]->() RETURN count(r)")

            return {
                "nodes": node_result.result_set[0][0] if node_result.result_set else 0,
                "edges": edge_result.result_set[0][0] if edge_result.result_set else 0,
                "connected": True,
                "host": self._host,
                "port": self._port,
                "graph_name": self._graph_name,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"nodes": 0, "edges": 0, "connected": False, "error": str(e)}

    def clear_graph(self) -> bool:
        """Clear all nodes and edges from the graph."""
        self._ensure_initialized()

        if not self._graph:
            return False

        try:
            self._graph.query("MATCH (n) DETACH DELETE n")
            return True
        except Exception as e:
            logger.error(f"Failed to clear graph: {e}")
            return False

    def close(self) -> None:
        """Close the FalkorDB connection."""
        self._db = None
        self._graph = None
        self._initialized = False
