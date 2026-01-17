"""
ContextFS - Universal AI Memory Layer

Cross-client, cross-repo context management with RAG capabilities.
Works with Claude Code, Claude Desktop, Gemini CLI, Codex CLI, and any MCP client.

Features:
- Semantic search with ChromaDB + sentence-transformers
- Cross-repo namespace isolation
- Session management and episodic memory
- Git-aware context (commits, branches)
- MCP server for universal client support
- Graph-based memory relationships and lineage tracking
- Plugins for Claude Code, Gemini, Codex

Example:
    from contextfs import ContextFS

    ctx = ContextFS()
    ctx.save("Important decision", type="decision", tags=["auth"])
    results = ctx.search("authentication")
    ctx.recall("abc123")

Graph Example:
    from contextfs import ContextFS, MemoryLineage

    ctx = ContextFS()
    lineage = MemoryLineage(ctx._storage)

    # Evolve a memory
    evolved = lineage.evolve("abc123", "Updated content...")

    # Merge memories
    merged = lineage.merge(["mem1", "mem2"], strategy="union")
"""

__version__ = "0.2.22"

from contextfs.core import ContextFS
from contextfs.memory_lineage import ConflictResolution, MemoryLineage, MergeStrategy
from contextfs.schemas import Memory, MemoryType, Namespace, Session
from contextfs.storage_protocol import (
    EdgeRelation,
    GraphBackend,
    GraphPath,
    GraphTraversalResult,
    MemoryEdge,
    StorageBackend,
    StorageCapabilities,
)

__all__ = [
    # Core
    "ContextFS",
    "Memory",
    "MemoryType",
    "Session",
    "Namespace",
    # Graph/Lineage
    "MemoryLineage",
    "MergeStrategy",
    "ConflictResolution",
    # Storage Protocol
    "StorageBackend",
    "GraphBackend",
    "StorageCapabilities",
    "EdgeRelation",
    "MemoryEdge",
    "GraphPath",
    "GraphTraversalResult",
    # Version
    "__version__",
]
