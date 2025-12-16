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
- Plugins for Claude Code, Gemini, Codex

Example:
    from contextfs import ContextFS

    ctx = ContextFS()
    ctx.save("Important decision", type="decision", tags=["auth"])
    results = ctx.search("authentication")
    ctx.recall("abc123")
"""

__version__ = "0.1.6"

from contextfs.core import ContextFS
from contextfs.schemas import Memory, MemoryType, Namespace, Session

__all__ = [
    "ContextFS",
    "Memory",
    "MemoryType",
    "Session",
    "Namespace",
    "__version__",
]
