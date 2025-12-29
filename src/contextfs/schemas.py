"""
Schemas for ContextFS memory and session management.
"""

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Types of memories."""

    # Core types
    FACT = "fact"  # Static facts, configurations
    DECISION = "decision"  # Architectural/design decisions
    PROCEDURAL = "procedural"  # How-to procedures
    EPISODIC = "episodic"  # Session/conversation memories
    USER = "user"  # User preferences
    CODE = "code"  # Code snippets
    ERROR = "error"  # Runtime errors, stack traces
    COMMIT = "commit"  # Git commit history

    # Extended types
    TODO = "todo"  # Tasks, work items
    ISSUE = "issue"  # Bugs, problems, tickets
    API = "api"  # API endpoints, contracts
    SCHEMA = "schema"  # Data models, DB schemas
    TEST = "test"  # Test cases, coverage
    REVIEW = "review"  # PR feedback, code reviews
    RELEASE = "release"  # Changelogs, versions
    CONFIG = "config"  # Environment configs
    DEPENDENCY = "dependency"  # Package versions
    DOC = "doc"  # Documentation


# Centralized type configuration - single source of truth
# To add a new type: 1) Add to MemoryType enum above, 2) Add config here
TYPE_CONFIG: dict[str, dict[str, Any]] = {
    # Core types
    "fact": {
        "label": "Fact",
        "color": "#58a6ff",
        "description": "Static facts, configurations",
        "category": "core",
    },
    "decision": {
        "label": "Decision",
        "color": "#a371f7",
        "description": "Architectural/design decisions",
        "category": "core",
    },
    "procedural": {
        "label": "Procedural",
        "color": "#3fb950",
        "description": "How-to procedures",
        "category": "core",
    },
    "episodic": {
        "label": "Episodic",
        "color": "#d29922",
        "description": "Session/conversation memories",
        "category": "core",
    },
    "user": {
        "label": "User",
        "color": "#f778ba",
        "description": "User preferences",
        "category": "core",
    },
    "code": {
        "label": "Code",
        "color": "#79c0ff",
        "description": "Code snippets",
        "category": "core",
    },
    "error": {
        "label": "Error",
        "color": "#f85149",
        "description": "Runtime errors, stack traces",
        "category": "core",
    },
    "commit": {
        "label": "Commit",
        "color": "#8b5cf6",
        "description": "Git commit history",
        "category": "core",
    },
    # Extended types
    "todo": {
        "label": "Todo",
        "color": "#f59e0b",
        "description": "Tasks, work items",
        "category": "extended",
    },
    "issue": {
        "label": "Issue",
        "color": "#ef4444",
        "description": "Bugs, problems, tickets",
        "category": "extended",
    },
    "api": {
        "label": "API",
        "color": "#06b6d4",
        "description": "API endpoints, contracts",
        "category": "extended",
    },
    "schema": {
        "label": "Schema",
        "color": "#8b5cf6",
        "description": "Data models, DB schemas",
        "category": "extended",
    },
    "test": {
        "label": "Test",
        "color": "#22c55e",
        "description": "Test cases, coverage",
        "category": "extended",
    },
    "review": {
        "label": "Review",
        "color": "#ec4899",
        "description": "PR feedback, code reviews",
        "category": "extended",
    },
    "release": {
        "label": "Release",
        "color": "#6366f1",
        "description": "Changelogs, versions",
        "category": "extended",
    },
    "config": {
        "label": "Config",
        "color": "#64748b",
        "description": "Environment configs",
        "category": "extended",
    },
    "dependency": {
        "label": "Dependency",
        "color": "#0ea5e9",
        "description": "Package versions",
        "category": "extended",
    },
    "doc": {
        "label": "Doc",
        "color": "#14b8a6",
        "description": "Documentation",
        "category": "extended",
    },
}


def get_memory_types() -> list[dict[str, Any]]:
    """Get all memory types with their configuration.

    Returns list of dicts with: value, label, color, description, category
    Use this to dynamically generate UI dropdowns, API schemas, etc.
    """
    return [
        {
            "value": t.value,
            **TYPE_CONFIG.get(
                t.value,
                {
                    "label": t.value.title(),
                    "color": "#888888",
                    "description": "",
                    "category": "unknown",
                },
            ),
        }
        for t in MemoryType
    ]


def get_memory_type_values() -> list[str]:
    """Get list of all memory type values (for JSON schema enums)."""
    return [t.value for t in MemoryType]


class Namespace(BaseModel):
    """
    Namespace for cross-repo memory isolation.

    Hierarchy:
    - global: Shared across all repos
    - org/team: Shared within organization
    - repo: Specific to repository
    - session: Specific to session
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str
    parent_id: str | None = None
    repo_path: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def global_ns(cls) -> "Namespace":
        return cls(id="global", name="global")

    @classmethod
    def for_repo(cls, repo_path: str) -> "Namespace":
        from pathlib import Path

        # Resolve symlinks to get canonical path for consistent namespace
        resolved_path = str(Path(repo_path).resolve())
        repo_id = hashlib.sha256(resolved_path.encode()).hexdigest()[:12]
        return cls(
            id=f"repo-{repo_id}",
            name=resolved_path.split("/")[-1],
            repo_path=resolved_path,
        )


class Memory(BaseModel):
    """A single memory item."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    content: str
    type: MemoryType = MemoryType.FACT
    tags: list[str] = Field(default_factory=list)
    summary: str | None = None

    # Namespace for cross-repo support
    namespace_id: str = "global"

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Source tracking
    source_file: str | None = None
    source_repo: str | None = None
    source_tool: str | None = None  # claude-code, claude-desktop, gemini, chatgpt, etc.
    project: str | None = None  # Project name for grouping memories across repos
    session_id: str | None = None

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Embedding (populated by RAG backend)
    embedding: list[float] | None = None

    def to_context_string(self) -> str:
        """Format for context injection."""
        prefix = f"[{self.type.value}]"
        if self.summary:
            return f"{prefix} {self.summary}: {self.content[:200]}..."
        return f"{prefix} {self.content[:300]}..."


class SessionMessage(BaseModel):
    """A message in a session."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    role: str  # user, assistant, system
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    """A conversation session."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str | None = None
    namespace_id: str = "global"

    # Tool that created session
    tool: str = "contextfs"  # claude-code, gemini, codex, etc.

    # Git context
    repo_path: str | None = None
    branch: str | None = None

    # Messages
    messages: list[SessionMessage] = Field(default_factory=list)

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.now)
    ended_at: datetime | None = None

    # Generated summary
    summary: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_message(self, role: str, content: str) -> SessionMessage:
        msg = SessionMessage(role=role, content=content)
        self.messages.append(msg)
        return msg

    def end(self) -> None:
        self.ended_at = datetime.now(timezone.utc)


class SearchResult(BaseModel):
    """Search result with relevance score."""

    memory: Memory
    score: float = Field(ge=0.0, le=1.0)
    highlights: list[str] = Field(default_factory=list)
    source: str | None = None  # "fts", "rag", or "hybrid"
