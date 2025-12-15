"""
Schemas for ContextFS memory and session management.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field
import uuid
import hashlib


class MemoryType(str, Enum):
    """Types of memories."""
    FACT = "fact"
    DECISION = "decision"
    PROCEDURAL = "procedural"
    EPISODIC = "episodic"
    USER = "user"
    CODE = "code"
    ERROR = "error"


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
    parent_id: Optional[str] = None
    repo_path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def global_ns(cls) -> "Namespace":
        return cls(id="global", name="global")

    @classmethod
    def for_repo(cls, repo_path: str) -> "Namespace":
        repo_id = hashlib.sha256(repo_path.encode()).hexdigest()[:12]
        return cls(
            id=f"repo-{repo_id}",
            name=repo_path.split("/")[-1],
            repo_path=repo_path,
        )


class Memory(BaseModel):
    """A single memory item."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    content: str
    type: MemoryType = MemoryType.FACT
    tags: list[str] = Field(default_factory=list)
    summary: Optional[str] = None

    # Namespace for cross-repo support
    namespace_id: str = "global"

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # Source tracking
    source_file: Optional[str] = None
    source_repo: Optional[str] = None
    session_id: Optional[str] = None

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Embedding (populated by RAG backend)
    embedding: Optional[list[float]] = None

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
    label: Optional[str] = None
    namespace_id: str = "global"

    # Tool that created session
    tool: str = "contextfs"  # claude-code, gemini, codex, etc.

    # Git context
    repo_path: Optional[str] = None
    branch: Optional[str] = None

    # Messages
    messages: list[SessionMessage] = Field(default_factory=list)

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None

    # Generated summary
    summary: Optional[str] = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_message(self, role: str, content: str) -> SessionMessage:
        msg = SessionMessage(role=role, content=content)
        self.messages.append(msg)
        return msg

    def end(self) -> None:
        self.ended_at = datetime.now()


class SearchResult(BaseModel):
    """Search result with relevance score."""
    memory: Memory
    score: float = Field(ge=0.0, le=1.0)
    highlights: list[str] = Field(default_factory=list)
