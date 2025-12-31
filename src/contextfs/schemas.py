"""
Schemas for ContextFS memory and session management.

Supports typed memory with optional structured_data validation.
Each memory type can have a JSON schema that validates its structured_data field.
"""

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


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


# JSON Schemas for structured_data validation per memory type
# Each schema defines the expected structure for that memory type's structured_data field
# If a type is not in TYPE_SCHEMAS, no validation is performed on structured_data
TYPE_SCHEMAS: dict[str, dict[str, Any]] = {
    "decision": {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "description": "The decision that was made",
            },
            "rationale": {
                "type": "string",
                "description": "Why this decision was made",
            },
            "alternatives": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Alternative options that were considered",
            },
            "constraints": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Constraints that influenced the decision",
            },
            "date": {
                "type": "string",
                "description": "When the decision was made",
            },
            "status": {
                "type": "string",
                "enum": ["proposed", "accepted", "deprecated", "superseded"],
                "description": "Current status of the decision",
            },
        },
        "required": ["decision"],
        "additionalProperties": True,
    },
    "procedural": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Title of the procedure",
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered list of steps to follow",
            },
            "prerequisites": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Prerequisites before starting",
            },
            "notes": {
                "type": "string",
                "description": "Additional notes or warnings",
            },
        },
        "required": ["steps"],
        "additionalProperties": True,
    },
    "error": {
        "type": "object",
        "properties": {
            "error_type": {
                "type": "string",
                "description": "Type/class of error",
            },
            "message": {
                "type": "string",
                "description": "Error message",
            },
            "stack_trace": {
                "type": "string",
                "description": "Full stack trace",
            },
            "file": {
                "type": "string",
                "description": "File where error occurred",
            },
            "line": {
                "type": "integer",
                "description": "Line number",
            },
            "resolution": {
                "type": "string",
                "description": "How the error was resolved",
            },
        },
        "required": ["error_type", "message"],
        "additionalProperties": True,
    },
    "api": {
        "type": "object",
        "properties": {
            "endpoint": {
                "type": "string",
                "description": "API endpoint path",
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                "description": "HTTP method",
            },
            "request_schema": {
                "type": "object",
                "description": "Request body schema",
            },
            "response_schema": {
                "type": "object",
                "description": "Response body schema",
            },
            "parameters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "required": {"type": "boolean"},
                        "description": {"type": "string"},
                    },
                },
                "description": "Query/path parameters",
            },
        },
        "required": ["endpoint"],
        "additionalProperties": True,
    },
    "todo": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Task title",
            },
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed", "blocked", "cancelled"],
                "description": "Task status",
            },
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
                "description": "Task priority",
            },
            "assignee": {
                "type": "string",
                "description": "Person assigned to task",
            },
            "due_date": {
                "type": "string",
                "description": "Due date for task",
            },
            "checklist": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "done": {"type": "boolean"},
                    },
                },
                "description": "Subtasks/checklist items",
            },
        },
        "required": ["title"],
        "additionalProperties": True,
    },
    "issue": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Issue title",
            },
            "severity": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
                "description": "Issue severity",
            },
            "status": {
                "type": "string",
                "enum": ["open", "investigating", "resolved", "closed", "wontfix"],
                "description": "Issue status",
            },
            "steps_to_reproduce": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Steps to reproduce the issue",
            },
            "expected_behavior": {
                "type": "string",
                "description": "Expected behavior",
            },
            "actual_behavior": {
                "type": "string",
                "description": "Actual behavior observed",
            },
            "resolution": {
                "type": "string",
                "description": "How the issue was resolved",
            },
        },
        "required": ["title"],
        "additionalProperties": True,
    },
    "test": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Test name",
            },
            "type": {
                "type": "string",
                "enum": ["unit", "integration", "e2e", "performance", "security"],
                "description": "Type of test",
            },
            "status": {
                "type": "string",
                "enum": ["passing", "failing", "skipped", "flaky"],
                "description": "Test status",
            },
            "file": {
                "type": "string",
                "description": "Test file path",
            },
            "assertions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Test assertions",
            },
            "coverage": {
                "type": "number",
                "description": "Code coverage percentage",
            },
        },
        "required": ["name"],
        "additionalProperties": True,
    },
    "config": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Configuration name",
            },
            "environment": {
                "type": "string",
                "enum": ["development", "staging", "production", "test"],
                "description": "Environment this config applies to",
            },
            "settings": {
                "type": "object",
                "description": "Configuration key-value pairs",
            },
            "secrets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of secret keys (values not stored)",
            },
        },
        "required": ["name"],
        "additionalProperties": True,
    },
    "dependency": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Package/dependency name",
            },
            "version": {
                "type": "string",
                "description": "Current version",
            },
            "latest_version": {
                "type": "string",
                "description": "Latest available version",
            },
            "type": {
                "type": "string",
                "enum": ["runtime", "dev", "peer", "optional"],
                "description": "Dependency type",
            },
            "vulnerabilities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Known vulnerabilities",
            },
            "changelog_url": {
                "type": "string",
                "description": "URL to changelog",
            },
        },
        "required": ["name", "version"],
        "additionalProperties": True,
    },
    "release": {
        "type": "object",
        "properties": {
            "version": {
                "type": "string",
                "description": "Release version",
            },
            "date": {
                "type": "string",
                "description": "Release date",
            },
            "changes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of changes in this release",
            },
            "breaking_changes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Breaking changes",
            },
            "deprecations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Deprecated features",
            },
            "contributors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Contributors to this release",
            },
        },
        "required": ["version"],
        "additionalProperties": True,
    },
    "review": {
        "type": "object",
        "properties": {
            "pr_number": {
                "type": "integer",
                "description": "Pull request number",
            },
            "reviewer": {
                "type": "string",
                "description": "Reviewer name",
            },
            "status": {
                "type": "string",
                "enum": ["pending", "approved", "changes_requested", "commented"],
                "description": "Review status",
            },
            "comments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "comment": {"type": "string"},
                    },
                },
                "description": "Review comments",
            },
            "summary": {
                "type": "string",
                "description": "Review summary",
            },
        },
        "required": ["status"],
        "additionalProperties": True,
    },
    "schema": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Schema/model name",
            },
            "type": {
                "type": "string",
                "enum": ["database", "api", "event", "message", "config"],
                "description": "Schema type",
            },
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "required": {"type": "boolean"},
                        "description": {"type": "string"},
                    },
                },
                "description": "Schema fields",
            },
            "relationships": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Related schemas/tables",
            },
        },
        "required": ["name"],
        "additionalProperties": True,
    },
}


def validate_structured_data(memory_type: str, data: dict[str, Any]) -> tuple[bool, str | None]:
    """
    Validate structured_data against the schema for the given memory type.

    Args:
        memory_type: The memory type (e.g., "decision", "error")
        data: The structured data to validate

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.
    """
    schema = TYPE_SCHEMAS.get(memory_type)
    if schema is None:
        # No schema defined for this type, accept any data
        return True, None

    try:
        import jsonschema

        jsonschema.validate(instance=data, schema=schema)
        return True, None
    except ImportError:
        # jsonschema not installed, skip validation
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e.message)
    except jsonschema.SchemaError as e:
        return False, f"Invalid schema: {e.message}"


def get_type_schema(memory_type: str) -> dict[str, Any] | None:
    """Get the JSON schema for a memory type, if one exists."""
    return TYPE_SCHEMAS.get(memory_type)


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
    """
    A single memory item.

    Supports optional structured_data for type-specific schema validation.
    When structured_data is provided, it is validated against TYPE_SCHEMAS
    for the memory's type. This enables typed memory with enforced structure.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    content: str
    type: MemoryType = MemoryType.FACT
    tags: list[str] = Field(default_factory=list)
    summary: str | None = None

    # Typed structured data (validated against TYPE_SCHEMAS)
    structured_data: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured data validated against the type's JSON schema",
    )

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

    @model_validator(mode="after")
    def validate_structured_data_schema(self) -> "Memory":
        """Validate structured_data against the type's JSON schema if provided."""
        if self.structured_data is not None:
            type_value = self.type.value if isinstance(self.type, MemoryType) else self.type
            is_valid, error = validate_structured_data(type_value, self.structured_data)
            if not is_valid:
                raise ValueError(
                    f"structured_data validation failed for type '{type_value}': {error}"
                )
        return self

    def to_context_string(self) -> str:
        """Format for context injection."""
        prefix = f"[{self.type.value}]"
        if self.summary:
            return f"{prefix} {self.summary}: {self.content[:200]}..."
        return f"{prefix} {self.content[:300]}..."

    def get_structured_field(self, field: str, default: Any = None) -> Any:
        """Get a field from structured_data with a default value."""
        if self.structured_data is None:
            return default
        return self.structured_data.get(field, default)


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
