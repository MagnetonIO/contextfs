"""
Schemas for ContextFS memory and session management.

Supports typed memory with optional structured_data validation.
Each memory type can have a JSON schema that validates its structured_data field.

Phase 2 adds Pydantic subclasses for type-safe structured_data per memory type.
"""

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

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


# ============================================================================
# Phase 2: Pydantic Typed Data Models
# ============================================================================
# These models provide type-safe structured_data for each memory type.
# They mirror the JSON schemas above but with full Pydantic validation.


class BaseStructuredData(BaseModel):
    """Base class for all typed structured data models."""

    model_config = {"extra": "allow"}  # Allow additional fields for extensibility


class DecisionData(BaseStructuredData):
    """Structured data for decision memories."""

    type: Literal["decision"] = "decision"
    decision: str = Field(..., description="The decision that was made")
    rationale: str | None = Field(None, description="Why this decision was made")
    alternatives: list[str] = Field(
        default_factory=list, description="Alternative options considered"
    )
    constraints: list[str] = Field(
        default_factory=list, description="Constraints that influenced the decision"
    )
    date: str | None = Field(None, description="When the decision was made")
    status: Literal["proposed", "accepted", "deprecated", "superseded"] | None = Field(
        None, description="Current status of the decision"
    )


class ProceduralData(BaseStructuredData):
    """Structured data for procedural memories."""

    type: Literal["procedural"] = "procedural"
    title: str | None = Field(None, description="Title of the procedure")
    steps: list[str] = Field(..., description="Ordered list of steps to follow")
    prerequisites: list[str] = Field(
        default_factory=list, description="Prerequisites before starting"
    )
    notes: str | None = Field(None, description="Additional notes or warnings")


class ErrorData(BaseStructuredData):
    """Structured data for error memories."""

    type: Literal["error"] = "error"
    error_type: str = Field(..., description="Type/class of error")
    message: str = Field(..., description="Error message")
    stack_trace: str | None = Field(None, description="Full stack trace")
    file: str | None = Field(None, description="File where error occurred")
    line: int | None = Field(None, description="Line number")
    resolution: str | None = Field(None, description="How the error was resolved")


class APIData(BaseStructuredData):
    """Structured data for API memories."""

    type: Literal["api"] = "api"
    endpoint: str = Field(..., description="API endpoint path")
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] | None = Field(
        None, description="HTTP method"
    )
    request_schema: dict[str, Any] | None = Field(None, description="Request body schema")
    response_schema: dict[str, Any] | None = Field(None, description="Response body schema")
    parameters: list[dict[str, Any]] = Field(
        default_factory=list, description="Query/path parameters"
    )


class TodoData(BaseStructuredData):
    """Structured data for todo memories."""

    type: Literal["todo"] = "todo"
    title: str = Field(..., description="Task title")
    status: Literal["pending", "in_progress", "completed", "blocked", "cancelled"] | None = Field(
        None, description="Task status"
    )
    priority: Literal["low", "medium", "high", "critical"] | None = Field(
        None, description="Task priority"
    )
    assignee: str | None = Field(None, description="Person assigned to task")
    due_date: str | None = Field(None, description="Due date for task")
    checklist: list[dict[str, Any]] = Field(
        default_factory=list, description="Subtasks/checklist items"
    )


class IssueData(BaseStructuredData):
    """Structured data for issue memories."""

    type: Literal["issue"] = "issue"
    title: str = Field(..., description="Issue title")
    severity: Literal["low", "medium", "high", "critical"] | None = Field(
        None, description="Issue severity"
    )
    status: Literal["open", "investigating", "resolved", "closed", "wontfix"] | None = Field(
        None, description="Issue status"
    )
    steps_to_reproduce: list[str] = Field(default_factory=list, description="Steps to reproduce")
    expected_behavior: str | None = Field(None, description="Expected behavior")
    actual_behavior: str | None = Field(None, description="Actual behavior observed")
    resolution: str | None = Field(None, description="How the issue was resolved")


class TestData(BaseStructuredData):
    """Structured data for test memories."""

    type: Literal["test"] = "test"
    name: str = Field(..., description="Test name")
    test_type: Literal["unit", "integration", "e2e", "performance", "security"] | None = Field(
        None, description="Type of test"
    )
    status: Literal["passing", "failing", "skipped", "flaky"] | None = Field(
        None, description="Test status"
    )
    file: str | None = Field(None, description="Test file path")
    assertions: list[str] = Field(default_factory=list, description="Test assertions")
    coverage: float | None = Field(None, description="Code coverage percentage")


class ConfigData(BaseStructuredData):
    """Structured data for config memories."""

    type: Literal["config"] = "config"
    name: str = Field(..., description="Configuration name")
    environment: Literal["development", "staging", "production", "test"] | None = Field(
        None, description="Environment this config applies to"
    )
    settings: dict[str, Any] = Field(
        default_factory=dict, description="Configuration key-value pairs"
    )
    secrets: list[str] = Field(
        default_factory=list, description="List of secret keys (values not stored)"
    )


class DependencyData(BaseStructuredData):
    """Structured data for dependency memories."""

    type: Literal["dependency"] = "dependency"
    name: str = Field(..., description="Package/dependency name")
    version: str = Field(..., description="Current version")
    latest_version: str | None = Field(None, description="Latest available version")
    dep_type: Literal["runtime", "dev", "peer", "optional"] | None = Field(
        None, description="Dependency type"
    )
    vulnerabilities: list[str] = Field(default_factory=list, description="Known vulnerabilities")
    changelog_url: str | None = Field(None, description="URL to changelog")


class ReleaseData(BaseStructuredData):
    """Structured data for release memories."""

    type: Literal["release"] = "release"
    version: str = Field(..., description="Release version")
    date: str | None = Field(None, description="Release date")
    changes: list[str] = Field(default_factory=list, description="List of changes in this release")
    breaking_changes: list[str] = Field(default_factory=list, description="Breaking changes")
    deprecations: list[str] = Field(default_factory=list, description="Deprecated features")
    contributors: list[str] = Field(
        default_factory=list, description="Contributors to this release"
    )


class ReviewComment(BaseModel):
    """A comment in a code review."""

    file: str | None = None
    line: int | None = None
    comment: str


class ReviewData(BaseStructuredData):
    """Structured data for review memories."""

    type: Literal["review"] = "review"
    pr_number: int | None = Field(None, description="Pull request number")
    reviewer: str | None = Field(None, description="Reviewer name")
    status: Literal["pending", "approved", "changes_requested", "commented"] = Field(
        ..., description="Review status"
    )
    comments: list[ReviewComment] = Field(default_factory=list, description="Review comments")
    summary: str | None = Field(None, description="Review summary")


class SchemaField(BaseModel):
    """A field in a schema definition."""

    name: str
    type: str
    required: bool = False
    description: str | None = None


class SchemaData(BaseStructuredData):
    """Structured data for schema memories."""

    type: Literal["schema"] = "schema"
    name: str = Field(..., description="Schema/model name")
    schema_type: Literal["database", "api", "event", "message", "config"] | None = Field(
        None, description="Schema type"
    )
    fields: list[SchemaField] = Field(default_factory=list, description="Schema fields")
    relationships: list[str] = Field(default_factory=list, description="Related schemas/tables")


# Union of all typed structured data models
# Use this for type hints when you want any typed structured data
TypedStructuredData = (
    DecisionData
    | ProceduralData
    | ErrorData
    | APIData
    | TodoData
    | IssueData
    | TestData
    | ConfigData
    | DependencyData
    | ReleaseData
    | ReviewData
    | SchemaData
)

# Mapping from memory type to its structured data class
STRUCTURED_DATA_CLASSES: dict[str, type[BaseStructuredData]] = {
    "decision": DecisionData,
    "procedural": ProceduralData,
    "error": ErrorData,
    "api": APIData,
    "todo": TodoData,
    "issue": IssueData,
    "test": TestData,
    "config": ConfigData,
    "dependency": DependencyData,
    "release": ReleaseData,
    "review": ReviewData,
    "schema": SchemaData,
}


def parse_structured_data(
    memory_type: str, data: dict[str, Any]
) -> BaseStructuredData | dict[str, Any]:
    """
    Parse raw dict into typed Pydantic model if available.

    Args:
        memory_type: The memory type (e.g., "decision", "error")
        data: The raw structured data dict

    Returns:
        Typed Pydantic model if type has a class, otherwise returns dict as-is.
    """
    data_class = STRUCTURED_DATA_CLASSES.get(memory_type)
    if data_class is None:
        return data
    try:
        return data_class(**data)
    except Exception:
        # If parsing fails, return raw dict (backward compatibility)
        return data


def serialize_structured_data(
    data: BaseStructuredData | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Serialize structured data to dict for storage.

    Handles both Pydantic models and raw dicts.
    """
    if data is None:
        return None
    if isinstance(data, BaseModel):
        return data.model_dump(by_alias=True, exclude_none=False)
    return data


# ============================================================================
# Validation Functions
# ============================================================================


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

    @property
    def typed_data(self) -> TypedStructuredData | dict[str, Any] | None:
        """
        Get structured_data as a typed Pydantic model if available.

        Returns the appropriate typed model (DecisionData, ErrorData, etc.)
        if the memory type has a corresponding class, otherwise returns
        the raw dict or None.
        """
        if self.structured_data is None:
            return None
        type_value = self.type.value if isinstance(self.type, MemoryType) else self.type
        return parse_structured_data(type_value, self.structured_data)

    # Factory methods for creating typed memories
    @classmethod
    def decision(
        cls,
        content: str,
        decision: str,
        rationale: str | None = None,
        alternatives: list[str] | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create a decision memory with typed structured_data."""
        structured = {"decision": decision}
        if rationale:
            structured["rationale"] = rationale
        if alternatives:
            structured["alternatives"] = alternatives
        return cls(
            content=content,
            type=MemoryType.DECISION,
            structured_data=structured,
            **kwargs,
        )

    @classmethod
    def procedural(
        cls,
        content: str,
        steps: list[str],
        title: str | None = None,
        prerequisites: list[str] | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create a procedural memory with typed structured_data."""
        structured: dict[str, Any] = {"steps": steps}
        if title:
            structured["title"] = title
        if prerequisites:
            structured["prerequisites"] = prerequisites
        return cls(
            content=content,
            type=MemoryType.PROCEDURAL,
            structured_data=structured,
            **kwargs,
        )

    @classmethod
    def error(
        cls,
        content: str,
        error_type: str,
        message: str,
        file: str | None = None,
        line: int | None = None,
        resolution: str | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create an error memory with typed structured_data."""
        structured: dict[str, Any] = {"error_type": error_type, "message": message}
        if file:
            structured["file"] = file
        if line:
            structured["line"] = line
        if resolution:
            structured["resolution"] = resolution
        return cls(
            content=content,
            type=MemoryType.ERROR,
            structured_data=structured,
            **kwargs,
        )

    @classmethod
    def api(
        cls,
        content: str,
        endpoint: str,
        method: str | None = None,
        parameters: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create an API memory with typed structured_data."""
        structured: dict[str, Any] = {"endpoint": endpoint}
        if method:
            structured["method"] = method
        if parameters:
            structured["parameters"] = parameters
        return cls(
            content=content,
            type=MemoryType.API,
            structured_data=structured,
            **kwargs,
        )

    @classmethod
    def todo(
        cls,
        content: str,
        title: str,
        status: str | None = None,
        priority: str | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create a todo memory with typed structured_data."""
        structured: dict[str, Any] = {"title": title}
        if status:
            structured["status"] = status
        if priority:
            structured["priority"] = priority
        return cls(
            content=content,
            type=MemoryType.TODO,
            structured_data=structured,
            **kwargs,
        )


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
