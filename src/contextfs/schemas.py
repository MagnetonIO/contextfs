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
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from contextfs.types.memory import Mem
    from contextfs.types.versioned import VersionedMem


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

    # Workflow/Agent types
    WORKFLOW = "workflow"  # Workflow definitions
    TASK = "task"  # Individual workflow tasks
    STEP = "step"  # Execution steps within tasks
    AGENT_RUN = "agent_run"  # LLM agent execution records


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
    # Workflow/Agent types
    "workflow": {
        "label": "Workflow",
        "color": "#7c3aed",
        "description": "Workflow definitions",
        "category": "workflow",
    },
    "task": {
        "label": "Task",
        "color": "#2563eb",
        "description": "Individual workflow tasks",
        "category": "workflow",
    },
    "step": {
        "label": "Step",
        "color": "#0891b2",
        "description": "Execution steps within tasks",
        "category": "workflow",
    },
    "agent_run": {
        "label": "Agent Run",
        "color": "#059669",
        "description": "LLM agent execution records",
        "category": "workflow",
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


# ============================================================================
# Additional structured data types for complete type coverage
# ============================================================================


class FactData(BaseStructuredData):
    """Structured data for fact memories."""

    type: Literal["fact"] = "fact"
    category: str | None = Field(
        None, description="Category of fact (e.g., 'configuration', 'constant', 'rule')"
    )
    source: str | None = Field(None, description="Where this fact came from")
    confidence: float | None = Field(None, ge=0.0, le=1.0, description="Confidence level (0.0-1.0)")
    verified_at: str | None = Field(None, description="When the fact was verified (ISO timestamp)")
    valid_until: str | None = Field(None, description="Expiration date if applicable")


class EpisodicData(BaseStructuredData):
    """Structured data for episodic memories (sessions/conversations)."""

    type: Literal["episodic"] = "episodic"
    session_type: str | None = Field(
        None, description="Type of session (e.g., 'conversation', 'debug', 'review')"
    )
    participants: list[str] = Field(
        default_factory=list, description="Participants (e.g., ['user', 'claude'])"
    )
    duration_seconds: int | None = Field(None, description="Session duration in seconds")
    outcome: Literal["resolved", "ongoing", "abandoned", "deferred"] | None = Field(
        None, description="Session outcome"
    )
    tool: str | None = Field(None, description="Tool used (e.g., 'claude-code', 'cursor')")
    messages_count: int | None = Field(None, description="Number of messages in session")


class UserData(BaseStructuredData):
    """Structured data for user preference memories."""

    type: Literal["user"] = "user"
    preference_key: str = Field(..., description="The preference identifier")
    preference_value: str | None = Field(None, description="The preference value")
    scope: Literal["global", "project", "session", "repo"] | None = Field(
        None, description="Scope of the preference"
    )
    priority: int | None = Field(None, description="Override priority (higher wins)")
    expires_at: str | None = Field(None, description="Expiration time if temporary")


class CodeData(BaseStructuredData):
    """Structured data for code snippet memories."""

    type: Literal["code"] = "code"
    language: str | None = Field(None, description="Programming language")
    purpose: Literal["snippet", "pattern", "example", "template", "fix"] | None = Field(
        None, description="Purpose of the code"
    )
    file_path: str | None = Field(None, description="Source file path")
    line_start: int | None = Field(None, description="Starting line number")
    line_end: int | None = Field(None, description="Ending line number")
    dependencies: list[str] = Field(
        default_factory=list, description="Required imports/dependencies"
    )
    framework: str | None = Field(None, description="Framework context (e.g., 'react', 'fastapi')")


class CommitData(BaseStructuredData):
    """Structured data for git commit memories."""

    type: Literal["commit"] = "commit"
    sha: str | None = Field(None, description="Commit SHA hash")
    short_sha: str | None = Field(None, description="Short commit SHA (7 chars)")
    author: str | None = Field(None, description="Commit author")
    author_email: str | None = Field(None, description="Author email")
    message: str | None = Field(None, description="Commit message")
    files_changed: list[str] = Field(default_factory=list, description="Files modified")
    insertions: int | None = Field(None, description="Lines added")
    deletions: int | None = Field(None, description="Lines removed")
    branch: str | None = Field(None, description="Branch name")
    timestamp: str | None = Field(None, description="Commit timestamp (ISO)")


class DocData(BaseStructuredData):
    """Structured data for documentation reference memories."""

    type: Literal["doc"] = "doc"
    doc_type: Literal["readme", "api", "tutorial", "reference", "guide", "changelog"] | None = (
        Field(None, description="Type of documentation")
    )
    title: str | None = Field(None, description="Document title")
    url: str | None = Field(None, description="URL to documentation")
    version: str | None = Field(None, description="Documentation version")
    last_updated: str | None = Field(None, description="Last update timestamp (ISO)")
    format: Literal["markdown", "html", "pdf", "rst", "asciidoc"] | None = Field(
        None, description="Document format"
    )
    sections: list[str] = Field(default_factory=list, description="Key sections covered")


# ============================================================================
# Workflow/Agent structured data types
# ============================================================================


class WorkflowData(BaseStructuredData):
    """Structured data for workflow definitions."""

    type: Literal["workflow"] = "workflow"
    name: str = Field(..., description="Workflow name")
    status: Literal["draft", "active", "paused", "completed", "failed"] = Field(
        "draft", description="Workflow execution status"
    )
    description: str | None = Field(None, description="Workflow description")
    steps: list[str] = Field(default_factory=list, description="Ordered step/task IDs")
    parallel_groups: list[list[str]] = Field(
        default_factory=list, description="Groups of steps to run in parallel"
    )
    dependencies: dict[str, list[str]] = Field(
        default_factory=dict, description="Step dependencies: step_id -> [dependency_ids]"
    )
    created_by: str | None = Field(None, description="Creator of workflow")
    started_at: str | None = Field(None, description="Start time (ISO)")
    completed_at: str | None = Field(None, description="Completion time (ISO)")
    error: str | None = Field(None, description="Error message if failed")


class TaskData(BaseStructuredData):
    """Structured data for workflow tasks."""

    type: Literal["task"] = "task"
    name: str = Field(..., description="Task name")
    workflow_id: str | None = Field(None, description="Parent workflow ID")
    status: Literal["pending", "running", "completed", "failed", "skipped", "cancelled"] = Field(
        "pending", description="Task execution status"
    )
    assigned_agent: str | None = Field(None, description="Agent assigned to this task")
    input_data: dict[str, Any] = Field(default_factory=dict, description="Task input data")
    output_data: dict[str, Any] = Field(default_factory=dict, description="Task output/result")
    error: str | None = Field(None, description="Error message if failed")
    retries: int = Field(0, description="Number of retry attempts made")
    max_retries: int = Field(3, description="Maximum retry attempts allowed")
    timeout_seconds: int | None = Field(None, description="Task timeout in seconds")
    started_at: str | None = Field(None, description="Start time (ISO)")
    completed_at: str | None = Field(None, description="Completion time (ISO)")
    depends_on: list[str] = Field(default_factory=list, description="Task IDs this depends on")


class StepData(BaseStructuredData):
    """Structured data for execution steps within tasks."""

    type: Literal["step"] = "step"
    task_id: str = Field(..., description="Parent task ID")
    step_number: int = Field(0, description="Step sequence number")
    action: Literal["tool_call", "llm_response", "decision", "observation", "error"] | None = Field(
        None, description="Type of action taken"
    )
    input: dict[str, Any] = Field(default_factory=dict, description="Step input")
    output: dict[str, Any] = Field(default_factory=dict, description="Step output")
    duration_ms: int | None = Field(None, description="Step duration in milliseconds")
    tokens_used: int | None = Field(None, description="LLM tokens consumed")
    reason: Literal["observation", "inference", "correction", "decay"] | None = Field(
        None, description="ChangeReason for this step"
    )
    tool_name: str | None = Field(None, description="Tool name if action=tool_call")
    error: str | None = Field(None, description="Error message if step failed")


class AgentRunData(BaseStructuredData):
    """Structured data for LLM agent execution records."""

    type: Literal["agent_run"] = "agent_run"
    agent_name: str = Field(..., description="Agent identifier/name")
    model: str | None = Field(None, description="LLM model used (e.g., 'claude-opus-4-5-20251101')")
    provider: str | None = Field(None, description="LLM provider (e.g., 'anthropic', 'openai')")
    workflow_id: str | None = Field(None, description="Parent workflow ID if part of workflow")
    task_id: str | None = Field(None, description="Parent task ID if part of task")
    prompt_tokens: int | None = Field(None, description="Input tokens consumed")
    completion_tokens: int | None = Field(None, description="Output tokens generated")
    total_tokens: int | None = Field(None, description="Total tokens used")
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list, description="Tool calls made: [{name, input, output}]"
    )
    status: Literal["running", "completed", "failed", "timeout", "cancelled"] = Field(
        "running", description="Agent run status"
    )
    started_at: str | None = Field(None, description="Start time (ISO)")
    completed_at: str | None = Field(None, description="Completion time (ISO)")
    error: str | None = Field(None, description="Error message if failed")
    system_prompt: str | None = Field(None, description="System prompt used")


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
    | FactData
    | EpisodicData
    | UserData
    | CodeData
    | CommitData
    | DocData
    # Workflow/Agent types
    | WorkflowData
    | TaskData
    | StepData
    | AgentRunData
)

# Mapping from memory type to its structured data class
# All 22 memory types now have structured schemas for complete type coverage
STRUCTURED_DATA_CLASSES: dict[str, type[BaseStructuredData]] = {
    # Core types
    "fact": FactData,
    "decision": DecisionData,
    "procedural": ProceduralData,
    "episodic": EpisodicData,
    "user": UserData,
    # Code types
    "code": CodeData,
    "error": ErrorData,
    "commit": CommitData,
    # Task management types
    "todo": TodoData,
    "issue": IssueData,
    # API/Schema types
    "api": APIData,
    "schema": SchemaData,
    # Testing types
    "test": TestData,
    "review": ReviewData,
    # Ops types
    "config": ConfigData,
    "dependency": DependencyData,
    "release": ReleaseData,
    "doc": DocData,
    # Workflow/Agent types
    "workflow": WorkflowData,
    "task": TaskData,
    "step": StepData,
    "agent_run": AgentRunData,
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


def _get_git_remote_url(repo_path: str) -> str | None:
    """Get the git remote origin URL for a repository."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _normalize_git_url(url: str) -> str:
    """
    Normalize git URL to a canonical form for consistent hashing.

    Handles:
    - SSH vs HTTPS: git@github.com:org/repo.git → github.com/org/repo
    - .git suffix removal
    - Trailing slashes
    - Case normalization for host
    """
    import re

    if not url:
        return ""

    # Remove trailing whitespace and .git suffix
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]

    # Convert SSH format to normalized form
    # git@github.com:org/repo → github.com/org/repo
    ssh_match = re.match(r"^(?:ssh://)?git@([^:]+):(.+)$", url)
    if ssh_match:
        host = ssh_match.group(1).lower()
        path = ssh_match.group(2)
        return f"{host}/{path}"

    # Convert HTTPS format to normalized form
    # https://github.com/org/repo → github.com/org/repo
    https_match = re.match(r"^https?://([^/]+)/(.+)$", url)
    if https_match:
        host = https_match.group(1).lower()
        path = https_match.group(2)
        return f"{host}/{path}"

    # Return as-is if no match (unusual URL format)
    return url.lower()


def _read_contextfs_namespace(repo_path: str) -> str | None:
    """Read stored namespace ID from .contextfs/config.yaml if it exists."""
    from pathlib import Path

    try:
        import yaml

        config_file = Path(repo_path) / ".contextfs" / "config.yaml"
        if config_file.exists():
            config = yaml.safe_load(config_file.read_text())
            if config and isinstance(config, dict):
                return config.get("namespace_id")
    except Exception:
        pass
    return None


class NamespaceSource:
    """Enum-like class for namespace derivation source."""

    EXPLICIT = "explicit"  # From .contextfs/config.yaml
    GIT_REMOTE = "git_remote"  # From git remote URL
    PATH = "path"  # Fallback to local path (not portable)


class Namespace(BaseModel):
    """
    Namespace for cross-repo memory isolation.

    Hierarchy:
    - global: Shared across all repos
    - org/team: Shared within organization
    - repo: Specific to repository
    - session: Specific to session

    Namespace ID derivation priority:
    1. Explicit ID from .contextfs/config.yaml (highest priority)
    2. Git remote URL (portable across machines)
    3. Local path fallback (not portable, for repos without remotes)
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str
    parent_id: str | None = None
    repo_path: str | None = None
    remote_url: str | None = None  # Git remote URL if available
    source: str | None = None  # How namespace was derived (explicit, git_remote, path)
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def global_ns(cls) -> "Namespace":
        return cls(id="global", name="global")

    @classmethod
    def for_repo(cls, repo_path: str, use_path_fallback: bool = True) -> "Namespace":
        """
        Create namespace for a repository.

        Derivation priority:
        1. Explicit namespace_id from .contextfs/config.yaml
        2. Git remote origin URL (portable across machines)
        3. Local path (fallback, not portable)

        Args:
            repo_path: Path to the repository
            use_path_fallback: If False, return None when no remote URL found

        Returns:
            Namespace object with stable ID
        """
        from pathlib import Path

        resolved_path = str(Path(repo_path).resolve())
        repo_name = resolved_path.split("/")[-1]

        # 1. Check for explicit namespace in .contextfs/config.yaml
        explicit_id = _read_contextfs_namespace(resolved_path)
        if explicit_id:
            return cls(
                id=explicit_id,
                name=repo_name,
                repo_path=resolved_path,
                source=NamespaceSource.EXPLICIT,
            )

        # 2. Try git remote URL (portable across machines)
        remote_url = _get_git_remote_url(resolved_path)
        if remote_url:
            normalized_url = _normalize_git_url(remote_url)
            repo_id = hashlib.sha256(normalized_url.encode()).hexdigest()[:12]
            return cls(
                id=f"repo-{repo_id}",
                name=repo_name,
                repo_path=resolved_path,
                remote_url=remote_url,
                source=NamespaceSource.GIT_REMOTE,
            )

        # 3. Fallback to path-based (not portable, but works for local-only repos)
        if use_path_fallback:
            repo_id = hashlib.sha256(resolved_path.encode()).hexdigest()[:12]
            return cls(
                id=f"local-{repo_id}",
                name=repo_name,
                repo_path=resolved_path,
                source=NamespaceSource.PATH,
            )

        return None

    @classmethod
    def for_repo_legacy(cls, repo_path: str) -> "Namespace":
        """
        Legacy path-based namespace generation.

        Use this only for migration purposes or when you specifically
        need the old path-based behavior.
        """
        from pathlib import Path

        resolved_path = str(Path(repo_path).resolve())
        repo_id = hashlib.sha256(resolved_path.encode()).hexdigest()[:12]
        return cls(
            id=f"repo-{repo_id}",
            name=resolved_path.split("/")[-1],
            repo_path=resolved_path,
            source=NamespaceSource.PATH,
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

    # Authoritative flag (Phase 3)
    # Marks this memory as the canonical/official version in a lineage chain
    authoritative: bool = Field(
        default=False,
        description="Whether this is the authoritative/canonical version in a lineage",
    )

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

    @classmethod
    def fact(
        cls,
        content: str,
        category: str | None = None,
        source: str | None = None,
        confidence: float | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create a fact memory with typed structured_data."""
        structured: dict[str, Any] = {}
        if category:
            structured["category"] = category
        if source:
            structured["source"] = source
        if confidence is not None:
            structured["confidence"] = confidence
        return cls(
            content=content,
            type=MemoryType.FACT,
            structured_data=structured if structured else None,
            **kwargs,
        )

    @classmethod
    def episodic(
        cls,
        content: str,
        session_type: str | None = None,
        participants: list[str] | None = None,
        outcome: str | None = None,
        tool: str | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create an episodic memory with typed structured_data."""
        structured: dict[str, Any] = {}
        if session_type:
            structured["session_type"] = session_type
        if participants:
            structured["participants"] = participants
        if outcome:
            structured["outcome"] = outcome
        if tool:
            structured["tool"] = tool
        return cls(
            content=content,
            type=MemoryType.EPISODIC,
            structured_data=structured if structured else None,
            **kwargs,
        )

    @classmethod
    def user(
        cls,
        content: str,
        preference_key: str,
        preference_value: str | None = None,
        scope: str | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create a user preference memory with typed structured_data."""
        structured: dict[str, Any] = {"preference_key": preference_key}
        if preference_value:
            structured["preference_value"] = preference_value
        if scope:
            structured["scope"] = scope
        return cls(
            content=content,
            type=MemoryType.USER,
            structured_data=structured,
            **kwargs,
        )

    @classmethod
    def code(
        cls,
        content: str,
        language: str | None = None,
        purpose: str | None = None,
        file_path: str | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create a code memory with typed structured_data."""
        structured: dict[str, Any] = {}
        if language:
            structured["language"] = language
        if purpose:
            structured["purpose"] = purpose
        if file_path:
            structured["file_path"] = file_path
        return cls(
            content=content,
            type=MemoryType.CODE,
            structured_data=structured if structured else None,
            **kwargs,
        )

    @classmethod
    def commit(
        cls,
        content: str,
        sha: str | None = None,
        author: str | None = None,
        message: str | None = None,
        files_changed: list[str] | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create a commit memory with typed structured_data."""
        structured: dict[str, Any] = {}
        if sha:
            structured["sha"] = sha
            structured["short_sha"] = sha[:7] if len(sha) >= 7 else sha
        if author:
            structured["author"] = author
        if message:
            structured["message"] = message
        if files_changed:
            structured["files_changed"] = files_changed
        return cls(
            content=content,
            type=MemoryType.COMMIT,
            structured_data=structured if structured else None,
            **kwargs,
        )

    @classmethod
    def doc(
        cls,
        content: str,
        doc_type: str | None = None,
        title: str | None = None,
        url: str | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create a documentation reference memory with typed structured_data."""
        structured: dict[str, Any] = {}
        if doc_type:
            structured["doc_type"] = doc_type
        if title:
            structured["title"] = title
        if url:
            structured["url"] = url
        return cls(
            content=content,
            type=MemoryType.DOC,
            structured_data=structured if structured else None,
            **kwargs,
        )

    # =========================================================================
    # Workflow/Agent Factory Methods
    # =========================================================================

    @classmethod
    def workflow(
        cls,
        content: str,
        name: str,
        status: str = "draft",
        description: str | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create a workflow memory with typed structured_data."""
        structured: dict[str, Any] = {"name": name, "status": status}
        if description:
            structured["description"] = description
        return cls(
            content=content,
            type=MemoryType.WORKFLOW,
            structured_data=structured,
            **kwargs,
        )

    @classmethod
    def task(
        cls,
        content: str,
        name: str,
        workflow_id: str | None = None,
        status: str = "pending",
        assigned_agent: str | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create a task memory with typed structured_data."""
        structured: dict[str, Any] = {"name": name, "status": status}
        if workflow_id:
            structured["workflow_id"] = workflow_id
        if assigned_agent:
            structured["assigned_agent"] = assigned_agent
        return cls(
            content=content,
            type=MemoryType.TASK,
            structured_data=structured,
            **kwargs,
        )

    @classmethod
    def step(
        cls,
        content: str,
        task_id: str,
        step_number: int = 0,
        action: str | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create a step memory with typed structured_data."""
        structured: dict[str, Any] = {"task_id": task_id, "step_number": step_number}
        if action:
            structured["action"] = action
        return cls(
            content=content,
            type=MemoryType.STEP,
            structured_data=structured,
            **kwargs,
        )

    @classmethod
    def agent_run(
        cls,
        content: str,
        agent_name: str,
        model: str | None = None,
        status: str = "running",
        workflow_id: str | None = None,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create an agent run memory with typed structured_data."""
        structured: dict[str, Any] = {"agent_name": agent_name, "status": status}
        if model:
            structured["model"] = model
        if workflow_id:
            structured["workflow_id"] = workflow_id
        if task_id:
            structured["task_id"] = task_id
        return cls(
            content=content,
            type=MemoryType.AGENT_RUN,
            structured_data=structured,
            **kwargs,
        )

    @classmethod
    def issue(
        cls,
        content: str,
        title: str,
        severity: str | None = None,
        status: str | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create an issue memory with typed structured_data."""
        structured: dict[str, Any] = {"title": title}
        if severity:
            structured["severity"] = severity
        if status:
            structured["status"] = status
        return cls(
            content=content,
            type=MemoryType.ISSUE,
            structured_data=structured,
            **kwargs,
        )

    @classmethod
    def schema_def(
        cls,
        content: str,
        name: str,
        schema_type: str | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create a schema memory with typed structured_data.

        Note: Named schema_def to avoid conflict with Pydantic's schema() method.
        """
        structured: dict[str, Any] = {"name": name}
        if schema_type:
            structured["schema_type"] = schema_type
        return cls(
            content=content,
            type=MemoryType.SCHEMA,
            structured_data=structured,
            **kwargs,
        )

    @classmethod
    def test_case(
        cls,
        content: str,
        name: str,
        test_type: str | None = None,
        status: str | None = None,
        file: str | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create a test memory with typed structured_data.

        Note: Named test_case to avoid conflict with pytest.
        """
        structured: dict[str, Any] = {"name": name}
        if test_type:
            structured["test_type"] = test_type
        if status:
            structured["status"] = status
        if file:
            structured["file"] = file
        return cls(
            content=content,
            type=MemoryType.TEST,
            structured_data=structured,
            **kwargs,
        )

    @classmethod
    def review(
        cls,
        content: str,
        pr_number: int,
        reviewer: str | None = None,
        status: str = "pending",
        **kwargs: Any,
    ) -> "Memory":
        """Create a review memory with typed structured_data."""
        structured: dict[str, Any] = {"pr_number": pr_number, "status": status}
        if reviewer:
            structured["reviewer"] = reviewer
        return cls(
            content=content,
            type=MemoryType.REVIEW,
            structured_data=structured,
            **kwargs,
        )

    @classmethod
    def release(
        cls,
        content: str,
        version: str,
        date: str | None = None,
        changes: list[str] | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create a release memory with typed structured_data."""
        structured: dict[str, Any] = {"version": version}
        if date:
            structured["date"] = date
        if changes:
            structured["changes"] = changes
        return cls(
            content=content,
            type=MemoryType.RELEASE,
            structured_data=structured,
            **kwargs,
        )

    @classmethod
    def config(
        cls,
        content: str,
        name: str,
        environment: str | None = None,
        settings: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create a config memory with typed structured_data."""
        structured: dict[str, Any] = {"name": name}
        if environment:
            structured["environment"] = environment
        if settings:
            structured["settings"] = settings
        return cls(
            content=content,
            type=MemoryType.CONFIG,
            structured_data=structured,
            **kwargs,
        )

    @classmethod
    def dependency(
        cls,
        content: str,
        name: str,
        version: str = "unknown",
        dep_type: str | None = None,
        **kwargs: Any,
    ) -> "Memory":
        """Create a dependency memory with typed structured_data."""
        structured: dict[str, Any] = {"name": name, "version": version}
        if dep_type:
            structured["dep_type"] = dep_type
        return cls(
            content=content,
            type=MemoryType.DEPENDENCY,
            structured_data=structured,
            **kwargs,
        )

    # =========================================================================
    # Formal Type System Integration
    # =========================================================================

    def as_typed(self, schema_type: type) -> "Mem":
        """
        Convert to schema-indexed Mem[S] type for type-safe structured_data access.

        This method wraps the Memory in a Mem[S] container that provides:
        - Type-safe access to structured_data via the .data property
        - Schema validation ensuring structured_data matches type S
        - Generic type parameter for static type checking

        Args:
            schema_type: The schema type (must be BaseSchema subclass).

        Returns:
            Mem[S] wrapper around this memory.

        Raises:
            ValueError: If structured_data doesn't match schema.

        Example:
            >>> from contextfs.schemas import DecisionData
            >>> memory = Memory.decision("DB choice", decision="PostgreSQL")
            >>> typed = memory.as_typed(DecisionData)
            >>> typed.data.decision  # Type-safe: str
            'PostgreSQL'
        """
        from contextfs.types.memory import Mem

        return Mem.wrap(self, schema_type)

    def as_versioned(self, schema_type: type) -> "VersionedMem":
        """
        Get versioned wrapper with timeline support for evolution tracking.

        This method wraps the Memory in a VersionedMem[S] container that provides:
        - Timeline of version entries with ChangeReason tracking
        - Type-safe evolution operations
        - Integration with memory lineage system

        Args:
            schema_type: The schema type (must be BaseSchema subclass).

        Returns:
            VersionedMem[S] wrapper with timeline support.

        Example:
            >>> from contextfs.schemas import DecisionData
            >>> from contextfs.types import ChangeReason
            >>> memory = Memory.decision("DB choice", decision="PostgreSQL")
            >>> versioned = memory.as_versioned(DecisionData)
            >>> versioned.evolve(
            ...     DecisionData(decision="SQLite"),
            ...     reason=ChangeReason.CORRECTION
            ... )
        """
        from contextfs.types.versioned import VersionedMem

        return VersionedMem.from_memory(self, schema_type)


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
