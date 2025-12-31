"""Unit tests for typed schema validation.

Tests the structured_data field and JSON schema validation per memory type.
"""

import pytest
from pydantic import ValidationError

from contextfs.schemas import (
    TYPE_SCHEMAS,
    Memory,
    MemoryType,
    get_type_schema,
    validate_structured_data,
)


class TestTypeSchemas:
    """Test TYPE_SCHEMAS definitions."""

    def test_type_schemas_exist(self):
        """TYPE_SCHEMAS should have schemas for key types."""
        assert "decision" in TYPE_SCHEMAS
        assert "procedural" in TYPE_SCHEMAS
        assert "error" in TYPE_SCHEMAS
        assert "todo" in TYPE_SCHEMAS
        assert "api" in TYPE_SCHEMAS

    def test_decision_schema_structure(self):
        """Decision schema should have expected properties."""
        schema = TYPE_SCHEMAS["decision"]
        assert schema["type"] == "object"
        assert "decision" in schema["properties"]
        assert "rationale" in schema["properties"]
        assert "alternatives" in schema["properties"]
        assert "decision" in schema["required"]

    def test_procedural_schema_structure(self):
        """Procedural schema should have expected properties."""
        schema = TYPE_SCHEMAS["procedural"]
        assert "steps" in schema["properties"]
        assert "prerequisites" in schema["properties"]
        assert "steps" in schema["required"]

    def test_error_schema_structure(self):
        """Error schema should have expected properties."""
        schema = TYPE_SCHEMAS["error"]
        assert "error_type" in schema["properties"]
        assert "message" in schema["properties"]
        assert "stack_trace" in schema["properties"]
        assert "error_type" in schema["required"]
        assert "message" in schema["required"]

    def test_get_type_schema_valid(self):
        """get_type_schema should return schema for valid type."""
        schema = get_type_schema("decision")
        assert schema is not None
        assert "properties" in schema

    def test_get_type_schema_invalid(self):
        """get_type_schema should return None for type without schema."""
        schema = get_type_schema("fact")  # fact has no special schema
        assert schema is None


class TestValidateStructuredData:
    """Test validate_structured_data function."""

    def test_valid_decision_data(self):
        """Valid decision data should pass validation."""
        data = {
            "decision": "Use JWT for authentication",
            "rationale": "Stateless, scales well",
            "alternatives": ["Sessions", "OAuth only"],
        }
        is_valid, error = validate_structured_data("decision", data)
        assert is_valid is True
        assert error is None

    def test_minimal_decision_data(self):
        """Minimal decision data (just required fields) should pass."""
        data = {"decision": "Use PostgreSQL"}
        is_valid, error = validate_structured_data("decision", data)
        assert is_valid is True

    def test_invalid_decision_missing_required(self):
        """Decision without required 'decision' field should fail."""
        data = {"rationale": "Some reason"}  # missing 'decision'
        is_valid, error = validate_structured_data("decision", data)
        assert is_valid is False
        assert error is not None
        assert "decision" in error.lower() or "required" in error.lower()

    def test_valid_error_data(self):
        """Valid error data should pass validation."""
        data = {
            "error_type": "ValueError",
            "message": "Invalid input",
            "stack_trace": "Traceback...",
            "file": "test.py",
            "line": 42,
        }
        is_valid, error = validate_structured_data("error", data)
        assert is_valid is True

    def test_invalid_error_wrong_type(self):
        """Error with wrong type for line number should fail."""
        data = {
            "error_type": "ValueError",
            "message": "Invalid input",
            "line": "not a number",  # should be integer
        }
        is_valid, error = validate_structured_data("error", data)
        assert is_valid is False

    def test_valid_procedural_data(self):
        """Valid procedural data should pass validation."""
        data = {
            "steps": ["Step 1", "Step 2", "Step 3"],
            "prerequisites": ["Python 3.10+"],
        }
        is_valid, error = validate_structured_data("procedural", data)
        assert is_valid is True

    def test_type_without_schema(self):
        """Type without schema should accept any data."""
        data = {"anything": "goes", "nested": {"data": True}}
        is_valid, error = validate_structured_data("fact", data)
        assert is_valid is True
        assert error is None

    def test_additional_properties_allowed(self):
        """Schemas allow additional properties by default."""
        data = {
            "decision": "Something",
            "custom_field": "Should be allowed",
            "another_custom": 123,
        }
        is_valid, error = validate_structured_data("decision", data)
        assert is_valid is True


class TestMemoryWithStructuredData:
    """Test Memory class with structured_data field."""

    def test_memory_without_structured_data(self):
        """Memory can be created without structured_data."""
        memory = Memory(content="Test content")
        assert memory.structured_data is None

    def test_memory_with_valid_structured_data(self):
        """Memory with valid structured_data should be created."""
        memory = Memory(
            content="Auth decision",
            type=MemoryType.DECISION,
            structured_data={"decision": "Use JWT"},
        )
        assert memory.structured_data == {"decision": "Use JWT"}

    def test_memory_with_full_structured_data(self):
        """Memory with full structured_data should preserve all fields."""
        data = {
            "decision": "Use PostgreSQL",
            "rationale": "ACID compliance, mature ecosystem",
            "alternatives": ["MySQL", "MongoDB"],
            "constraints": ["Must support JSON", "Team experience"],
            "status": "accepted",
        }
        memory = Memory(
            content="Database decision",
            type=MemoryType.DECISION,
            structured_data=data,
        )
        assert memory.structured_data == data
        assert memory.get_structured_field("decision") == "Use PostgreSQL"
        assert memory.get_structured_field("alternatives") == ["MySQL", "MongoDB"]

    def test_memory_invalid_structured_data(self):
        """Memory with invalid structured_data should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Memory(
                content="Missing decision field",
                type=MemoryType.DECISION,
                structured_data={"rationale": "No decision specified"},
            )
        assert "structured_data" in str(exc_info.value).lower()

    def test_get_structured_field_with_default(self):
        """get_structured_field should return default for missing fields."""
        memory = Memory(
            content="Test",
            type=MemoryType.DECISION,
            structured_data={"decision": "Something"},
        )
        assert memory.get_structured_field("rationale", "Not specified") == "Not specified"
        assert memory.get_structured_field("nonexistent") is None

    def test_get_structured_field_no_structured_data(self):
        """get_structured_field should return default when no structured_data."""
        memory = Memory(content="Test")
        assert memory.get_structured_field("anything", "default") == "default"


class TestTodoSchema:
    """Test todo schema validation."""

    def test_valid_todo(self):
        """Valid todo data should pass."""
        data = {
            "title": "Implement feature X",
            "status": "pending",
            "priority": "high",
        }
        is_valid, error = validate_structured_data("todo", data)
        assert is_valid is True

    def test_invalid_todo_status(self):
        """Todo with invalid status enum should fail."""
        data = {
            "title": "Some task",
            "status": "invalid_status",  # not in enum
        }
        is_valid, error = validate_structured_data("todo", data)
        assert is_valid is False


class TestApiSchema:
    """Test API schema validation."""

    def test_valid_api_endpoint(self):
        """Valid API endpoint data should pass."""
        data = {
            "endpoint": "/api/users",
            "method": "GET",
            "parameters": [
                {"name": "limit", "type": "integer", "required": False},
            ],
        }
        is_valid, error = validate_structured_data("api", data)
        assert is_valid is True

    def test_minimal_api_endpoint(self):
        """Minimal API data (just endpoint) should pass."""
        data = {"endpoint": "/api/health"}
        is_valid, error = validate_structured_data("api", data)
        assert is_valid is True


class TestConfigSchema:
    """Test config schema validation."""

    def test_valid_config(self):
        """Valid config data should pass."""
        data = {
            "name": "database",
            "environment": "production",
            "settings": {"host": "localhost", "port": 5432},
        }
        is_valid, error = validate_structured_data("config", data)
        assert is_valid is True


class TestDependencySchema:
    """Test dependency schema validation."""

    def test_valid_dependency(self):
        """Valid dependency data should pass."""
        data = {
            "name": "requests",
            "version": "2.31.0",
            "latest_version": "2.32.0",
            "type": "runtime",
        }
        is_valid, error = validate_structured_data("dependency", data)
        assert is_valid is True

    def test_missing_required_dependency_fields(self):
        """Dependency missing required fields should fail."""
        data = {"name": "requests"}  # missing version
        is_valid, error = validate_structured_data("dependency", data)
        assert is_valid is False
