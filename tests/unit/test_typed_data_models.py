"""Unit tests for Phase 2: Pydantic typed data models."""

import pytest

from contextfs.schemas import (
    STRUCTURED_DATA_CLASSES,
    APIData,
    BaseStructuredData,
    ConfigData,
    DecisionData,
    DependencyData,
    ErrorData,
    IssueData,
    Memory,
    MemoryType,
    ProceduralData,
    ReleaseData,
    ReviewComment,
    ReviewData,
    SchemaData,
    SchemaField,
    TestData,
    TodoData,
    parse_structured_data,
    serialize_structured_data,
)


class TestTypedDataModels:
    """Test individual typed data model classes."""

    def test_decision_data_required_field(self):
        """Test DecisionData requires decision field."""
        data = DecisionData(decision="Use PostgreSQL")
        assert data.decision == "Use PostgreSQL"
        assert data.type == "decision"
        assert data.rationale is None
        assert data.alternatives == []

    def test_decision_data_all_fields(self):
        """Test DecisionData with all fields."""
        data = DecisionData(
            decision="Use PostgreSQL",
            rationale="ACID compliance",
            alternatives=["MySQL", "MongoDB"],
            constraints=["Must support JSON"],
            date="2024-01-15",
            status="accepted",
        )
        assert data.decision == "Use PostgreSQL"
        assert data.rationale == "ACID compliance"
        assert len(data.alternatives) == 2
        assert data.status == "accepted"

    def test_decision_data_invalid_status(self):
        """Test DecisionData rejects invalid status."""
        with pytest.raises(ValueError):
            DecisionData(decision="Test", status="invalid_status")

    def test_procedural_data_required_steps(self):
        """Test ProceduralData requires steps."""
        with pytest.raises(ValueError):
            ProceduralData()  # Missing required 'steps'

    def test_procedural_data_with_steps(self):
        """Test ProceduralData with steps."""
        data = ProceduralData(steps=["Step 1", "Step 2", "Step 3"])
        assert len(data.steps) == 3
        assert data.type == "procedural"

    def test_error_data_required_fields(self):
        """Test ErrorData requires error_type and message."""
        data = ErrorData(error_type="ValueError", message="Invalid input")
        assert data.error_type == "ValueError"
        assert data.message == "Invalid input"
        assert data.type == "error"

    def test_error_data_with_location(self):
        """Test ErrorData with file location."""
        data = ErrorData(
            error_type="TypeError",
            message="Expected str",
            file="app.py",
            line=42,
            resolution="Cast to string",
        )
        assert data.file == "app.py"
        assert data.line == 42
        assert data.resolution == "Cast to string"

    def test_api_data_required_endpoint(self):
        """Test APIData requires endpoint."""
        data = APIData(endpoint="/api/users")
        assert data.endpoint == "/api/users"
        assert data.type == "api"

    def test_api_data_with_method(self):
        """Test APIData with HTTP method."""
        data = APIData(endpoint="/api/users", method="POST")
        assert data.method == "POST"

    def test_api_data_invalid_method(self):
        """Test APIData rejects invalid method."""
        with pytest.raises(ValueError):
            APIData(endpoint="/api/users", method="INVALID")

    def test_todo_data_required_title(self):
        """Test TodoData requires title."""
        data = TodoData(title="Fix bug")
        assert data.title == "Fix bug"
        assert data.type == "todo"

    def test_todo_data_with_status_priority(self):
        """Test TodoData with status and priority."""
        data = TodoData(
            title="Implement feature",
            status="in_progress",
            priority="high",
        )
        assert data.status == "in_progress"
        assert data.priority == "high"

    def test_issue_data(self):
        """Test IssueData."""
        data = IssueData(
            title="Login fails",
            severity="high",
            status="investigating",
            steps_to_reproduce=["Go to login", "Enter credentials", "Click submit"],
        )
        assert data.title == "Login fails"
        assert data.severity == "high"
        assert len(data.steps_to_reproduce) == 3

    def test_test_data(self):
        """Test TestData."""
        data = TestData(
            name="test_login",
            test_type="unit",  # Note: use test_type, not type (type is reserved for discriminator)
            status="passing",
            file="tests/test_auth.py",
        )
        assert data.name == "test_login"
        assert data.test_type == "unit"
        assert data.status == "passing"
        assert data.type == "test"  # Discriminator type

    def test_config_data(self):
        """Test ConfigData."""
        data = ConfigData(
            name="database",
            environment="production",
            settings={"host": "localhost", "port": 5432},
            secrets=["password", "api_key"],
        )
        assert data.name == "database"
        assert data.environment == "production"
        assert data.settings["port"] == 5432

    def test_dependency_data(self):
        """Test DependencyData."""
        data = DependencyData(
            name="pydantic",
            version="2.5.0",
            latest_version="2.6.0",
            dep_type="runtime",
        )
        assert data.name == "pydantic"
        assert data.version == "2.5.0"
        assert data.dep_type == "runtime"

    def test_release_data(self):
        """Test ReleaseData."""
        data = ReleaseData(
            version="1.0.0",
            date="2024-01-15",
            changes=["Added feature X", "Fixed bug Y"],
            breaking_changes=["Removed deprecated API"],
        )
        assert data.version == "1.0.0"
        assert len(data.changes) == 2
        assert len(data.breaking_changes) == 1

    def test_review_data(self):
        """Test ReviewData."""
        data = ReviewData(
            pr_number=123,
            reviewer="alice",
            status="approved",
            comments=[
                ReviewComment(file="app.py", line=42, comment="Good fix!"),
            ],
        )
        assert data.pr_number == 123
        assert data.status == "approved"
        assert len(data.comments) == 1

    def test_schema_data(self):
        """Test SchemaData."""
        data = SchemaData(
            name="User",
            schema_type="database",
            fields=[
                SchemaField(name="id", type="integer", required=True),
                SchemaField(name="email", type="string", required=True),
            ],
            relationships=["posts", "comments"],
        )
        assert data.name == "User"
        assert data.schema_type == "database"
        assert len(data.fields) == 2

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed for extensibility."""
        data = DecisionData(
            decision="Test",
            custom_field="custom_value",  # Extra field
        )
        assert data.decision == "Test"
        # Extra fields should be accessible via model_extra
        assert data.model_extra.get("custom_field") == "custom_value"


class TestParseStructuredData:
    """Test parse_structured_data function."""

    def test_parse_decision_data(self):
        """Test parsing decision data into typed model."""
        raw = {"decision": "Use Redis", "rationale": "Fast caching"}
        result = parse_structured_data("decision", raw)
        assert isinstance(result, DecisionData)
        assert result.decision == "Use Redis"
        assert result.rationale == "Fast caching"

    def test_parse_procedural_data(self):
        """Test parsing procedural data."""
        raw = {"steps": ["Step 1", "Step 2"]}
        result = parse_structured_data("procedural", raw)
        assert isinstance(result, ProceduralData)
        assert len(result.steps) == 2

    def test_parse_unknown_type_returns_dict(self):
        """Test parsing unknown type returns raw dict."""
        raw = {"custom": "data"}
        result = parse_structured_data("unknown_type_xyz", raw)  # Unknown type has no schema
        assert isinstance(result, dict)
        assert result == raw

    def test_parse_invalid_data_returns_dict(self):
        """Test parsing invalid data falls back to dict."""
        raw = {"invalid": "data"}  # Missing required 'decision'
        result = parse_structured_data("decision", raw)
        # Should fall back to raw dict for backward compatibility
        assert isinstance(result, dict)


class TestSerializeStructuredData:
    """Test serialize_structured_data function."""

    def test_serialize_pydantic_model(self):
        """Test serializing Pydantic model."""
        data = DecisionData(decision="Test")
        result = serialize_structured_data(data)
        assert isinstance(result, dict)
        assert result["decision"] == "Test"
        assert result["type"] == "decision"

    def test_serialize_dict(self):
        """Test serializing raw dict."""
        data = {"custom": "value"}
        result = serialize_structured_data(data)
        assert result == data

    def test_serialize_none(self):
        """Test serializing None."""
        result = serialize_structured_data(None)
        assert result is None


class TestStructuredDataClassesMapping:
    """Test STRUCTURED_DATA_CLASSES mapping."""

    def test_all_types_have_classes(self):
        """Test that expected types have class mappings."""
        expected_types = [
            "decision",
            "procedural",
            "error",
            "api",
            "todo",
            "issue",
            "test",
            "config",
            "dependency",
            "release",
            "review",
            "schema",
        ]
        for t in expected_types:
            assert t in STRUCTURED_DATA_CLASSES
            assert issubclass(STRUCTURED_DATA_CLASSES[t], BaseStructuredData)

    def test_classes_have_type_literal(self):
        """Test that each class has a type literal field."""
        for type_name, cls in STRUCTURED_DATA_CLASSES.items():
            # The type field should have a default matching the type name
            field_info = cls.model_fields.get("type")
            assert field_info is not None, f"Class {cls.__name__} missing type field"
            assert field_info.default == type_name


class TestMemoryTypedDataProperty:
    """Test Memory.typed_data property."""

    def test_typed_data_returns_model(self):
        """Test typed_data returns typed model when available."""
        memory = Memory(
            content="Database choice",
            type=MemoryType.DECISION,
            structured_data={"decision": "PostgreSQL"},
        )
        typed = memory.typed_data
        assert isinstance(typed, DecisionData)
        assert typed.decision == "PostgreSQL"

    def test_typed_data_returns_dict_for_unknown_type(self):
        """Test typed_data returns dict for types without models."""
        memory = Memory(
            content="A fact",
            type=MemoryType.FACT,
            structured_data={"category": "config", "custom": "data"},
        )
        typed = memory.typed_data
        # FACT now has a schema, so it returns FactData (which allows extra fields)
        from contextfs.schemas import FactData

        assert isinstance(typed, FactData)
        assert typed.category == "config"
        # Extra fields are preserved due to extra="allow" config
        assert getattr(typed, "custom", None) == "data"

    def test_typed_data_returns_none_when_empty(self):
        """Test typed_data returns None when structured_data is None."""
        memory = Memory(content="Simple memory", type=MemoryType.FACT)
        assert memory.typed_data is None


class TestMemoryFactoryMethods:
    """Test Memory factory methods."""

    def test_decision_factory(self):
        """Test Memory.decision() factory method."""
        memory = Memory.decision(
            content="Database selection",
            decision="PostgreSQL",
            rationale="ACID compliance",
            alternatives=["MySQL", "MongoDB"],
        )
        assert memory.type == MemoryType.DECISION
        assert memory.structured_data["decision"] == "PostgreSQL"
        assert memory.structured_data["rationale"] == "ACID compliance"
        assert len(memory.structured_data["alternatives"]) == 2

    def test_procedural_factory(self):
        """Test Memory.procedural() factory method."""
        memory = Memory.procedural(
            content="Setup guide",
            steps=["Install", "Configure", "Run"],
            title="Getting Started",
            prerequisites=["Python 3.10+"],
        )
        assert memory.type == MemoryType.PROCEDURAL
        assert len(memory.structured_data["steps"]) == 3
        assert memory.structured_data["title"] == "Getting Started"

    def test_error_factory(self):
        """Test Memory.error() factory method."""
        memory = Memory.error(
            content="Connection failed",
            error_type="ConnectionError",
            message="Could not connect to database",
            file="db.py",
            line=42,
            resolution="Check credentials",
        )
        assert memory.type == MemoryType.ERROR
        assert memory.structured_data["error_type"] == "ConnectionError"
        assert memory.structured_data["line"] == 42

    def test_api_factory(self):
        """Test Memory.api() factory method."""
        memory = Memory.api(
            content="User endpoint",
            endpoint="/api/users",
            method="GET",
            parameters=[{"name": "id", "type": "string", "required": True}],
        )
        assert memory.type == MemoryType.API
        assert memory.structured_data["endpoint"] == "/api/users"
        assert memory.structured_data["method"] == "GET"

    def test_todo_factory(self):
        """Test Memory.todo() factory method."""
        memory = Memory.todo(
            content="Fix login bug",
            title="Fix authentication issue",
            status="in_progress",
            priority="high",
        )
        assert memory.type == MemoryType.TODO
        assert memory.structured_data["title"] == "Fix authentication issue"
        assert memory.structured_data["status"] == "in_progress"

    def test_factory_with_additional_kwargs(self):
        """Test factory methods accept additional Memory fields."""
        memory = Memory.decision(
            content="Test",
            decision="Test decision",
            tags=["test", "example"],
            summary="A test decision",
        )
        assert memory.tags == ["test", "example"]
        assert memory.summary == "A test decision"


class TestBackwardCompatibility:
    """Test backward compatibility with raw dicts."""

    def test_raw_dict_still_works(self):
        """Test that raw dicts still work without typed models."""
        memory = Memory(
            content="Legacy memory",
            type=MemoryType.DECISION,
            structured_data={"decision": "Legacy", "custom_field": "custom"},
        )
        assert memory.structured_data["decision"] == "Legacy"
        assert memory.structured_data["custom_field"] == "custom"

    def test_typed_data_preserves_extra_fields(self):
        """Test that typed_data preserves extra fields."""
        memory = Memory(
            content="Test",
            type=MemoryType.DECISION,
            structured_data={"decision": "Test", "extra_field": "preserved"},
        )
        typed = memory.typed_data
        assert isinstance(typed, DecisionData)
        assert typed.decision == "Test"
        # Extra field should be in model_extra
        assert typed.model_extra.get("extra_field") == "preserved"
