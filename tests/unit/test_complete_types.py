"""Unit tests for Phase 0: Complete type system schemas."""

import pytest

from contextfs.schemas import (
    STRUCTURED_DATA_CLASSES,
    BaseStructuredData,
    CodeData,
    CommitData,
    DocData,
    EpisodicData,
    FactData,
    Memory,
    MemoryType,
    UserData,
    parse_structured_data,
)


class TestFactData:
    """Test FactData structured data model."""

    def test_fact_data_defaults(self):
        """Test FactData with defaults."""
        data = FactData()
        assert data.type == "fact"
        assert data.category is None
        assert data.source is None
        assert data.confidence is None
        assert data.verified_at is None
        assert data.valid_until is None

    def test_fact_data_all_fields(self):
        """Test FactData with all fields."""
        data = FactData(
            category="configuration",
            source="documentation",
            confidence=0.95,
            verified_at="2024-01-15T10:00:00Z",
            valid_until="2025-01-15T10:00:00Z",
        )
        assert data.category == "configuration"
        assert data.source == "documentation"
        assert data.confidence == 0.95
        assert data.verified_at == "2024-01-15T10:00:00Z"

    def test_fact_data_confidence_range(self):
        """Test FactData confidence should be 0.0-1.0."""
        data = FactData(confidence=0.5)
        assert data.confidence == 0.5

        # Values outside 0-1 are rejected (validated)
        with pytest.raises(ValueError):
            FactData(confidence=1.5)


class TestEpisodicData:
    """Test EpisodicData structured data model."""

    def test_episodic_data_defaults(self):
        """Test EpisodicData with defaults."""
        data = EpisodicData()
        assert data.type == "episodic"
        assert data.session_type is None
        assert data.participants == []
        assert data.duration_seconds is None
        assert data.outcome is None
        assert data.tool is None

    def test_episodic_data_all_fields(self):
        """Test EpisodicData with all fields."""
        data = EpisodicData(
            session_type="conversation",
            participants=["user", "claude"],
            duration_seconds=3600,
            outcome="resolved",
            tool="claude-code",
        )
        assert data.session_type == "conversation"
        assert len(data.participants) == 2
        assert data.duration_seconds == 3600
        assert data.outcome == "resolved"
        assert data.tool == "claude-code"


class TestUserData:
    """Test UserData structured data model."""

    def test_user_data_required_key(self):
        """Test UserData requires preference_key."""
        data = UserData(preference_key="theme")
        assert data.preference_key == "theme"
        assert data.type == "user"
        assert data.preference_value is None
        assert data.scope is None
        assert data.priority is None

    def test_user_data_all_fields(self):
        """Test UserData with all fields."""
        data = UserData(
            preference_key="editor.theme",
            preference_value="dark",
            scope="project",
            priority=10,
        )
        assert data.preference_key == "editor.theme"
        assert data.preference_value == "dark"
        assert data.scope == "project"
        assert data.priority == 10

    def test_user_data_invalid_scope(self):
        """Test UserData rejects invalid scope."""
        with pytest.raises(ValueError):
            UserData(preference_key="test", scope="invalid_scope")

    def test_user_data_valid_scopes(self):
        """Test all valid user data scopes."""
        valid_scopes = ["global", "project", "session"]
        for scope in valid_scopes:
            data = UserData(preference_key="test", scope=scope)
            assert data.scope == scope


class TestCodeData:
    """Test CodeData structured data model."""

    def test_code_data_defaults(self):
        """Test CodeData with defaults."""
        data = CodeData()
        assert data.type == "code"
        assert data.language is None
        assert data.purpose is None
        assert data.file_path is None
        assert data.line_start is None
        assert data.line_end is None
        assert data.dependencies == []

    def test_code_data_all_fields(self):
        """Test CodeData with all fields."""
        data = CodeData(
            language="python",
            purpose="snippet",
            file_path="src/app.py",
            line_start=10,
            line_end=25,
            dependencies=["pydantic", "fastapi"],
        )
        assert data.language == "python"
        assert data.purpose == "snippet"
        assert data.file_path == "src/app.py"
        assert data.line_start == 10
        assert data.line_end == 25
        assert len(data.dependencies) == 2


class TestCommitData:
    """Test CommitData structured data model."""

    def test_commit_data_defaults(self):
        """Test CommitData with defaults."""
        data = CommitData()
        assert data.type == "commit"
        assert data.sha is None
        assert data.author is None
        assert data.message is None
        assert data.files_changed == []
        assert data.insertions is None
        assert data.deletions is None
        assert data.branch is None

    def test_commit_data_all_fields(self):
        """Test CommitData with all fields."""
        data = CommitData(
            sha="abc123def456",
            author="developer@example.com",
            message="Add new feature",
            files_changed=["src/app.py", "tests/test_app.py"],
            insertions=50,
            deletions=10,
            branch="feature/new-feature",
        )
        assert data.sha == "abc123def456"
        assert data.author == "developer@example.com"
        assert data.message == "Add new feature"
        assert len(data.files_changed) == 2
        assert data.insertions == 50
        assert data.deletions == 10
        assert data.branch == "feature/new-feature"


class TestDocData:
    """Test DocData structured data model."""

    def test_doc_data_defaults(self):
        """Test DocData with defaults."""
        data = DocData()
        assert data.type == "doc"
        assert data.doc_type is None
        assert data.url is None
        assert data.version is None
        assert data.last_updated is None
        assert data.format is None

    def test_doc_data_all_fields(self):
        """Test DocData with all fields."""
        data = DocData(
            doc_type="api",
            url="https://docs.example.com/api",
            version="2.0.0",
            last_updated="2024-01-15T10:00:00Z",
            format="markdown",
        )
        assert data.doc_type == "api"
        assert data.url == "https://docs.example.com/api"
        assert data.version == "2.0.0"
        assert data.last_updated == "2024-01-15T10:00:00Z"
        assert data.format == "markdown"


class TestPhase0TypesInStructuredDataClasses:
    """Test that Phase 0 types are in STRUCTURED_DATA_CLASSES."""

    def test_fact_type_registered(self):
        """Test fact type is registered."""
        assert "fact" in STRUCTURED_DATA_CLASSES
        assert STRUCTURED_DATA_CLASSES["fact"] == FactData
        assert issubclass(FactData, BaseStructuredData)

    def test_episodic_type_registered(self):
        """Test episodic type is registered."""
        assert "episodic" in STRUCTURED_DATA_CLASSES
        assert STRUCTURED_DATA_CLASSES["episodic"] == EpisodicData
        assert issubclass(EpisodicData, BaseStructuredData)

    def test_user_type_registered(self):
        """Test user type is registered."""
        assert "user" in STRUCTURED_DATA_CLASSES
        assert STRUCTURED_DATA_CLASSES["user"] == UserData
        assert issubclass(UserData, BaseStructuredData)

    def test_code_type_registered(self):
        """Test code type is registered."""
        assert "code" in STRUCTURED_DATA_CLASSES
        assert STRUCTURED_DATA_CLASSES["code"] == CodeData
        assert issubclass(CodeData, BaseStructuredData)

    def test_commit_type_registered(self):
        """Test commit type is registered."""
        assert "commit" in STRUCTURED_DATA_CLASSES
        assert STRUCTURED_DATA_CLASSES["commit"] == CommitData
        assert issubclass(CommitData, BaseStructuredData)

    def test_doc_type_registered(self):
        """Test doc type is registered."""
        assert "doc" in STRUCTURED_DATA_CLASSES
        assert STRUCTURED_DATA_CLASSES["doc"] == DocData
        assert issubclass(DocData, BaseStructuredData)


class TestPhase0TypesParsing:
    """Test parsing Phase 0 types."""

    def test_parse_fact_data(self):
        """Test parsing fact data."""
        raw = {"category": "rule", "confidence": 0.9}
        result = parse_structured_data("fact", raw)
        assert isinstance(result, FactData)
        assert result.category == "rule"
        assert result.confidence == 0.9

    def test_parse_episodic_data(self):
        """Test parsing episodic data."""
        raw = {"session_type": "debug", "participants": ["user"]}
        result = parse_structured_data("episodic", raw)
        assert isinstance(result, EpisodicData)
        assert result.session_type == "debug"

    def test_parse_user_data(self):
        """Test parsing user data."""
        raw = {"preference_key": "theme", "preference_value": "dark"}
        result = parse_structured_data("user", raw)
        assert isinstance(result, UserData)
        assert result.preference_key == "theme"

    def test_parse_code_data(self):
        """Test parsing code data."""
        raw = {"language": "python", "file_path": "app.py"}
        result = parse_structured_data("code", raw)
        assert isinstance(result, CodeData)
        assert result.language == "python"

    def test_parse_commit_data(self):
        """Test parsing commit data."""
        raw = {"sha": "abc123", "message": "Fix bug"}
        result = parse_structured_data("commit", raw)
        assert isinstance(result, CommitData)
        assert result.sha == "abc123"

    def test_parse_doc_data(self):
        """Test parsing doc data."""
        raw = {"doc_type": "readme", "format": "markdown"}
        result = parse_structured_data("doc", raw)
        assert isinstance(result, DocData)
        assert result.doc_type == "readme"


class TestPhase0MemoryFactoryMethods:
    """Test Memory factory methods for Phase 0 types."""

    def test_fact_factory(self):
        """Test Memory.fact() factory method."""
        memory = Memory.fact(
            content="The default port is 8080",
            category="configuration",
            source="documentation",
            confidence=0.95,
        )
        assert memory.type == MemoryType.FACT
        assert memory.structured_data["category"] == "configuration"
        assert memory.structured_data["source"] == "documentation"
        assert memory.structured_data["confidence"] == 0.95

    def test_episodic_factory(self):
        """Test Memory.episodic() factory method."""
        memory = Memory.episodic(
            content="Debug session for authentication issue",
            session_type="debug",
            participants=["user", "claude"],
            outcome="resolved",
            tool="claude-code",
        )
        assert memory.type == MemoryType.EPISODIC
        assert memory.structured_data["session_type"] == "debug"
        assert len(memory.structured_data["participants"]) == 2

    def test_user_factory(self):
        """Test Memory.user() factory method."""
        memory = Memory.user(
            content="User prefers dark theme",
            preference_key="theme",
            preference_value="dark",
            scope="global",
        )
        assert memory.type == MemoryType.USER
        assert memory.structured_data["preference_key"] == "theme"
        assert memory.structured_data["scope"] == "global"

    def test_code_factory(self):
        """Test Memory.code() factory method."""
        memory = Memory.code(
            content="def hello(): print('Hello')",
            language="python",
            purpose="snippet",
            file_path="app.py",
        )
        assert memory.type == MemoryType.CODE
        assert memory.structured_data["language"] == "python"
        assert memory.structured_data["purpose"] == "snippet"
        assert memory.structured_data["file_path"] == "app.py"

    def test_commit_factory(self):
        """Test Memory.commit() factory method."""
        memory = Memory.commit(
            content="Added new feature",
            sha="abc123def456",
            author="dev@example.com",
            message="Add feature X",
            files_changed=["app.py"],
        )
        assert memory.type == MemoryType.COMMIT
        assert memory.structured_data["sha"] == "abc123def456"
        assert memory.structured_data["author"] == "dev@example.com"
        assert memory.structured_data["message"] == "Add feature X"

    def test_doc_factory(self):
        """Test Memory.doc() factory method."""
        memory = Memory.doc(
            content="API documentation",
            doc_type="api",
            url="https://docs.example.com",
        )
        assert memory.type == MemoryType.DOC
        assert memory.structured_data["doc_type"] == "api"
        assert memory.structured_data["url"] == "https://docs.example.com"


class TestAllTypesHaveStructuredSchemas:
    """Verify all 22 memory types have structured schemas."""

    def test_total_type_count(self):
        """Test that we have 22 structured data classes."""
        assert len(STRUCTURED_DATA_CLASSES) == 22

    def test_all_memory_types_have_schema(self):
        """Test that all MemoryType enum values have a schema."""
        for mem_type in MemoryType:
            assert mem_type.value in STRUCTURED_DATA_CLASSES, (
                f"Missing schema for type: {mem_type.value}"
            )
