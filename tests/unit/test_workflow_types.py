"""Unit tests for workflow memory types."""

import pytest

from contextfs.schemas import (
    STRUCTURED_DATA_CLASSES,
    AgentRunData,
    BaseStructuredData,
    Memory,
    MemoryType,
    StepData,
    TaskData,
    WorkflowData,
    parse_structured_data,
)


class TestWorkflowData:
    """Test WorkflowData structured data model."""

    def test_workflow_data_required_name(self):
        """Test WorkflowData requires name field."""
        data = WorkflowData(name="test-workflow")
        assert data.name == "test-workflow"
        assert data.type == "workflow"
        assert data.status == "draft"
        assert data.steps == []
        assert data.parallel_groups == []
        assert data.dependencies == {}

    def test_workflow_data_all_fields(self):
        """Test WorkflowData with all fields."""
        data = WorkflowData(
            name="etl-pipeline",
            status="active",
            description="Extract, transform, load data",
            steps=["extract", "transform", "load"],
            parallel_groups=[["validate", "backup"]],
            dependencies={"transform": ["extract"], "load": ["transform"]},
            created_by="user@example.com",
            started_at="2024-01-15T10:00:00Z",
            completed_at="2024-01-15T10:30:00Z",
        )
        assert data.name == "etl-pipeline"
        assert data.status == "active"
        assert len(data.steps) == 3
        assert len(data.dependencies) == 2
        assert data.dependencies["transform"] == ["extract"]

    def test_workflow_data_invalid_status(self):
        """Test WorkflowData rejects invalid status."""
        with pytest.raises(ValueError):
            WorkflowData(name="test", status="invalid_status")

    def test_workflow_data_valid_statuses(self):
        """Test all valid workflow statuses."""
        valid_statuses = ["draft", "active", "paused", "completed", "failed"]
        for status in valid_statuses:
            data = WorkflowData(name="test", status=status)
            assert data.status == status


class TestTaskData:
    """Test TaskData structured data model."""

    def test_task_data_required_name(self):
        """Test TaskData requires name field."""
        data = TaskData(name="fetch-data")
        assert data.name == "fetch-data"
        assert data.type == "task"
        assert data.status == "pending"
        assert data.retries == 0
        assert data.max_retries == 3

    def test_task_data_all_fields(self):
        """Test TaskData with all fields."""
        data = TaskData(
            name="analyze",
            workflow_id="wf-123",
            status="running",
            assigned_agent="analyst-agent",
            input_data={"source": "database"},
            output_data={"result": "success"},
            error=None,
            retries=1,
            max_retries=5,
            timeout_seconds=600,
            started_at="2024-01-15T10:00:00Z",
            completed_at="2024-01-15T10:05:00Z",
        )
        assert data.name == "analyze"
        assert data.workflow_id == "wf-123"
        assert data.status == "running"
        assert data.assigned_agent == "analyst-agent"
        assert data.input_data["source"] == "database"
        assert data.timeout_seconds == 600

    def test_task_data_invalid_status(self):
        """Test TaskData rejects invalid status."""
        with pytest.raises(ValueError):
            TaskData(name="test", status="invalid_status")

    def test_task_data_valid_statuses(self):
        """Test all valid task statuses."""
        valid_statuses = ["pending", "running", "completed", "failed", "skipped", "cancelled"]
        for status in valid_statuses:
            data = TaskData(name="test", status=status)
            assert data.status == status

    def test_task_data_with_error(self):
        """Test TaskData with error message."""
        data = TaskData(
            name="failed-task",
            status="failed",
            error="Connection timeout",
            retries=3,
        )
        assert data.error == "Connection timeout"
        assert data.retries == 3


class TestStepData:
    """Test StepData structured data model."""

    def test_step_data_required_task_id(self):
        """Test StepData requires task_id field."""
        data = StepData(task_id="task-123")
        assert data.task_id == "task-123"
        assert data.type == "step"
        assert data.step_number == 0
        assert data.input == {}
        assert data.output == {}

    def test_step_data_all_fields(self):
        """Test StepData with all fields."""
        data = StepData(
            task_id="task-123",
            step_number=2,
            action="tool_call",
            input={"tool": "search", "query": "test"},
            output={"results": ["item1", "item2"]},
            duration_ms=150,
            tokens_used=500,
            model="claude-sonnet-4-20250514",
            reason="observation",
            error=None,
        )
        assert data.task_id == "task-123"
        assert data.step_number == 2
        assert data.action == "tool_call"
        assert data.duration_ms == 150
        assert data.tokens_used == 500
        assert data.model == "claude-sonnet-4-20250514"

    def test_step_data_with_error(self):
        """Test StepData with error."""
        data = StepData(
            task_id="task-123",
            action="llm_response",
            error="Rate limit exceeded",
        )
        assert data.error == "Rate limit exceeded"


class TestAgentRunData:
    """Test AgentRunData structured data model."""

    def test_agent_run_required_name(self):
        """Test AgentRunData requires agent_name field."""
        data = AgentRunData(agent_name="data-analyst")
        assert data.agent_name == "data-analyst"
        assert data.type == "agent_run"
        assert data.status == "running"
        assert data.tool_calls == []

    def test_agent_run_all_fields(self):
        """Test AgentRunData with all fields."""
        data = AgentRunData(
            agent_name="code-reviewer",
            model="claude-opus-4-5-20251101",
            provider="anthropic",
            workflow_id="wf-123",
            task_id="task-456",
            prompt_tokens=1000,
            completion_tokens=500,
            tool_calls=[
                {"name": "read_file", "input": {"path": "app.py"}, "output": "content..."},
            ],
            status="completed",
            started_at="2024-01-15T10:00:00Z",
            completed_at="2024-01-15T10:01:00Z",
            error=None,
        )
        assert data.agent_name == "code-reviewer"
        assert data.model == "claude-opus-4-5-20251101"
        assert data.provider == "anthropic"
        assert data.prompt_tokens == 1000
        assert len(data.tool_calls) == 1
        assert data.tool_calls[0]["name"] == "read_file"

    def test_agent_run_invalid_status(self):
        """Test AgentRunData rejects invalid status."""
        with pytest.raises(ValueError):
            AgentRunData(agent_name="test", status="invalid_status")

    def test_agent_run_valid_statuses(self):
        """Test all valid agent run statuses."""
        valid_statuses = ["running", "completed", "failed", "timeout", "cancelled"]
        for status in valid_statuses:
            data = AgentRunData(agent_name="test", status=status)
            assert data.status == status


class TestWorkflowTypesInStructuredDataClasses:
    """Test that workflow types are in STRUCTURED_DATA_CLASSES."""

    def test_workflow_type_registered(self):
        """Test workflow type is registered."""
        assert "workflow" in STRUCTURED_DATA_CLASSES
        assert STRUCTURED_DATA_CLASSES["workflow"] == WorkflowData
        assert issubclass(WorkflowData, BaseStructuredData)

    def test_task_type_registered(self):
        """Test task type is registered."""
        assert "task" in STRUCTURED_DATA_CLASSES
        assert STRUCTURED_DATA_CLASSES["task"] == TaskData
        assert issubclass(TaskData, BaseStructuredData)

    def test_step_type_registered(self):
        """Test step type is registered."""
        assert "step" in STRUCTURED_DATA_CLASSES
        assert STRUCTURED_DATA_CLASSES["step"] == StepData
        assert issubclass(StepData, BaseStructuredData)

    def test_agent_run_type_registered(self):
        """Test agent_run type is registered."""
        assert "agent_run" in STRUCTURED_DATA_CLASSES
        assert STRUCTURED_DATA_CLASSES["agent_run"] == AgentRunData
        assert issubclass(AgentRunData, BaseStructuredData)


class TestWorkflowTypesParsing:
    """Test parsing workflow types."""

    def test_parse_workflow_data(self):
        """Test parsing workflow data into typed model."""
        raw = {"name": "test-workflow", "status": "active", "steps": ["a", "b"]}
        result = parse_structured_data("workflow", raw)
        assert isinstance(result, WorkflowData)
        assert result.name == "test-workflow"
        assert result.status == "active"
        assert result.steps == ["a", "b"]

    def test_parse_task_data(self):
        """Test parsing task data into typed model."""
        raw = {"name": "test-task", "status": "running", "retries": 2}
        result = parse_structured_data("task", raw)
        assert isinstance(result, TaskData)
        assert result.name == "test-task"
        assert result.retries == 2

    def test_parse_step_data(self):
        """Test parsing step data into typed model."""
        raw = {"task_id": "task-123", "step_number": 1, "action": "tool_call"}
        result = parse_structured_data("step", raw)
        assert isinstance(result, StepData)
        assert result.task_id == "task-123"
        assert result.action == "tool_call"

    def test_parse_agent_run_data(self):
        """Test parsing agent_run data into typed model."""
        raw = {"agent_name": "test-agent", "model": "gpt-4", "status": "completed"}
        result = parse_structured_data("agent_run", raw)
        assert isinstance(result, AgentRunData)
        assert result.agent_name == "test-agent"
        assert result.model == "gpt-4"


class TestWorkflowMemoryFactoryMethods:
    """Test Memory factory methods for workflow types."""

    def test_workflow_factory(self):
        """Test Memory.workflow() factory method."""
        memory = Memory.workflow(
            content="ETL Pipeline",
            name="etl-pipeline",
            status="active",
            description="Data processing workflow",
        )
        assert memory.type == MemoryType.WORKFLOW
        assert memory.structured_data["name"] == "etl-pipeline"
        assert memory.structured_data["status"] == "active"
        assert memory.structured_data["description"] == "Data processing workflow"

    def test_task_factory(self):
        """Test Memory.task() factory method."""
        memory = Memory.task(
            content="Data extraction task",
            name="extract-data",
            workflow_id="wf-123",
            status="pending",
            assigned_agent="extractor",
        )
        assert memory.type == MemoryType.TASK
        assert memory.structured_data["name"] == "extract-data"
        assert memory.structured_data["workflow_id"] == "wf-123"

    def test_step_factory(self):
        """Test Memory.step() factory method."""
        memory = Memory.step(
            content="Tool call step",
            task_id="task-123",
            step_number=1,
            action="tool_call",
            input={"tool": "search"},
            output={"results": []},
        )
        assert memory.type == MemoryType.STEP
        assert memory.structured_data["task_id"] == "task-123"
        assert memory.structured_data["action"] == "tool_call"

    def test_agent_run_factory(self):
        """Test Memory.agent_run() factory method."""
        memory = Memory.agent_run(
            content="Agent execution",
            agent_name="analyst",
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            status="completed",
        )
        assert memory.type == MemoryType.AGENT_RUN
        assert memory.structured_data["agent_name"] == "analyst"
        assert memory.structured_data["model"] == "claude-sonnet-4-20250514"


class TestMemoryTypedDataForWorkflowTypes:
    """Test Memory.typed_data property for workflow types."""

    def test_typed_data_returns_workflow_data(self):
        """Test typed_data returns WorkflowData."""
        memory = Memory(
            content="Workflow",
            type=MemoryType.WORKFLOW,
            structured_data={"name": "test", "status": "active"},
        )
        typed = memory.typed_data
        assert isinstance(typed, WorkflowData)
        assert typed.name == "test"

    def test_typed_data_returns_task_data(self):
        """Test typed_data returns TaskData."""
        memory = Memory(
            content="Task",
            type=MemoryType.TASK,
            structured_data={"name": "test-task", "status": "running"},
        )
        typed = memory.typed_data
        assert isinstance(typed, TaskData)
        assert typed.name == "test-task"

    def test_typed_data_returns_agent_run_data(self):
        """Test typed_data returns AgentRunData."""
        memory = Memory(
            content="Agent Run",
            type=MemoryType.AGENT_RUN,
            structured_data={"agent_name": "test-agent", "status": "completed"},
        )
        typed = memory.typed_data
        assert isinstance(typed, AgentRunData)
        assert typed.agent_name == "test-agent"
