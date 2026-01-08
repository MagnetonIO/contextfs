"""Unit tests for Agent framework."""

import pytest

from contextfs.agents.base import Agent, AgentProvider, Tool
from contextfs.workflows.engine import Task, TaskStatus, Workflow, WorkflowStatus


class TestTool:
    """Test Tool dataclass."""

    def test_tool_creation(self):
        """Test creating a Tool."""
        tool = Tool(
            name="search",
            description="Search the codebase",
            func=lambda x: x,
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        assert tool.name == "search"
        assert tool.description == "Search the codebase"
        assert "query" in tool.parameters["properties"]

    def test_tool_from_function(self):
        """Test creating Tool from function."""

        def my_search(query: str, limit: int = 10) -> list:
            """Search for items matching query."""
            return []

        tool = Tool.from_function(my_search)
        assert tool.name == "my_search"
        assert "Search for items" in tool.description
        assert tool.func == my_search

    def test_tool_from_function_with_name_override(self):
        """Test Tool.from_function with name override."""

        def helper(x: int) -> int:
            """Original description."""
            return x * 2

        tool = Tool.from_function(helper, name="double")
        assert tool.name == "double"
        # Description comes from docstring
        assert "Original description" in tool.description

    def test_tool_to_schema(self):
        """Test Tool.to_schema() for LLM API."""
        tool = Tool(
            name="read_file",
            description="Read a file",
            func=lambda x: x,
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        schema = tool.to_schema()
        assert schema["name"] == "read_file"
        assert schema["description"] == "Read a file"
        assert schema["input_schema"]["properties"]["path"]["type"] == "string"


class TestAgentProvider:
    """Test AgentProvider abstract base class."""

    def test_provider_is_abstract(self):
        """Test that AgentProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AgentProvider()

    def test_provider_subclass_must_implement_complete(self):
        """Test that subclasses must implement complete()."""

        class IncompleteProvider(AgentProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()

    def test_provider_subclass_works(self):
        """Test that proper subclass works."""

        class MockProvider(AgentProvider):
            async def complete(self, messages, tools=None, **kwargs):
                return {"content": "mock response", "tool_calls": None}

        provider = MockProvider()
        assert hasattr(provider, "complete")


class TestTaskStatus:
    """Test TaskStatus enum."""

    def test_task_status_values(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.SKIPPED.value == "skipped"
        assert TaskStatus.CANCELLED.value == "cancelled"

    def test_task_status_is_str(self):
        """Test TaskStatus inherits from str."""
        assert isinstance(TaskStatus.PENDING, str)
        assert TaskStatus.PENDING == "pending"


class TestWorkflowStatus:
    """Test WorkflowStatus enum."""

    def test_workflow_status_values(self):
        """Test WorkflowStatus enum values."""
        assert WorkflowStatus.DRAFT.value == "draft"
        assert WorkflowStatus.ACTIVE.value == "active"
        assert WorkflowStatus.PAUSED.value == "paused"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"


class TestTask:
    """Test Task dataclass."""

    def test_task_with_handler(self):
        """Test creating task with handler."""

        def my_handler(data):
            return data

        task = Task(name="process", handler=my_handler)
        assert task.name == "process"
        assert task.handler == my_handler
        assert task.agent is None
        assert task.depends_on == []
        assert task.timeout == 300
        assert task.retries == 3

    def test_task_requires_agent_or_handler(self):
        """Test task requires either agent or handler."""
        with pytest.raises(ValueError) as exc_info:
            Task(name="invalid")
        assert "must have either an agent or handler" in str(exc_info.value)

    def test_task_cannot_have_both(self):
        """Test task cannot have both agent and handler."""

        class MockProvider(AgentProvider):
            async def complete(self, messages, tools=None, **kwargs):
                return {"content": "mock", "tool_calls": None}

        # Create a mock agent (need to avoid ContextFS dependency)
        mock_agent = object()  # Placeholder

        with pytest.raises(ValueError) as exc_info:
            Task(name="invalid", agent=mock_agent, handler=lambda x: x)
        assert "cannot have both agent and handler" in str(exc_info.value)

    def test_task_with_dependencies(self):
        """Test task with dependencies."""
        task = Task(
            name="transform",
            handler=lambda x: x,
            depends_on=["extract", "validate"],
        )
        assert task.depends_on == ["extract", "validate"]

    def test_task_with_custom_timeout_retries(self):
        """Test task with custom timeout and retries."""
        task = Task(
            name="slow-task",
            handler=lambda x: x,
            timeout=600,
            retries=5,
        )
        assert task.timeout == 600
        assert task.retries == 5


class TestWorkflow:
    """Test Workflow dataclass."""

    def test_workflow_creation(self):
        """Test creating a workflow."""
        workflow = Workflow(name="test-workflow")
        assert workflow.name == "test-workflow"
        assert workflow.tasks == []
        assert workflow.ctx is None

    def test_workflow_with_description(self):
        """Test workflow with description."""
        workflow = Workflow(
            name="etl-pipeline",
            description="Extract, transform, load data",
        )
        assert workflow.description == "Extract, transform, load data"

    def test_add_task(self):
        """Test adding tasks to workflow."""
        workflow = Workflow(name="test")

        task1 = Task(name="task1", handler=lambda x: x)
        task2 = Task(name="task2", handler=lambda x: x)

        workflow.add_task(task1).add_task(task2)

        assert len(workflow.tasks) == 2
        assert workflow.tasks[0].name == "task1"
        assert workflow.tasks[1].name == "task2"

    def test_add_duplicate_task_fails(self):
        """Test adding duplicate task name fails."""
        workflow = Workflow(name="test")
        task1 = Task(name="same-name", handler=lambda x: x)
        task2 = Task(name="same-name", handler=lambda x: x)

        workflow.add_task(task1)

        with pytest.raises(ValueError) as exc_info:
            workflow.add_task(task2)
        assert "already exists" in str(exc_info.value)

    def test_parallel_tasks(self):
        """Test adding parallel tasks."""
        workflow = Workflow(name="test")

        task1 = Task(name="parallel1", handler=lambda x: x)
        task2 = Task(name="parallel2", handler=lambda x: x)
        task3 = Task(name="parallel3", handler=lambda x: x)

        workflow.parallel(task1, task2, task3)

        assert len(workflow.tasks) == 3

    def test_get_task(self):
        """Test getting task by name."""
        workflow = Workflow(name="test")
        task = Task(name="target", handler=lambda x: x)
        workflow.add_task(task)

        found = workflow.get_task("target")
        assert found is task

        not_found = workflow.get_task("nonexistent")
        assert not_found is None

    def test_get_dependencies(self):
        """Test getting task dependencies."""
        workflow = Workflow(name="test")

        task1 = Task(name="base", handler=lambda x: x)
        task2 = Task(name="dependent", handler=lambda x: x, depends_on=["base"])

        workflow.add_task(task1).add_task(task2)

        deps = workflow.get_dependencies("dependent")
        assert len(deps) == 1
        assert deps[0].name == "base"

    def test_get_dependents(self):
        """Test getting tasks that depend on a task."""
        workflow = Workflow(name="test")

        task1 = Task(name="base", handler=lambda x: x)
        task2 = Task(name="child1", handler=lambda x: x, depends_on=["base"])
        task3 = Task(name="child2", handler=lambda x: x, depends_on=["base"])

        workflow.add_task(task1).add_task(task2).add_task(task3)

        dependents = workflow.get_dependents("base")
        assert len(dependents) == 2
        names = [t.name for t in dependents]
        assert "child1" in names
        assert "child2" in names

    def test_validate_success(self):
        """Test workflow validation passes for valid workflow."""
        workflow = Workflow(name="test")

        task1 = Task(name="first", handler=lambda x: x)
        task2 = Task(name="second", handler=lambda x: x, depends_on=["first"])

        workflow.add_task(task1).add_task(task2)

        errors = workflow.validate()
        assert errors == []

    def test_validate_missing_dependency(self):
        """Test validation catches missing dependencies."""
        workflow = Workflow(name="test")

        task = Task(name="orphan", handler=lambda x: x, depends_on=["nonexistent"])
        workflow.add_task(task)

        errors = workflow.validate()
        assert len(errors) == 1
        assert "unknown task 'nonexistent'" in errors[0]

    def test_validate_circular_dependency(self):
        """Test validation catches circular dependencies."""
        workflow = Workflow(name="test")

        task1 = Task(name="a", handler=lambda x: x, depends_on=["c"])
        task2 = Task(name="b", handler=lambda x: x, depends_on=["a"])
        task3 = Task(name="c", handler=lambda x: x, depends_on=["b"])

        workflow.add_task(task1).add_task(task2).add_task(task3)

        errors = workflow.validate()
        assert len(errors) >= 1
        assert any("Circular dependency" in e for e in errors)


class TestAgentInit:
    """Test Agent initialization (without execution)."""

    def test_agent_requires_provider(self):
        """Test Agent requires a provider."""

        class MockProvider(AgentProvider):
            async def complete(self, messages, tools=None, **kwargs):
                return {"content": "mock", "tool_calls": None}

        # Verify MockProvider can be instantiated
        _provider = MockProvider()
        assert _provider is not None

        # Agent requires ctx which we don't have in unit tests
        # This test just validates the class structure exists
        assert hasattr(Agent, "__init__")
        assert hasattr(Agent, "run")
        assert hasattr(Agent, "_execute")
