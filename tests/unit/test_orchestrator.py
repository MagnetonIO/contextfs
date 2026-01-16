"""Unit tests for Agent Orchestrator."""

from unittest.mock import MagicMock, patch

import pytest

from contextfs.orchestrator import (
    AgentRole,
    AgentStatus,
    Job,
    JobStatus,
    WorkerResult,
)
from contextfs.schemas import MemoryType


class TestAgentRole:
    """Test AgentRole enum."""

    def test_role_values(self):
        """Test AgentRole enum values."""
        assert AgentRole.TEST.value == "test"
        assert AgentRole.CODE.value == "code"
        assert AgentRole.REVIEW.value == "review"
        assert AgentRole.DOCS.value == "docs"
        assert AgentRole.DEBUG.value == "debug"
        assert AgentRole.CUSTOM.value == "custom"
        assert AgentRole.ORCHESTRATOR.value == "orchestrator"

    def test_role_from_string(self):
        """Test creating role from string."""
        role = AgentRole("test")
        assert role == AgentRole.TEST

    def test_invalid_role_raises(self):
        """Test invalid role raises ValueError."""
        with pytest.raises(ValueError):
            AgentRole("invalid_role")


class TestAgentStatus:
    """Test AgentStatus enum."""

    def test_status_values(self):
        """Test AgentStatus enum values."""
        assert AgentStatus.PENDING.value == "pending"
        assert AgentStatus.STARTING.value == "starting"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.COMPLETED.value == "completed"
        assert AgentStatus.FAILED.value == "failed"
        assert AgentStatus.CANCELLED.value == "cancelled"


class TestJobStatus:
    """Test JobStatus enum."""

    def test_status_values(self):
        """Test JobStatus enum values."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.TIMEOUT.value == "timeout"
        assert JobStatus.CANCELLED.value == "cancelled"


class TestJob:
    """Test Job dataclass."""

    def test_job_creation(self):
        """Test creating a job."""
        job = Job(
            id="abc12345",
            role=AgentRole.TEST,
            directory="/path/to/tests",
            task="Write unit tests",
        )
        assert job.role == AgentRole.TEST
        assert job.directory == "/path/to/tests"
        assert job.task == "Write unit tests"
        assert job.status == JobStatus.PENDING
        assert job.id == "abc12345"

    def test_job_with_custom_timeout(self):
        """Test job with custom timeout."""
        job = Job(
            id="job12345",
            role=AgentRole.CODE,
            directory="/src",
            task="Implement feature",
            timeout=300.0,
        )
        assert job.timeout == 300.0

    def test_job_duration(self):
        """Test job duration calculation."""
        from datetime import datetime, timedelta

        job = Job(
            id="test123",
            role=AgentRole.TEST,
            directory="/tests",
            task="Run tests",
        )
        job.started_at = datetime.now() - timedelta(seconds=10)
        job.completed_at = datetime.now()

        assert job.duration is not None
        assert job.duration >= 9.0  # Allow for some timing variance


class TestWorkerResult:
    """Test WorkerResult dataclass."""

    def test_result_creation(self):
        """Test creating a worker result."""
        result = WorkerResult(
            job_id="abc12345",
            agent_id="agent-1",
            role="test",
            status=JobStatus.COMPLETED,
            result="Tests passed",
            result_data={"tests_run": 10, "passed": 10},
            duration=10.5,
        )
        assert result.job_id == "abc12345"
        assert result.agent_id == "agent-1"
        assert result.role == "test"
        assert result.status == JobStatus.COMPLETED
        assert result.duration == 10.5
        assert result.result == "Tests passed"
        assert result.result_data == {"tests_run": 10, "passed": 10}
        assert result.error is None

    def test_result_with_error(self):
        """Test result with error."""
        result = WorkerResult(
            job_id="abc12345",
            agent_id="agent-1",
            role="test",
            status=JobStatus.FAILED,
            result=None,
            result_data=None,
            duration=5.0,
            error="Test failed: assertion error",
        )
        assert result.status == JobStatus.FAILED
        assert result.error == "Test failed: assertion error"
        assert result.result is None


class TestOrchestratorTypeEnforcement:
    """Test that orchestrator uses proper types."""

    def test_orchestrator_uses_memory_type(self):
        """Test Orchestrator uses MemoryType enum, not strings."""

        # Verify MemoryType is imported in orchestrator
        import contextfs.orchestrator.orchestrator as orch_module

        assert hasattr(orch_module, "MemoryType")

        # Verify the type is the enum class
        assert orch_module.MemoryType is MemoryType

    def test_memory_type_task_exists(self):
        """Test MemoryType.TASK exists and has correct value."""
        assert hasattr(MemoryType, "TASK")
        assert MemoryType.TASK.value == "task"


class TestMemoryMonitor:
    """Test MemoryMonitor class."""

    def test_monitor_creation(self):
        """Test creating a memory monitor."""
        from contextfs.orchestrator import MemoryMonitor

        # Mock ContextFS to avoid database dependency
        with patch("contextfs.orchestrator.monitor.ContextFS") as MockCtx:
            mock_ctx = MagicMock()
            MockCtx.return_value = mock_ctx

            monitor = MemoryMonitor(project="test-project")
            assert monitor.project == "test-project"

    def test_monitor_callbacks(self):
        """Test registering activity callbacks."""
        from contextfs.orchestrator import MemoryMonitor

        with patch("contextfs.orchestrator.monitor.ContextFS") as MockCtx:
            mock_ctx = MagicMock()
            MockCtx.return_value = mock_ctx

            monitor = MemoryMonitor(project="test")
            callback = MagicMock()

            monitor.on_activity(callback)
            assert callback in monitor._callbacks


class TestAgentPrompts:
    """Test agent role prompts."""

    def test_role_prompts_exist(self):
        """Test all roles have prompts defined."""
        from contextfs.orchestrator.agent import ROLE_PROMPTS

        for role in AgentRole:
            assert role.value in ROLE_PROMPTS, f"Missing prompt for {role.value}"

    def test_prompts_contain_memory_instructions(self):
        """Test prompts include memory operation instructions."""
        from contextfs.orchestrator.agent import ROLE_PROMPTS

        # Test agent should save errors
        assert "error" in ROLE_PROMPTS["test"].lower()

        # Code agent should save decisions
        assert "decision" in ROLE_PROMPTS["code"].lower()

        # Review agent should link findings
        assert "link" in ROLE_PROMPTS["review"].lower()


class TestAgentConfig:
    """Test Agent configuration."""

    def test_agent_creation(self, tmp_path):
        """Test creating an agent."""
        from contextfs.orchestrator import Agent

        agent = Agent(
            role=AgentRole.TEST,
            directory=str(tmp_path),
            task="Write tests",
            project="test-project",
        )

        assert agent.role == AgentRole.TEST
        assert agent.directory == str(tmp_path)
        assert agent.task == "Write tests"
        assert agent.project == "test-project"
        assert agent.status == AgentStatus.PENDING
        assert agent.id is not None
        assert len(agent.id) == 8

    def test_agent_to_dict(self, tmp_path):
        """Test agent serialization."""
        from contextfs.orchestrator import Agent

        agent = Agent(
            role=AgentRole.CODE,
            directory=str(tmp_path),
            task="Implement feature",
            project="my-project",
        )

        data = agent.to_dict()
        assert data["role"] == "code"
        assert data["directory"] == str(tmp_path)
        assert data["task"] == "Implement feature"
        assert data["project"] == "my-project"
        assert data["status"] == "pending"

    def test_agent_invalid_directory_raises(self):
        """Test creating agent with invalid directory raises ValueError."""
        from contextfs.orchestrator import Agent

        with pytest.raises(ValueError) as exc_info:
            Agent(
                role=AgentRole.TEST,
                directory="/nonexistent/path/that/does/not/exist",
                task="Test",
            )
        assert "Directory does not exist" in str(exc_info.value)

    def test_agent_system_prompt(self, tmp_path):
        """Test agent has proper system prompt."""
        from contextfs.orchestrator import Agent

        agent = Agent(
            role=AgentRole.TEST,
            directory=str(tmp_path),
            task="Write tests",
            project="test-proj",
        )

        prompt = agent.system_prompt
        assert "test-proj" in prompt.lower()
        assert agent.id in prompt
