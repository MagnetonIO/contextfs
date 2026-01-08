"""
Workflow definition classes.

Provides dataclass-based workflow definitions that can be executed
by the WorkflowExecutor with full memory persistence.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from contextfs import ContextFS
    from contextfs.agents import Agent


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class WorkflowStatus(str, Enum):
    """Workflow execution status."""

    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """
    Definition of a workflow task.

    A task can be executed by either:
    - An LLM agent (via the `agent` parameter)
    - A handler function (via the `handler` parameter)

    Tasks can have dependencies on other tasks, and will only
    execute after all dependencies have completed successfully.

    Example:
        # Agent-based task
        Task(name="analyze", agent=analyst_agent, depends_on=["fetch"])

        # Handler-based task
        Task(name="fetch", handler=fetch_data_fn)
    """

    name: str
    agent: Agent | None = None
    handler: Callable[..., Any] | Callable[..., Coroutine[Any, Any, Any]] | None = None
    depends_on: list[str] = field(default_factory=list)
    timeout: int = 300  # seconds
    retries: int = 3
    prompt_template: str | None = None  # Template for agent prompt

    def __post_init__(self) -> None:
        """Validate task configuration."""
        if self.agent is None and self.handler is None:
            raise ValueError(f"Task '{self.name}' must have either an agent or handler")
        if self.agent is not None and self.handler is not None:
            raise ValueError(f"Task '{self.name}' cannot have both agent and handler")


@dataclass
class Workflow:
    """
    Workflow definition with tasks.

    Workflows manage a collection of tasks with dependencies.
    Tasks can run sequentially or in parallel based on their
    dependency graph.

    Example:
        workflow = Workflow(name="etl-pipeline", ctx=ctx)
        workflow.add_task(Task(name="extract", handler=extract_fn))
        workflow.add_task(Task(name="transform", agent=transformer, depends_on=["extract"]))
        workflow.add_task(Task(name="load", handler=load_fn, depends_on=["transform"]))

        results = await workflow.run({"source": "s3://data"})
    """

    name: str
    ctx: ContextFS | None = None
    description: str | None = None
    tasks: list[Task] = field(default_factory=list)

    def add_task(self, task: Task) -> Workflow:
        """
        Add a task to the workflow.

        Args:
            task: Task to add.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If task name already exists.
        """
        if any(t.name == task.name for t in self.tasks):
            raise ValueError(f"Task '{task.name}' already exists in workflow")
        self.tasks.append(task)
        return self

    def parallel(self, *tasks: Task) -> Workflow:
        """
        Add multiple tasks to run in parallel.

        All tasks added this way should have the same dependencies
        (or no dependencies) to actually run in parallel.

        Args:
            *tasks: Tasks to add.

        Returns:
            Self for chaining.
        """
        for task in tasks:
            self.add_task(task)
        return self

    def get_task(self, name: str) -> Task | None:
        """Get a task by name."""
        for task in self.tasks:
            if task.name == name:
                return task
        return None

    def get_dependencies(self, task_name: str) -> list[Task]:
        """Get all tasks that the given task depends on."""
        task = self.get_task(task_name)
        if not task:
            return []
        return [t for t in self.tasks if t.name in task.depends_on]

    def get_dependents(self, task_name: str) -> list[Task]:
        """Get all tasks that depend on the given task."""
        return [t for t in self.tasks if task_name in t.depends_on]

    def validate(self) -> list[str]:
        """
        Validate the workflow configuration.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: list[str] = []

        # Check for missing dependencies
        task_names = {t.name for t in self.tasks}
        for task in self.tasks:
            for dep in task.depends_on:
                if dep not in task_names:
                    errors.append(f"Task '{task.name}' depends on unknown task '{dep}'")

        # Check for circular dependencies
        def has_cycle(task_name: str, visited: set[str], path: set[str]) -> bool:
            if task_name in path:
                return True
            if task_name in visited:
                return False

            visited.add(task_name)
            path.add(task_name)

            task = self.get_task(task_name)
            if task:
                for dep in task.depends_on:
                    if has_cycle(dep, visited, path):
                        return True

            path.remove(task_name)
            return False

        visited: set[str] = set()
        for task in self.tasks:
            if has_cycle(task.name, visited, set()):
                errors.append(f"Circular dependency detected involving task '{task.name}'")
                break

        return errors

    async def run(self, input_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Execute the workflow.

        Args:
            input_data: Initial input data for the workflow.

        Returns:
            Dict mapping task names to their results.

        Raises:
            ValueError: If workflow is invalid.
            Exception: If workflow execution fails.
        """
        from contextfs.workflows.executor import WorkflowExecutor

        # Validate first
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid workflow: {'; '.join(errors)}")

        if self.ctx is None:
            raise ValueError("Workflow requires a ContextFS instance")

        executor = WorkflowExecutor(self, self.ctx)
        return await executor.execute(input_data or {})
