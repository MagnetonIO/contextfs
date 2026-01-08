"""
Workflow execution engine.

Handles the actual execution of workflows with:
- Dependency resolution and topological ordering
- Parallel execution of independent tasks
- Retry logic with exponential backoff
- Full state persistence to ContextFS memories
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from contextfs.schemas import Memory
from contextfs.storage_protocol import EdgeRelation
from contextfs.types.versioned import ChangeReason
from contextfs.workflows.engine import Task, TaskStatus, WorkflowStatus

if TYPE_CHECKING:
    from contextfs import ContextFS
    from contextfs.workflows.engine import Workflow


class WorkflowExecutor:
    """
    Executes workflows with dependency resolution and parallel execution.

    All execution state is persisted to ContextFS memories:
    - workflow: Overall workflow record
    - task: Individual task records linked to workflow

    Example:
        executor = WorkflowExecutor(workflow, ctx)
        results = await executor.execute({"input": "data"})
    """

    def __init__(self, workflow: Workflow, ctx: ContextFS):
        """
        Initialize executor.

        Args:
            workflow: Workflow definition to execute.
            ctx: ContextFS instance for memory persistence.
        """
        self.workflow = workflow
        self.ctx = ctx
        self._results: dict[str, Any] = {}
        self._task_status: dict[str, TaskStatus] = {}
        self._workflow_memory: Memory | None = None
        self._task_memories: dict[str, Memory] = {}

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the workflow.

        Args:
            input_data: Initial input data for tasks.

        Returns:
            Dict mapping task names to their results.

        Raises:
            Exception: If any task fails after all retries.
        """
        start_time = datetime.now(timezone.utc)

        # Create workflow memory
        self._workflow_memory = self.ctx.save(
            Memory.workflow(
                content=f"Workflow: {self.workflow.name}",
                name=self.workflow.name,
                status="active",
                description=self.workflow.description,
                summary=f"Executing workflow: {self.workflow.name}",
            )
        )

        # Initialize task statuses
        for task in self.workflow.tasks:
            self._task_status[task.name] = TaskStatus.PENDING

        try:
            # Execute tasks in dependency order
            await self._execute_tasks(input_data)

            # Update workflow to completed
            self._update_workflow_status(
                status=WorkflowStatus.COMPLETED,
                started_at=start_time.isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(),
            )

            return self._results

        except Exception as e:
            # Update workflow to failed
            self._update_workflow_status(
                status=WorkflowStatus.FAILED,
                error=str(e),
                started_at=start_time.isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            raise

    async def _execute_tasks(self, input_data: dict[str, Any]) -> None:
        """Execute all tasks respecting dependencies."""
        while True:
            # Find tasks ready to execute (dependencies satisfied)
            ready_tasks = self._get_ready_tasks()

            if not ready_tasks:
                # Check if all tasks are done
                pending = [
                    t
                    for t in self.workflow.tasks
                    if self._task_status[t.name] in (TaskStatus.PENDING, TaskStatus.RUNNING)
                ]
                if not pending:
                    break
                # Should not happen - means circular dep or bug
                raise RuntimeError("No ready tasks but workflow not complete")

            # Execute ready tasks in parallel
            await asyncio.gather(*[self._execute_task(task, input_data) for task in ready_tasks])

    def _get_ready_tasks(self) -> list[Task]:
        """Get tasks whose dependencies are all satisfied."""
        ready = []
        for task in self.workflow.tasks:
            if self._task_status[task.name] != TaskStatus.PENDING:
                continue

            # Check if all dependencies are completed
            deps_satisfied = all(
                self._task_status.get(dep) == TaskStatus.COMPLETED for dep in task.depends_on
            )

            if deps_satisfied:
                ready.append(task)

        return ready

    async def _execute_task(self, task: Task, input_data: dict[str, Any]) -> None:
        """Execute a single task with retries."""
        self._task_status[task.name] = TaskStatus.RUNNING
        start_time = datetime.now(timezone.utc)

        # Create task memory
        task_memory = self.ctx.save(
            Memory.task(
                content=f"Task: {task.name}",
                name=task.name,
                workflow_id=self._workflow_memory.id if self._workflow_memory else None,
                status="running",
                summary=f"Executing task: {task.name}",
            )
        )
        self._task_memories[task.name] = task_memory

        # Link task to workflow
        if self._workflow_memory:
            self.ctx._lineage.link(
                self._workflow_memory.id,
                task_memory.id,
                relation=EdgeRelation.PART_OF,
            )

        # Retry loop
        last_error: Exception | None = None
        for attempt in range(task.retries + 1):
            try:
                # Build task input (combine workflow input with dependency results)
                task_input = self._build_task_input(task, input_data)

                # Execute based on type
                if task.agent:
                    result = await self._execute_agent_task(task, task_input)
                else:
                    result = await self._execute_handler_task(task, task_input)

                # Success
                self._results[task.name] = result
                self._task_status[task.name] = TaskStatus.COMPLETED

                # Update task memory
                self._update_task_memory(
                    task.name,
                    status=TaskStatus.COMPLETED,
                    output_data=result,
                    started_at=start_time.isoformat(),
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )

                return

            except Exception as e:
                last_error = e
                if attempt < task.retries:
                    # Exponential backoff
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                    # Update retry count
                    self._update_task_memory(
                        task.name,
                        retries=attempt + 1,
                    )

        # All retries exhausted
        self._task_status[task.name] = TaskStatus.FAILED
        self._update_task_memory(
            task.name,
            status=TaskStatus.FAILED,
            error=str(last_error),
            started_at=start_time.isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

        raise last_error or RuntimeError(f"Task {task.name} failed")

    def _build_task_input(self, task: Task, input_data: dict[str, Any]) -> dict[str, Any]:
        """Build input for a task including dependency results."""
        result: dict[str, Any] = {"workflow_input": input_data}

        # Add results from dependencies
        for dep_name in task.depends_on:
            if dep_name in self._results:
                result[dep_name] = self._results[dep_name]

        return result

    async def _execute_agent_task(self, task: Task, task_input: dict[str, Any]) -> Any:
        """Execute a task using an LLM agent."""
        if not task.agent:
            raise ValueError(f"Task {task.name} has no agent")

        # Build prompt
        if task.prompt_template:
            prompt = task.prompt_template.format(**task_input)
        else:
            # Default prompt includes task input as context
            input_str = json.dumps(task_input, indent=2)
            prompt = f"Execute task '{task.name}' with the following input:\n\n{input_str}"

        return await task.agent.run(prompt, context=task_input)

    async def _execute_handler_task(self, task: Task, task_input: dict[str, Any]) -> Any:
        """Execute a task using a handler function."""
        if not task.handler:
            raise ValueError(f"Task {task.name} has no handler")

        # Check if handler is async
        if asyncio.iscoroutinefunction(task.handler):
            return await task.handler(task_input)
        else:
            # Run sync handler in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, task.handler, task_input)

    def _update_workflow_status(self, **updates: Any) -> None:
        """Update workflow memory with new status."""
        if not self._workflow_memory:
            return

        status = updates.get("status", WorkflowStatus.ACTIVE)
        content = f"Workflow ({status.value if isinstance(status, WorkflowStatus) else status}): {self.workflow.name}"
        if updates.get("error"):
            content += f" - Error: {updates['error']}"

        self.ctx._lineage.evolve(
            self._workflow_memory.id,
            new_content=content,
            reason=ChangeReason.OBSERVATION,
        )

    def _update_task_memory(self, task_name: str, **updates: Any) -> None:
        """Update task memory with new status/data."""
        task_memory = self._task_memories.get(task_name)
        if not task_memory:
            return

        status = updates.get("status", TaskStatus.RUNNING)
        status_str = status.value if isinstance(status, TaskStatus) else status
        content = f"Task ({status_str}): {task_name}"
        if updates.get("error"):
            content += f" - Error: {updates['error']}"

        self.ctx._lineage.evolve(
            task_memory.id,
            new_content=content,
            reason=ChangeReason.OBSERVATION,
        )
