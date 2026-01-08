"""
ContextFS Workflow Engine.

Provides memory-native workflow orchestration with:
- Sequential and parallel task execution
- Dependency resolution
- Retry logic with exponential backoff
- Full state persistence to ContextFS memories

Example:
    from contextfs import ContextFS
    from contextfs.agents import Agent, AnthropicProvider
    from contextfs.workflows import Workflow, Task

    ctx = ContextFS()
    provider = AnthropicProvider()

    analyst = Agent(name="analyst", provider=provider, ctx=ctx)

    workflow = Workflow(name="analysis-pipeline", ctx=ctx)
    workflow.add_task(Task(name="fetch", handler=fetch_data))
    workflow.add_task(Task(name="analyze", agent=analyst, depends_on=["fetch"]))

    results = await workflow.run({"source": "database"})
"""

from contextfs.workflows.engine import Task, TaskStatus, Workflow
from contextfs.workflows.executor import WorkflowExecutor

__all__ = [
    "Workflow",
    "Task",
    "TaskStatus",
    "WorkflowExecutor",
]
