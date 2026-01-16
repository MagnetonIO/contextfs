"""
ContextFS Agent Orchestrator

Spawn and coordinate multiple AI agents across directories with shared memory.

Example:
    from contextfs.orchestrator import Orchestrator, AgentRole

    orch = Orchestrator(project="my-project")

    # Spawn agents
    orch.spawn_agent(
        role=AgentRole.TEST,
        directory="/path/to/tests",
        task="Write unit tests for auth module"
    )
    orch.spawn_agent(
        role=AgentRole.CODE,
        directory="/path/to/src",
        task="Implement auth module"
    )

    # Monitor via shared memory
    orch.monitor()
"""

from .agent import Agent, AgentRole, AgentStatus
from .monitor import MemoryMonitor
from .orchestrator import Orchestrator, run_fleet
from .worker import (
    Job,
    JobStatus,
    WorkerPool,  # Legacy alias
    WorkerQueue,
    WorkerResult,
    run_agents,
    run_agents_sync,
)

__all__ = [
    "Orchestrator",
    "run_fleet",
    "Agent",
    "AgentRole",
    "AgentStatus",
    "MemoryMonitor",
    "WorkerQueue",
    "WorkerPool",
    "WorkerResult",
    "Job",
    "JobStatus",
    "run_agents",
    "run_agents_sync",
]
