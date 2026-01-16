"""
Main orchestrator for coordinating multiple AI agents.
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from contextfs.core import ContextFS
from contextfs.schemas import MemoryType

from .agent import Agent, AgentRole, AgentStatus
from .monitor import MemoryMonitor


@dataclass
class Orchestrator:
    """
    Coordinate multiple AI agents with shared ContextFS memory.

    The orchestrator spawns agents in different directories, monitors their
    progress via shared memory, and can coordinate their work.

    Example:
        orch = Orchestrator(project="my-feature")

        # Spawn agents
        test_agent = orch.spawn_agent(
            role=AgentRole.TEST,
            directory="/project/tests",
            task="Write tests for auth"
        )
        code_agent = orch.spawn_agent(
            role=AgentRole.CODE,
            directory="/project/src",
            task="Implement auth"
        )

        # Wait for completion
        orch.wait_all()

        # Check results via memory
        findings = orch.get_findings()
    """

    project: str = "default"
    tool: str = "claude"
    ctx: ContextFS | None = field(default=None, repr=False)
    _agents: list[Agent] = field(default_factory=list, repr=False)
    _monitor: MemoryMonitor | None = field(default=None, repr=False)
    _state_file: Path | None = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize orchestrator."""
        if self.ctx is None:
            self.ctx = ContextFS()
        self._monitor = MemoryMonitor(project=self.project, ctx=self.ctx)

        # State persistence
        state_dir = Path(os.path.expanduser("~/.contextfs/orchestrator"))
        state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = state_dir / f"orchestrator-{self.project}.json"

    @property
    def agents(self) -> list[Agent]:
        """Get all agents."""
        return self._agents

    @property
    def running_agents(self) -> list[Agent]:
        """Get currently running agents."""
        return [a for a in self._agents if a.is_running]

    @property
    def completed_agents(self) -> list[Agent]:
        """Get completed agents."""
        return [a for a in self._agents if a.status == AgentStatus.COMPLETED]

    @property
    def failed_agents(self) -> list[Agent]:
        """Get failed agents."""
        return [a for a in self._agents if a.status == AgentStatus.FAILED]

    def spawn_agent(
        self,
        role: AgentRole | str,
        directory: str,
        task: str,
        background: bool = True,
    ) -> Agent:
        """
        Spawn a new agent.

        Args:
            role: Agent role (determines behavior)
            directory: Working directory for the agent
            task: The task to perform
            background: Run in background (non-blocking)

        Returns:
            The spawned agent
        """
        if isinstance(role, str):
            role = AgentRole(role)

        agent = Agent(
            role=role,
            directory=directory,
            task=task,
            project=self.project,
        )

        # Log spawn to memory
        if self.ctx is not None:
            self.ctx.save(
                content=f"Spawned {role.value} agent in {directory}\nTask: {task}",
                type=MemoryType.TASK,
                summary=f"Spawned {role.value} agent: {task[:50]}...",
                tags=["orchestrator", f"{role.value}-agent", "spawn"],
            )

        agent.spawn(tool=self.tool, background=background)
        self._agents.append(agent)
        self._save_state()

        return agent

    def spawn_fleet(
        self,
        agents: list[dict],
        background: bool = True,
    ) -> list[Agent]:
        """
        Spawn multiple agents at once.

        Args:
            agents: List of agent configs with role, directory, task
            background: Run in background

        Returns:
            List of spawned agents
        """
        spawned = []
        for config in agents:
            agent = self.spawn_agent(
                role=config["role"],
                directory=config["directory"],
                task=config["task"],
                background=background,
            )
            spawned.append(agent)
        return spawned

    def wait_all(self, timeout: float | None = None) -> bool:
        """
        Wait for all agents to complete.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            True if all completed successfully
        """
        start = time.time()

        while self.running_agents:
            if timeout and (time.time() - start) > timeout:
                return False
            time.sleep(1)
            self._save_state()

        return all(a.status == AgentStatus.COMPLETED for a in self._agents)

    def wait_any(self, timeout: float | None = None) -> Agent | None:
        """
        Wait for any agent to complete.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            First completed agent, or None if timeout
        """
        start = time.time()

        while True:
            for agent in self._agents:
                if agent.is_done:
                    return agent

            if timeout and (time.time() - start) > timeout:
                return None

            time.sleep(0.5)

    def cancel_all(self) -> None:
        """Cancel all running agents."""
        for agent in self.running_agents:
            agent.cancel()
        self._save_state()

    def poll_all(self) -> dict[str, AgentStatus]:
        """
        Poll status of all agents.

        Returns:
            Dict mapping agent ID to status
        """
        return {agent.id: agent.poll() for agent in self._agents}

    def get_findings(self, agent_role: str | None = None) -> list[dict]:
        """
        Get findings from agents via memory.

        Args:
            agent_role: Filter by role

        Returns:
            List of memory findings
        """
        if self._monitor is None:
            return []
        return self._monitor.get_agent_findings(agent_role)

    def get_errors(self) -> list[dict]:
        """Get all errors reported by agents."""
        if self._monitor is None:
            return []
        return self._monitor.get_errors()

    def get_decisions(self) -> list[dict]:
        """Get all decisions made by agents."""
        if self._monitor is None:
            return []
        return self._monitor.get_decisions()

    def monitor(
        self,
        interval: float = 5.0,
        duration: float | None = None,
    ) -> None:
        """
        Monitor agent activity in real-time.

        Args:
            interval: Seconds between checks
            duration: Total seconds to monitor
        """
        if self._monitor is None:
            return
        self._monitor.watch(interval=interval, duration=duration)

    def status(self) -> dict:
        """
        Get orchestrator status summary.

        Returns:
            Status dictionary
        """
        return {
            "project": self.project,
            "total_agents": len(self._agents),
            "running": len(self.running_agents),
            "completed": len(self.completed_agents),
            "failed": len(self.failed_agents),
            "agents": [a.to_dict() for a in self._agents],
        }

    def print_status(self) -> None:
        """Print status to console."""
        status = self.status()
        print(f"\n{'='*60}")
        print(f"Orchestrator: {self.project}")
        print(f"{'='*60}")
        print(
            f"Total: {status['total_agents']} | Running: {status['running']} | "
            f"Completed: {status['completed']} | Failed: {status['failed']}"
        )
        print(f"{'-'*60}")

        for agent in self._agents:
            status_icon = {
                AgentStatus.PENDING: "⏳",
                AgentStatus.STARTING: "🚀",
                AgentStatus.RUNNING: "🔄",
                AgentStatus.COMPLETED: "✅",
                AgentStatus.FAILED: "❌",
                AgentStatus.CANCELLED: "🚫",
            }.get(agent.status, "?")

            duration = f" ({agent.duration:.1f}s)" if agent.duration else ""
            print(f"{status_icon} [{agent.id}] {agent.role.value}: {agent.status.value}{duration}")
            if agent.error:
                print(f"   Error: {agent.error}")

        print(f"{'='*60}\n")

    def _save_state(self) -> None:
        """Persist orchestrator state."""
        if self._state_file:
            state = {
                "project": self.project,
                "tool": self.tool,
                "agents": [a.to_dict() for a in self._agents],
                "updated_at": datetime.now().isoformat(),
            }
            self._state_file.write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, project: str) -> "Orchestrator":
        """
        Load orchestrator state from file.

        Args:
            project: Project name

        Returns:
            Orchestrator instance (agents not restored)
        """
        state_file = Path(
            os.path.expanduser(f"~/.contextfs/orchestrator/orchestrator-{project}.json")
        )
        if not state_file.exists():
            raise FileNotFoundError(f"No saved state for project: {project}")

        state = json.loads(state_file.read_text())
        return cls(project=state["project"], tool=state.get("tool", "claude"))


def run_fleet(
    agents: list[dict],
    project: str = "default",
    tool: str = "claude",
    wait: bool = True,
    monitor: bool = False,
) -> Orchestrator:
    """
    Convenience function to run a fleet of agents.

    Args:
        agents: List of agent configs with role, directory, task
        project: ContextFS project for grouping
        tool: CLI tool to use
        wait: Wait for completion
        monitor: Print activity in real-time

    Returns:
        Orchestrator instance

    Example:
        run_fleet([
            {"role": "test", "directory": "./tests", "task": "Write unit tests"},
            {"role": "code", "directory": "./src", "task": "Implement feature"},
        ])
    """
    orch = Orchestrator(project=project, tool=tool)
    orch.spawn_fleet(agents)

    if monitor:
        # Monitor in background while waiting
        import threading

        monitor_thread = threading.Thread(
            target=orch.monitor,
            kwargs={"duration": None if wait else 10},
            daemon=True,
        )
        monitor_thread.start()

    if wait:
        orch.wait_all()
        orch.print_status()

    return orch
