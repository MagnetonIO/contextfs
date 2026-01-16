"""
Agent definitions and spawning logic.
"""

import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class AgentRole(str, Enum):
    """Predefined agent roles with memory protocols."""

    ORCHESTRATOR = "orchestrator"
    TEST = "test"
    CODE = "code"
    REVIEW = "review"
    DOCS = "docs"
    DEBUG = "debug"
    CUSTOM = "custom"


class AgentStatus(str, Enum):
    """Agent lifecycle status."""

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Role-specific system prompts that enforce memory operations
ROLE_PROMPTS = {
    AgentRole.ORCHESTRATOR: """You are the Orchestrator Agent. Your role is to:
- Monitor other agents via shared ContextFS memory
- Search for agent findings: contextfs_search(query="agent-", cross_repo=True)
- Coordinate work between agents by saving task assignments
- Save coordination decisions as type="decision"
- Link related findings across agents using contextfs_link""",
    AgentRole.TEST: """You are the Test Agent. Your role is to:
- Write and run tests for the codebase
- ALWAYS save test failures as type="error" with structured_data
- ALWAYS save test patterns as type="code" memories
- Search for existing patterns before writing new tests
- Tag all memories with "test-agent" for cross-agent visibility""",
    AgentRole.CODE: """You are the Code Agent. Your role is to:
- Implement features and fix bugs
- ALWAYS save architecture decisions as type="decision" with rationale
- ALWAYS save code patterns as type="code" memories
- Search for existing decisions before making new ones
- Evolve existing patterns when you improve them
- Tag all memories with "code-agent" for cross-agent visibility""",
    AgentRole.REVIEW: """You are the Review Agent. Your role is to:
- Review code for quality, bugs, and best practices
- Search for related decisions before reviewing
- ALWAYS save review findings as type="fact" memories
- Link related memories using contextfs_link
- Tag all memories with "review-agent" for cross-agent visibility""",
    AgentRole.DOCS: """You are the Documentation Agent. Your role is to:
- Write and update documentation
- Search for existing docs and decisions
- ALWAYS save documentation as type="doc" memories
- Tag all memories with "docs-agent" for cross-agent visibility""",
    AgentRole.DEBUG: """You are the Debug Agent. Your role is to:
- Investigate and fix bugs
- Search for similar errors: contextfs_search(query="error", type="error")
- ALWAYS save error resolutions as type="error" with resolution
- Tag all memories with "debug-agent" for cross-agent visibility""",
    AgentRole.CUSTOM: """You are a specialized agent. Follow your assigned task.
- Use ContextFS for persistent memory
- Save important findings with appropriate types
- Search before acting to leverage existing knowledge""",
}


@dataclass
class Agent:
    """
    An AI agent that runs in a specific directory with ContextFS memory.

    Attributes:
        id: Unique agent identifier
        role: Agent's role (determines behavior)
        directory: Working directory for the agent
        task: The task to perform
        project: ContextFS project for grouping memories
        status: Current agent status
        process: Subprocess handle (when running)
        started_at: When the agent started
        completed_at: When the agent completed
        output: Captured output from the agent
        error: Error message if failed
    """

    role: AgentRole
    directory: str
    task: str
    project: str = "default"
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: AgentStatus = AgentStatus.PENDING
    process: subprocess.Popen | None = field(default=None, repr=False)
    pid: int | None = None  # Track PID for status checks across CLI invocations
    started_at: datetime | None = None
    completed_at: datetime | None = None
    output: str = ""
    error: str = ""
    _output_file: Path | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate agent configuration."""
        self.directory = os.path.expanduser(self.directory)
        if not os.path.isdir(self.directory):
            raise ValueError(f"Directory does not exist: {self.directory}")

    @property
    def system_prompt(self) -> str:
        """Get the role-specific system prompt."""
        base_prompt = ROLE_PROMPTS.get(self.role, ROLE_PROMPTS[AgentRole.CUSTOM])
        return f"""{base_prompt}

PROJECT: {self.project}
DIRECTORY: {self.directory}
AGENT_ID: {self.id}

Remember: All memories are shared with other agents. Use tags and project to organize."""

    def get_full_prompt(self) -> str:
        """Get the complete prompt including task."""
        return f"""{self.system_prompt}

---
TASK: {self.task}
---

Begin by searching ContextFS for relevant context, then proceed with the task.
When complete, save a summary of your work to memory."""

    def spawn(self, tool: str = "claude", background: bool = True) -> "Agent":
        """
        Spawn the agent as a subprocess.

        Args:
            tool: CLI tool to use ("claude", "gemini", etc.)
            background: Run in background (non-blocking)

        Returns:
            Self for chaining
        """
        if self.status not in (AgentStatus.PENDING, AgentStatus.FAILED):
            raise RuntimeError(f"Agent {self.id} already spawned (status: {self.status})")

        # Check if tool is available
        tool_path = shutil.which(tool)
        if not tool_path:
            raise RuntimeError(f"Tool '{tool}' not found in PATH")

        # Set up environment
        env = os.environ.copy()
        env["CONTEXTFS_DATA_DIR"] = os.path.expanduser("~/.contextfs")
        env["CONTEXTFS_SOURCE_TOOL"] = f"agent-{self.role.value}"
        env["CONTEXTFS_PROJECT"] = self.project  # Share project across agents

        # Create output file for capturing results
        output_dir = Path(os.path.expanduser("~/.contextfs/orchestrator"))
        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_file = output_dir / f"agent-{self.id}.log"

        # Set up agent-specific hooks for auto-extraction
        self._setup_agent_hooks()

        # Build command based on tool
        if tool == "claude":
            cmd = [
                tool,
                "--print",
                "--permission-mode",
                "bypassPermissions",  # Allow MCP tools
                "--output-format",
                "json",  # Structured output
                "--no-session-persistence",  # Don't save to disk
                "-p",
                self.get_full_prompt(),
            ]
        elif tool == "gemini":
            # Gemini CLI (if available)
            cmd = [tool, "--non-interactive", self.get_full_prompt()]
        else:
            # Generic fallback
            cmd = [tool, self.get_full_prompt()]

        self.status = AgentStatus.STARTING
        self.started_at = datetime.now()

        try:
            with open(self._output_file, "w") as outfile:
                self.process = subprocess.Popen(
                    cmd,
                    cwd=self.directory,
                    env=env,
                    stdout=outfile,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            self.pid = self.process.pid  # Track PID for status checks
            self.status = AgentStatus.RUNNING
        except Exception as e:
            self.status = AgentStatus.FAILED
            self.error = str(e)
            raise

        if not background:
            self.wait()

        return self

    def _setup_agent_hooks(self) -> None:
        """Set up Claude Code hooks for this agent's directory."""
        claude_dir = Path(self.directory) / ".claude"
        claude_dir.mkdir(exist_ok=True)

        settings_file = claude_dir / "settings.local.json"

        # Agent-specific hooks for auto-extraction
        hooks_config = {
            "hooks": {
                "SessionStart": [
                    {
                        "matcher": "",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "python -m contextfs.cli memory auto-recall --quiet",
                            }
                        ],
                    }
                ],
                "Stop": [
                    {
                        "matcher": "",
                        "hooks": [
                            {
                                "type": "command",
                                "command": f"python -m contextfs.cli extract transcript --save --quiet --agent-role {self.role.value} --project {self.project}",
                            }
                        ],
                    }
                ],
            }
        }

        settings_file.write_text(json.dumps(hooks_config, indent=2))

    def wait(self, timeout: float | None = None) -> int:
        """
        Wait for the agent to complete.

        Args:
            timeout: Maximum seconds to wait (None = infinite)

        Returns:
            Exit code
        """
        if not self.process:
            raise RuntimeError("Agent not spawned")

        try:
            exit_code = self.process.wait(timeout=timeout)
            self.completed_at = datetime.now()

            # Read output
            if self._output_file and self._output_file.exists():
                self.output = self._output_file.read_text()
                self._parse_json_output()

            if exit_code == 0:
                self.status = AgentStatus.COMPLETED
            else:
                self.status = AgentStatus.FAILED
                self.error = f"Exit code: {exit_code}"

            return exit_code

        except subprocess.TimeoutExpired:
            self.cancel()
            raise

    def _parse_json_output(self) -> None:
        """Parse JSON output from claude CLI."""
        if not self.output:
            return

        try:
            data = json.loads(self.output.strip())
            self._result_data = data
            # Extract the actual result text
            if "result" in data:
                self._result_text = data["result"]
            # Check for errors
            if data.get("is_error"):
                self.error = data.get("result", "Unknown error")
        except json.JSONDecodeError:
            # Not JSON, keep raw output
            self._result_data = None
            self._result_text = self.output

    @property
    def result(self) -> str | None:
        """Get the agent's result text."""
        return getattr(self, "_result_text", None)

    @property
    def result_data(self) -> dict | None:
        """Get the full JSON result data."""
        return getattr(self, "_result_data", None)

    def cancel(self) -> None:
        """Cancel the running agent."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.status = AgentStatus.CANCELLED
        self.completed_at = datetime.now()

    def poll(self) -> AgentStatus:
        """Check agent status without blocking."""
        if self.process and self.status == AgentStatus.RUNNING:
            exit_code = self.process.poll()
            if exit_code is not None:
                self.completed_at = datetime.now()
                if self._output_file and self._output_file.exists():
                    self.output = self._output_file.read_text()
                if exit_code == 0:
                    self.status = AgentStatus.COMPLETED
                else:
                    self.status = AgentStatus.FAILED
                    self.error = f"Exit code: {exit_code}"
        return self.status

    @property
    def is_running(self) -> bool:
        """Check if agent is still running."""
        self.poll()
        return self.status == AgentStatus.RUNNING

    @property
    def is_done(self) -> bool:
        """Check if agent has finished (success or failure)."""
        self.poll()
        return self.status in (
            AgentStatus.COMPLETED,
            AgentStatus.FAILED,
            AgentStatus.CANCELLED,
        )

    @property
    def duration(self) -> float | None:
        """Get agent run duration in seconds."""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Serialize agent state to dictionary."""
        return {
            "id": self.id,
            "role": self.role.value,
            "directory": self.directory,
            "task": self.task,
            "project": self.project,
            "status": self.status.value,
            "pid": self.pid,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration": self.duration,
            "error": self.error if self.error else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Agent":
        """Restore agent from dictionary (for loading persisted state)."""
        agent = object.__new__(cls)
        agent.id = data["id"]
        agent.role = AgentRole(data["role"])
        agent.directory = data["directory"]
        agent.task = data["task"]
        agent.project = data.get("project", "default")
        agent.status = AgentStatus(data["status"])
        agent.pid = data.get("pid")
        agent.process = None  # Can't restore process handle
        agent.output = ""
        agent.error = data.get("error", "")
        agent._output_file = None

        # Restore timestamps
        if data.get("started_at"):
            agent.started_at = datetime.fromisoformat(data["started_at"])
        else:
            agent.started_at = None

        if data.get("completed_at"):
            agent.completed_at = datetime.fromisoformat(data["completed_at"])
        else:
            agent.completed_at = None

        return agent

    def is_process_alive(self) -> bool:
        """Check if the agent's process is still running (by PID)."""
        if self.pid is None:
            return False
        try:
            os.kill(self.pid, 0)  # Signal 0 just checks if process exists
            return True
        except OSError:
            return False

    def refresh_status(self) -> AgentStatus:
        """Refresh status by checking if process is still alive."""
        if self.status == AgentStatus.RUNNING and not self.is_process_alive():
            # Process finished - mark as completed (or failed if we can detect)
            self.status = AgentStatus.COMPLETED
            self.completed_at = datetime.now()
        return self.status

    def __str__(self) -> str:
        return f"Agent({self.id}, {self.role.value}, {self.status.value})"
