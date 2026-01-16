"""
Memory-based monitoring for agent orchestration.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from contextfs.core import ContextFS


@dataclass
class AgentActivity:
    """Activity detected from an agent via memory."""

    agent_id: str
    agent_role: str
    memory_id: str
    memory_type: str
    summary: str
    timestamp: datetime
    tags: list[str] = field(default_factory=list)


class MemoryMonitor:
    """
    Monitor agent activity via shared ContextFS memory.

    This allows the orchestrator to see what agents are doing
    by watching for new memories they create.
    """

    def __init__(self, project: str = "default", ctx: ContextFS | None = None):
        """
        Initialize the memory monitor.

        Args:
            project: Project to monitor
            ctx: ContextFS instance (creates one if not provided)
        """
        self.project = project
        self.ctx = ctx or ContextFS()
        self._last_check = datetime.now()
        self._seen_ids: set[str] = set()
        self._callbacks: list[Callable[[AgentActivity], None]] = []

    def on_activity(self, callback: Callable[[AgentActivity], None]) -> None:
        """Register a callback for new agent activity."""
        self._callbacks.append(callback)

    def check_activity(self) -> list[AgentActivity]:
        """
        Check for new agent activity since last check.

        Returns:
            List of new activities detected
        """
        activities = []

        # Search for recent memories from agents
        results = self.ctx.search(
            query="agent-",
            limit=50,
            cross_repo=True,
        )

        for result in results:
            memory = result.memory

            # Skip already seen
            if memory.id in self._seen_ids:
                continue

            # Check if from an agent (tagged with agent-*)
            agent_tags = [t for t in (memory.tags or []) if t.startswith("agent-") or "-agent" in t]
            if not agent_tags:
                continue

            # Extract agent info from tags
            agent_role = "unknown"
            for tag in agent_tags:
                if "-agent" in tag:
                    agent_role = tag.replace("-agent", "")
                    break

            activity = AgentActivity(
                agent_id=memory.source_tool or "unknown",
                agent_role=agent_role,
                memory_id=memory.id,
                memory_type=memory.type.value,
                summary=memory.summary or memory.content[:100],
                timestamp=memory.created_at,
                tags=memory.tags or [],
            )

            activities.append(activity)
            self._seen_ids.add(memory.id)

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(activity)
                except Exception:
                    pass  # Don't let callback errors break monitoring

        self._last_check = datetime.now()
        return activities

    def get_agent_findings(self, agent_role: str | None = None) -> list[dict]:
        """
        Get all findings from agents.

        Args:
            agent_role: Filter by agent role (e.g., "test", "code")

        Returns:
            List of memory dictionaries
        """
        query = f"{agent_role}-agent" if agent_role else "agent-"

        results = self.ctx.search(
            query=query,
            limit=100,
            cross_repo=True,
        )

        return [
            {
                "id": r.memory.id,
                "type": r.memory.type.value,
                "summary": r.memory.summary,
                "content": r.memory.content,
                "tags": r.memory.tags,
                "score": r.score,
            }
            for r in results
        ]

    def get_errors(self) -> list[dict]:
        """Get all error memories from agents."""
        results = self.ctx.search(
            query="agent error",
            limit=50,
            cross_repo=True,
        )

        return [
            {
                "id": r.memory.id,
                "summary": r.memory.summary,
                "content": r.memory.content,
                "source": r.memory.source_tool,
            }
            for r in results
            if r.memory.type.value == "error"
        ]

    def get_decisions(self) -> list[dict]:
        """Get all decision memories from agents."""
        results = self.ctx.search(
            query="agent decision",
            limit=50,
            cross_repo=True,
        )

        return [
            {
                "id": r.memory.id,
                "summary": r.memory.summary,
                "content": r.memory.content,
                "source": r.memory.source_tool,
            }
            for r in results
            if r.memory.type.value == "decision"
        ]

    def watch(
        self,
        interval: float = 5.0,
        duration: float | None = None,
        on_activity: Callable[[AgentActivity], None] | None = None,
    ) -> None:
        """
        Continuously watch for agent activity.

        Args:
            interval: Seconds between checks
            duration: Total seconds to watch (None = forever)
            on_activity: Callback for each activity
        """
        if on_activity:
            self.on_activity(on_activity)

        start = time.time()
        try:
            while True:
                activities = self.check_activity()

                for activity in activities:
                    print(f"[{activity.timestamp}] {activity.agent_role}: {activity.summary}")

                if duration and (time.time() - start) > duration:
                    break

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
