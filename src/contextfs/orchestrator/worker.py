"""
Lightweight async worker engine for agent orchestration.

Uses asyncio Queue + subprocess for concurrent agent execution.
Pattern: N workers consuming from a queue, each running subprocess.

Features:
- Job queue with backpressure
- Per-job timeout
- Streaming output support
- Cancellation
- No external dependencies
"""

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .agent import Agent, AgentRole


class JobStatus(str, Enum):
    """Job lifecycle status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class Job:
    """A job to be executed by a worker."""

    id: str
    role: AgentRole
    directory: str
    task: str
    project: str = "default"
    timeout: float | None = None
    status: JobStatus = JobStatus.PENDING
    result: str | None = None
    result_data: dict | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def duration(self) -> float | None:
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()


@dataclass
class WorkerResult:
    """Result from a completed job."""

    job_id: str
    agent_id: str
    role: str
    status: JobStatus
    result: str | None
    result_data: dict | None
    duration: float
    error: str | None = None


class WorkerQueue:
    """
    Async worker queue for running CLI agents concurrently.

    Pattern:
    - Jobs added to asyncio.Queue
    - N worker tasks consume from queue
    - Each worker runs subprocess via asyncio.create_subprocess_exec
    - Supports timeout, cancellation, streaming, backpressure

    Example:
        async with WorkerQueue(num_workers=3, project="my-feature") as queue:
            # Add jobs
            await queue.submit("test", "./tests", "Write tests")
            await queue.submit("code", "./src", "Implement feature")

            # Wait for all jobs
            results = await queue.join()

        # Or with streaming results
        async with WorkerQueue(num_workers=3) as queue:
            await queue.submit("test", "./tests", "Write tests")
            await queue.submit("code", "./src", "Implement feature")

            async for result in queue.results():
                print(f"Completed: {result.role}")
    """

    def __init__(
        self,
        num_workers: int = 3,
        project: str = "default",
        tool: str = "claude",
        default_timeout: float | None = None,
        max_queue_size: int = 0,  # 0 = unlimited
    ):
        self.num_workers = num_workers
        self.project = project
        self.tool = tool
        self.default_timeout = default_timeout

        self._queue: asyncio.Queue[Job | None] = asyncio.Queue(maxsize=max_queue_size)
        self._results_queue: asyncio.Queue[WorkerResult] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._jobs: dict[str, Job] = {}
        self._running = False
        self._job_counter = 0

    async def __aenter__(self) -> "WorkerQueue":
        """Start workers on context enter."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop workers on context exit."""
        await self.stop()

    async def start(self) -> None:
        """Start worker tasks."""
        if self._running:
            return

        self._running = True
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker(i), name=f"worker-{i}")
            self._workers.append(worker)

    async def stop(self, wait: bool = True) -> None:
        """
        Stop all workers.

        Args:
            wait: If True, wait for current jobs to complete
        """
        if not self._running:
            return

        # Signal workers to stop
        for _ in range(self.num_workers):
            await self._queue.put(None)

        if wait:
            # Wait for workers to finish current jobs
            await asyncio.gather(*self._workers, return_exceptions=True)

        self._running = False
        self._workers.clear()

    async def submit(
        self,
        role: AgentRole | str,
        directory: str,
        task: str,
        timeout: float | None = None,
    ) -> str:
        """
        Submit a job to the queue.

        Args:
            role: Agent role
            directory: Working directory
            task: Task description
            timeout: Per-job timeout (overrides default)

        Returns:
            Job ID
        """
        if isinstance(role, str):
            role = AgentRole(role)

        self._job_counter += 1
        job_id = f"job-{self._job_counter:04d}"

        job = Job(
            id=job_id,
            role=role,
            directory=directory,
            task=task,
            project=self.project,
            timeout=timeout or self.default_timeout,
        )

        self._jobs[job_id] = job
        await self._queue.put(job)

        return job_id

    async def cancel(self, job_id: str) -> bool:
        """
        Cancel a pending job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancelled, False if already running/completed
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            return True

        return False

    async def join(self) -> list[WorkerResult]:
        """
        Wait for all submitted jobs to complete.

        Returns:
            List of all results
        """
        # Wait for queue to be empty
        await self._queue.join()

        # Collect all results
        results = []
        while not self._results_queue.empty():
            results.append(await self._results_queue.get())

        return results

    async def results(self) -> AsyncIterator[WorkerResult]:
        """
        Async iterator yielding results as they complete.

        Example:
            async for result in queue.results():
                print(f"Completed: {result.role}")
        """
        pending_jobs = len(self._jobs)
        completed = 0

        while completed < pending_jobs:
            result = await self._results_queue.get()
            completed += 1
            yield result

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID."""
        return self._jobs.get(job_id)

    @property
    def pending_count(self) -> int:
        """Number of jobs waiting in queue."""
        return self._queue.qsize()

    @property
    def running_count(self) -> int:
        """Number of jobs currently running."""
        return sum(1 for j in self._jobs.values() if j.status == JobStatus.RUNNING)

    async def _worker(self, worker_id: int) -> None:
        """Worker task that processes jobs from queue."""
        while True:
            job = await self._queue.get()

            # None signals shutdown
            if job is None:
                self._queue.task_done()
                break

            # Skip cancelled jobs
            if job.status == JobStatus.CANCELLED:
                self._queue.task_done()
                continue

            try:
                result = await self._execute_job(job)
                await self._results_queue.put(result)
            except Exception as e:
                # Shouldn't happen, but handle gracefully
                result = WorkerResult(
                    job_id=job.id,
                    agent_id="unknown",
                    role=job.role.value,
                    status=JobStatus.FAILED,
                    result=None,
                    result_data=None,
                    duration=job.duration or 0,
                    error=str(e),
                )
                await self._results_queue.put(result)
            finally:
                self._queue.task_done()

    async def _execute_job(self, job: Job) -> WorkerResult:
        """Execute a single job."""
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()

        # Create agent for prompt generation
        agent = Agent(
            role=job.role,
            directory=job.directory,
            task=job.task,
            project=job.project,
        )

        try:
            result_data = await self._run_cli(agent, job.timeout)
            job.completed_at = datetime.now()

            is_error = result_data.get("is_error", False) if result_data else True
            job.status = JobStatus.FAILED if is_error else JobStatus.COMPLETED
            job.result = result_data.get("result") if result_data else None
            job.result_data = result_data
            if is_error:
                job.error = job.result

            return WorkerResult(
                job_id=job.id,
                agent_id=agent.id,
                role=job.role.value,
                status=job.status,
                result=job.result,
                result_data=job.result_data,
                duration=job.duration or 0,
                error=job.error,
            )

        except asyncio.TimeoutError:
            job.completed_at = datetime.now()
            job.status = JobStatus.TIMEOUT
            job.error = f"Timeout after {job.timeout}s"

            return WorkerResult(
                job_id=job.id,
                agent_id=agent.id,
                role=job.role.value,
                status=JobStatus.TIMEOUT,
                result=None,
                result_data=None,
                duration=job.duration or 0,
                error=job.error,
            )

        except Exception as e:
            job.completed_at = datetime.now()
            job.status = JobStatus.FAILED
            job.error = str(e)

            return WorkerResult(
                job_id=job.id,
                agent_id=agent.id,
                role=job.role.value,
                status=JobStatus.FAILED,
                result=None,
                result_data=None,
                duration=job.duration or 0,
                error=job.error,
            )

    async def _run_cli(
        self,
        agent: Agent,
        timeout: float | None = None,
    ) -> dict | None:
        """Run agent CLI via subprocess."""
        env = os.environ.copy()
        env["CONTEXTFS_DATA_DIR"] = os.path.expanduser("~/.contextfs")
        env["CONTEXTFS_SOURCE_TOOL"] = f"agent-{agent.role.value}"
        env["CONTEXTFS_PROJECT"] = self.project

        args = [
            self.tool,
            "--print",
            "--permission-mode",
            "bypassPermissions",
            "--output-format",
            "json",
            "--no-session-persistence",
            "-p",
            agent.get_full_prompt(),
        ]

        process = await asyncio.create_subprocess_exec(
            *args,
            cwd=agent.directory,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            output = stdout.decode() if stdout else ""

            try:
                return json.loads(output.strip())
            except json.JSONDecodeError:
                return {"result": output, "is_error": False}

        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise


# Convenience functions


async def run_agents(
    agents: list[dict],
    project: str = "default",
    tool: str = "claude",
    num_workers: int = 3,
    timeout: float | None = None,
    on_complete: Callable[[WorkerResult], None] | None = None,
) -> list[WorkerResult]:
    """
    Run multiple agents using worker queue.

    Args:
        agents: List of agent configs with role, directory, task
        project: ContextFS project for grouping
        tool: CLI tool to use
        num_workers: Number of concurrent workers
        timeout: Per-job timeout in seconds
        on_complete: Callback for each completed job

    Returns:
        List of WorkerResults

    Example:
        results = await run_agents([
            {"role": "test", "directory": "./tests", "task": "Write tests"},
            {"role": "code", "directory": "./src", "task": "Implement feature"},
        ])
    """
    async with WorkerQueue(
        num_workers=num_workers,
        project=project,
        tool=tool,
        default_timeout=timeout,
    ) as queue:
        # Submit all jobs
        for config in agents:
            await queue.submit(
                role=config["role"],
                directory=config["directory"],
                task=config["task"],
            )

        # Collect results with optional callback
        results = []
        async for result in queue.results():
            results.append(result)
            if on_complete:
                on_complete(result)

        return results


def run_agents_sync(
    agents: list[dict],
    project: str = "default",
    tool: str = "claude",
    num_workers: int = 3,
    timeout: float | None = None,
) -> list[WorkerResult]:
    """
    Synchronous wrapper for run_agents.

    Example:
        results = run_agents_sync([
            {"role": "test", "directory": "./tests", "task": "Write tests"},
        ])
    """
    return asyncio.run(
        run_agents(
            agents=agents,
            project=project,
            tool=tool,
            num_workers=num_workers,
            timeout=timeout,
        )
    )


# Legacy aliases for backwards compatibility
WorkerPool = WorkerQueue
