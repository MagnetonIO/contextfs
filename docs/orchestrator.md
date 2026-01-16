# ContextFS Agent Orchestrator

Run multiple AI agents concurrently with shared memory. Built on asyncio with zero external dependencies.

## Quick Start

```bash
# 1. Create agent config
contextfs orchestrator init-config -o agents.json

# 2. Edit agents.json with your tasks

# 3. Run agents concurrently
contextfs orchestrator run agents.json --workers 3
```

## CLI Commands

### `orchestrator run` (Recommended)

Run agents concurrently using async worker queue.

```bash
contextfs orchestrator run agents.json [OPTIONS]

Options:
  -p, --project TEXT   Project name for memory grouping [default: default]
  -t, --tool TEXT      CLI tool: claude, gemini [default: claude]
  -w, --workers INT    Max concurrent agents [default: 3]
  --timeout FLOAT      Per-task timeout in seconds
```

### `orchestrator spawn`

Spawn a single agent.

```bash
contextfs orchestrator spawn <role> <directory> "<task>" [OPTIONS]

# Example
contextfs orchestrator spawn test ./tests "Write unit tests for auth module" --wait
```

### `orchestrator roles`

List available agent roles.

```bash
contextfs orchestrator roles

# Output:
# orchestrator - Monitors and coordinates other agents
# test         - Writes and runs tests, saves failures as errors
# code         - Implements features, saves decisions and patterns
# review       - Reviews code quality, links related findings
# docs         - Writes documentation
# debug        - Investigates and fixes bugs
# custom       - Generic agent for custom tasks
```

### `orchestrator findings`

View memories from agents.

```bash
contextfs orchestrator findings --project myproject --limit 10
```

### `orchestrator errors`

View errors reported by agents.

```bash
contextfs orchestrator errors --project myproject
```

### `orchestrator monitor`

Watch agent activity in real-time.

```bash
contextfs orchestrator monitor --project myproject --interval 5
```

## Configuration File Format

```json
[
  {
    "role": "test",
    "directory": "./tests",
    "task": "Write unit tests for the auth module. Search memory for existing patterns first."
  },
  {
    "role": "code",
    "directory": "./src",
    "task": "Implement the auth module. Save architecture decisions to memory."
  },
  {
    "role": "review",
    "directory": ".",
    "task": "Review auth implementation for security issues. Link findings to decisions."
  }
]
```

## Python API

### Basic Usage

```python
from contextfs.orchestrator import run_agents_sync

# Run agents and get results
results = run_agents_sync([
    {"role": "test", "directory": "./tests", "task": "Write tests"},
    {"role": "code", "directory": "./src", "task": "Implement feature"},
], num_workers=3, timeout=120)

for r in results:
    print(f"{r.role}: {r.status.value} ({r.duration:.1f}s)")
    if r.result:
        print(r.result)
```

### Async Usage

```python
import asyncio
from contextfs.orchestrator import run_agents, WorkerQueue

async def main():
    # Simple: run_agents convenience function
    results = await run_agents([
        {"role": "test", "directory": "./tests", "task": "Write tests"},
    ], num_workers=3)

    # Advanced: WorkerQueue for more control
    async with WorkerQueue(num_workers=3, project="my-feature") as queue:
        # Submit jobs
        job1 = await queue.submit("test", "./tests", "Write tests")
        job2 = await queue.submit("code", "./src", "Implement feature")

        # Stream results as they complete
        async for result in queue.results():
            print(f"Done: {result.role} in {result.duration:.1f}s")

        # Or wait for all
        # results = await queue.join()

asyncio.run(main())
```

### WorkerQueue API

```python
from contextfs.orchestrator import WorkerQueue, Job, JobStatus

async with WorkerQueue(
    num_workers=3,           # Concurrent workers
    project="my-project",    # ContextFS project for memory grouping
    tool="claude",           # CLI tool to use
    default_timeout=120,     # Default per-job timeout
    max_queue_size=10,       # Backpressure (0 = unlimited)
) as queue:

    # Submit jobs (returns job ID)
    job_id = await queue.submit(
        role="test",
        directory="./tests",
        task="Write unit tests",
        timeout=60,  # Override default timeout
    )

    # Check job status
    job = queue.get_job(job_id)
    print(job.status)  # JobStatus.PENDING, RUNNING, COMPLETED, FAILED, TIMEOUT

    # Cancel pending job
    cancelled = await queue.cancel(job_id)

    # Get queue stats
    print(f"Pending: {queue.pending_count}")
    print(f"Running: {queue.running_count}")

    # Wait for all jobs
    results = await queue.join()

    # Or stream results
    async for result in queue.results():
        print(result)
```

## Agent Roles & Memory Protocol

Each agent role has built-in memory operations:

| Role | Auto-saves | Auto-searches |
|------|-----------|---------------|
| **test** | Test failures as `error`, patterns as `code` | Existing test patterns |
| **code** | Decisions as `decision`, patterns as `code` | Prior decisions |
| **review** | Findings as `fact`, links related memories | Related decisions |
| **docs** | Documentation as `doc` | Existing docs |
| **debug** | Error resolutions as `error` | Similar errors |

### Memory Tags

All agent memories are tagged for cross-agent visibility:

- `test-agent`, `code-agent`, `review-agent`, etc.
- Searchable with: `contextfs_search(query="test-agent", cross_repo=True)`

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      WorkerQueue                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   submit() ──► asyncio.Queue ──┬──► Worker 0 ──► claude CLI │
│                                ├──► Worker 1 ──► claude CLI │
│                                └──► Worker N ──► claude CLI │
│                                                             │
│   results() ◄── asyncio.Queue ◄────────────────────────────┘
│                                                             │
│                         │                                   │
│                         ▼                                   │
│              ┌──────────────────┐                          │
│              │  Shared Memory   │                          │
│              │  ~/.contextfs/   │                          │
│              └──────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## Examples

### Run Test + Code Agents in Parallel

```bash
cat > agents.json << 'EOF'
[
  {"role": "test", "directory": "./tests", "task": "Write tests for UserService"},
  {"role": "code", "directory": "./src", "task": "Implement UserService"}
]
EOF

contextfs orchestrator run agents.json --project user-service --workers 2
```

### Cross-Repo Coordination

```bash
cat > cross-repo.json << 'EOF'
[
  {"role": "code", "directory": "/path/to/backend", "task": "Add /api/users endpoint"},
  {"role": "code", "directory": "/path/to/frontend", "task": "Add user list component that calls /api/users"}
]
EOF

contextfs orchestrator run cross-repo.json --project user-feature
```

### Continuous Integration

```python
#!/usr/bin/env python3
"""CI script that runs test agents before deploying."""

from contextfs.orchestrator import run_agents_sync, JobStatus

results = run_agents_sync([
    {"role": "test", "directory": "./tests", "task": "Run all tests and report failures"},
    {"role": "review", "directory": ".", "task": "Check for security issues in recent changes"},
], num_workers=2, timeout=300)

# Check results
failed = [r for r in results if r.status != JobStatus.COMPLETED]
if failed:
    print("CI Failed:")
    for r in failed:
        print(f"  {r.role}: {r.error}")
    exit(1)

print("CI Passed!")
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CONTEXTFS_DATA_DIR` | Shared memory directory | `~/.contextfs` |
| `CONTEXTFS_PROJECT` | Default project name | `default` |
| `CONTEXTFS_SOURCE_TOOL` | Agent identifier | `agent-{role}` |
