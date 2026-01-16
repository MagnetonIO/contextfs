# Memory Types

ContextFS categorizes memories by type to enable better organization, filtering, and retrieval.

## Available Types

### `fact`

Technical facts, configurations, and static information.

```python
ctx.save(
    content="Database connection pool size is 20",
    type=MemoryType.FACT,
    tags=["database", "config"]
)
```

**Use for:**

- Configuration values
- API endpoints and URLs
- Technical specifications
- Environment details

### `decision`

Architectural decisions and their rationale.

```python
ctx.save(
    content="Using PostgreSQL over MySQL because of JSON column support and better concurrency handling for our read-heavy workload",
    type=MemoryType.DECISION,
    tags=["database", "architecture"]
)
```

**Use for:**

- Technology choices
- Trade-off analyses
- Design patterns selected
- Why something was done a certain way

### `procedural`

How-to guides, workflows, and step-by-step processes.

```python
ctx.save(
    content="To deploy: 1) Run tests 2) Build Docker image 3) Push to registry 4) Update k8s deployment",
    type=MemoryType.PROCEDURAL,
    tags=["deploy", "workflow"]
)
```

**Use for:**

- Deployment procedures
- Setup instructions
- Debugging workflows
- Common task sequences

### `episodic`

Session summaries, events, and temporal information.

```python
ctx.save(
    content="Debugging session: Found race condition in payment processing. Fixed by adding mutex lock around balance update.",
    type=MemoryType.EPISODIC,
    tags=["debugging", "payments"]
)
```

**Use for:**

- Session summaries
- Incident reports
- Meeting notes
- Timeline of changes

### `code`

Code snippets, patterns, and implementations.

```python
ctx.save(
    content='''def retry_with_backoff(fn, max_retries=3):
    for i in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if i == max_retries - 1:
                raise
            time.sleep(2 ** i)''',
    type=MemoryType.CODE,
    tags=["python", "retry", "pattern"]
)
```

**Use for:**

- Reusable code snippets
- Common patterns
- Boilerplate templates
- API usage examples

### `error`

Bug fixes, troubleshooting, and error resolutions.

```python
ctx.save(
    content="ImportError: cannot import 'foo' from 'bar' - Fixed by upgrading bar to v2.0",
    type=MemoryType.ERROR,
    tags=["python", "imports", "dependencies"]
)
```

**Use for:**

- Error messages and fixes
- Troubleshooting steps
- Known issues
- Workarounds

### `user`

User preferences and personalization.

```python
ctx.save(
    content="Prefers TypeScript over JavaScript. Uses Vim keybindings.",
    type=MemoryType.USER,
    tags=["preferences"]
)
```

**Use for:**

- User preferences
- Coding style preferences
- Tool preferences
- Personal conventions

## Memory Schema

Each memory has the following structure:

```python
class Memory(BaseModel):
    id: str                    # Unique identifier
    content: str               # Main content
    type: MemoryType           # Category
    tags: list[str]            # Searchable tags
    summary: str | None        # Brief summary

    # Namespace
    namespace_id: str          # Repo/global namespace

    # Timestamps
    created_at: datetime
    updated_at: datetime

    # Source tracking
    source_file: str | None    # Origin file
    source_repo: str | None    # Origin repository
    source_tool: str | None    # Creating tool
    project: str | None        # Project grouping
    session_id: str | None     # Session reference

    # Metadata
    metadata: dict[str, Any]   # Additional data

    # Computed
    embedding: list[float]     # Vector embedding
```

## Best Practices

### Use Appropriate Types

Choose the type that best describes the **purpose** of the memory:

| If you're saving... | Use type... |
|---------------------|-------------|
| A configuration value | `fact` |
| Why you chose a library | `decision` |
| How to run tests | `procedural` |
| What you did in a session | `episodic` |
| A utility function | `code` |
| How you fixed a bug | `error` |
| User preferences | `user` |

### Add Meaningful Tags

Tags improve searchability:

```python
# Good: specific, categorical
tags=["auth", "jwt", "security", "api"]

# Avoid: too generic
tags=["code", "backend"]
```

### Include Context

Memories should be self-contained:

```python
# Good: includes context
ctx.save(
    content="JWT tokens expire after 24h. Refresh tokens last 7 days. Configured in auth/config.py",
    type=MemoryType.FACT
)

# Poor: lacks context
ctx.save(
    content="24 hours",
    type=MemoryType.FACT
)
```

### Use Summaries for Long Content

For detailed memories, add a summary:

```python
ctx.save(
    content="[Long detailed deployment procedure...]",
    summary="Production deployment checklist",
    type=MemoryType.PROCEDURAL
)
```

## Formal Type System

ContextFS implements a formal type-theoretic memory system based on Definition 5.1 Type Grammar. This enables type-safe access to `structured_data` with both runtime (Pydantic) and static (mypy/pyright) enforcement.

### Type-Safe Memory Access

Convert any memory to a typed wrapper for IDE autocomplete and type checking:

```python
from contextfs.schemas import Memory, DecisionData
from contextfs.types import Mem

# Create memory
memory = Memory.decision("DB choice", decision="PostgreSQL")

# Type-safe access
typed: Mem[DecisionData] = memory.as_typed(DecisionData)
print(typed.data.decision)  # IDE knows this is str
```

### Versioned Memory with Timeline

Track memory evolution with formal change reasons:

```python
from contextfs.types import VersionedMem, ChangeReason

versioned = memory.as_versioned(DecisionData)

# Evolve with reason tracking
versioned.evolve(
    DecisionData(decision="SQLite"),
    reason=ChangeReason.CORRECTION
)

# Query history
print(versioned.timeline.root.content.decision)     # "PostgreSQL"
print(versioned.timeline.current.content.decision)  # "SQLite"
```

### Change Reasons

| Reason | When to Use |
|--------|-------------|
| `OBSERVATION` | New external information |
| `INFERENCE` | Derived from existing knowledge |
| `CORRECTION` | Fixing an error |
| `DECAY` | Knowledge becoming stale |

See the type system implementation in `contextfs.types` for full documentation.
