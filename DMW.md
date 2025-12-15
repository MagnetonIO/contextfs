# Developer Memory Workflow (DMW)

A comprehensive guide to implementing persistent, searchable memory for software development with ContextFS.

## Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Solo Developer Workflow](#solo-developer-workflow)
- [Team Workflow](#team-workflow)
- [Memory Types](#memory-types)
- [Search Strategies](#search-strategies)
- [Integration Patterns](#integration-patterns)
- [Best Practices](#best-practices)

---

## Overview

Developer Memory Workflow (DMW) is a methodology for capturing, organizing, and retrieving development knowledge using ContextFS. It addresses the common problem of **context loss** - when developers (or their AI assistants) repeatedly re-discover the same information across sessions.

### The Problem

```
Session 1: "Use PostgreSQL for the database" → AI understands
Session 2: "What database should I use?" → AI doesn't know
Session 3: Debug CORS issue → Fix it
Session 4: Same CORS issue → Debug from scratch
```

### The Solution

```
Session 1: Save decision → Memory: "Use PostgreSQL"
Session 2: Search "database" → Instant recall
Session 3: Save fix → Memory: "CORS allow_methods=['*']"
Session 4: Search "CORS" → Fix in seconds
```

---

## Core Concepts

### 1. Memory Persistence

Memories are stored in two complementary backends:

| Backend | Storage | Best For |
|---------|---------|----------|
| **SQLite + FTS5** | `~/.contextfs/context.db` | Keyword search, sessions, fast queries |
| **ChromaDB** | `~/.contextfs/chroma_db/` | Semantic search, code similarity |

### 2. Namespace Isolation

Memories are isolated by namespace (typically per-repo):

```python
# Auto-detected from git repo
ctx = ContextFS()  # namespace = "repo-<hash>"

# Explicit namespace for sharing
ctx = ContextFS(namespace_id="team-acme")

# Global namespace
ctx = ContextFS(namespace_id="global")
```

### 3. Smart Routing

ContextFS automatically routes queries to the optimal backend:

| Memory Type | Primary Backend | Reason |
|-------------|-----------------|--------|
| `episodic`, `user` | FTS | Keyword-heavy session data |
| `code`, `error` | RAG | Semantic code similarity |
| `fact`, `decision`, `procedural` | Hybrid | Both keyword + semantic |

---

## Solo Developer Workflow

### Phase 1: Capture

As you work, capture important context:

```python
from contextfs import ContextFS, MemoryType

ctx = ContextFS()

# Architectural decision
ctx.save(
    content="Use PostgreSQL for production, SQLite for testing",
    type=MemoryType.DECISION,
    tags=["database", "architecture", "testing"],
    summary="Database technology choice"
)

# Bug fix
ctx.save(
    content="""CORS preflight failing: Fixed by setting allow_methods=['*']
    instead of explicit list. FastAPI CORSMiddleware requires this for
    credentials=True.""",
    type=MemoryType.ERROR,
    tags=["cors", "fastapi", "api", "bug-fix"],
    summary="CORS preflight fix for FastAPI"
)

# Code pattern
ctx.save(
    content="""Spline length approximation:
    def approximate_length(spline, segments=100):
        points = list(spline.approximate(segments=segments))
        return sum(p1.distance(p2) for p1, p2 in zip(points, points[1:]))
    """,
    type=MemoryType.CODE,
    tags=["geometry", "algorithm", "spline"],
    summary="Spline length calculation algorithm"
)

# Deployment procedure
ctx.save(
    content="""Deploy to production:
    1. Run tests: pytest
    2. Build: docker build -t app .
    3. Push: docker push registry/app:latest
    4. Deploy: kubectl rollout restart deployment/app
    5. Verify: curl https://api.example.com/health
    """,
    type=MemoryType.PROCEDURAL,
    tags=["deployment", "kubernetes", "docker"],
    summary="Production deployment steps"
)
```

### Phase 2: Recall

Later sessions automatically benefit from captured context:

```python
# Find relevant database decisions
results = ctx.search("database configuration")
for r in results:
    print(f"[{r.memory.type.value}] {r.memory.summary}")
    # [decision] Database technology choice

# Find past bug fixes
results = ctx.search("API errors", type=MemoryType.ERROR)
# Returns CORS fix and other API-related errors

# Get deployment steps
results = ctx.search("how to deploy", type=MemoryType.PROCEDURAL)
# Returns deployment procedure
```

### Phase 3: Context Injection

ContextFS can inject relevant context into AI prompts:

```python
# Get formatted context for a task
context = ctx.get_context_for_task("implement authentication")
# Returns formatted string with relevant memories

# Use with your AI assistant
prompt = f"""
{context}

Task: Implement JWT authentication for the API.
"""
```

---

## Team Workflow

### Setup: Shared Namespace

```python
# All team members use same namespace
TEAM_NAMESPACE = "team-acme-project"

ctx = ContextFS(namespace_id=TEAM_NAMESPACE)
```

### Onboarding New Members

New team members instantly access all project knowledge:

```python
# New developer joins
ctx = ContextFS(namespace_id="team-acme-project")

# Search for conventions
results = ctx.search("coding conventions style guide")

# Search for architecture
results = ctx.search("system architecture design")

# Search for common issues
results = ctx.search("common bugs errors", type=MemoryType.ERROR)
```

### Cross-Repo Context

For multi-repo projects (frontend + backend):

```python
# In frontend repo
ctx = ContextFS(namespace_id="haven-project")
ctx.save(
    content="Frontend uploads DXF to POST /api/v2/analyze",
    type=MemoryType.FACT,
    tags=["integration", "api", "dxf"],
    summary="Frontend-backend DXF upload integration"
)

# In backend repo (same namespace)
ctx = ContextFS(namespace_id="haven-project")
results = ctx.search("DXF upload endpoint")
# Finds the integration documentation
```

### Team Conventions

Document and enforce team standards:

```python
# Save team conventions
conventions = [
    ("API uses snake_case for JSON keys", ["api", "convention"]),
    ("All dates in ISO 8601 format", ["api", "dates", "convention"]),
    ("Error responses include 'code' and 'message' fields", ["api", "errors"]),
    ("Use conventional commits for git messages", ["git", "convention"]),
]

for content, tags in conventions:
    ctx.save(
        content=content,
        type=MemoryType.FACT,
        tags=tags + ["team-standard"],
        summary=f"Convention: {content[:50]}..."
    )
```

---

## Memory Types

### FACT
**Purpose:** Objective information about the project

```python
ctx.save(
    content="PostgreSQL 15 on AWS RDS, 2 replicas",
    type=MemoryType.FACT,
    tags=["infrastructure", "database"],
)
```

**Best for:** Configurations, dependencies, integrations, specifications

### DECISION
**Purpose:** Architectural decisions with rationale

```python
ctx.save(
    content="""Decision: Use Redis for caching

    Rationale:
    - Sub-millisecond latency required
    - Already have Redis expertise on team
    - Excellent Python client (redis-py)

    Alternatives considered:
    - Memcached (less features)
    - In-memory dict (not distributed)
    """,
    type=MemoryType.DECISION,
    tags=["caching", "architecture", "redis"],
)
```

**Best for:** Technology choices, design patterns, trade-off decisions

### CODE
**Purpose:** Important algorithms, patterns, and snippets

```python
ctx.save(
    content="""Rate limiting decorator:

    from functools import wraps
    from time import time

    def rate_limit(calls_per_minute):
        def decorator(func):
            last_called = [0.0]
            @wraps(func)
            def wrapper(*args, **kwargs):
                elapsed = time() - last_called[0]
                if elapsed < 60 / calls_per_minute:
                    raise RateLimitExceeded()
                last_called[0] = time()
                return func(*args, **kwargs)
            return wrapper
        return decorator
    """,
    type=MemoryType.CODE,
    tags=["rate-limiting", "decorator", "pattern"],
)
```

**Best for:** Algorithms, patterns, reusable code, complex logic

### ERROR
**Purpose:** Bug patterns and their solutions

```python
ctx.save(
    content="""Error: Connection pool exhausted

    Symptom: SQLAlchemy raising 'QueuePool limit reached'

    Root cause: Not closing sessions in async code

    Fix: Use async context manager:
        async with AsyncSession() as session:
            # queries here
        # session auto-closed

    Prevention: Add connection pool monitoring
    """,
    type=MemoryType.ERROR,
    tags=["sqlalchemy", "async", "connection-pool"],
)
```

**Best for:** Bug fixes, error patterns, debugging solutions

### PROCEDURAL
**Purpose:** Step-by-step processes

```python
ctx.save(
    content="""Setting up local development:

    1. Clone repo: git clone ...
    2. Create venv: python -m venv venv
    3. Install deps: pip install -e ".[dev]"
    4. Copy env: cp .env.example .env
    5. Start DB: docker-compose up -d postgres
    6. Run migrations: alembic upgrade head
    7. Start server: uvicorn main:app --reload
    """,
    type=MemoryType.PROCEDURAL,
    tags=["setup", "development", "onboarding"],
)
```

**Best for:** Setup guides, deployment steps, operational runbooks

### EPISODIC
**Purpose:** Session transcripts and conversations

```python
ctx.save(
    content="""Session: Debugging auth flow

    User: Login fails with 401
    Investigation: Token was expired
    Fix: Added token refresh logic
    Outcome: Auth working correctly
    """,
    type=MemoryType.EPISODIC,
    tags=["debugging", "auth", "session-2024-01-15"],
)
```

**Best for:** Session summaries, conversation context, debugging sessions

---

## Search Strategies

### Keyword Search (FTS)

Best for exact terms and session data:

```python
# Exact term search
results = ctx.search("PostgreSQL", semantic=False)

# Phrase search
results = ctx.search('"connection pool"', semantic=False)
```

### Semantic Search (RAG)

Best for conceptual queries and code:

```python
# Conceptual search
results = ctx.search("how to handle database connections")

# Code similarity
results = ctx.search("rate limiting implementation")
```

### Smart Search

Let ContextFS choose the best backend:

```python
# Automatic routing based on query and type
results = ctx.smart_search("deployment steps", type=MemoryType.PROCEDURAL)
# Routes to FTS (keyword-heavy)

results = ctx.smart_search("similar algorithm", type=MemoryType.CODE)
# Routes to RAG (semantic)
```

### Dual Search

Compare results from both backends:

```python
# Get results from both
results = ctx.search_both("authentication")

print("FTS Results:")
for r in results["fts"]:
    print(f"  {r.memory.summary}")

print("RAG Results:")
for r in results["rag"]:
    print(f"  {r.memory.summary}")
```

---

## Integration Patterns

### MCP Server (Claude Code, etc.)

```json
{
  "mcpServers": {
    "contextfs": {
      "command": "python",
      "args": ["-m", "contextfs.mcp_server"]
    }
  }
}
```

Available tools:
- `contextfs_save` - Save a memory
- `contextfs_search` - Semantic search
- `contextfs_recall` - Get specific memory by ID
- `contextfs_list` - List recent memories
- `contextfs_sessions` - List sessions
- `contextfs_load_session` - Load session messages

### Pre-commit Hook

Auto-save decisions from commit messages:

```bash
#!/bin/bash
# .git/hooks/commit-msg

MSG=$(cat "$1")

if [[ "$MSG" == *"DECISION:"* ]]; then
    python -c "
from contextfs import ContextFS, MemoryType
ctx = ContextFS()
ctx.save(
    content='$MSG',
    type=MemoryType.DECISION,
    tags=['git-commit', 'auto-captured']
)
"
fi
```

### CI/CD Integration

Save deployment context:

```yaml
# .github/workflows/deploy.yml
- name: Record deployment
  run: |
    python -c "
    from contextfs import ContextFS, MemoryType
    ctx = ContextFS(namespace_id='${{ github.repository }}')
    ctx.save(
        content='Deployed ${{ github.sha }} to production',
        type=MemoryType.EPISODIC,
        tags=['deployment', 'production', 'ci-cd']
    )
    "
```

---

## Best Practices

### 1. Capture Early, Capture Often

Save context as soon as you learn something:
- Made a decision? Save it.
- Fixed a bug? Document the fix.
- Found a useful pattern? Store it.

### 2. Use Meaningful Tags

Tags enable efficient filtering:
```python
# Good tags
tags=["auth", "jwt", "security", "api-v2"]

# Poor tags
tags=["stuff", "code", "thing"]
```

### 3. Write for Your Future Self

Include context that will help later:
```python
# Good: Explains why
content="""Use bcrypt for password hashing.
Rationale: Industry standard, adaptive cost factor,
built-in salt. Alternatives (SHA256, MD5) are too fast
and vulnerable to brute force."""

# Poor: Just states what
content="Use bcrypt"
```

### 4. Organize by Namespace

- One namespace per project/team
- Use `global` sparingly (truly universal knowledge)
- Consider repo-specific vs shared namespaces

### 5. Review and Prune

Periodically review memories:
```python
# Find old memories
old = ctx.list_recent(limit=100)
for m in old:
    if m.created_at < one_year_ago:
        # Review for relevance
        pass
```

### 6. Integrate with Workflow

- Add to IDE snippets
- Include in onboarding docs
- Reference in code reviews
- Use in sprint retrospectives

---

## Conclusion

Developer Memory Workflow transforms how you and your AI assistants work with code. By capturing decisions, bugs, patterns, and procedures, you build a persistent knowledge base that compounds over time.

Start small:
1. Save one decision per day
2. Document bug fixes as you make them
3. Search before you solve

Over time, you'll build an invaluable resource that accelerates development and reduces repeated work.

---

*For more information, visit the [ContextFS GitHub repository](https://github.com/MagnetonIO/contextfs) or the [documentation site](https://magnetonio.github.io/contextfs).*
