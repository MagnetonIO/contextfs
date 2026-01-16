# Recall - Search and Load Relevant Context

Search ContextFS memory for relevant context based on the current task or query.

## Available Tools

| Tool | Purpose |
|------|---------|
| `contextfs_search` | Hybrid semantic + keyword search |
| `contextfs_sessions` | List recent sessions |
| `contextfs_load_session` | Load session messages |
| `contextfs_recall` | Get specific memory by ID |
| `contextfs_list` | List recent memories |

---

## Search Strategy

### If Starting a New Task
- Search for prior work on the same topic
- Search for relevant decisions and procedures
- Search for known errors and solutions

### If Debugging
- Search for similar errors (type="error")
- Search for related code patterns
- Search for prior debugging sessions

### If Exploring Code
- Search for architectural decisions (type="decision")
- Search for code patterns (type="code")
- Search for API documentation (type="api")

---

## Search Patterns

### General Search
```python
contextfs_search(
    query="<topic or task>",
    limit=5,
    cross_repo=True  # Search across all repositories
)
```

### Type-Filtered Search
```python
# Find decisions
contextfs_search(query="<topic>", type="decision", limit=3)

# Find errors and solutions
contextfs_search(query="<error or symptom>", type="error", limit=5)

# Find procedures
contextfs_search(query="<workflow>", type="procedural", limit=3)

# Find code patterns
contextfs_search(query="<pattern>", type="code", limit=5)
```

### Project-Scoped Search
```python
contextfs_search(
    query="<topic>",
    project="my-project",  # Filter by project
    limit=5
)
```

---

## Session Management

### List Recent Sessions
```python
contextfs_sessions(limit=5, label="<optional filter>")
```

### Load Specific Session
```python
contextfs_load_session(session_id="<id>", max_messages=20)
```

---

## Get Specific Memory
```python
# Recall by ID (can use partial ID, min 8 chars)
contextfs_recall(id="abc12345")
```

---

## Output

Present findings organized by relevance:

1. **Most Relevant Memories** - Direct matches to the query
2. **Related Decisions** - Prior decisions that may affect current work
3. **Known Issues** - Errors and solutions that may be relevant
4. **Procedures to Follow** - Established workflows for similar tasks

Include memory IDs for reference so user can request more details.

---

## Example Usage

User: "I need to work on the authentication system"

Searches to run:
```python
# General search
contextfs_search(query="authentication system", limit=5, cross_repo=True)

# Decisions about auth
contextfs_search(query="auth", type="decision", limit=3)

# Known auth errors
contextfs_search(query="auth login", type="error", limit=3)

# Auth procedures
contextfs_search(query="authentication", type="procedural", limit=2)
```

Then synthesize findings into actionable context.
