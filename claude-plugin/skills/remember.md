# Remember - Save Conversation Insights to Memory

Extract and save important information from this conversation to ContextFS long-term memory.

## TYPE-SAFE Memory Operations

**CRITICAL: Some types REQUIRE structured_data with specific fields.**

### Types WITH Required structured_data

| Type | Required Fields |
|------|-----------------|
| `decision` | `decision`, `rationale`, `alternatives[]` |
| `procedural` | `steps[]` (title, prerequisites optional) |
| `error` | `error_type`, `message`, `resolution` |
| `api` | `endpoint`, `method` |
| `config` | `name`, `settings{}` |

### Types WITHOUT Required structured_data

| Type | Use Case |
|------|----------|
| `fact` | Learned information about codebase/user |
| `episodic` | Session summaries, conversations |
| `code` | Code snippets, patterns |
| `user` | User preferences |

---

## Extraction Categories

### 1. Decisions Made (REQUIRED: structured_data)
```python
contextfs_save(
    type="decision",
    summary="Chose PostgreSQL over MongoDB",
    content="Full rationale and context...",
    tags=["decision", "<topic>"],
    structured_data={
        "decision": "Use PostgreSQL for user data",      # REQUIRED
        "rationale": "Team expertise, ACID compliance",  # REQUIRED
        "alternatives": ["MongoDB", "DynamoDB"]          # REQUIRED
    }
)
```

### 2. Errors and Solutions (REQUIRED: structured_data)
```python
contextfs_save(
    type="error",
    summary="ChromaDB collection not found",
    content="Full error context and stack trace...",
    tags=["error", "<technology>"],
    structured_data={
        "error_type": "CollectionNotFoundError",  # REQUIRED
        "message": "Collection does not exist",   # REQUIRED
        "resolution": "Reconnect MCP server"      # REQUIRED
    }
)
```

### 3. Procedures (REQUIRED: structured_data with steps[])
```python
contextfs_save(
    type="procedural",
    summary="Deploy to Railway",
    content="Detailed deployment procedure...",
    tags=["procedure", "<topic>"],
    structured_data={
        "title": "Railway Deployment",                                    # optional
        "steps": ["Build Docker", "Push to registry", "Redeploy"],        # REQUIRED
        "prerequisites": ["Docker installed", "Railway CLI configured"]   # optional
    }
)
```

### 4. Facts Learned (no structured_data required)
```python
contextfs_save(
    type="fact",
    summary="API uses JWT auth with 1hr expiry",
    content="The auth system uses JWT tokens...",
    tags=["fact", "<topic>"]
)
```

### 5. Code Patterns (no structured_data required)
```python
contextfs_save(
    type="code",
    summary="Auth middleware pattern",
    content="```python\ndef auth_middleware():\n    ...\n```",
    tags=["code", "<language>", "<pattern>"]
)
```

---

## Execution Steps

1. **Review the conversation** - Identify all memorable content

2. **Search before saving** - Check for duplicates:
   ```python
   contextfs_search(query="<topic>", limit=3)
   ```

3. **Evolve don't duplicate** - If similar memory exists, update it:
   ```python
   contextfs_evolve(memory_id="<existing_id>", new_content="Updated info...")
   ```

4. **Save new memories** - Use appropriate type WITH required structured_data

5. **Link related memories** - Connect to existing context:
   ```python
   contextfs_link(from_id="<new_id>", to_id="<existing_id>", relation="related_to")
   ```

6. **Save session** - Finally, save the session summary:
   ```python
   contextfs_save(save_session="current", label="<descriptive-label>")
   ```

---

## Output

After extraction, report:
- Number of memories saved by type
- Any memories that were evolved (updated) instead of created new
- Session label used

**Important**: Be thorough but avoid duplicating. Always search first, evolve if exists.
