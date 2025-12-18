# Testing Memory Operations

This guide covers testing all ContextFS memory operations from basic CRUD to advanced lineage, merging, and graph operations.

## Prerequisites

```bash
cd /path/to/contextfs
pip install -e .
```

## Quick Start

```bash
# Check status
contextfs status
contextfs graph-status

# Run automated test
python scripts/test_memory_operations.py

# Or use the shell script
./scripts/test_memory_cli.sh
```

## Basic Operations

### Save Memories

```bash
# Save a fact
contextfs save "Database host is localhost:5432" --type fact --tags "db,config"

# Save a decision
contextfs save "Chose PostgreSQL for ACID compliance" --type decision --tags "database"

# Save with summary
contextfs save "API rate limit is 100/min" --type fact --tags "api" --summary "Rate limiting"
```

### List and Search

```bash
# List recent
contextfs list --limit 10

# Filter by type
contextfs list --type fact

# Search semantically
contextfs search "database configuration"

# Search with filters
contextfs search "api" --type fact --limit 5
```

### Recall and Delete

```bash
# Recall by ID (can use partial ID, min 8 chars)
contextfs recall abc12345

# Delete
contextfs delete abc12345
```

## Lineage Operations

### Evolve (Update with History)

Evolving creates a new memory linked to the original, preserving history.

```bash
# Create initial memory
contextfs save "Max upload size is 5MB" --type fact --tags "config"
# Output: Created memory abc12345...

# Evolve it (use ID from above)
contextfs evolve abc12345 "Max upload size increased to 10MB" --summary "Doubled limit"
# Output: Created memory def67890...

# Evolve again
contextfs evolve def67890 "Max upload size is 25MB for premium users" --tags "premium"
```

### View Lineage

```bash
# See full lineage (ancestors and descendants)
contextfs lineage <memory_id>

# Only ancestors (history)
contextfs lineage <memory_id> --direction ancestors

# Only descendants (what evolved from this)
contextfs lineage <memory_id> --direction descendants
```

### Merge Memories

Combine multiple memories into one, with configurable tag strategies.

```bash
# Create memories to merge
contextfs save "Frontend uses React 18" --type fact --tags "frontend,react"
contextfs save "Frontend uses TypeScript" --type fact --tags "frontend,typescript"

# Merge with union strategy (all tags)
contextfs merge <id1> <id2> --summary "Frontend stack" --strategy union

# Merge with custom content
contextfs merge <id1> <id2> --content "Stack: React 18 + TypeScript 5" --strategy union

# Merge strategies:
#   union: All tags from all memories (default)
#   intersection: Only common tags
#   latest: Tags from newest memory
#   oldest: Tags from oldest memory
```

### Split Memories

Divide a memory containing multiple topics into separate parts.

```bash
# Create a memory with multiple topics
contextfs save "Config: DB_HOST=localhost, REDIS_HOST=localhost" --type fact --tags "config"

# Split into parts
contextfs split <id> "DB_HOST=localhost" "REDIS_HOST=localhost" \
  --summaries "Database config|Redis config"
```

## Graph Operations

### Link Memories

Create relationships between memories.

```bash
# Create memories
contextfs save "UserService handles auth" --type fact --tags "service"
contextfs save "UserService uses Redis for sessions" --type fact --tags "service"

# Link them
contextfs link <user_service_id> <redis_id> related_to

# Bidirectional link
contextfs link <id1> <id2> related_to --bidirectional

# Available relations:
#   references, referenced_by, related_to, contradicts, supersedes,
#   superseded_by, part_of, contains, parent_of, child_of,
#   caused_by, causes, evolved_from, merged_from, split_from
```

### Find Related Memories

```bash
# Direct relationships
contextfs related <memory_id>

# Filter by relation type
contextfs related <memory_id> --relation depends_on

# Multi-hop traversal
contextfs related <memory_id> --depth 2
```

### Graph Status

```bash
contextfs graph-status
```

## Python API

```python
from contextfs import ContextFS

ctx = ContextFS()

# Save
m1 = ctx.save("API endpoint is /users", type="fact", tags=["api"])

# Evolve
m2 = ctx.evolve(m1.id, "API endpoint changed to /v2/users", summary="v2 migration")

# Get lineage
lineage = ctx.get_lineage(m2.id)
print(f"Ancestors: {lineage['ancestors']}")

# Link
m3 = ctx.save("Auth uses JWT", type="fact", tags=["auth"])
ctx.link(m2.id, m3.id, "references")

# Get related
related = ctx.get_related(m2.id)

# Merge
merged = ctx.merge([m1.id, m3.id], summary="API and auth")

# Split
parts = ctx.split(merged.id, ["Part 1", "Part 2"], summaries=["First", "Second"])

# Search
results = ctx.search("API endpoint")
```

## Backend Configuration

### SQLite (Default - Local)

```bash
# No configuration needed, works out of the box
export CONTEXTFS_BACKEND=sqlite
```

### PostgreSQL (Hosted)

```bash
export CONTEXTFS_BACKEND=postgres
export CONTEXTFS_POSTGRES_URL=postgresql://user:pass@host:5432/contextfs
```

### With FalkorDB (Advanced Graph)

```bash
# SQLite + FalkorDB
export CONTEXTFS_BACKEND=sqlite+falkordb
docker-compose up -d falkordb

# PostgreSQL + FalkorDB
export CONTEXTFS_BACKEND=postgres+falkordb
docker-compose up -d postgres falkordb
```

## Troubleshooting

### Reset Database

```bash
# Reset ChromaDB (vector search)
contextfs reset-chroma

# Rebuild ChromaDB from SQLite
contextfs rebuild-chroma
```

### Check Sync Status

```bash
contextfs status
# Shows SQLite and ChromaDB memory counts
```

## MCP Prompting (Claude Desktop / Claude Code)

Test memory operations via natural language prompts:

### Save Operations

```
Save a fact that the API rate limit is 100 requests per minute

Remember that we decided to use PostgreSQL for the database

Save this procedure: To deploy, run ./deploy.sh then check /health endpoint
```

### Search and Recall

```
Search my memories for anything about database configuration

What do I have saved about API endpoints?

Recall memory abc12345
```

### Evolve (Update with History)

```
The rate limit has changed - evolve my memory about rate limits to say it's now 500 per minute

Update the database memory - we've upgraded to PostgreSQL 16
```

### Lineage

```
Show me the history of changes to the rate limit memory

What's the lineage of memory abc12345?

How has our API documentation evolved over time?
```

### Linking

```
Link the auth memory to the session memory as related

Connect the deployment procedure to the CI/CD configuration

Mark these two memories as contradicting each other
```

### Find Related

```
What memories are related to the authentication config?

Find everything connected to the database setup

Show me memories linked to abc12345
```

### Merge

```
Merge all my frontend configuration memories into one

Combine the React and TypeScript memories into a single frontend stack memory

Consolidate my deployment-related memories
```

### Split

```
Split the environment config memory into separate memories for each variable

Break apart the all-in-one settings memory into individual configs
```

### Graph Status

```
What's the status of the memory graph?

Is the graph backend available?
```

## Test Coverage Matrix

| Operation    | CLI | Python | MCP Prompt |
|--------------|-----|--------|------------|
| Save         | ✅  | ✅     | ✅         |
| Search       | ✅  | ✅     | ✅         |
| Recall       | ✅  | ✅     | ✅         |
| Delete       | ✅  | ✅     | ✅         |
| Evolve       | ✅  | ✅     | ✅         |
| Get Lineage  | ✅  | ✅     | ✅         |
| Link         | ✅  | ✅     | ✅         |
| Get Related  | ✅  | ✅     | ✅         |
| Merge        | ✅  | ✅     | ✅         |
| Split        | ✅  | ✅     | ✅         |
| Graph Status | ✅  | ✅     | ✅         |

