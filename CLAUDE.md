# ContextFS Development Guidelines

## Configuration Rules (CRITICAL)
**NEVER use static/hardcoded values for configuration. ALWAYS use environment variables.**

All configuration values must be:
1. Defined in `config.py` with the `CONTEXTFS_` prefix
2. Configurable via environment variables
3. Never hardcoded in application code

Examples:
```python
# WRONG - hardcoded values
chroma_port = 8000
mcp_port = 8003

# CORRECT - from config (loaded from env)
from contextfs.config import get_config
config = get_config()
chroma_port = config.chroma_port  # CONTEXTFS_CHROMA_PORT
mcp_port = config.mcp_port        # CONTEXTFS_MCP_PORT
```

Key environment variables:
- `CONTEXTFS_CHROMA_HOST` - ChromaDB server host (default: localhost)
- `CONTEXTFS_CHROMA_PORT` - ChromaDB server port (default: 8000)
- `CONTEXTFS_MCP_PORT` - MCP server port (default: 8003)
- `CONTEXTFS_DATA_DIR` - Data directory (default: ~/.contextfs)

## Type System Enforcement (CRITICAL)
**ALWAYS use typed enums and classes. NEVER use raw strings for typed fields.**

The Memory model has a `type: MemoryType` field. Always use the `MemoryType` enum:

```python
# WRONG - raw string (breaks type safety)
ctx.save(content="...", type="task")
ctx.save(content="...", type="decision")

# CORRECT - use MemoryType enum
from contextfs.schemas import MemoryType

ctx.save(content="...", type=MemoryType.TASK)
ctx.save(content="...", type=MemoryType.DECISION)
```

### Type Safety Rules

1. **Import enums from schemas**: `from contextfs.schemas import MemoryType`
2. **Use enum values, not strings**: `MemoryType.TASK` not `"task"`
3. **Run mypy before committing**: `python -m mypy src/contextfs/ --ignore-missing-imports`
4. **Handle Optional types**: Check for `None` before accessing `Optional[T]` fields

### Available Memory Types

```python
# Core types
MemoryType.FACT      # Static facts, configurations
MemoryType.DECISION  # Architectural/design decisions
MemoryType.PROCEDURAL  # How-to procedures
MemoryType.EPISODIC  # Session/conversation memories
MemoryType.USER      # User preferences
MemoryType.CODE      # Code snippets
MemoryType.ERROR     # Runtime errors, stack traces
MemoryType.COMMIT    # Git commit history

# Extended types
MemoryType.TODO      # Tasks, work items
MemoryType.ISSUE     # Bugs, problems, tickets
MemoryType.API       # API endpoints, contracts
MemoryType.SCHEMA    # Data models, DB schemas
MemoryType.TEST      # Test cases, coverage
MemoryType.REVIEW    # PR feedback, code reviews
MemoryType.RELEASE   # Changelogs, versions
MemoryType.CONFIG    # Environment configs
MemoryType.DEPENDENCY  # Package versions
MemoryType.DOC       # Documentation

# Workflow/Agent types
MemoryType.WORKFLOW  # Multi-step workflows
MemoryType.TASK      # Individual workflow tasks
MemoryType.STEP      # Execution steps within tasks
MemoryType.AGENT_RUN # LLM agent execution records
```

## Git Workflow (GitFlow)
Always follow GitFlow for changes:
1. Create a new branch for changes (feature/*, bugfix/*, hotfix/*)
2. Make changes on the feature branch
3. **Validate work before committing** (run relevant tests, verify functionality)
4. Create PR to merge into main
5. Never commit directly to main

### Release Workflow
When the user says **"release"**, execute this full workflow:

```bash
# 1. Commit all changes on current branch (develop or feature)
git add -A && git commit -m "description"
git push origin <branch>

# 2. If on feature branch, merge to develop first
git checkout develop && git merge <feature-branch> && git push origin develop

# 3. Run tests locally
uv run pytest tests/ -x -q

# 4. Merge develop to main
git checkout main && git pull origin main
git merge develop --no-edit && git push origin main

# 5. Switch back to develop for continued work
git checkout develop && git merge main && git push origin develop
```

This ensures:
- All changes flow through develop before main
- Tests pass before merging to main
- main is always in a releasable state
- develop stays in sync with main after release

## Releasing New Versions (MANDATORY)
**ALWAYS use the release script to create new versions:**

```bash
./scripts/release.sh <version>
# Example: ./scripts/release.sh 0.2.15
```

The script handles:
1. Version format validation (X.Y.Z)
2. Checks for uncommitted changes
3. Updates pyproject.toml and __init__.py
4. Commits with proper message
5. Creates git tag and pushes

**NEVER manually edit version files or create tags.** Always use the script.

## Testing Requirements
**Each feature must have a test. Tests must pass locally before committing.**

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/integration/test_autoindex.py -x -q

# Run with coverage
pytest tests/ --cov=contextfs
```

### Test Guidelines
1. **Every new feature needs a test** - No exceptions
2. **Run tests locally before committing** - Avoid CI failures
3. **Tests must work without optional dependencies** - Use `auto` mode for embedding backend
4. **Fix failing tests before pushing** - Don't break the build

### Common CI Failures to Avoid
- **FastEmbed not installed**: Use `embedding_backend: str = "auto"` (falls back to sentence_transformers)
- **Missing test fixtures**: Ensure pytest fixtures are properly scoped
- **Database state**: Tests should be isolated, use temp directories

## Validation Before Commit
Before committing any changes:
1. Run relevant tests: `pytest tests/` or specific test files
2. Verify the fix/feature works as expected
3. Check for regressions in related functionality

## Search Strategy
Always search contextfs memories FIRST before searching code directly:
1. Use `contextfs_search` to find relevant memories
2. Only search code with Glob/Grep if memories don't have the answer
3. The repo is self-indexed - semantic search can find code snippets

## Database Architecture

### CRITICAL: PostgreSQL ONLY for Hosted Services
**NEVER use SQLite for hosted/cloud services.** The sync service (`service/`) MUST use PostgreSQL exclusively.

### SQLite vs PostgreSQL

| Component | Database | Location | Purpose |
|-----------|----------|----------|---------|
| **Local CLI** | SQLite | `~/.contextfs/context.db` | Local memory storage, caching |
| **Cloud Sync Service** | PostgreSQL | Docker/Cloud | Source of truth for all data |

### What Lives Where

**SQLite (Local Client - `src/contextfs/`):**
- Local memories and sessions (user's machine)
- Local ChromaDB embeddings
- Memory edges (relationships)
- Sync state (for tracking what's been synced)
- Index status for auto-indexing

**PostgreSQL (Cloud Service - `service/`):**
- User accounts (source of truth)
- API keys and authentication
- Subscriptions and billing (Stripe integration)
- Team management (teams, members, invitations)
- Device registration and limits
- Synced memories (cloud copies)
- Usage tracking

**NOT in SQLite:**
- Users, API keys, subscriptions (PostgreSQL only)
- Teams, team members, invitations (PostgreSQL only)
- Devices (PostgreSQL only)

### Database Migrations

**CRITICAL: Always use Alembic for database migrations.** Both local and cloud services use Alembic for version-controlled, safe schema changes.

**Local Client (SQLite):**
- Migrations in `src/contextfs/migrations/versions/` (001-007)
- Core memory/session schema only
- Run automatically on CLI startup via `run_migrations()`

**Cloud Service (PostgreSQL):**
- Migrations in `service/migrations/versions/`
- Models defined in `service/db/models.py`
- Run automatically on service startup via `run_migrations()`
- Includes: users, auth, subscriptions, teams, devices

### Creating New Migrations

**Local Client:**
```bash
cd src/contextfs
alembic revision -m "description_of_change"
# Edit the generated file in migrations/versions/
```

**Cloud Service:**
```bash
cd service
alembic revision -m "description_of_change"
# Edit the generated file in migrations/versions/
```

### Migration Best Practices
1. **Never use raw SQL files** - Always use Alembic Python migrations
2. **Test locally first** - Run `docker restart contextfs-sync` before Railway deploy
3. **Include downgrade** - Every migration should have a working `downgrade()` function
4. **One change per migration** - Keep migrations atomic and focused

## Documentation in Memory
**When adding new features, always save to contextfs memory:**
1. After implementing a new CLI command, MCP tool, or API endpoint, save to memory with type `api`
2. Use `contextfs_evolve` on memory ID `f9b4bb25` (API reference) to update the complete endpoint list
3. Include: endpoint/command name, parameters, and brief description
4. This keeps the API reference memory up-to-date for future sessions

## Saving Plans to Memory (TYPE-SAFE)
**After completing a plan, always save it to contextfs memory:**
1. Save the plan with `type="procedural"` and **required** `structured_data`
2. The `steps` array is REQUIRED for procedural type
3. This preserves implementation context for future sessions

Example:
```python
contextfs_save(
    type="procedural",
    summary="Feature X implementation plan",
    content="Detailed implementation notes...",
    tags=["plan", "feature-x"],
    structured_data={
        "title": "Feature X Implementation",
        "steps": [
            "Create new module in src/",
            "Add API endpoint",
            "Write tests",
            "Update documentation"
        ],
        "prerequisites": ["Read existing codebase"],
        "notes": "Key decisions: Used PostgreSQL, followed existing patterns"
    }
)
```

## ChromaDB and MCP Server Testing
**The MCP server caches ChromaDB collection references.** If you get "Collection does not exist" errors, the fix is usually just reconnecting MCP - NOT rebuilding ChromaDB.

### When MCP Tools Fail with Collection Errors
**FIRST: Try reconnecting MCP (don't rebuild!):**
1. Run `/mcp` in Claude Code to see MCP server status
2. Disconnect and reconnect the contextfs MCP server
3. Or restart Claude Code entirely

**Rebuilding ChromaDB should be a last resort** - it's slow and usually unnecessary.

### Avoiding ChromaDB Issues During Testing
1. **Use CLI for testing, not MCP tools**: `python -m contextfs.cli search "query"` instead of MCP `contextfs_search`
2. **Never rebuild ChromaDB while MCP server is running** - it will cache stale collection IDs
3. **If MCP fails**: Try `/mcp` reconnect FIRST before any rebuild

### Only If Reconnect Doesn't Work
```bash
# rebuild-chroma preserves ALL data (rebuilds from SQLite, no re-indexing needed)
echo "y" | python -m contextfs.cli rebuild-chroma

# Then reconnect MCP via /mcp command
```

### Testing Best Practices
- Always use `python -m contextfs.cli` for testing (not `contextfs` or `uv run contextfs`)
- This ensures you're testing the local code, not an installed version
- The CLI creates fresh ChromaDB connections, avoiding cache issues

## Railway Deployment (ALWAYS USE DOCKER)

**CRITICAL: Always deploy to Railway using Docker images, NOT `railway up`.**

### Deployment Steps (Automatic via GitHub Actions)

1. **Push to main branch** - GitHub Actions automatically builds and pushes Docker image
2. **Redeploy Railway:**
```bash
railway redeploy --service sync-api-prod --yes
```

### Manual Build (if needed)
```bash
cd /path/to/contextfs
docker build -f docker/Dockerfile.sync -t magnetonio/contextfs-sync:latest .
docker push magnetonio/contextfs-sync:latest
railway redeploy --service sync-api-prod --yes
```

### Railway Configuration
- **Project**: contextfs-sync
- **Service**: sync-api-prod
- **Image**: `magnetonio/contextfs-sync:latest`
- **Database**: PostgreSQL (postgres-pgvector)

### SSH into Railway (for debugging)
```bash
railway ssh --service sync-api-prod -- 'command'
```

### Check logs
```bash
railway logs --service sync-api-prod
railway logs --service sync-api-prod --build
```

### Environment Variables (Railway)
Key variables configured in Railway:
- `CONTEXTFS_POSTGRES_URL` - PostgreSQL connection string
- `STRIPE_SECRET_KEY` - Stripe live/test key
- `STRIPE_WEBHOOK_SECRET` - Webhook signing secret
- `STRIPE_PRICE_PRO` / `STRIPE_PRICE_TEAM` - Price IDs

### Why Docker Instead of `railway up`
- `railway up` uploads source and builds remotely - often times out
- Docker images are pre-built and deploy instantly
- More reliable and consistent deployments

<!-- CONTEXTFS_MEMORY_START -->
# ContextFS Memory Management Protocol

## Memory Operations (MANDATORY)

You have access to ContextFS for persistent memory across sessions. Use these operations actively throughout your work - memory management is infrastructure, not optional.

### When to Save Memories (TYPE-SAFE)

**ALWAYS save immediately when you:**

1. **Learn a new fact** about the codebase, architecture, or user preferences
   - Use: `contextfs_save` with `type="fact"` (no structured_data required)
   - Example: `contextfs_save(type="fact", summary="API uses JWT", content="...")`

2. **Make or discover a decision** with rationale
   - Use: `contextfs_save` with `type="decision"`
   - **REQUIRED structured_data**: `{decision, rationale, alternatives[]}`
   - Example: `structured_data={"decision": "Use PostgreSQL", "rationale": "JSON support", "alternatives": ["MySQL"]}`

3. **Encounter and solve an error**
   - Use: `contextfs_save` with `type="error"`
   - **REQUIRED structured_data**: `{error_type, message, resolution}`
   - Example: `structured_data={"error_type": "CollectionNotFound", "message": "...", "resolution": "Reconnect MCP"}`

4. **Discover a procedure or workflow**
   - Use: `contextfs_save` with `type="procedural"`
   - **REQUIRED structured_data**: `{steps[]}`
   - Example: `structured_data={"title": "Deploy to Railway", "steps": ["Build", "Push", "Deploy"]}`

5. **Complete a significant task**
   - Use: `contextfs_save` with `type="episodic"` (no structured_data required)
   - Or save session: `contextfs_save(save_session="current", label="task-name")`

6. **Complete a plan or implementation**
   - Use: `contextfs_save` with `type="procedural"` and **required** `steps[]`
   - Include files modified and key decisions in the steps or notes

7. **Create or update research papers**
   - Use: `contextfs_save` with `type="doc"`
   - Save summary of paper, key concepts, and file location
   - Example: `contextfs_save(type="doc", summary="Paper title", content="Overview and key concepts...", tags=["research", "topic"])`

### When to Search Memories

**ALWAYS search FIRST when you:**

1. **Start a new task** - Check for prior context
   ```
   contextfs_search(query="<task topic>", limit=5)
   ```

2. **Encounter an error** - Check if it's been solved before
   ```
   contextfs_search(query="<error message>", type="error", limit=5)
   ```

3. **Need to make a decision** - Check for prior decisions on the topic
   ```
   contextfs_search(query="<decision topic>", type="decision", limit=5)
   ```

4. **Explore unfamiliar code** - Check for documented patterns
   ```
   contextfs_search(query="<code area>", type="code", limit=10)
   ```

### Memory Type Reference (TYPE-SAFE)

Types with required `structured_data` schemas:

| Type | Required structured_data | Example |
|------|-------------------------|---------|
| `decision` | `{decision, rationale, alternatives[]}` | Decision with rationale |
| `procedural` | `{steps[], title?, prerequisites[]?}` | Step-by-step workflows |
| `error` | `{error_type, message, resolution}` | Errors and solutions |
| `api` | `{endpoint, method, parameters[]?}` | API endpoints |
| `config` | `{name, environment?, settings{}}` | Configuration details |

Types WITHOUT required structured_data (simple content only):

| Type | When to Use | Example |
|------|------------|---------|
| `fact` | Learned information about codebase/user | "API uses JWT auth" |
| `episodic` | Session summaries, conversations | "Debugged auth flow" |
| `code` | Code snippets, patterns | "Auth middleware pattern" |
| `user` | User preferences | "Prefers TypeScript" |

### Session Management

**Before ending a session:**
1. Save important learnings that haven't been saved yet
2. Use `contextfs_save` with `save_session="current"` and a descriptive label
3. Or use `/remember` skill for automatic extraction

**At session start:**
1. Search for relevant context based on working directory
2. Load previous session if continuing work: `contextfs_load_session`

### Memory Evolution

When knowledge updates or corrections are needed:
- Use `contextfs_evolve` to update existing memories (preserves history)
- Use `contextfs_link` to connect related memories
- Use `contextfs_merge` to consolidate duplicate information

### Quick Reference Commands (TYPE-SAFE)

```python
# Save a fact (no structured_data required)
contextfs_save(content="...", type="fact", tags=["topic"], summary="Brief summary")

# Save a decision (REQUIRED: structured_data with decision, rationale)
contextfs_save(
    type="decision",
    summary="Chose X over Y",
    content="Detailed reasoning...",
    structured_data={
        "decision": "Use PostgreSQL",
        "rationale": "Better JSON support",
        "alternatives": ["MySQL", "MongoDB"]
    }
)

# Save a procedure (REQUIRED: structured_data with steps array)
contextfs_save(
    type="procedural",
    summary="How to deploy",
    content="Deployment procedure...",
    structured_data={
        "title": "Deploy to Railway",
        "steps": ["Build Docker image", "Push to registry", "Redeploy service"],
        "prerequisites": ["Docker installed", "Railway CLI configured"]
    }
)

# Save an error (REQUIRED: structured_data with error_type, message, resolution)
contextfs_save(
    type="error",
    summary="ChromaDB collection error",
    content="Full error context...",
    structured_data={
        "error_type": "CollectionNotFound",
        "message": "Collection does not exist",
        "resolution": "Reconnect MCP server"
    }
)

# Search before acting
contextfs_search(query="relevant topic", limit=5, cross_repo=True)

# Save session at end
contextfs_save(save_session="current", label="descriptive-label")
```

## Integration with Development Workflow

1. **Before coding**: Search memory for patterns, decisions, gotchas
2. **During coding**: Save errors encountered and solutions found
3. **After coding**: Save decisions made and lessons learned
4. **Before committing**: Ensure important context is saved to memory

Memory is your persistent context. Use it actively, not as an afterthought.

<!-- CONTEXTFS_MEMORY_END -->
