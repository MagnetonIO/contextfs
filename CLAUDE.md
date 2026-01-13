# ContextFS Development Guidelines

## Git Workflow (GitFlow)
Always follow GitFlow for changes:
1. Create a new branch for changes (feature/*, bugfix/*, hotfix/*)
2. Make changes on the feature branch
3. **Validate work before committing** (run relevant tests, verify functionality)
4. Create PR to merge into main
5. Never commit directly to main

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

**Local Client (SQLite):**
- Migrations in `src/contextfs/migrations/versions/` (001-007)
- Core memory/session schema only
- Run automatically on CLI startup

**Cloud Service (PostgreSQL):**
- SQL scripts in `migrations/` (sync-*.sql)
- Models defined in `service/db/models.py`
- Run via Docker or direct SQL execution
- Includes: users, auth, subscriptions, teams, devices

## Documentation in Memory
**When adding new features, always save to contextfs memory:**
1. After implementing a new CLI command, MCP tool, or API endpoint, save to memory with type `api`
2. Use `contextfs_evolve` on memory ID `f9b4bb25` (API reference) to update the complete endpoint list
3. Include: endpoint/command name, parameters, and brief description
4. This keeps the API reference memory up-to-date for future sessions

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
