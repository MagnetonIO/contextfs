# ContextFS Development Guidelines

## Git Workflow (GitFlow)
Always follow GitFlow for changes:
1. Create a new branch for changes (feature/*, bugfix/*, hotfix/*)
2. Make changes on the feature branch
3. **Validate work before committing** (run relevant tests, verify functionality)
4. Create PR to merge into main
5. Never commit directly to main

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

## Database Changes
- Core tables (memories, sessions): Use Alembic migrations in `src/contextfs/migrations/`
- Index tables (index_status, indexed_files, indexed_commits): Managed by AutoIndexer._init_db() directly, no migration needed
