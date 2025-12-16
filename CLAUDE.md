# ContextFS Development Guidelines

## Git Workflow (GitFlow)
Always follow GitFlow for changes:
1. Create a new branch for changes (feature/*, bugfix/*, hotfix/*)
2. Make changes on the feature branch
3. Create PR to merge into main
4. Never commit directly to main

## Search Strategy
Always search contextfs memories FIRST before searching code directly:
1. Use `contextfs_search` to find relevant memories
2. Only search code with Glob/Grep if memories don't have the answer
3. The repo is self-indexed - semantic search can find code snippets

## Database Changes
- Core tables (memories, sessions): Use Alembic migrations in `src/contextfs/migrations/`
- Index tables (index_status, indexed_files, indexed_commits): Managed by AutoIndexer._init_db() directly, no migration needed
