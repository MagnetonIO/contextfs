# Sync - Synchronize Memories with Cloud

Sync local memories with ContextFS Cloud for backup and cross-device access.

## Tool Priority (MANDATORY)

**ALWAYS use MCP tools first. Only fall back to CLI if MCP fails.**

1. **First**: Use the `contextfs_sync()` MCP tool
2. **If MCP fails**: Fall back to CLI: `contextfs cloud sync --all`
3. **NEVER** use `python -m contextfs.cli` for user-facing commands — use `contextfs` directly

## Prerequisites

Before syncing, ensure you're logged in:
```bash
contextfs cloud login
contextfs cloud configure --enabled
```

## Usage

### Step 1: Try MCP Tool (Preferred)
```python
# Two-way sync (default)
contextfs_sync()

# Push only (local -> cloud)
contextfs_sync(direction="push")

# Pull only (cloud -> local)
contextfs_sync(direction="pull")

# Push all memories (not just changed)
contextfs_sync(push_all=True)
```

### Step 2: CLI Fallback (Only if MCP fails)
```bash
# Two-way sync
contextfs cloud sync

# Push all memories
contextfs cloud sync --all
```

## Sync Behavior

| Direction | What Happens |
|-----------|--------------|
| `push` | Upload local changes to cloud |
| `pull` | Download cloud changes to local |
| `both` | Push then pull (default) |

### What Gets Synced
- All memory types (facts, decisions, procedures, errors, etc.)
- Memory metadata (tags, timestamps, relationships)
- Session summaries

### What Doesn't Sync
- Local ChromaDB embeddings (regenerated on pull)
- Temporary/draft memories

## Status Check

Check sync status:
```bash
contextfs cloud status
```

## Troubleshooting

### "Cloud sync is disabled"
Run: `contextfs cloud configure --enabled`

### "No API key configured"
Run: `contextfs cloud login`

### Sync conflicts
Cloud version wins by default. Local changes are preserved with `_local` suffix.

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `contextfs_sync()` | Two-way sync via MCP |
| `contextfs cloud sync` | Two-way sync via CLI |
| `contextfs cloud status` | Check sync status |
| `contextfs cloud login` | Authenticate with cloud |
