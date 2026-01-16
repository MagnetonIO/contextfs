# Sync - Synchronize Memories with Cloud

Sync local memories with ContextFS Cloud for backup and cross-device access.

## Prerequisites

Before syncing, ensure you're logged in:
```bash
contextfs cloud login
contextfs cloud configure --enabled
```

## Usage

Use the MCP tool or CLI to sync:

### MCP Tool
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

### CLI
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
