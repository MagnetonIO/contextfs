# Claude Code Integration

ContextFS integrates seamlessly with Claude Code for persistent memory across coding sessions.

## Setup

Add ContextFS to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "contextfs": {
      "command": "contextfs-mcp",
      "env": {
        "CONTEXTFS_SOURCE_TOOL": "claude-code"
      }
    }
  }
}
```

Or use the Python module directly:

```json
{
  "mcpServers": {
    "contextfs": {
      "command": "python",
      "args": ["-m", "contextfs.mcp_server"],
      "env": {
        "CONTEXTFS_SOURCE_TOOL": "claude-code"
      }
    }
  }
}
```

## Available Tools

### Memory Operations

| Tool | Description |
|------|-------------|
| `contextfs_save` | Save memories with type and tags |
| `contextfs_search` | Semantic search over memories |
| `contextfs_list` | List recent memories |
| `contextfs_recall` | Recall by ID |
| `contextfs_update` | Update existing memory content, type, tags, or project |
| `contextfs_delete` | Delete a memory by ID |

### Repository Operations

| Tool | Description |
|------|-------------|
| `contextfs_init` | Initialize repo for auto-indexing (creates `.contextfs/config.yaml`) |
| `contextfs_index` | Index codebase for search |
| `contextfs_index_status` | Check or cancel background indexing |
| `contextfs_list_indexes` | List all indexed repositories with stats |
| `contextfs_list_repos` | List indexed repositories |
| `contextfs_list_tools` | List source tools (claude-code, claude-desktop, etc.) |
| `contextfs_list_projects` | List project groupings |

### Session Operations

| Tool | Description |
|------|-------------|
| `contextfs_sessions` | List sessions |
| `contextfs_load_session` | Load session context |
| `contextfs_message` | Log session message |
| `contextfs_update_session` | Update session label or summary |
| `contextfs_delete_session` | Delete session and its messages |
| `contextfs_import_conversation` | Import JSON conversation as episodic memory |

## Lifecycle Hooks

ContextFS can install Claude Code hooks for automatic context capture:

```bash
# Install hooks and MCP server
python -c "from contextfs.plugins.claude_code import install_claude_code; install_claude_code()"
```

This installs:

| Hook | Action |
|------|--------|
| `SessionStart` | Auto-index initialized repos in background |
| `PreCompact` | Save session before context compaction |

### Opt-in Indexing

The SessionStart hook only indexes repos that have been initialized with `contextfs init`:

```bash
# Initialize a repo for auto-indexing
contextfs init

# Now SessionStart will auto-index this repo
```

This prevents unwanted indexing of random directories. The hook runs:
```bash
contextfs index --quiet --background --require-init
```

### Manual Hook Configuration

To manually configure hooks in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "uvx contextfs index --quiet --background --require-init"
      }]
    }],
    "PreCompact": [{
      "hooks": [{
        "type": "command",
        "command": "uvx contextfs save-session --label 'auto-compact'"
      }]
    }]
  }
}
```

## Workflow Examples

### Starting a New Feature

```
1. Search for related prior work:
   contextfs_search("authentication implementation")

2. Index the codebase if not already:
   contextfs_index()

3. Implement the feature with context from prior decisions
```

### Debugging Session

```
1. Search for similar errors:
   contextfs_search("connection timeout error", type="error")

2. After fixing, save the solution:
   contextfs_save(
     "Fixed connection timeout by increasing pool size to 50",
     type="error",
     tags=["database", "connection", "timeout"]
   )
```

### Code Review

```
1. Search for coding standards:
   contextfs_search("coding standards", type="decision")

2. Search for similar patterns:
   contextfs_search("validation patterns", type="code")
```

## Session Management

Track what happens in Claude Code sessions:

```python
# Claude Code automatically tracks sessions
# Sessions include:
# - User messages
# - Assistant responses
# - Tool calls and results
# - Files modified
```

### Load Previous Session

Continue from a previous conversation:

```
Load the session from yesterday about OAuth implementation
```

### Search Session History

Find relevant past conversations:

```
contextfs_sessions(label="auth")
```

## Best Practices

### 1. Initialize Repos for Auto-Indexing

Initialize repositories you work with frequently:

```bash
# One-time setup per repo
contextfs init
```

This enables automatic indexing when you start Claude Code sessions.

### 2. Index Early

Index your repository at the start of a project:

```
Please index this repository so we can search the codebase
```

### 3. Save Decisions Explicitly

When making important choices:

```
We decided to use SQLAlchemy 2.0 with async support.
Please save this decision with tags: database, orm, async
```

### 4. Reference Prior Context

Before implementing:

```
Before we implement the payment system, search our memory
for any payment-related decisions or patterns
```

### 5. Document Errors

When you fix bugs:

```
Save this fix to memory as an error type:
"TypeError in user serialization - fixed by adding null check"
```

### 6. Use Projects for Multi-Repo

Group related repositories:

```
Save this with project="my-saas" so it's findable from other repos
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CONTEXTFS_SOURCE_TOOL` | Tool identifier | auto-detected |
| `CONTEXTFS_DATA_DIR` | Data directory | `~/.contextfs` |
| `CONTEXTFS_PROJECT` | Default project | auto-detect |

**Note:** ContextFS automatically detects whether it's running under Claude Code or Claude Desktop based on terminal environment indicators (`TERM`, `SHELL`). You only need to set `CONTEXTFS_SOURCE_TOOL` to override auto-detection.

## Troubleshooting

### MCP Server Not Starting

Check that contextfs is installed:

```bash
which contextfs-mcp
# or
python -m contextfs.mcp_server --help
```

### Memories Not Persisting

Verify the data directory:

```bash
ls ~/.contextfs/
# Should contain: context.db, chroma/
```

### Search Not Finding Results

Check if indexing completed:

```bash
contextfs status
```
