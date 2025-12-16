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
| `contextfs_index` | Index codebase for search |
| `contextfs_index_status` | Check or cancel background indexing |
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

## Auto-Save Hooks

Configure Claude Code to automatically save sessions:

```bash
# In your shell config (.bashrc, .zshrc)
export CLAUDE_CODE_HOOKS='{"post_session": "contextfs save-session"}'
```

Or create a hook script:

```bash
#!/bin/bash
# ~/.claude-code/hooks/post_session.sh
contextfs save-session --label "$CLAUDE_SESSION_ID"
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

### 1. Index Early

Index your repository at the start of a project:

```
Please index this repository so we can search the codebase
```

### 2. Save Decisions Explicitly

When making important choices:

```
We decided to use SQLAlchemy 2.0 with async support.
Please save this decision with tags: database, orm, async
```

### 3. Reference Prior Context

Before implementing:

```
Before we implement the payment system, search our memory
for any payment-related decisions or patterns
```

### 4. Document Errors

When you fix bugs:

```
Save this fix to memory as an error type:
"TypeError in user serialization - fixed by adding null check"
```

### 5. Use Projects for Multi-Repo

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
