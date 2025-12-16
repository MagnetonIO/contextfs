# Claude Desktop Integration

ContextFS integrates with Claude Desktop via the Model Context Protocol (MCP).

## Installation

```bash
contextfs install-claude-desktop
```

This automatically configures Claude Desktop. Restart the app to activate.

## Manual Configuration

If automatic installation doesn't work, add to your Claude Desktop config:

=== "macOS"

    Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

    ```json
    {
      "mcpServers": {
        "contextfs": {
          "command": "/path/to/contextfs-mcp"
        }
      }
    }
    ```

=== "Windows"

    Edit `%APPDATA%\Claude\claude_desktop_config.json`:

    ```json
    {
      "mcpServers": {
        "contextfs": {
          "command": "C:\\path\\to\\contextfs-mcp.exe"
        }
      }
    }
    ```

=== "Linux"

    Edit `~/.config/Claude/claude_desktop_config.json`:

    ```json
    {
      "mcpServers": {
        "contextfs": {
          "command": "/path/to/contextfs-mcp"
        }
      }
    }
    ```

**Note:** ContextFS auto-detects the source tool (`claude-desktop` vs `claude-code`) based on environment. No manual configuration needed.

## Available Tools

After installation, Claude Desktop has access to these tools:

### `contextfs_save`

Save memories to the store.

```
Save to memory: We decided to use PostgreSQL for better JSON support.

Type: decision
Tags: database, architecture
```

### `contextfs_search`

Search memories semantically.

```
Search memory for: database decisions
```

### `contextfs_list`

List recent memories.

```
Show my recent memories
```

### `contextfs_recall`

Recall a specific memory by ID.

```
Recall memory abc123
```

### `contextfs_update`

Update an existing memory.

```
Update memory abc123 with new tags: auth, jwt, production
```

### `contextfs_delete`

Delete a memory by ID.

```
Delete memory abc123
```

### `contextfs_index`

Index the current repository.

```
Index this codebase for search
```

### `contextfs_index_status`

Check or cancel background indexing.

```
Check indexing progress
```

### `contextfs_sessions`

List conversation sessions.

```
Show recent sessions
```

### `contextfs_update_session`

Update session label or summary.

```
Update session xyz with label "feature-complete"
```

### `contextfs_delete_session`

Delete a session and its messages.

```
Delete session xyz
```

### `contextfs_import_conversation`

Import a JSON conversation export as episodic memory.

```
Import this conversation JSON to memory
```

### `contextfs_list_tools`

List source tools that have created memories.

```
Show which tools have saved memories
```

## Usage Examples

### Saving Decisions

> **You:** We just decided to use React 18 with TypeScript for the frontend. Save this to memory.
>
> **Claude:** I'll save that decision to memory.
>
> *Uses contextfs_save with type="decision", tags=["frontend", "react", "typescript"]*

### Searching Context

> **You:** What database decisions have we made?
>
> **Claude:** Let me search our memory for database decisions.
>
> *Uses contextfs_search with query="database decisions"*
>
> Found 3 relevant memories:
> 1. [decision] PostgreSQL for user data - JSON support
> 2. [decision] Redis for session caching
> 3. [fact] Database connection pool size is 20

### Building on Prior Context

> **You:** Implement user authentication following our patterns
>
> **Claude:** Let me check what authentication decisions we've made.
>
> *Uses contextfs_search with query="authentication patterns"*
>
> Based on our prior decisions, I'll implement using JWT with RS256...

## Prompts

ContextFS provides built-in prompts accessible via Claude Desktop:

### Session Guide

Helps Claude understand how to use ContextFS effectively:

```
Use the contextfs-session-guide prompt
```

### Index Repository

Guides through indexing a codebase:

```
Use the contextfs-index-repo prompt
```

## Best Practices

### 1. Save Important Decisions

When you make architectural decisions, explicitly save them:

> "Save to memory: We're using GraphQL instead of REST for the mobile API because of bandwidth constraints"

### 2. Search Before Implementing

Ask Claude to check memory before major implementations:

> "Before implementing auth, check our memory for any authentication decisions"

### 3. Use Tags Consistently

Establish tag conventions for your project:

- `auth`, `database`, `api` - feature areas
- `production`, `development` - environments
- `urgent`, `deprecated` - status

### 4. Summarize Sessions

At the end of important conversations:

> "Summarize what we accomplished and save it to memory"

## Troubleshooting

### Tools Not Appearing

1. Restart Claude Desktop completely
2. Check the config file syntax (valid JSON)
3. Verify `contextfs-mcp` is in your PATH:
   ```bash
   which contextfs-mcp
   ```

### Permission Errors

Ensure the MCP executable has proper permissions:

```bash
chmod +x $(which contextfs-mcp)
```

### Connection Issues

Check Claude Desktop logs:

- macOS: `~/Library/Logs/Claude/`
- Windows: `%LOCALAPPDATA%\Claude\logs\`

### Uninstall

```bash
contextfs install-claude-desktop --uninstall
```
