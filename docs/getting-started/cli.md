# CLI Reference

ContextFS provides a full-featured command-line interface. The main command is `contextfs` with a short alias `ctx`.

## Commands

### `contextfs save`

Save a memory to the store.

```bash
contextfs save CONTENT [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--type, -t` | Memory type: fact, decision, procedural, episodic, code, error, user |
| `--tags` | Comma-separated tags |
| `--summary, -s` | Brief summary |

**Examples:**

```bash
contextfs save "Use React 18 with TypeScript" --type decision --tags frontend,react

contextfs save "Deploy with: docker compose up -d" --type procedural --summary "Deployment command"
```

### `contextfs search`

Search memories using semantic similarity.

```bash
contextfs search QUERY [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--limit, -n` | Maximum results (default: 10) |
| `--type, -t` | Filter by memory type |

**Examples:**

```bash
contextfs search "how to deploy"
contextfs search "database" --type decision --limit 5
```

### `contextfs init`

Initialize a repository for ContextFS indexing. Creates a `.contextfs/config.yaml` marker file that opts the repo into automatic indexing.

```bash
contextfs init [PATH] [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--no-index` | Don't run index after init |
| `--auto-index/--no-auto-index` | Enable/disable auto-indexing on session start (default: enabled) |
| `--max-commits` | Maximum commits to index (default: 100) |
| `--force, -f` | Reinitialize even if already initialized |
| `--quiet, -q` | Minimal output |

**Examples:**

```bash
# Initialize current repo for ContextFS
contextfs init

# Initialize without immediate indexing
contextfs init --no-index

# Initialize with custom settings
contextfs init --max-commits 500 --no-auto-index

# Reinitialize existing repo
contextfs init --force
```

The config file (`.contextfs/config.yaml`) controls:
- `auto_index`: Whether SessionStart hooks should auto-index this repo
- `max_commits`: How much commit history to index

### `contextfs index`

Index a repository for semantic code search.

```bash
contextfs index [PATH] [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--force, -f` | Force re-index even if already indexed |
| `--incremental/--full` | Incremental (default) or full re-index |
| `--background` | Run indexing in background subprocess |
| `--quiet, -q` | Suppress output |
| `--require-init` | Only index if repo has `.contextfs/config.yaml` (used by hooks) |
| `--mode` | Index mode: `all`, `files_only`, or `commits_only` |

**Examples:**

```bash
# Index current repo
contextfs index

# Index specific path
contextfs index /path/to/repo

# Force full re-index
contextfs index --force --full

# Background indexing (for hooks)
contextfs index --quiet --background --require-init
```

### `contextfs list`

List recent memories.

```bash
contextfs list [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--limit, -n` | Maximum results (default: 10) |
| `--type, -t` | Filter by memory type |

### `contextfs recall`

Recall a specific memory by ID.

```bash
contextfs recall MEMORY_ID
```

The ID can be partial (first 8 characters).

### `contextfs delete`

Delete a memory.

```bash
contextfs delete MEMORY_ID [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--yes, -y` | Skip confirmation |

### `contextfs sessions`

List recent sessions.

```bash
contextfs sessions [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--limit, -n` | Maximum results (default: 10) |
| `--tool` | Filter by tool (claude-code, gemini, etc.) |
| `--label` | Filter by label |

### `contextfs status`

Show ContextFS status and statistics.

```bash
contextfs status
```

Displays:
- Data directory location
- Current namespace
- Memory counts by type
- Vector store statistics
- Active session info

### `contextfs serve`

Start the MCP server.

```bash
contextfs serve
```

Typically not run directly - used by Claude Desktop integration.

### `contextfs web`

Start the web UI server.

```bash
contextfs web [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--host, -h` | Host to bind (default: 127.0.0.1) |
| `--port, -p` | Port to bind (default: 8000) |

### `contextfs install-claude-desktop`

Install/uninstall MCP server for Claude Desktop.

```bash
contextfs install-claude-desktop [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--uninstall` | Remove from Claude Desktop |

### `contextfs save-session`

Save session for use with hooks.

```bash
contextfs save-session [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--label, -l` | Session label |
| `--transcript, -t` | Path to transcript JSONL file |

### `contextfs chroma-server`

Manage the ChromaDB server for multi-process safe access.

```bash
contextfs chroma-server [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--daemon` | Start server in background |
| `--status` | Check if server is running |
| `--install` | Install as system service (auto-start on boot) |
| `--uninstall` | Remove system service |
| `--port` | Port to use (default: 8000) |
| `--path` | Data directory path |

**Examples:**

```bash
# Start ChromaDB server in background
contextfs chroma-server --daemon

# Check server status
contextfs chroma-server --status

# Install as system service (macOS/Linux/Windows)
contextfs chroma-server --install

# Uninstall system service
contextfs chroma-server --uninstall
```

**Why use ChromaDB server mode?**

When multiple MCP clients (Claude Code, Claude Desktop) access ChromaDB simultaneously, the embedded mode can cause corruption. Server mode provides:
- Safe multi-process access
- Auto-recovery from connection issues
- Persistent service across reboots (with `--install`)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CONTEXTFS_DATA_DIR` | Data storage directory | `~/.contextfs` |
| `CONTEXTFS_SOURCE_TOOL` | Tool identifier | auto-detect |
| `CONTEXTFS_EMBEDDING_MODEL` | Embedding model | `all-MiniLM-L6-v2` |
| `CONTEXTFS_CHROMA_HOST` | ChromaDB server host (enables server mode) | - |
| `CONTEXTFS_CHROMA_PORT` | ChromaDB server port | `8000` |
| `CONTEXTFS_EMBEDDING_BACKEND` | Embedding backend: `auto`, `fastembed`, `sentence_transformers` | `auto` |

## Shell Completion

Install shell completion:

```bash
# Bash
contextfs --install-completion bash

# Zsh
contextfs --install-completion zsh

# Fish
contextfs --install-completion fish
```
