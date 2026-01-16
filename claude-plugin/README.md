# ContextFS Plugin for Claude Code

Persistent memory management for Claude Code - save, search, and recall context across sessions.

## Features

- **Persistent Memory**: Save decisions, procedures, errors, and facts to long-term storage
- **Semantic Search**: Find relevant memories using hybrid keyword + semantic search
- **Session Management**: Automatically save and restore session context
- **Cross-Repo Support**: Share knowledge across multiple repositories
- **MCP Integration**: Full MCP server with 15+ tools

## Installation

### Option 1: Install via Claude Plugin Registry

```bash
claude plugin add contextfs
```

### Option 2: Install via npm

```bash
npm install -g claude-plugin-contextfs
claude plugin add claude-plugin-contextfs
```

### Option 3: Install via ContextFS CLI (Recommended)

```bash
# Install ContextFS
pip install contextfs

# Install Claude Code integration
contextfs install claude
```

## Prerequisites

The plugin requires the ContextFS Python package to be installed:

```bash
pip install contextfs
```

## Usage

### Starting the MCP Server

Before using the plugin, start the MCP server:

```bash
contextfs server
```

### Available Skills

- **/remember** - Extract and save important information from conversations
- **/recall** - Search memories for relevant context

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `contextfs_save` | Save a new memory |
| `contextfs_search` | Search memories using hybrid search |
| `contextfs_recall` | Get a specific memory by ID |
| `contextfs_list` | List recent memories |
| `contextfs_evolve` | Update a memory with version history |
| `contextfs_link` | Create relationships between memories |
| `contextfs_delete` | Delete a memory |
| `contextfs_sessions` | List recent sessions |
| `contextfs_load_session` | Load a session's messages |

### Hooks

The plugin includes automatic hooks:

- **SessionStart**: Auto-recall relevant memories for current project
- **Stop**: Extract and save important information from the session
- **PreCompact**: Save session before context compaction

## Memory Types

| Type | Use Case |
|------|----------|
| `fact` | Learned information about codebase/user |
| `decision` | Architectural/design decisions with rationale |
| `procedural` | Step-by-step workflows and procedures |
| `error` | Errors encountered and their solutions |
| `code` | Code snippets and patterns |
| `episodic` | Session summaries |

## Links

- [ContextFS Documentation](https://github.com/magneticion/contextfs)
- [MCP Server Documentation](https://github.com/magneticion/contextfs#mcp-server)
- [Claude Code Plugins](https://claude-plugins.dev/)

## License

MIT
