# ContextFS

**Universal AI Memory Layer** - Cross-client, cross-repo context management with RAG.

Works with Claude Code, Claude Desktop, Gemini CLI, Codex CLI, and any MCP client.

[![PyPI](https://img.shields.io/pypi/v/contextfs)](https://pypi.org/project/contextfs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**[Documentation](https://magnetonio.github.io/contextfs)** | **[Developer Memory Workflow Guide](DMW.md)** | **[GitHub](https://github.com/MagnetonIO/contextfs)**

## Features

- **Semantic Search** - ChromaDB + sentence-transformers for intelligent retrieval
- **Auto Code Indexing** - Automatically index repositories for semantic code search
- **Dual Storage** - Smart routing between FTS (keywords) and RAG (semantic)
- **Cross-Repo Memory** - Memories track source repository automatically
- **Session Management** - Automatic capture and replay of conversation context
- **MCP Server** - Standard protocol for universal client support
- **Plugins** - Native integrations for Claude Code, Gemini CLI, Codex CLI
- **Web UI** - Browse and search memories with side-by-side FTS/RAG comparison

## Quick Start

```bash
# Run with uvx (no install needed)
uvx contextfs --help
uvx contextfs-mcp  # Start MCP server

# Or install with pip
pip install contextfs

# Or install with uv
uv pip install contextfs

# Or install from source
git clone https://github.com/MagnetonIO/contextfs.git
cd contextfs
pip install -e .
```

## Upgrading

```bash
# Upgrade with pip
pip install --upgrade contextfs

# Upgrade with uv
uv pip install --upgrade contextfs

# Upgrade with uvx (automatic on next run)
uvx --upgrade contextfs --help
```

## Usage

### CLI

```bash
# Save memories
contextfs save "Use PostgreSQL for the database" --type decision --tags db,architecture
contextfs save "API uses snake_case keys" --type fact --tags api,style

# Search
contextfs search "database decisions"
contextfs search "api conventions" --type fact

# Recall specific memory
contextfs recall abc123

# List recent
contextfs list --limit 20 --type decision

# Sessions
contextfs sessions
```

### Python API

```python
from contextfs import ContextFS, MemoryType

ctx = ContextFS()

# Save
ctx.save(
    "Use JWT for authentication",
    type=MemoryType.DECISION,
    tags=["auth", "security"],
)

# Search
results = ctx.search("authentication")
for r in results:
    print(f"[{r.score:.2f}] {r.memory.content}")

# Get context for a task
context = ctx.get_context_for_task("implement login")
# Returns formatted strings ready for prompt injection
```

### MCP Server

Add to your MCP client config (Claude Code, Claude Desktop):

```json
{
  "mcpServers": {
    "contextfs": {
      "command": "uvx",
      "args": ["contextfs-mcp"]
    }
  }
}
```

Or with Python directly:
```json
{
  "mcpServers": {
    "contextfs": {
      "command": "python",
      "args": ["-m", "contextfs.mcp_server"]
    }
  }
}
```

**MCP Tools:**

| Tool | Description |
|------|-------------|
| `contextfs_save` | Save memory (auto-indexes repo, logs to session) |
| `contextfs_search` | Semantic search with cross-repo support |
| `contextfs_recall` | Get specific memory by ID |
| `contextfs_list` | List recent memories |
| `contextfs_update` | Update existing memory content, type, tags, or project |
| `contextfs_delete` | Delete a memory by ID |
| `contextfs_index` | Index current repository for code search |
| `contextfs_index_status` | Check or cancel background indexing progress |
| `contextfs_list_repos` | List all repositories with memories |
| `contextfs_list_tools` | List source tools (claude-code, claude-desktop, etc.) |
| `contextfs_list_projects` | List all projects |
| `contextfs_sessions` | List sessions |
| `contextfs_load_session` | Load session messages |
| `contextfs_message` | Add message to current session |
| `contextfs_update_session` | Update session label or summary |
| `contextfs_delete_session` | Delete a session and its messages |
| `contextfs_import_conversation` | Import JSON conversation as episodic memory |

**MCP Prompts:**

| Prompt | Description |
|--------|-------------|
| `contextfs-save-memory` | Guided memory save with type selection |
| `contextfs-index` | Index repository for semantic search |
| `contextfs-session-guide` | Instructions for session capture |
| `contextfs-save-session` | Save current session |

## Plugins

### Claude Code

```bash
# Install hooks for automatic context capture
python -c "from contextfs.plugins.claude_code import install_claude_code; install_claude_code()"
```

### Gemini CLI / Codex CLI

```python
from contextfs.plugins.gemini import install_gemini
from contextfs.plugins.codex import install_codex

install_gemini()  # For Gemini CLI
install_codex()   # For Codex CLI
```

## Cross-Repo Namespaces

ContextFS automatically detects your git repository and isolates memories:

```python
# In repo A
ctx = ContextFS()  # namespace = "repo-<hash-of-repo-a>"
ctx.save("Repo A specific fact")

# In repo B
ctx = ContextFS()  # namespace = "repo-<hash-of-repo-b>"
# Won't see Repo A's memories

# Global namespace (shared across repos)
ctx = ContextFS(namespace_id="global")
ctx.save("Shared across all repos")
```

## Configuration

Environment variables:

```bash
CONTEXTFS_DATA_DIR=~/.contextfs
CONTEXTFS_EMBEDDING_MODEL=all-MiniLM-L6-v2
CONTEXTFS_CHUNK_SIZE=1000
CONTEXTFS_DEFAULT_SEARCH_LIMIT=10
CONTEXTFS_AUTO_SAVE_SESSIONS=true
CONTEXTFS_AUTO_LOAD_ON_STARTUP=true
```

## Supported Languages

ContextFS supports 50+ file types for code ingestion:

**Top 20 Programming Languages:**
Python, JavaScript, TypeScript, Java, C++, C, C#, Go, Rust, PHP, Ruby, Swift, Kotlin, Scala, R, MATLAB, Perl, Lua, Haskell, Elixir

**Plus:** Dart, Julia, Clojure, Erlang, F#, Zig, Nim, Crystal, Groovy, and more.

**Web:** HTML, CSS, SCSS, JSX, TSX, Vue, Svelte

**Config:** JSON, YAML, TOML, XML, INI

**Documentation:** Markdown, RST, TeX, Org

## Developer Memory Workflow (DMW)

ContextFS enables **persistent developer memory** that follows you across sessions:

```python
# Save decisions as you make them
ctx.save(
    "Use PostgreSQL for production",
    type=MemoryType.DECISION,
    tags=["database", "architecture"]
)

# Document bug fixes for future reference
ctx.save(
    "CORS fix: allow_methods=['*'] required for credentials",
    type=MemoryType.ERROR,
    tags=["cors", "api", "bug-fix"]
)

# Later sessions can search for context
results = ctx.search("database decisions")
results = ctx.search("CORS issues", type=MemoryType.ERROR)
```

### Memory Types

| Type | Use Case |
|------|----------|
| `fact` | Project configurations, conventions |
| `decision` | Architectural choices with rationale |
| `code` | Algorithms, patterns, important snippets |
| `error` | Bug fixes, error patterns, solutions |
| `procedural` | Setup guides, deployment steps |
| `episodic` | Session transcripts, conversations |

### Solo vs Team Workflows

- **Solo**: Per-repo namespace, personal memory bank
- **Team**: Shared namespace, collective knowledge base

See the full **[Developer Memory Workflow Guide](DMW.md)** for detailed patterns.

### Session & Memory Event Workflow

ContextFS tracks sessions and provides CRUD operations for managing your memory lifecycle:

```python
# Start a labeled session
ctx = ContextFS()
session = ctx.start_session(tool="claude-code", label="feature-auth")

# Log important exchanges during conversation
ctx.add_message("user", "Implement OAuth2 login")
ctx.add_message("assistant", "Created OAuth2 flow in auth/oauth.py")

# Save memories as decisions are made
memory = ctx.save(
    "Use PKCE flow for OAuth2 - more secure for SPAs",
    type=MemoryType.DECISION,
    tags=["auth", "oauth", "security"]
)

# Update memory if context changes
ctx.update(memory.id, tags=["auth", "oauth", "security", "production"])

# End session with auto-generated summary
ctx.end_session(generate_summary=True)
```

**Session Events via MCP:**

```
# During conversation - log key exchanges
contextfs_message(role="user", content="How should we handle auth?")
contextfs_message(role="assistant", content="Implementing JWT with refresh tokens")

# Save decisions with context
contextfs_save(content="...", type="decision", tags=["auth"])

# Update existing memories
contextfs_update(id="abc123", tags=["auth", "jwt", "v2"])

# Import external conversations (from Claude Desktop JSON exports)
contextfs_import_conversation(json_content="...", summary="Auth discussion")

# Manage sessions
contextfs_update_session(session_id="xyz", label="auth-feature-complete")
contextfs_delete_session(session_id="old-session")

# End of session - save for future reference
contextfs_save(save_session="current", label="auth-implementation")
```

**Source Tool Detection:**

ContextFS automatically detects whether it's running under Claude Code or Claude Desktop:
- `claude-code`: Terminal environment (has `TERM`, `SHELL` env vars)
- `claude-desktop`: GUI environment (no terminal indicators)
- Override with `CONTEXTFS_SOURCE_TOOL` environment variable

## Web UI

Start the web server to browse and search memories:

```bash
contextfs web
# Opens at http://localhost:8000

contextfs web --port 3000  # Custom port
```

Features:
- Browse all memories with filtering by type, repo, and project
- Side-by-side FTS vs RAG search comparison
- Session browser and message viewer
- Real-time memory statistics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ContextFS Core                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │   CLI    │  │   MCP    │  │  Web UI  │  │  Python  │    │
│  │          │  │  Server  │  │          │  │   API    │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │             │             │             │           │
│       └─────────────┴──────┬──────┴─────────────┘           │
│                            │                                │
│                    ┌───────▼───────┐                        │
│                    │  ContextFS()  │                        │
│                    │   core.py     │                        │
│                    └───────┬───────┘                        │
│                            │                                │
│          ┌─────────────────┼─────────────────┐              │
│          │                 │                 │              │
│  ┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐     │
│  │  AutoIndexer  │ │  RAG Backend  │ │  FTS Backend  │     │
│  │  (Code Index) │ │  (ChromaDB)   │ │  (SQLite)     │     │
│  └───────────────┘ └───────────────┘ └───────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Dual Storage

| SQLite + FTS5 | ChromaDB |
|---------------|----------|
| Fast keyword queries | Vector similarity search |
| Session/message storage | Code semantic search |
| Exact matches | Conceptual matching |
| Metadata & filtering | Embedding-based retrieval |

## License

MIT

## Authors

Matthew Long and The YonedaAI Collaboration
