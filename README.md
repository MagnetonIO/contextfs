# ContextFS

**Universal AI Memory Layer** - Cross-client, cross-repo context management with RAG.

Works with Claude Code, Claude Desktop, Gemini CLI, Codex CLI, and any MCP client.

## Features

- **Semantic Search** - ChromaDB + sentence-transformers for intelligent retrieval
- **Cross-Repo Namespaces** - Memory isolation per repository with sharing options
- **Session Management** - Automatic capture and replay of conversation context
- **MCP Server** - Standard protocol for universal client support
- **Plugins** - Native integrations for Claude Code, Gemini, Codex
- **Auto-Load/Save** - Context automatically loaded on startup, saved on exit

## Quick Start

```bash
# Clone and install
git clone https://github.com/MagnetonIO/contextfs.git
cd contextfs
./setup.sh

# Or pip install
pip install contextfs
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

Add to your MCP client config:

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

Available MCP tools:
- `contextfs_save` - Save a memory
- `contextfs_search` - Semantic search
- `contextfs_recall` - Get specific memory
- `contextfs_list` - List recent memories
- `contextfs_sessions` - List sessions
- `contextfs_load_session` - Load session messages
- `contextfs_message` - Add message to current session

## Plugins

### Claude Code

```python
from contextfs.plugins.claude_code import install_claude_code
install_claude_code()
```

Installs:
- Session start/end hooks for automatic context injection
- Pre/post tool execution hooks for observation capture
- Memory search skill

### Gemini CLI

```python
from contextfs.plugins.gemini import install_gemini
install_gemini()
```

### Codex CLI

```python
from contextfs.plugins.codex import install_codex
install_codex()
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

## Architecture

```
contextfs/
├── src/contextfs/
│   ├── core.py          # Main ContextFS class
│   ├── rag.py           # RAG backend (ChromaDB + embeddings)
│   ├── schemas.py       # Pydantic models
│   ├── config.py        # Configuration
│   ├── mcp_server.py    # MCP protocol server
│   ├── cli.py           # CLI commands
│   └── plugins/
│       ├── claude_code.py
│       ├── gemini.py
│       └── codex.py
└── pyproject.toml
```

## License

MIT

## Authors

Matthew Long and The YonedaAI Collaboration
