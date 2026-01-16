# ContextFS

**Persistent Memory for AI Agents** - Give your AI tools memory that persists across sessions.

[![PyPI](https://img.shields.io/pypi/v/contextfs)](https://pypi.org/project/contextfs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**[Full Documentation](https://contextfs.ai/docs)** | **[Get Started](https://contextfs.ai)**

## Install

```bash
# With pip
pip install contextfs

# With uv (recommended)
uv pip install contextfs

# Run directly without installing
uvx contextfs
```

## Quick Start

```bash
# Initialize your repo for indexing
contextfs index init

# Save a memory
contextfs memory save "Use PostgreSQL for database" --type decision

# Search memories
contextfs memory search "database"

# Index your codebase for semantic search
contextfs index index
```

## MCP Integration

Add to your AI tool's MCP config:

```json
{
  "mcpServers": {
    "contextfs": {
      "command": "uvx",
      "args": ["contextfs"]
    }
  }
}
```

**Works with:** Claude Code, Claude Desktop, Cursor, VS Code, and any MCP-compatible client.

See [tool-specific setup guides](https://contextfs.ai/docs) for detailed instructions.

## Key Features

- **Semantic Search** - Find relevant memories using natural language
- **Auto Code Indexing** - Index your entire codebase for context-aware AI
- **Cross-Session Memory** - Decisions, facts, and patterns persist across conversations
- **Multi-Tool Sync** - Share memory between Claude, Cursor, VS Code, and more

## Python SDK

```python
from contextfs import ContextFS

ctx = ContextFS()

# Save
ctx.save("Use JWT for auth", type="decision", tags=["auth"])

# Search
results = ctx.search("authentication")
```

## Cloud Sync

Enable cross-device memory sync:

```bash
contextfs cloud login
contextfs cloud sync
```

Sign up at [contextfs.ai](https://contextfs.ai) for cloud features.

## Documentation

Visit **[contextfs.ai/docs](https://contextfs.ai/docs)** for:
- Installation guides for each AI tool
- API reference
- Memory types and best practices
- Cloud sync setup

## License

MIT - Matthew Long and The YonedaAI Collaboration
