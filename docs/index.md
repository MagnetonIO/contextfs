# ContextFS

**Universal AI Memory Layer** — Cross-client, cross-repo context management with semantic search

---

## Features

| | Feature | Description |
|---|---------|-------------|
| :material-brain: | **Semantic Memory** | Store and retrieve context using semantic search powered by ChromaDB and sentence transformers. [Learn more →](architecture/overview.md) |
| :material-source-repository-multiple: | **Cross-Repo** | Memories are automatically namespaced by repository, with support for cross-repo search and project grouping. [Learn more →](architecture/namespaces.md) |
| :material-robot: | **Multi-Client** | Works with Claude Desktop, Claude Code, Gemini, ChatGPT, and any MCP-compatible client. [Learn more →](integration/claude-desktop.md) |
| :material-console: | **CLI & MCP** | Full-featured CLI for memory management plus MCP server for AI tool integration. [Learn more →](getting-started/cli.md) |

---

## Installation

=== "pip"

    ```bash
    pip install contextfs
    ```

=== "uv"

    ```bash
    uv pip install contextfs
    ```

=== "pipx"

    ```bash
    pipx install contextfs
    ```

## Quick Example

```python
from contextfs import ContextFS

# Initialize (auto-detects current repo)
ctx = ContextFS()

# Save a memory
ctx.save(
    content="Authentication uses JWT tokens with 24h expiry",
    type="decision",
    tags=["auth", "security"]
)

# Search memories
results = ctx.search("how does authentication work?")
for r in results:
    print(f"{r.score:.2f}: {r.memory.content}")
```

## CLI Usage

```bash
# Save a memory
contextfs save "API uses REST with JSON responses" --type decision --tags api,design

# Search memories
contextfs search "API design patterns"

# Index a repository
contextfs index

# List recent memories
contextfs list
```

## Why ContextFS?

Modern AI development involves multiple tools, repositories, and long-running projects. ContextFS solves the **context fragmentation problem**:

- :material-refresh: **Memory across sessions** — Don't repeat yourself to AI tools
- :material-swap-horizontal: **Memory across tools** — Share context between Claude, Gemini, and others
- :material-folder-multiple: **Memory across repos** — Find related decisions from other projects
- :material-magnify: **Semantic search** — Natural language queries over your entire context history

## Theoretical Foundation

ContextFS is built on principles from [Type-Safe Context Engineering](research/typesafe-context.md), applying insights from protein folding and type theory to AI memory systems.

---

[:material-github: GitHub](https://github.com/MagnetonIO/contextfs){ .md-button }
[:material-package: PyPI](https://pypi.org/project/contextfs/){ .md-button }
