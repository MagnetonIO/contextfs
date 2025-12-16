# Installation

## Requirements

- Python 3.10 or higher
- ~500MB disk space for embeddings model

## Package Installation

=== "pip"

    ```bash
    pip install contextfs
    ```

=== "uv"

    ```bash
    uv pip install contextfs
    ```

=== "pipx (isolated)"

    ```bash
    pipx install contextfs
    ```

=== "From source"

    ```bash
    git clone https://github.com/MagnetonIO/contextfs.git
    cd contextfs
    pip install -e .
    ```

## Upgrading

=== "pip"

    ```bash
    pip install --upgrade contextfs
    ```

=== "uv"

    ```bash
    uv pip install --upgrade contextfs
    ```

=== "uvx"

    ```bash
    # uvx auto-upgrades, or force upgrade:
    uvx --upgrade contextfs --help
    ```

=== "pipx"

    ```bash
    pipx upgrade contextfs
    ```

## Optional Dependencies

```bash
# Web UI support
pip install contextfs[web]

# PostgreSQL backend (enterprise)
pip install contextfs[postgres]

# Everything
pip install contextfs[all]
```

## Verify Installation

```bash
# Check version
contextfs status

# Or use the short alias
ctx status
```

## Claude Desktop Integration

Install ContextFS as an MCP server for Claude Desktop:

```bash
contextfs install-claude-desktop
```

This automatically configures Claude Desktop to use ContextFS. Restart Claude Desktop to activate.

## First-Time Setup

ContextFS works out of the box with sensible defaults:

- Data stored in `~/.contextfs/`
- Automatic repo detection via git
- Local embeddings (no API keys needed)

To initialize a project-specific configuration:

```bash
cd your-project
contextfs init
```
