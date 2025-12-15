#!/bin/bash
# ContextFS - Easy Setup Script
# Universal AI Memory Layer

set -e

echo "================================"
echo "ContextFS Setup"
echo "================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "Error: Python $REQUIRED_VERSION or higher required (found $PYTHON_VERSION)"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION"

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

# Install package in editable mode
echo "Installing ContextFS..."
pip install -e ".[dev]" -q

echo "✓ ContextFS installed"

# Create data directory
DATA_DIR="${CONTEXTFS_DATA_DIR:-$HOME/.contextfs}"
mkdir -p "$DATA_DIR"
echo "✓ Data directory: $DATA_DIR"

# Initialize ChromaDB (lazy, but create dir)
mkdir -p "$DATA_DIR/chroma_db"

# Show commands
echo ""
echo "================================"
echo "Installation Complete!"
echo "================================"
echo ""
echo "Commands:"
echo "  contextfs save \"content\" --type fact --tags tag1,tag2"
echo "  contextfs search \"query\""
echo "  contextfs recall <id>"
echo "  contextfs list"
echo "  contextfs sessions"
echo "  contextfs status"
echo "  contextfs serve  # Start MCP server"
echo ""
echo "Install plugins:"
echo "  python -c \"from contextfs.plugins.claude_code import install_claude_code; install_claude_code()\""
echo "  python -c \"from contextfs.plugins.gemini import install_gemini; install_gemini()\""
echo "  python -c \"from contextfs.plugins.codex import install_codex; install_codex()\""
echo ""
echo "MCP Config (add to claude_desktop_config.json):"
echo '  "mcpServers": {'
echo '    "contextfs": {'
echo '      "command": "python",'
echo '      "args": ["-m", "contextfs.mcp_server"]'
echo '    }'
echo '  }'
echo ""
echo "To activate environment: source .venv/bin/activate"
