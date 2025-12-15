#!/usr/bin/env bash
# Build the package
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== Building Package ==="

# Clean previous builds
rm -rf dist/ build/ *.egg-info src/*.egg-info

# Install build tool if needed
pip install --quiet build

# Build
python -m build

echo ""
echo "=== Build Complete ==="
echo "Artifacts in dist/:"
ls -la dist/
