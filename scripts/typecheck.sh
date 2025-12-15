#!/usr/bin/env bash
# Run type checking with mypy
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== Running Type Check ==="

mypy src/contextfs --ignore-missing-imports

echo "=== Type Check Complete ==="
