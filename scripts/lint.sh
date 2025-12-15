#!/usr/bin/env bash
# Lint the codebase using ruff
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== Running Ruff Linter ==="

# Check if --fix flag is passed
if [[ "${1:-}" == "--fix" ]]; then
    echo "Running with auto-fix enabled..."
    ruff check src/ tests/ --fix
    ruff format src/ tests/
else
    echo "Running lint check (use --fix to auto-fix)..."
    ruff check src/ tests/
    ruff format --check src/ tests/
fi

echo "=== Lint Complete ==="
