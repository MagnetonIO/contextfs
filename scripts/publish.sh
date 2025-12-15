#!/usr/bin/env bash
# Publish package to PyPI
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== Publishing Package ==="

# Check if dist exists
if [[ ! -d "dist" ]] || [[ -z "$(ls -A dist)" ]]; then
    echo "No dist/ directory found. Run ./scripts/build.sh first."
    exit 1
fi

# Install twine if needed
pip install --quiet twine

# Check if --test flag for TestPyPI
if [[ "${1:-}" == "--test" ]]; then
    echo "Publishing to TestPyPI..."
    twine upload --repository testpypi dist/*
else
    echo "Publishing to PyPI..."
    echo "Use --test to publish to TestPyPI instead."
    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        twine upload dist/*
    else
        echo "Aborted."
        exit 1
    fi
fi

echo "=== Publish Complete ==="
