#!/usr/bin/env bash
# Run tests with pytest
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== Running Tests ==="

# Parse arguments
COVERAGE=false
SLOW=false
UNIT_ONLY=false
INTEGRATION_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage|-c)
            COVERAGE=true
            shift
            ;;
        --slow|-s)
            SLOW=true
            shift
            ;;
        --unit|-u)
            UNIT_ONLY=true
            shift
            ;;
        --integration|-i)
            INTEGRATION_ONLY=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Build pytest command
PYTEST_ARGS="-v --tb=short"

if [[ "$COVERAGE" == true ]]; then
    PYTEST_ARGS="$PYTEST_ARGS --cov=src/contextfs --cov-report=term --cov-report=html"
fi

if [[ "$SLOW" == false ]]; then
    PYTEST_ARGS="$PYTEST_ARGS -m 'not slow and not postgres'"
fi

# Determine test paths
if [[ "$UNIT_ONLY" == true ]]; then
    TEST_PATHS="tests/unit"
elif [[ "$INTEGRATION_ONLY" == true ]]; then
    TEST_PATHS="tests/integration"
else
    TEST_PATHS="tests/"
fi

echo "Running: pytest $TEST_PATHS $PYTEST_ARGS"
pytest $TEST_PATHS $PYTEST_ARGS

echo "=== Tests Complete ==="
