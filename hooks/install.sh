#!/bin/bash
# ContextFS Git Hooks Installer
# Installs post-commit and post-merge hooks to enable automatic indexing
#
# Usage: ./install.sh [repo_path]
#        If no repo_path provided, installs to current directory

set -e

# Determine target repo
if [ -n "$1" ]; then
    REPO_PATH="$1"
else
    REPO_PATH="."
fi

# Resolve to absolute path
REPO_PATH=$(cd "$REPO_PATH" && pwd)

# Verify it's a git repo
if [ ! -d "$REPO_PATH/.git" ]; then
    echo "Error: $REPO_PATH is not a git repository"
    exit 1
fi

HOOKS_DIR="$REPO_PATH/.git/hooks"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing ContextFS git hooks to: $REPO_PATH"

# Install post-commit hook
if [ -f "$HOOKS_DIR/post-commit" ]; then
    echo "  post-commit: exists, backing up to post-commit.bak"
    mv "$HOOKS_DIR/post-commit" "$HOOKS_DIR/post-commit.bak"
fi
cp "$SCRIPT_DIR/post-commit" "$HOOKS_DIR/post-commit"
chmod +x "$HOOKS_DIR/post-commit"
echo "  post-commit: installed"

# Install post-merge hook
if [ -f "$HOOKS_DIR/post-merge" ]; then
    echo "  post-merge: exists, backing up to post-merge.bak"
    mv "$HOOKS_DIR/post-merge" "$HOOKS_DIR/post-merge.bak"
fi
cp "$SCRIPT_DIR/post-merge" "$HOOKS_DIR/post-merge"
chmod +x "$HOOKS_DIR/post-merge"
echo "  post-merge: installed"

echo ""
echo "Done! ContextFS will now automatically index on:"
echo "  - git commit (indexes changed files)"
echo "  - git pull/merge (indexes new files and commits)"
