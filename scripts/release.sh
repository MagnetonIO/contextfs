#!/bin/bash
set -e

# ContextFS Release Script (Non-Interactive)
# Usage: ./scripts/release.sh <version>
# Example: ./scripts/release.sh 0.1.6

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.1.6"
    echo ""
    echo "Current versions:"
    echo "  pyproject.toml:  $(grep '^version = ' pyproject.toml | cut -d'"' -f2)"
    echo "  __init__.py:     $(grep '__version__' src/contextfs/__init__.py | cut -d'"' -f2)"
    echo "  Latest git tag:  $(git describe --tags --abbrev=0 2>/dev/null || echo 'none')"
    exit 1
fi

# Validate version format (semver)
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in format X.Y.Z (e.g., 0.1.6)"
    exit 1
fi

TAG="v$VERSION"

# Check if tag already exists
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "Error: Tag $TAG already exists"
    exit 1
fi

# Check for uncommitted changes
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: You have uncommitted changes. Please commit or stash them first."
    exit 1
fi

echo "Releasing version $VERSION..."
echo ""

# Show changelog preview
echo "=== What's Changed (since last release) ==="
python scripts/generate_changelog.py --format markdown 2>/dev/null || echo "No previous tag found, will include recent commits"
echo ""
echo "============================================"
echo ""

# Update pyproject.toml
sed -i '' "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
echo "✓ Updated pyproject.toml"

# Update __init__.py
sed -i '' "s/__version__ = \".*\"/__version__ = \"$VERSION\"/" src/contextfs/__init__.py
echo "✓ Updated src/contextfs/__init__.py"

# Commit changes
git add pyproject.toml src/contextfs/__init__.py
git commit -m "Bump version to $VERSION"
echo "✓ Committed version bump"

# Create tag
git tag "$TAG"
echo "✓ Created tag $TAG"

# Push to remote
git push origin main
git push origin "$TAG"
echo "✓ Pushed to remote"

echo ""
echo "To publish to PyPI:"
echo "  python -m build"
echo "  twine upload dist/*"

echo ""
echo "Done! Version $VERSION is ready."
