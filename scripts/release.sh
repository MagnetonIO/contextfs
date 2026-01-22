#!/bin/bash
set -e

# ContextFS Release Script (GitFlow)
# Usage: ./scripts/release.sh <version>
# Example: ./scripts/release.sh 0.1.6
#
# This script follows gitflow with protected main branch:
# 1. Updates versions on develop
# 2. Creates PR from develop to main
# 3. After PR is merged, creates and pushes tag

# Handle --tag option for creating tag after PR merge
if [ "$1" = "--tag" ]; then
    VERSION=$2
    if [ -z "$VERSION" ]; then
        echo "Usage: $0 --tag <version>"
        exit 1
    fi
    TAG="v$VERSION"

    echo "Creating tag $TAG after PR merge..."
    git checkout main
    git pull origin main
    git tag "$TAG"
    git push origin "$TAG"

    echo ""
    echo "✓ Tag $TAG pushed!"
    echo ""
    echo "GitHub Actions will automatically:"
    echo "  • Publish CLI to PyPI"
    echo "  • Publish plugin to npm (with provenance)"
    echo "  • Create GitHub release"
    echo "  • Send Slack notification"
    exit 0
fi

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.1.6"
    echo ""
    echo "Current versions:"
    echo "  pyproject.toml:       $(grep '^version = ' pyproject.toml | cut -d'"' -f2)"
    echo "  __init__.py:          $(grep '__version__' src/contextfs/__init__.py | cut -d'"' -f2)"
    echo "  plugin package.json:  $(grep '"version"' claude-plugin/package.json | head -1 | cut -d'"' -f4)"
    echo "  plugin plugin.json:   $(grep '"version"' claude-plugin/plugin.json | head -1 | cut -d'"' -f4)"
    echo "  Latest git tag:       $(git describe --tags --abbrev=0 2>/dev/null || echo 'none')"
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

# Ensure we're on develop branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "develop" ]; then
    echo "Switching to develop branch..."
    git checkout develop
fi

# Pull latest from main to ensure develop is up to date
echo "Syncing develop with main..."
git pull origin main --no-edit 2>/dev/null || git merge origin/main --no-edit

echo ""
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

# Update claude-plugin/package.json
sed -i '' 's/"version": "[^"]*"/"version": "'"$VERSION"'"/' claude-plugin/package.json
echo "✓ Updated claude-plugin/package.json"

# Update claude-plugin/plugin.json
sed -i '' 's/"version": "[^"]*"/"version": "'"$VERSION"'"/' claude-plugin/plugin.json
echo "✓ Updated claude-plugin/plugin.json"

# Commit changes (CLI + plugin together)
git add pyproject.toml src/contextfs/__init__.py claude-plugin/package.json claude-plugin/plugin.json
git commit -m "Bump version to $VERSION"
echo "✓ Committed version bump (CLI + plugin)"

# Push to develop
git push origin develop
echo "✓ Pushed to develop"

# Create PR from develop to main
echo ""
echo "Creating PR from develop to main..."
PR_URL=$(gh pr create --base main --head develop \
    --title "Release v$VERSION" \
    --body "## Release v$VERSION

### What's Changed
$(python scripts/generate_changelog.py --format markdown 2>/dev/null || echo 'See commit history')

---
*After merging, run \`./scripts/release.sh --tag $VERSION\` to create and push the tag.*" 2>&1)

echo "✓ Created PR: $PR_URL"
echo ""
echo "Next steps:"
echo "  1. Review and merge the PR: $PR_URL"
echo "  2. After merge, run: ./scripts/release.sh --tag $VERSION"
echo ""
echo "Or merge now and tag automatically:"
read -p "Merge PR and create tag now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Merging PR..."
    gh pr merge --merge --delete-branch

    echo "Pulling main..."
    git checkout main
    git pull origin main

    echo "Creating and pushing tag..."
    git tag "$TAG"
    git push origin "$TAG"

    echo ""
    echo "✓ Released v$VERSION!"
    echo ""
    echo "GitHub Actions will automatically:"
    echo "  • Publish CLI to PyPI"
    echo "  • Publish plugin to npm (with provenance)"
    echo "  • Create GitHub release"
    echo "  • Send Slack notification"
fi
