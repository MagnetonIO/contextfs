# AI Development Workflow for ContextFS

A standardized procedure for AI assistants (Claude, Gemini, GPT, etc.) to complete multiple development tasks efficiently.

## Overview

This workflow ensures consistent, high-quality contributions through:
- Proper GitFlow branching
- Comprehensive testing before commits
- Clear task tracking
- Systematic releases

## Phase 1: Task Analysis

### Break Down Requirements
```
User Request → Discrete Tasks → Prioritized Todo List
```

**Priority Order:**
1. Critical bugs (breaking functionality)
2. Regular bugs
3. New features
4. Improvements/refactoring
5. Documentation

### Create Task List
Track each task with:
- Clear description
- Status: pending | in_progress | completed
- One task in_progress at a time

## Phase 2: Implementation Loop

For each task, follow this cycle:

### 2.1 Branch Creation
```bash
git checkout main && git pull
git checkout -b <type>/<descriptive-name>
```

Branch types:
- `feature/` - new functionality
- `bugfix/` - bug fixes
- `hotfix/` - critical production fixes
- `docs/` - documentation only

### 2.2 Implementation
- Make focused changes (one logical change per branch)
- Follow existing code patterns
- Don't over-engineer or add unrequested features

### 2.3 Testing
```bash
# Always test with local code, not installed version
python -m contextfs.cli <command>

# Run test suite
pytest tests/ -x -q --tb=short

# Test specific areas affected by changes
pytest tests/integration/test_<relevant>.py -x -q
```

**Testing Rules:**
- All tests must pass before commit
- Test the actual functionality, not just the tests
- For core changes, verify sync flow still works

### 2.4 Commit
```bash
git add <specific-files>
git commit -m "Clear, descriptive message

Explain what changed and why.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: <AI Assistant> <noreply@anthropic.com>"
```

### 2.5 Pull Request
```bash
git push -u origin <branch-name>
gh pr create --title "Title" --body "## Summary
- Change 1
- Change 2

## Test Plan
- [x] Tested X
- [x] Verified Y"
```

### 2.6 Merge & Cleanup
```bash
gh pr merge <number> --merge --delete-branch
git checkout main && git pull
```

### 2.7 Update Progress
- Mark task as completed immediately
- Move to next pending task

## Phase 3: Release

When all tasks are complete:

### 3.1 Version Bump
Update both files:
- `pyproject.toml`: `version = "X.Y.Z"`
- `src/contextfs/__init__.py`: `__version__ = "X.Y.Z"`

### 3.2 Commit & Tag
```bash
git add pyproject.toml src/contextfs/__init__.py
git commit -m "Bump version to X.Y.Z

Release highlights:
- Feature 1
- Fix 2"

git tag -a vX.Y.Z -m "vX.Y.Z - Release Title

- Highlight 1
- Highlight 2"
```

### 3.3 Push & Release
```bash
git push origin main --tags
gh release create vX.Y.Z --title "vX.Y.Z - Title" --notes "Release notes"
```

## Phase 4: Documentation

### Save to Memory
```bash
# Save decisions
python -m contextfs.cli save --type decision --tags "tag1,tag2" "Content"

# Save procedures
python -m contextfs.cli save --type procedural --tags "workflow" "Content"
```

### Update Project Docs
- Update CLAUDE.md for workflow changes
- Keep API reference current

## Troubleshooting

### MCP Tools Failing
1. **First**: Try `/mcp` reconnect in Claude Code
2. **Fallback**: Use CLI directly: `python -m contextfs.cli <command>`
3. **Last resort**: `echo "y" | python -m contextfs.cli rebuild-chroma`

### Test Failures
1. Read error messages carefully
2. Fix the root cause, not symptoms
3. Re-run full test suite after fixes

### Merge Conflicts
1. `git checkout main && git pull`
2. `git checkout <branch> && git rebase main`
3. Resolve conflicts, then `git rebase --continue`

## Key Principles

| Principle | Rationale |
|-----------|-----------|
| One branch per change | Easier review, revert, tracking |
| Test before commit | Never push broken code |
| Use local CLI for testing | Avoids cache/version issues |
| Complete todos immediately | Accurate progress tracking |
| Save knowledge to memory | Helps future sessions |

## Example Session

```
User: "Fix bug X, add feature Y, then release"

1. TodoWrite: [Fix bug X (pending), Add feature Y (pending), Release (pending)]
2. Mark "Fix bug X" as in_progress
3. git checkout -b bugfix/fix-x
4. Implement fix, test, commit, PR, merge
5. Mark "Fix bug X" as completed
6. Mark "Add feature Y" as in_progress
7. git checkout -b feature/add-y
8. Implement, test, commit, PR, merge
9. Mark "Add feature Y" as completed
10. Mark "Release" as in_progress
11. Bump version, tag, push, create release
12. Mark "Release" as completed
```
