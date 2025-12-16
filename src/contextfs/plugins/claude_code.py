"""
Claude Code Plugin for ContextFS.

Provides:
- Lifecycle hooks for automatic context capture
- Skills for memory search
- Auto-save sessions on exit
"""

import json
from pathlib import Path

from contextfs.core import ContextFS


class ClaudeCodePlugin:
    """
    Claude Code integration plugin.

    Hooks into Claude Code's lifecycle events to automatically
    capture and inject context.
    """

    def __init__(self, ctx: ContextFS | None = None):
        """
        Initialize Claude Code plugin.

        Args:
            ctx: ContextFS instance (creates one if not provided)
        """
        self.ctx = ctx or ContextFS(auto_load=True)
        self._settings_file = Path.home() / ".claude" / "settings.json"
        self._skills_dir = Path.home() / ".claude" / "skills"

    def install(self) -> None:
        """Install Claude Code hooks and skills."""
        self._install_hooks()
        self._install_skills()
        print("Claude Code plugin installed successfully.")
        print(f"Settings: {self._settings_file}")
        print(f"Skills: {self._skills_dir}")
        print("\nInstalled components:")
        print("  - SessionStart hook (auto-index in background)")
        print("  - PreCompact hook (auto-save before context compaction)")
        print("  - MCP server (contextfs)")
        print("\nRestart Claude Code for changes to take effect.")

    def uninstall(self) -> None:
        """Uninstall Claude Code hooks from settings."""
        if self._settings_file.exists():
            settings = json.loads(self._settings_file.read_text())
            if "hooks" in settings:
                # Remove contextfs hooks
                for hook_type in ["SessionStart", "PreCompact", "SessionEnd"]:
                    if hook_type in settings["hooks"]:
                        settings["hooks"][hook_type] = [
                            h
                            for h in settings["hooks"][hook_type]
                            if "contextfs" not in h.get("hooks", [{}])[0].get("command", "")
                        ]
            # Remove MCP server
            if "mcpServers" in settings and "contextfs" in settings["mcpServers"]:
                del settings["mcpServers"]["contextfs"]

            self._settings_file.write_text(json.dumps(settings, indent=2))

        # Remove skill files
        skill_file = self._skills_dir / "contextfs-search.md"
        if skill_file.exists():
            skill_file.unlink()

        print("Claude Code plugin uninstalled.")

    def _install_hooks(self) -> None:
        """Install lifecycle hooks into Claude Code settings."""
        self._settings_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing settings or create new
        if self._settings_file.exists():
            settings = json.loads(self._settings_file.read_text())
        else:
            settings = {}

        # Ensure hooks section exists
        if "hooks" not in settings:
            settings["hooks"] = {}

        # Add SessionStart hook (indexes on session start in background)
        settings["hooks"]["SessionStart"] = [
            {
                "matcher": {},
                "hooks": [
                    {
                        "type": "command",
                        "command": "uvx contextfs index --quiet --background",
                    }
                ],
            }
        ]

        # Add PreCompact hook (saves before context compaction)
        settings["hooks"]["PreCompact"] = [
            {
                "matcher": {},
                "hooks": [
                    {
                        "type": "command",
                        "command": "uvx contextfs save-session --label 'auto-compact'",
                    }
                ],
            }
        ]

        # Ensure mcpServers section exists
        if "mcpServers" not in settings:
            settings["mcpServers"] = {}

        # Add contextfs MCP server
        settings["mcpServers"]["contextfs"] = {
            "command": "uvx",
            "args": ["contextfs", "serve"],
            "env": {"CONTEXTFS_SOURCE_TOOL": "claude-code"},
        }

        # Write updated settings
        self._settings_file.write_text(json.dumps(settings, indent=2))

    def _install_skills(self) -> None:
        """Install search skill."""
        self._skills_dir.mkdir(parents=True, exist_ok=True)

        search_skill = """# ContextFS Search Skill

Search your AI memory for relevant context.

## Usage

Use this skill to find relevant memories, decisions, and context from previous sessions.

## Parameters

- `query`: What to search for
- `type`: Filter by type (fact, decision, procedural, episodic, user, code, error)
- `limit`: Maximum results (default: 5)

## Example Prompts

- "Search my memory for authentication decisions"
- "Find previous discussions about database design"
- "What do I know about the user's preferences?"

## Implementation

```python
from contextfs import ContextFS

ctx = ContextFS()
results = ctx.search(query, limit=5)

for r in results:
    print(f"[{r.memory.type.value}] {r.memory.content}")
```
"""
        (self._skills_dir / "contextfs-search.md").write_text(search_skill)


# CLI commands


def install_claude_code():
    """Install Claude Code plugin."""
    plugin = ClaudeCodePlugin()
    plugin.install()


def uninstall_claude_code():
    """Uninstall Claude Code plugin."""
    plugin = ClaudeCodePlugin()
    plugin.uninstall()
