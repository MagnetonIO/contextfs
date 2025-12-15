"""
Claude Code Plugin for ContextFS.

Provides:
- Lifecycle hooks for automatic context capture
- Skills for memory search
- Auto-save sessions on exit
"""

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
        self._hooks_dir = Path.home() / ".claude" / "hooks"
        self._skills_dir = Path.home() / ".claude" / "skills"

    def install(self) -> None:
        """Install Claude Code hooks and skills."""
        self._install_hooks()
        self._install_skills()
        print("Claude Code plugin installed successfully.")
        print(f"Hooks: {self._hooks_dir}")
        print(f"Skills: {self._skills_dir}")

    def uninstall(self) -> None:
        """Uninstall Claude Code hooks and skills."""
        # Remove hook files
        for hook_name in ["PreToolExecution", "PostToolExecution", "SessionStart", "SessionEnd"]:
            hook_file = self._hooks_dir / f"{hook_name}.py"
            if hook_file.exists():
                hook_file.unlink()

        # Remove skill files
        skill_file = self._skills_dir / "contextfs-search.md"
        if skill_file.exists():
            skill_file.unlink()

        print("Claude Code plugin uninstalled.")

    def _install_hooks(self) -> None:
        """Install lifecycle hooks."""
        self._hooks_dir.mkdir(parents=True, exist_ok=True)

        # Session Start Hook
        session_start_hook = '''#!/usr/bin/env python3
"""ContextFS: Session start hook - load relevant context."""
import sys
import json

def main():
    from contextfs import ContextFS

    ctx = ContextFS(auto_load=True)
    session = ctx.start_session(tool="claude-code")

    # Get recent context for injection
    context = ctx.get_context_for_task("current task", limit=3)

    if context:
        print("## Loaded Context from ContextFS")
        for c in context:
            print(f"- {c}")
        print()

if __name__ == "__main__":
    main()
'''
        (self._hooks_dir / "SessionStart.py").write_text(session_start_hook)

        # Session End Hook
        session_end_hook = '''#!/usr/bin/env python3
"""ContextFS: Session end hook - save session."""
import sys

def main():
    from contextfs import ContextFS

    ctx = ContextFS(auto_load=False)
    ctx.end_session(generate_summary=True)
    print("[ContextFS] Session saved.")

if __name__ == "__main__":
    main()
'''
        (self._hooks_dir / "SessionEnd.py").write_text(session_end_hook)

        # Pre-Tool Execution Hook
        pre_tool_hook = '''#!/usr/bin/env python3
"""ContextFS: Pre-tool execution hook - inject context."""
import sys
import json
import os

def main():
    # Read tool info from stdin or env
    tool_name = os.environ.get("CLAUDE_TOOL_NAME", "")

    if tool_name in ["Read", "Write", "Edit"]:
        from contextfs import ContextFS
        ctx = ContextFS(auto_load=False)

        # Get file-related context
        file_path = os.environ.get("CLAUDE_TOOL_FILE", "")
        if file_path:
            context = ctx.search(file_path, limit=2)
            if context:
                print(f"## Related Context for {file_path}")
                for r in context:
                    print(f"- [{r.memory.type.value}] {r.memory.content[:100]}...")

if __name__ == "__main__":
    main()
'''
        (self._hooks_dir / "PreToolExecution.py").write_text(pre_tool_hook)

        # Post-Tool Execution Hook
        post_tool_hook = '''#!/usr/bin/env python3
"""ContextFS: Post-tool execution hook - capture observations."""
import sys
import json
import os

def main():
    tool_name = os.environ.get("CLAUDE_TOOL_NAME", "")
    tool_result = os.environ.get("CLAUDE_TOOL_RESULT", "")

    # Capture significant observations
    if tool_name == "Bash" and "error" in tool_result.lower():
        from contextfs import ContextFS
        from contextfs.schemas import MemoryType

        ctx = ContextFS(auto_load=False)
        ctx.save(
            content=f"Error in {tool_name}: {tool_result[:500]}",
            type=MemoryType.ERROR,
            tags=["error", "bash"],
        )

if __name__ == "__main__":
    main()
'''
        (self._hooks_dir / "PostToolExecution.py").write_text(post_tool_hook)

        # Make hooks executable
        for hook_file in self._hooks_dir.glob("*.py"):
            hook_file.chmod(0o755)

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


# Hook entry points for direct execution


def session_start_hook():
    """Entry point for session start hook."""
    plugin = ClaudeCodePlugin()
    plugin.ctx.start_session(tool="claude-code")

    # Load and display relevant context
    context = plugin.ctx.get_context_for_task("current session", limit=5)
    if context:
        print("\n## ContextFS: Loaded Context\n")
        for c in context:
            print(f"- {c}\n")


def session_end_hook():
    """Entry point for session end hook."""
    plugin = ClaudeCodePlugin()
    plugin.ctx.end_session(generate_summary=True)


def capture_message_hook(role: str, content: str):
    """Capture a message during the session."""
    plugin = ClaudeCodePlugin()
    plugin.ctx.add_message(role, content)


# CLI commands


def install_claude_code():
    """Install Claude Code plugin."""
    plugin = ClaudeCodePlugin()
    plugin.install()


def uninstall_claude_code():
    """Uninstall Claude Code plugin."""
    plugin = ClaudeCodePlugin()
    plugin.uninstall()
