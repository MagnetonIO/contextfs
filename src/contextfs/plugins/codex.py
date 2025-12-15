"""
Codex CLI Plugin for ContextFS.

Provides integration with OpenAI's Codex CLI tool.
"""

import json
from pathlib import Path

from contextfs.core import ContextFS


class CodexPlugin:
    """
    Codex CLI integration plugin.

    Provides context injection and session capture for Codex CLI.
    """

    def __init__(self, ctx: ContextFS | None = None):
        """
        Initialize Codex plugin.

        Args:
            ctx: ContextFS instance
        """
        self.ctx = ctx or ContextFS(auto_load=True)
        self._config_dir = Path.home() / ".codex"

    def install(self) -> None:
        """Install Codex CLI integration."""
        self._config_dir.mkdir(parents=True, exist_ok=True)

        # Create contextfs integration config
        config = {
            "contextfs": {
                "enabled": True,
                "auto_inject_context": True,
                "auto_save_sessions": True,
                "context_limit": 5,
            }
        }

        config_file = self._config_dir / "contextfs.json"
        config_file.write_text(json.dumps(config, indent=2))

        # Create wrapper script
        wrapper_script = '''#!/usr/bin/env python3
"""ContextFS wrapper for Codex CLI."""
import sys
import os
import subprocess

from contextfs import ContextFS
from contextfs.schemas import MemoryType


def main():
    ctx = ContextFS(auto_load=True)

    # Start session
    session = ctx.start_session(tool="codex")

    # Get relevant context
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        context = ctx.get_context_for_task(query, limit=3)

        if context:
            print("## ContextFS: Loaded Context")
            for c in context:
                print(f"- {c}")
            print()

    # Run actual codex command
    try:
        result = subprocess.run(
            ["codex"] + sys.argv[1:],
            capture_output=False,
        )
    finally:
        # End session
        ctx.end_session(generate_summary=True)

    sys.exit(result.returncode if result else 0)


if __name__ == "__main__":
    main()
'''
        wrapper_file = self._config_dir / "contextfs_wrapper.py"
        wrapper_file.write_text(wrapper_script)
        wrapper_file.chmod(0o755)

        # Create MCP config for Codex
        mcp_config = {
            "mcpServers": {
                "contextfs": {
                    "command": "python",
                    "args": ["-m", "contextfs.mcp_server"],
                }
            }
        }

        mcp_config_file = self._config_dir / "mcp.json"
        mcp_config_file.write_text(json.dumps(mcp_config, indent=2))

        print("Codex plugin installed successfully.")
        print(f"Config: {config_file}")
        print(f"MCP Config: {mcp_config_file}")
        print(f"Wrapper: {wrapper_file}")
        print("\nTo use with MCP, add to your Codex config:")
        print(f'  "mcpServers": {json.dumps(mcp_config["mcpServers"], indent=4)}')
        print("\nOr use wrapper: python ~/.codex/contextfs_wrapper.py <your prompt>")

    def uninstall(self) -> None:
        """Uninstall Codex CLI integration."""
        for filename in ["contextfs.json", "contextfs_wrapper.py", "mcp.json"]:
            filepath = self._config_dir / filename
            if filepath.exists():
                filepath.unlink()

        print("Codex plugin uninstalled.")

    def inject_context(self, prompt: str) -> str:
        """
        Inject relevant context into a prompt.

        Args:
            prompt: Original prompt

        Returns:
            Prompt with context prepended
        """
        context = self.ctx.get_context_for_task(prompt, limit=3)

        if not context:
            return prompt

        context_str = "\n".join(f"- {c}" for c in context)
        return f"""## Relevant Context
{context_str}

## Task
{prompt}"""

    def capture_response(self, prompt: str, response: str) -> None:
        """
        Capture a prompt/response pair.

        Args:
            prompt: User prompt
            response: Model response
        """
        self.ctx.add_message("user", prompt)
        self.ctx.add_message("assistant", response)

    def get_mcp_config(self) -> dict:
        """Get MCP server configuration for Codex."""
        return {
            "contextfs": {
                "command": "python",
                "args": ["-m", "contextfs.mcp_server"],
            }
        }


# CLI commands


def install_codex():
    """Install Codex plugin."""
    plugin = CodexPlugin()
    plugin.install()


def uninstall_codex():
    """Uninstall Codex plugin."""
    plugin = CodexPlugin()
    plugin.uninstall()
