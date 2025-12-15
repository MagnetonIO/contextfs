"""
Gemini CLI Plugin for ContextFS.

Provides integration with Google's Gemini CLI tool.
"""

import json
from pathlib import Path

from contextfs.core import ContextFS


class GeminiPlugin:
    """
    Gemini CLI integration plugin.

    Provides context injection and session capture for Gemini CLI.
    """

    def __init__(self, ctx: ContextFS | None = None):
        """
        Initialize Gemini plugin.

        Args:
            ctx: ContextFS instance
        """
        self.ctx = ctx or ContextFS(auto_load=True)
        self._config_dir = Path.home() / ".gemini"

    def install(self) -> None:
        """Install Gemini CLI integration."""
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
"""ContextFS wrapper for Gemini CLI."""
import sys
import os
import subprocess

from contextfs import ContextFS
from contextfs.schemas import MemoryType


def main():
    ctx = ContextFS(auto_load=True)

    # Start session
    session = ctx.start_session(tool="gemini")

    # Get relevant context
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        context = ctx.get_context_for_task(query, limit=3)

        if context:
            print("## ContextFS: Loaded Context")
            for c in context:
                print(f"- {c}")
            print()

    # Run actual gemini command
    try:
        result = subprocess.run(
            ["gemini"] + sys.argv[1:],
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

        print("Gemini plugin installed successfully.")
        print(f"Config: {config_file}")
        print(f"Wrapper: {wrapper_file}")
        print("\nTo use: python ~/.gemini/contextfs_wrapper.py <your prompt>")
        print("Or add an alias: alias gemini-ctx='python ~/.gemini/contextfs_wrapper.py'")

    def uninstall(self) -> None:
        """Uninstall Gemini CLI integration."""
        config_file = self._config_dir / "contextfs.json"
        wrapper_file = self._config_dir / "contextfs_wrapper.py"

        if config_file.exists():
            config_file.unlink()
        if wrapper_file.exists():
            wrapper_file.unlink()

        print("Gemini plugin uninstalled.")

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


# CLI commands


def install_gemini():
    """Install Gemini plugin."""
    plugin = GeminiPlugin()
    plugin.install()


def uninstall_gemini():
    """Uninstall Gemini plugin."""
    plugin = GeminiPlugin()
    plugin.uninstall()
