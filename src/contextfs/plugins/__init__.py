"""
Plugins for various AI CLI tools.

Provides integrations for:
- Claude Code (hooks + skills)
- Gemini CLI
- Codex CLI
"""

from contextfs.plugins.claude_code import ClaudeCodePlugin
from contextfs.plugins.codex import CodexPlugin
from contextfs.plugins.gemini import GeminiPlugin

__all__ = [
    "ClaudeCodePlugin",
    "GeminiPlugin",
    "CodexPlugin",
]
