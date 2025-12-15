"""
Configuration for ContextFS.
"""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """
    ContextFS configuration.

    Loaded from environment variables with CONTEXTFS_ prefix.
    """
    # Data directory
    data_dir: Path = Field(default=Path.home() / ".contextfs")

    # Embedding model
    embedding_model: str = "all-MiniLM-L6-v2"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Search
    default_search_limit: int = 10
    min_similarity_score: float = 0.3

    # Session
    auto_save_sessions: bool = True
    auto_load_on_startup: bool = True
    session_timeout_minutes: int = 60

    # API keys (optional, for LLM features)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # Default AI model for summaries
    default_ai_model: str = "claude"  # or "openai"
    claude_model: str = "claude-3-sonnet-20240229"
    openai_model: str = "gpt-3.5-turbo"

    # Top 20 programming languages + extras
    supported_extensions: list[str] = Field(default=[
        # Top 20 programming languages
        ".py",      # 1. Python
        ".js",      # 2. JavaScript
        ".ts",      # 3. TypeScript
        ".java",    # 4. Java
        ".cpp", ".cc", ".cxx", ".hpp",  # 5. C++
        ".c", ".h",  # 6. C
        ".cs",      # 7. C#
        ".go",      # 8. Go
        ".rs",      # 9. Rust
        ".php",     # 10. PHP
        ".rb",      # 11. Ruby
        ".swift",   # 12. Swift
        ".kt", ".kts",  # 13. Kotlin
        ".scala",   # 14. Scala
        ".r", ".R",  # 15. R
        ".m", ".mm",  # 16. Objective-C / MATLAB
        ".pl", ".pm",  # 17. Perl
        ".lua",     # 18. Lua
        ".hs", ".lhs",  # 19. Haskell
        ".ex", ".exs",  # 20. Elixir

        # Additional languages
        ".dart",    # Dart
        ".jl",      # Julia
        ".clj", ".cljs",  # Clojure
        ".erl", ".hrl",  # Erlang
        ".fs", ".fsx",  # F#
        ".v",       # V / Verilog
        ".zig",     # Zig
        ".nim",     # Nim
        ".cr",      # Crystal
        ".groovy",  # Groovy

        # Web
        ".html", ".htm",
        ".css", ".scss", ".sass", ".less",
        ".jsx", ".tsx",
        ".vue", ".svelte",

        # Config/Data
        ".json", ".yaml", ".yml", ".toml",
        ".xml", ".ini", ".conf", ".env",

        # Documentation
        ".md", ".rst", ".txt", ".tex",
        ".org", ".adoc",

        # Shell
        ".sh", ".bash", ".zsh", ".fish",
        ".ps1", ".bat", ".cmd",

        # Database
        ".sql", ".prisma", ".graphql",

        # DevOps
        ".dockerfile", ".tf", ".hcl",

        # Other
        ".proto", ".thrift",
    ])

    ignored_directories: list[str] = Field(default=[
        ".git", ".svn", ".hg",
        "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
        "node_modules", "vendor", "venv", ".venv", "env",
        "build", "dist", "target", "out", "bin", "obj",
        ".idea", ".vscode", ".vs",
        "coverage", ".coverage", "htmlcov",
        ".next", ".nuxt", ".output",
        ".contextfs",  # Don't index our own data
    ])

    class Config:
        env_prefix = "CONTEXTFS_"
        env_nested_delimiter = "__"


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set global config instance."""
    global _config
    _config = config
