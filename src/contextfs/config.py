"""
Configuration for ContextFS.
"""

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class BackendType(str, Enum):
    """Storage backend types."""

    SQLITE = "sqlite"  # SQLite + ChromaDB (default)
    POSTGRES = "postgres"  # PostgreSQL with pgvector + AGE
    SQLITE_FALKORDB = "sqlite+falkordb"  # SQLite + ChromaDB + FalkorDB
    POSTGRES_FALKORDB = "postgres+falkordb"  # PostgreSQL + FalkorDB


class MergeStrategyType(str, Enum):
    """Default merge strategy for lineage operations."""

    UNION = "union"
    INTERSECTION = "intersection"
    LATEST = "latest"
    OLDEST = "oldest"


class Config(BaseSettings):
    """
    ContextFS configuration.

    Loaded from environment variables with CONTEXTFS_ prefix.
    """

    # =================================================================
    # Backend Selection
    # =================================================================
    backend: BackendType = BackendType.SQLITE

    # Data directory (for sqlite/chromadb backends)
    data_dir: Path = Field(default=Path.home() / ".contextfs")

    # =================================================================
    # SQLite Configuration
    # =================================================================
    sqlite_filename: str = "context.db"

    # =================================================================
    # PostgreSQL Configuration
    # =================================================================
    postgres_url: str = "postgresql://contextfs:contextfs@localhost:5432/contextfs"
    postgres_pgvector: bool = True  # Enable pgvector extension
    postgres_age: bool = True  # Enable Apache AGE extension

    # =================================================================
    # ChromaDB Configuration
    # =================================================================
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_collection: str = "contextfs_memories"
    # None = embedded mode (for tests/dev), "localhost" = HTTP server mode (recommended for production)
    # Set CONTEXTFS_CHROMA_HOST=localhost in production to prevent corruption from concurrent access
    chroma_host: str | None = None
    chroma_port: int = 8000
    # Auto-start ChromaDB server if not running (only applies when chroma_host is set)
    chroma_auto_server: bool = True

    # =================================================================
    # Embedding Backend Configuration
    # =================================================================
    # Backend: "auto" (fastembed if installed, else sentence_transformers),
    #          "fastembed" (ONNX, faster), or "sentence_transformers"
    embedding_backend: str = "auto"
    # GPU acceleration: None = auto-detect, True = force GPU, False = force CPU
    use_gpu: bool | None = None
    # Parallel workers for embedding (None = auto, 0 = all cores)
    embedding_parallel_workers: int | None = None
    # Batch size for embedding generation (larger = fewer model calls but more memory)
    embedding_batch_size: int = 256

    # =================================================================
    # FalkorDB Configuration
    # =================================================================
    falkordb_enabled: bool = False
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_password: str | None = None
    falkordb_graph_name: str = "contextfs_memory"

    # =================================================================
    # Memory Lineage Settings (CORE FEATURE)
    # =================================================================
    lineage_auto_track: bool = True  # Auto-track evolution on updates
    lineage_merge_strategy: MergeStrategyType = MergeStrategyType.UNION
    lineage_preserve_tags: bool = True  # Preserve tags when evolving
    auto_link_enabled: bool = True  # Auto-link memories based on semantic similarity
    auto_link_threshold: float = 0.55  # Minimum similarity score to auto-link (0.0-1.0)
    auto_link_max: int = 3  # Maximum auto-links per memory save

    # =================================================================
    # Search Settings
    # =================================================================
    default_search_limit: int = 10
    min_similarity_score: float = 0.3

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # =================================================================
    # Indexing Settings
    # =================================================================
    max_commits: int = 500  # Maximum git commits to index (0 = unlimited)

    # =================================================================
    # Session Settings
    # =================================================================
    auto_save_sessions: bool = True
    auto_load_on_startup: bool = True
    session_timeout_minutes: int = 60

    # =================================================================
    # API Keys (optional, for LLM features)
    # =================================================================
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    # Default AI model for summaries
    default_ai_model: str = "claude"  # or "openai"
    claude_model: str = "claude-3-sonnet-20240229"
    openai_model: str = "gpt-3.5-turbo"

    # =================================================================
    # MCP Server Settings
    # =================================================================
    mcp_enabled: bool = True
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8003
    mcp_sse_path: str = "/sse"  # SSE endpoint path (FastMCP standard)
    mcp_message_path: str = "/messages/"  # Message endpoint path

    # =================================================================
    # Logging
    # =================================================================
    log_level: str = "INFO"
    log_file: str | None = None

    # =================================================================
    # Development
    # =================================================================
    debug: bool = False
    test_mode: bool = False

    # Top 20 programming languages + extras
    supported_extensions: list[str] = Field(
        default=[
            # Top 20 programming languages
            ".py",  # 1. Python
            ".js",  # 2. JavaScript
            ".ts",  # 3. TypeScript
            ".java",  # 4. Java
            ".cpp",
            ".cc",
            ".cxx",
            ".hpp",  # 5. C++
            ".c",
            ".h",  # 6. C
            ".cs",  # 7. C#
            ".go",  # 8. Go
            ".rs",  # 9. Rust
            ".php",  # 10. PHP
            ".rb",  # 11. Ruby
            ".swift",  # 12. Swift
            ".kt",
            ".kts",  # 13. Kotlin
            ".scala",  # 14. Scala
            ".r",
            ".R",  # 15. R
            ".m",
            ".mm",  # 16. Objective-C / MATLAB
            ".pl",
            ".pm",  # 17. Perl
            ".lua",  # 18. Lua
            ".hs",
            ".lhs",  # 19. Haskell
            ".ex",
            ".exs",  # 20. Elixir
            # Additional languages
            ".dart",  # Dart
            ".jl",  # Julia
            ".clj",
            ".cljs",  # Clojure
            ".erl",
            ".hrl",  # Erlang
            ".fs",
            ".fsx",  # F#
            ".v",  # V / Verilog
            ".zig",  # Zig
            ".nim",  # Nim
            ".cr",  # Crystal
            ".groovy",  # Groovy
            # Web
            ".html",
            ".htm",
            ".css",
            ".scss",
            ".sass",
            ".less",
            ".jsx",
            ".tsx",
            ".vue",
            ".svelte",
            # Config/Data
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".xml",
            ".ini",
            ".conf",
            ".env",
            # Documentation
            ".md",
            ".rst",
            ".txt",
            ".tex",
            ".org",
            ".adoc",
            # Shell
            ".sh",
            ".bash",
            ".zsh",
            ".fish",
            ".ps1",
            ".bat",
            ".cmd",
            # Database
            ".sql",
            ".prisma",
            ".graphql",
            # DevOps
            ".dockerfile",
            ".tf",
            ".hcl",
            # Other
            ".proto",
            ".thrift",
        ]
    )

    ignored_directories: list[str] = Field(
        default=[
            ".git",
            ".svn",
            ".hg",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "node_modules",
            "vendor",
            "venv",
            ".venv",
            "env",
            "build",
            "dist",
            "target",
            "out",
            "bin",
            "obj",
            ".idea",
            ".vscode",
            ".vs",
            "coverage",
            ".coverage",
            "htmlcov",
            ".next",
            ".nuxt",
            ".output",
            ".contextfs",  # Don't index our own data
        ]
    )

    class Config:
        env_prefix = "CONTEXTFS_"
        env_nested_delimiter = "__"


# Global config instance
_config: Config | None = None


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


def reset_config() -> None:
    """Reset global config instance (useful for tests)."""
    global _config
    _config = None
