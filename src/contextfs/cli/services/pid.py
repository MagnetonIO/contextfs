"""PID file utilities for server process management."""

from pathlib import Path


def get_pid_dir() -> Path:
    """Get the directory for PID files."""
    pid_dir = Path.home() / ".contextfs" / "run"
    pid_dir.mkdir(parents=True, exist_ok=True)
    return pid_dir


def get_mcp_pid_file() -> Path:
    """Get the MCP server PID file path."""
    return get_pid_dir() / "mcp.pid"


def get_chroma_pid_file() -> Path:
    """Get the ChromaDB server PID file path."""
    return get_pid_dir() / "chroma.pid"


def read_pid_file(pid_file: Path) -> int | None:
    """Read PID from file, return None if file doesn't exist or PID is stale."""
    import os

    if not pid_file.exists():
        return None

    try:
        pid = int(pid_file.read_text().strip())
        # Check if process is still running
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        # PID file is stale, clean it up
        pid_file.unlink(missing_ok=True)
        return None


def write_pid_file(pid_file: Path, pid: int) -> None:
    """Write PID to file."""
    pid_file.write_text(str(pid))
