"""Service management utilities for MCP and ChromaDB servers."""

from .chroma import (
    check_chroma_running,
    find_chroma_bin,
    get_chroma_pid,
    get_chroma_service_paths,
    install_chroma_linux_service,
    install_chroma_macos_service,
    install_chroma_windows_service,
    stop_chroma,
    uninstall_chroma_service,
)
from .mcp import (
    check_mcp_running,
    get_mcp_pid,
    get_mcp_service_paths,
    install_mcp_linux_service,
    install_mcp_macos_service,
    stop_mcp,
    uninstall_mcp_service,
)
from .pid import (
    get_chroma_pid_file,
    get_mcp_pid_file,
    get_pid_dir,
    read_pid_file,
    write_pid_file,
)

__all__ = [
    # PID utilities
    "get_pid_dir",
    "get_mcp_pid_file",
    "get_chroma_pid_file",
    "read_pid_file",
    "write_pid_file",
    # MCP
    "check_mcp_running",
    "get_mcp_pid",
    "get_mcp_service_paths",
    "stop_mcp",
    "install_mcp_macos_service",
    "install_mcp_linux_service",
    "uninstall_mcp_service",
    # ChromaDB
    "check_chroma_running",
    "get_chroma_pid",
    "find_chroma_bin",
    "get_chroma_service_paths",
    "stop_chroma",
    "install_chroma_macos_service",
    "install_chroma_linux_service",
    "install_chroma_windows_service",
    "uninstall_chroma_service",
]
