#!/usr/bin/env python3
"""
Install ContextFS MCP server for Claude Desktop.

Cross-platform script that:
1. Finds the installed contextfs-mcp path
2. Updates Claude Desktop config
3. Works on Windows, macOS, and Linux
"""

import json
import os
import platform
import shutil
import sys
from pathlib import Path


def get_claude_desktop_config_path() -> Path:
    """Get Claude Desktop config path for current platform."""
    system = platform.system()

    if system == "Darwin":  # macOS
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    elif system == "Windows":
        return Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json"
    else:  # Linux
        return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def find_contextfs_mcp_path() -> str:
    """Find the contextfs-mcp executable path."""
    # Try finding in PATH first
    path = shutil.which("contextfs-mcp")
    if path:
        # On Windows, return as-is; on Unix, resolve symlinks
        if platform.system() != "Windows":
            path = os.path.realpath(path)
        return path

    # Try finding via pip
    import subprocess

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "-f", "contextfs"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            lines = result.stdout.split("\n")
            location = None
            in_files = False
            for line in lines:
                if line.startswith("Location:"):
                    location = line.split(":", 1)[1].strip()
                if line.strip() == "Files:":
                    in_files = True
                    continue
                if in_files and "contextfs-mcp" in line:
                    # Construct full path
                    rel_path = line.strip()
                    if location:
                        full_path = Path(location) / rel_path
                        if full_path.exists():
                            return str(full_path.resolve())
    except Exception:
        pass

    # Fallback: use python -m
    return None


def get_mcp_config(contextfs_path: str | None) -> dict:
    """Generate MCP server config."""
    if contextfs_path:
        # Use the direct executable path
        return {"command": contextfs_path, "env": {"CONTEXTFS_SOURCE_TOOL": "claude-desktop"}}
    else:
        # Fallback to python -m (requires python in PATH)
        return {
            "command": sys.executable,
            "args": ["-m", "contextfs.mcp_server"],
            "env": {"CONTEXTFS_SOURCE_TOOL": "claude-desktop"},
        }


def install_claude_desktop():
    """Install ContextFS MCP server for Claude Desktop."""
    print("Installing ContextFS MCP server for Claude Desktop...")

    # Find contextfs-mcp path
    contextfs_path = find_contextfs_mcp_path()
    if contextfs_path:
        print(f"Found contextfs-mcp at: {contextfs_path}")
    else:
        print(f"Using Python module fallback: {sys.executable}")

    # Get config path
    config_path = get_claude_desktop_config_path()
    print(f"Claude Desktop config: {config_path}")

    # Load existing config or create new
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print("Loaded existing config")
    else:
        config = {}
        config_path.parent.mkdir(parents=True, exist_ok=True)
        print("Creating new config")

    # Ensure mcpServers exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Add/update contextfs
    config["mcpServers"]["contextfs"] = get_mcp_config(contextfs_path)

    # Write config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("\n✅ ContextFS MCP server installed successfully!")
    print("\nConfiguration added:")
    print(json.dumps(config["mcpServers"]["contextfs"], indent=2))
    print("\n⚠️  Restart Claude Desktop to activate the new MCP server.")

    # Show available tools
    print("\nAvailable MCP tools:")
    print("  • contextfs_save - Save memories (with project grouping)")
    print("  • contextfs_search - Search memories (cross-repo support)")
    print("  • contextfs_list - List recent memories")
    print("  • contextfs_list_repos - List repositories")
    print("  • contextfs_list_projects - List projects")
    print("  • contextfs_list_tools - List source tools")
    print("  • contextfs_recall - Recall by ID")
    print("  • contextfs_sessions - List sessions")


def uninstall_claude_desktop():
    """Remove ContextFS from Claude Desktop config."""
    config_path = get_claude_desktop_config_path()

    if not config_path.exists():
        print("Claude Desktop config not found.")
        return

    with open(config_path) as f:
        config = json.load(f)

    if "mcpServers" in config and "contextfs" in config["mcpServers"]:
        del config["mcpServers"]["contextfs"]

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print("✅ ContextFS removed from Claude Desktop config.")
        print("⚠️  Restart Claude Desktop to apply changes.")
    else:
        print("ContextFS not found in config.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Install ContextFS for Claude Desktop")
    parser.add_argument("--uninstall", action="store_true", help="Remove from Claude Desktop")
    args = parser.parse_args()

    if args.uninstall:
        uninstall_claude_desktop()
    else:
        install_claude_desktop()


if __name__ == "__main__":
    main()
