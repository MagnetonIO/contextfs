"""
Claude Code Plugin for ContextFS.

Provides:
- Lifecycle hooks for automatic context capture
- Skills for memory operations (/remember, /recall)
- Memory enforcement via CLAUDE.md
- Memory-extractor agent
- Auto-save sessions on exit
"""

import json
import os
import subprocess
import sys
from importlib import resources
from pathlib import Path

from contextfs.core import ContextFS

# Marker to identify ContextFS-managed sections in CLAUDE.md
CLAUDE_MD_MARKER_START = "<!-- CONTEXTFS_MEMORY_START -->"
CLAUDE_MD_MARKER_END = "<!-- CONTEXTFS_MEMORY_END -->"


class ClaudeCodePlugin:
    """
    Claude Code integration plugin.

    Hooks into Claude Code's lifecycle events to automatically
    capture and inject context. Supports both user-level (~/.claude/)
    and project-level (.claude/) installation.
    """

    def __init__(self, ctx: ContextFS | None = None, project_path: Path | None = None):
        """
        Initialize Claude Code plugin.

        Args:
            ctx: ContextFS instance (creates one if not provided)
            project_path: Path to project for project-level installation (optional)
        """
        self.ctx = ctx or ContextFS(auto_load=True)
        self._project_path = project_path

        # User-level paths
        self._user_settings_file = Path.home() / ".claude" / "settings.json"
        self._user_commands_dir = Path.home() / ".claude" / "commands"
        self._user_agents_dir = Path.home() / ".claude" / "agents"

        # Project-level paths (if project_path provided)
        if project_path:
            self._project_claude_dir = project_path / ".claude"
            self._project_commands_dir = self._project_claude_dir / "commands"
            self._project_agents_dir = self._project_claude_dir / "agents"
            self._project_claude_md = project_path / "CLAUDE.md"
        else:
            self._project_claude_dir = None
            self._project_commands_dir = None
            self._project_agents_dir = None
            self._project_claude_md = None

        # Templates directory
        self._templates_dir = self._get_templates_dir()

    def _get_templates_dir(self) -> Path:
        """Get the templates directory from package resources."""
        try:
            # Try to get from installed package
            with resources.files("contextfs") as pkg_path:
                templates_path = Path(pkg_path) / "templates"
                if templates_path.exists():
                    return templates_path
        except Exception:
            pass

        # Fallback: relative to this file (for development)
        return Path(__file__).parent.parent / "templates"

    def install(self, include_project: bool = True) -> None:
        """
        Install Claude Code hooks, skills, and memory enforcement.

        Args:
            include_project: If True and project_path is set, install project-level components
        """
        # Always install user-level components
        self._install_user_hooks()
        self._install_user_commands()

        print("ContextFS Claude Code plugin installed.")
        print("\nUser-level installation (~/.claude/):")
        print(f"  - Settings: {self._user_settings_file}")
        print(f"  - Commands: {self._user_commands_dir}")
        print("\nInstalled hooks:")
        print("  - SessionStart: Auto-index in background")
        print("  - PreCompact: Auto-save session before compaction")
        print("  - Stop: Auto-save session on exit")
        print("\nInstalled commands:")
        print("  - /remember: Extract and save conversation insights")
        print("  - /recall: Search and load relevant context")

        # Start MCP server and optionally install as service
        print("\nMCP server:")
        self._start_mcp_server()
        if sys.platform == "darwin":
            self._install_launchd_service()
        print("  URL: http://localhost:8003/mcp/sse")

        # Install project-level components if requested and path is set
        if include_project and self._project_path:
            self._install_project_components()
            print(f"\nProject-level installation ({self._project_path}):")
            print(f"  - Commands: {self._project_commands_dir}")
            print(f"  - Agents: {self._project_agents_dir}")
            print("  - CLAUDE.md: Memory protocol merged")

        print("\nRestart Claude Code for changes to take effect.")

    def install_to_project(self, project_path: Path) -> None:
        """
        Install project-level components to a specific project.

        Args:
            project_path: Path to the project directory
        """
        self._project_path = project_path
        self._project_claude_dir = project_path / ".claude"
        self._project_commands_dir = self._project_claude_dir / "commands"
        self._project_agents_dir = self._project_claude_dir / "agents"
        self._project_claude_md = project_path / "CLAUDE.md"

        self._install_project_components()

        print(f"ContextFS project components installed to {project_path}")
        print(f"  - Commands: {self._project_commands_dir}")
        print(f"  - Agents: {self._project_agents_dir}")
        print(
            f"  - CLAUDE.md: Memory protocol {'merged' if self._project_claude_md.exists() else 'created'}"
        )

    def _install_project_components(self) -> None:
        """Install project-level commands, agents, and merge CLAUDE.md."""
        if not self._project_path:
            return

        # Create directories
        self._project_commands_dir.mkdir(parents=True, exist_ok=True)
        self._project_agents_dir.mkdir(parents=True, exist_ok=True)

        # Install commands
        self._copy_template_file("commands/remember.md", self._project_commands_dir / "remember.md")
        self._copy_template_file("commands/recall.md", self._project_commands_dir / "recall.md")

        # Install agents
        self._copy_template_file(
            "agents/memory-extractor.md", self._project_agents_dir / "memory-extractor.md"
        )

        # Merge CLAUDE.md
        self._merge_claude_md()

    def _copy_template_file(self, template_name: str, dest_path: Path) -> None:
        """Copy a template file to destination."""
        template_path = self._templates_dir / template_name
        if template_path.exists():
            dest_path.write_text(template_path.read_text())

    def _merge_claude_md(self) -> None:
        """Merge memory protocol into project's CLAUDE.md."""
        if not self._project_claude_md:
            return

        # Load memory protocol template
        memory_template_path = self._templates_dir / "CLAUDE_MEMORY.md"
        if not memory_template_path.exists():
            print(f"Warning: Memory template not found at {memory_template_path}")
            return

        memory_content = memory_template_path.read_text()
        wrapped_content = f"\n{CLAUDE_MD_MARKER_START}\n{memory_content}\n{CLAUDE_MD_MARKER_END}\n"

        if self._project_claude_md.exists():
            existing_content = self._project_claude_md.read_text()

            # Check if already has ContextFS section
            if CLAUDE_MD_MARKER_START in existing_content:
                # Replace existing section
                import re

                pattern = f"{re.escape(CLAUDE_MD_MARKER_START)}.*?{re.escape(CLAUDE_MD_MARKER_END)}"
                new_content = re.sub(
                    pattern, wrapped_content.strip(), existing_content, flags=re.DOTALL
                )
                self._project_claude_md.write_text(new_content)
            else:
                # Append to existing file
                self._project_claude_md.write_text(existing_content + wrapped_content)
        else:
            # Create new file with just the memory protocol
            self._project_claude_md.write_text(wrapped_content.strip())

    def uninstall(self) -> None:
        """Uninstall Claude Code hooks from user settings."""
        if self._user_settings_file.exists():
            settings = json.loads(self._user_settings_file.read_text())
            if "hooks" in settings:
                # Remove contextfs hooks
                for hook_type in ["SessionStart", "PreCompact", "Stop"]:
                    if hook_type in settings["hooks"]:
                        settings["hooks"][hook_type] = [
                            h
                            for h in settings["hooks"][hook_type]
                            if not self._is_contextfs_hook(h)
                        ]
            # Remove MCP server
            if "mcpServers" in settings and "contextfs" in settings["mcpServers"]:
                del settings["mcpServers"]["contextfs"]

            self._user_settings_file.write_text(json.dumps(settings, indent=2))

        # Remove user-level command files
        for cmd_name in ["remember.md", "recall.md", "contextfs-search.md"]:
            cmd_file = self._user_commands_dir / cmd_name
            if cmd_file.exists():
                cmd_file.unlink()

        print("Claude Code plugin uninstalled from user settings.")

    def uninstall_from_project(self, project_path: Path) -> None:
        """
        Remove ContextFS components from a project.

        Args:
            project_path: Path to the project directory
        """
        claude_md = project_path / "CLAUDE.md"
        commands_dir = project_path / ".claude" / "commands"
        agents_dir = project_path / ".claude" / "agents"

        # Remove CLAUDE.md section
        if claude_md.exists():
            content = claude_md.read_text()
            if CLAUDE_MD_MARKER_START in content:
                import re

                pattern = (
                    f"\n?{re.escape(CLAUDE_MD_MARKER_START)}.*?{re.escape(CLAUDE_MD_MARKER_END)}\n?"
                )
                new_content = re.sub(pattern, "", content, flags=re.DOTALL)
                claude_md.write_text(new_content)

        # Remove command files
        for cmd_name in ["remember.md", "recall.md"]:
            cmd_file = commands_dir / cmd_name
            if cmd_file.exists():
                cmd_file.unlink()

        # Remove agent files
        agent_file = agents_dir / "memory-extractor.md"
        if agent_file.exists():
            agent_file.unlink()

        print(f"ContextFS components removed from {project_path}")

    def _is_contextfs_hook(self, hook_config: dict) -> bool:
        """Check if a hook configuration is from ContextFS."""
        hooks = hook_config.get("hooks", [])
        for h in hooks:
            command = h.get("command", "")
            prompt = h.get("prompt", "")
            if "contextfs" in command or "contextfs" in prompt.lower():
                return True
            if "memory" in prompt.lower() and "save" in prompt.lower():
                return True
        return False

    def _install_user_hooks(self) -> None:
        """Install lifecycle hooks into user-level Claude Code settings."""
        self._user_settings_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing settings or create new
        if self._user_settings_file.exists():
            settings = json.loads(self._user_settings_file.read_text())
        else:
            settings = {}

        # Ensure hooks section exists
        if "hooks" not in settings:
            settings["hooks"] = {}

        # Add SessionStart hook (indexes on session start in background)
        settings["hooks"]["SessionStart"] = [
            {
                "matcher": "",
                "hooks": [
                    {
                        "type": "command",
                        "command": "uvx contextfs index --quiet --background --require-init",
                    }
                ],
            }
        ]

        # Add PreCompact hook (saves before context compaction)
        settings["hooks"]["PreCompact"] = [
            {
                "matcher": "",
                "hooks": [
                    {
                        "type": "command",
                        "command": "uvx contextfs save-session --label 'auto-compact'",
                    }
                ],
            }
        ]

        # Add Stop hook (auto-save session on exit)
        # Note: Using command type instead of prompt type to avoid Claude Code bug
        # where prompt hooks show errors during non-stop evaluations
        # See: https://github.com/anthropics/claude-code/issues/10463
        settings["hooks"]["Stop"] = [
            {
                "matcher": "",
                "hooks": [
                    {
                        "type": "command",
                        "command": "uvx contextfs save-session --label 'session-end' --quiet",
                    }
                ],
            }
        ]

        # Ensure mcpServers section exists
        if "mcpServers" not in settings:
            settings["mcpServers"] = {}

        # Add contextfs MCP server (SSE transport)
        # Note: Requires `contextfs mcp-server` to be running
        settings["mcpServers"]["contextfs"] = {
            "type": "sse",
            "url": "http://localhost:8003/mcp/sse",
        }

        # Write updated settings
        self._user_settings_file.write_text(json.dumps(settings, indent=2))

    def _install_user_commands(self) -> None:
        """Install commands to user-level directory."""
        self._user_commands_dir.mkdir(parents=True, exist_ok=True)

        # Copy template commands
        self._copy_template_file("commands/remember.md", self._user_commands_dir / "remember.md")
        self._copy_template_file("commands/recall.md", self._user_commands_dir / "recall.md")

    def _start_mcp_server(self, port: int = 8003) -> bool:
        """
        Start the MCP server if not already running.

        Args:
            port: Port to run the server on

        Returns:
            True if server is running (started or was already running)
        """
        import socket

        # Check if server is already running
        def is_port_in_use(port: int) -> bool:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(("127.0.0.1", port)) == 0

        if is_port_in_use(port):
            print(f"  MCP server already running on port {port}")
            return True

        # Start the server in background
        try:
            # Use subprocess to start in background
            cmd = [sys.executable, "-m", "contextfs.mcp.server"]
            env = os.environ.copy()
            env["CONTEXTFS_MCP_PORT"] = str(port)

            # Start detached process
            if sys.platform == "win32":
                subprocess.Popen(
                    cmd,
                    env=env,
                    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.Popen(
                    cmd,
                    env=env,
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            # Wait briefly and verify it started
            import time

            time.sleep(1)

            if is_port_in_use(port):
                print(f"  MCP server started on port {port}")
                return True
            else:
                print("  Warning: MCP server may not have started correctly")
                return False

        except Exception as e:
            print(f"  Warning: Could not start MCP server: {e}")
            return False

    def _install_launchd_service(self, port: int = 8003) -> bool:
        """
        Install launchd service for macOS to auto-start MCP server on login.

        Args:
            port: Port to run the server on

        Returns:
            True if service was installed successfully
        """
        if sys.platform != "darwin":
            return False

        plist_dir = Path.home() / "Library" / "LaunchAgents"
        plist_file = plist_dir / "com.contextfs.mcp-server.plist"

        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.contextfs.mcp-server</string>
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>-m</string>
        <string>contextfs.mcp.server</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>CONTEXTFS_MCP_PORT</key>
        <string>{port}</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/contextfs-mcp.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/contextfs-mcp.err</string>
</dict>
</plist>
"""
        try:
            plist_dir.mkdir(parents=True, exist_ok=True)
            plist_file.write_text(plist_content)

            # Load the service
            subprocess.run(["launchctl", "unload", str(plist_file)], capture_output=True)
            subprocess.run(["launchctl", "load", str(plist_file)], capture_output=True, check=True)

            print(f"  Launchd service installed: {plist_file}")
            return True

        except Exception as e:
            print(f"  Warning: Could not install launchd service: {e}")
            return False


# CLI commands


def install_claude_code(project_path: Path | None = None):
    """
    Install Claude Code plugin.

    Args:
        project_path: Optional project path for project-level installation
    """
    plugin = ClaudeCodePlugin(project_path=project_path)
    plugin.install(include_project=project_path is not None)


def install_claude_code_to_project(project_path: Path):
    """
    Install ContextFS memory enforcement to a specific project.

    Args:
        project_path: Path to the project directory
    """
    plugin = ClaudeCodePlugin()
    plugin.install_to_project(project_path)


def uninstall_claude_code():
    """Uninstall Claude Code plugin from user settings."""
    plugin = ClaudeCodePlugin()
    plugin.uninstall()


def uninstall_claude_code_from_project(project_path: Path):
    """
    Remove ContextFS components from a project.

    Args:
        project_path: Path to the project directory
    """
    plugin = ClaudeCodePlugin()
    plugin.uninstall_from_project(project_path)
