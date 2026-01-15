"""MCP server management utilities."""

import subprocess
from pathlib import Path

from .pid import get_mcp_pid_file, read_pid_file


def check_mcp_running(host: str = "127.0.0.1", port: int = 8003) -> dict | None:
    """Check if MCP server is running. Returns status dict or None if not running."""
    import json
    import urllib.error
    import urllib.request

    try:
        url = f"http://{host}:{port}/health"
        with urllib.request.urlopen(url, timeout=2) as response:
            data = json.loads(response.read().decode())
            return {"running": True, "status": data.get("status", "ok")}
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def get_mcp_pid(port: int = 8003) -> int | None:
    """Get PID of running MCP server.

    First checks PID file, then falls back to port/process scanning.
    """
    # Check PID file first
    pid = read_pid_file(get_mcp_pid_file())
    if pid:
        return pid

    # Fallback: scan for process on port
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().split("\n")[0])
    except FileNotFoundError:
        pass

    try:
        result = subprocess.run(
            ["pgrep", "-f", f"contextfs.mcp.server.*{port}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().split("\n")[0])
    except FileNotFoundError:
        pass

    return None


def get_mcp_service_paths() -> dict:
    """Get platform-specific service file paths for MCP."""
    import platform

    system = platform.system()
    home = Path.home()

    if system == "Darwin":
        return {
            "platform": "macos",
            "service_file": home / "Library/LaunchAgents/com.contextfs.mcp-server.plist",
            "service_name": "com.contextfs.mcp-server",
        }
    elif system == "Linux":
        return {
            "platform": "linux",
            "service_file": home / ".config/systemd/user/contextfs-mcp.service",
            "service_name": "contextfs-mcp",
        }
    else:
        return {"platform": "unknown"}


def stop_mcp(port: int = 8003, timeout: float = 5.0) -> bool:
    """Stop the MCP server and wait for it to terminate.

    Uses PID file for tracking. If a system service is managing MCP,
    use 'contextfs uninstall-service mcp' to stop and remove it first.
    """
    import os
    import signal
    import time

    pid = get_mcp_pid(port)
    if not pid:
        # Clean up stale PID file if exists
        get_mcp_pid_file().unlink(missing_ok=True)
        return False

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        get_mcp_pid_file().unlink(missing_ok=True)
        return False

    # Wait for process to actually terminate
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            os.kill(pid, 0)  # Check if process exists
            time.sleep(0.1)
        except ProcessLookupError:
            get_mcp_pid_file().unlink(missing_ok=True)
            return True  # Process is gone

    # Process didn't terminate gracefully, force kill
    try:
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.2)
        get_mcp_pid_file().unlink(missing_ok=True)
        return True
    except ProcessLookupError:
        get_mcp_pid_file().unlink(missing_ok=True)
        return True


def install_mcp_macos_service(host: str, port: int) -> bool:
    """Install launchd service for MCP server on macOS."""
    import plistlib
    import sys

    paths = get_mcp_service_paths()
    plist_path = paths["service_file"]
    plist_path.parent.mkdir(parents=True, exist_ok=True)

    # Find Python executable
    python_bin = sys.executable

    plist_content = {
        "Label": paths["service_name"],
        "ProgramArguments": [
            python_bin,
            "-m",
            "contextfs.mcp.server",
            "--host",
            host,
            "--port",
            str(port),
        ],
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(Path.home() / ".contextfs/logs/mcp.log"),
        "StandardErrorPath": str(Path.home() / ".contextfs/logs/mcp.err"),
    }

    # Ensure log directory exists
    (Path.home() / ".contextfs/logs").mkdir(parents=True, exist_ok=True)

    with open(plist_path, "wb") as f:
        plistlib.dump(plist_content, f)

    # Load the service
    subprocess.run(["launchctl", "load", str(plist_path)], check=True)
    return True


def install_mcp_linux_service(host: str, port: int) -> bool:
    """Install systemd user service for MCP server on Linux."""
    import sys

    paths = get_mcp_service_paths()
    service_path = paths["service_file"]
    service_path.parent.mkdir(parents=True, exist_ok=True)

    python_bin = sys.executable

    service_content = f"""[Unit]
Description=ContextFS MCP Server
After=network.target

[Service]
Type=simple
ExecStart={python_bin} -m contextfs.mcp.server --host {host} --port {port}
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
"""

    service_path.write_text(service_content)

    # Enable and start the service
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", paths["service_name"]], check=True)
    subprocess.run(["systemctl", "--user", "start", paths["service_name"]], check=True)
    return True


def uninstall_mcp_service() -> bool:
    """Uninstall the MCP service for the current platform."""
    paths = get_mcp_service_paths()
    platform = paths["platform"]

    if platform == "macos":
        plist_path = paths["service_file"]
        if plist_path.exists():
            subprocess.run(["launchctl", "unload", "-w", str(plist_path)], check=False)
            plist_path.unlink()
        return True
    elif platform == "linux":
        service_path = paths["service_file"]
        if service_path.exists():
            subprocess.run(["systemctl", "--user", "stop", paths["service_name"]], check=False)
            subprocess.run(["systemctl", "--user", "disable", paths["service_name"]], check=False)
            service_path.unlink()
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
        return True
    return False
