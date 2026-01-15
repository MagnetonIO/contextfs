"""ChromaDB server management utilities."""

import shutil
import subprocess
from pathlib import Path

from .pid import get_chroma_pid_file, read_pid_file


def check_chroma_running(host: str, port: int) -> dict | None:
    """Check if ChromaDB server is running. Returns status dict or None if not running."""
    import json
    import urllib.error
    import urllib.request

    try:
        url = f"http://{host}:{port}/api/v2/heartbeat"
        with urllib.request.urlopen(url, timeout=2) as response:
            data = json.loads(response.read().decode())
            return {"running": True, "heartbeat": data.get("nanosecond heartbeat")}
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def get_chroma_pid(port: int = 8000) -> int | None:
    """Get PID of running ChromaDB process.

    First checks PID file, then falls back to port/process scanning.
    """
    # Check PID file first
    pid = read_pid_file(get_chroma_pid_file())
    if pid:
        return pid

    # Fallback: scan for process on port
    try:
        # Try lsof first (macOS/Linux)
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
        # Fallback: pgrep for chroma run
        result = subprocess.run(
            ["pgrep", "-f", f"chroma run.*{port}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().split("\n")[0])
    except FileNotFoundError:
        pass

    return None


def find_chroma_bin() -> str | None:
    """Find the chroma CLI executable."""
    import sys

    chroma_bin = shutil.which("chroma")
    if chroma_bin:
        return chroma_bin

    # Try to find it relative to the Python executable (e.g., in same venv)
    python_dir = Path(sys.executable).parent
    possible_paths = [
        python_dir / "chroma",
        python_dir.parent / "bin" / "chroma",
    ]
    for p in possible_paths:
        if p.exists():
            return str(p)

    return None


def get_chroma_service_paths() -> dict:
    """Get platform-specific service file paths for ChromaDB."""
    import platform

    system = platform.system()
    home = Path.home()

    if system == "Darwin":
        return {
            "platform": "macos",
            "service_file": home / "Library/LaunchAgents/com.contextfs.chromadb.plist",
            "service_name": "com.contextfs.chromadb",
        }
    elif system == "Linux":
        return {
            "platform": "linux",
            "service_file": home / ".config/systemd/user/contextfs-chromadb.service",
            "service_name": "contextfs-chromadb",
        }
    elif system == "Windows":
        return {
            "platform": "windows",
            "service_name": "ContextFS-ChromaDB",
        }
    else:
        return {"platform": "unknown"}


def stop_chroma(port: int = 8000, timeout: float = 5.0) -> bool:
    """Stop the ChromaDB server and wait for it to terminate.

    Uses PID file for tracking. If a system service is managing ChromaDB,
    use 'contextfs uninstall-service chroma' to stop and remove it first.
    """
    import os
    import signal
    import time

    pid = get_chroma_pid(port)
    if not pid:
        get_chroma_pid_file().unlink(missing_ok=True)
        return False

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        get_chroma_pid_file().unlink(missing_ok=True)
        return False

    # Wait for process to actually terminate
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            os.kill(pid, 0)  # Check if process exists
            time.sleep(0.1)
        except ProcessLookupError:
            get_chroma_pid_file().unlink(missing_ok=True)
            return True  # Process is gone

    # Process didn't terminate gracefully, force kill
    try:
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.2)
        get_chroma_pid_file().unlink(missing_ok=True)
        return True
    except ProcessLookupError:
        get_chroma_pid_file().unlink(missing_ok=True)
        return True


def install_chroma_macos_service(host: str, port: int, data_path: Path, chroma_bin: str) -> bool:
    """Install launchd service on macOS."""
    import plistlib

    paths = get_chroma_service_paths()
    plist_path = paths["service_file"]
    plist_path.parent.mkdir(parents=True, exist_ok=True)

    plist_content = {
        "Label": paths["service_name"],
        "ProgramArguments": [
            chroma_bin,
            "run",
            "--path",
            str(data_path),
            "--host",
            host,
            "--port",
            str(port),
        ],
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(Path.home() / ".contextfs/logs/chromadb.log"),
        "StandardErrorPath": str(Path.home() / ".contextfs/logs/chromadb.err"),
    }

    # Ensure log directory exists
    (Path.home() / ".contextfs/logs").mkdir(parents=True, exist_ok=True)

    with open(plist_path, "wb") as f:
        plistlib.dump(plist_content, f)

    # Load the service
    subprocess.run(["launchctl", "load", str(plist_path)], check=True)
    return True


def install_chroma_linux_service(host: str, port: int, data_path: Path, chroma_bin: str) -> bool:
    """Install systemd user service on Linux."""
    paths = get_chroma_service_paths()
    service_path = paths["service_file"]
    service_path.parent.mkdir(parents=True, exist_ok=True)

    service_content = f"""[Unit]
Description=ChromaDB Server for ContextFS
After=network.target

[Service]
Type=simple
ExecStart={chroma_bin} run --path {data_path} --host {host} --port {port}
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


def install_chroma_windows_service(host: str, port: int, data_path: Path, chroma_bin: str) -> bool:
    """Install Windows Task Scheduler task."""
    paths = get_chroma_service_paths()
    task_name = paths["service_name"]

    # Create XML for scheduled task
    xml_content = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>ChromaDB Server for ContextFS</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
  </Triggers>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>3</Count>
    </RestartOnFailure>
  </Settings>
  <Actions>
    <Exec>
      <Command>{chroma_bin}</Command>
      <Arguments>run --path {data_path} --host {host} --port {port}</Arguments>
    </Exec>
  </Actions>
</Task>
"""

    # Write temp XML file and import
    temp_xml = Path.home() / ".contextfs" / "chromadb_task.xml"
    temp_xml.parent.mkdir(parents=True, exist_ok=True)
    temp_xml.write_text(xml_content, encoding="utf-16")

    subprocess.run(
        ["schtasks", "/create", "/tn", task_name, "/xml", str(temp_xml), "/f"],
        check=True,
    )
    temp_xml.unlink()

    # Start the task now
    subprocess.run(["schtasks", "/run", "/tn", task_name], check=True)
    return True


def uninstall_chroma_service() -> bool:
    """Uninstall the ChromaDB service for the current platform."""
    paths = get_chroma_service_paths()
    platform = paths["platform"]

    if platform == "macos":
        plist_path = paths["service_file"]
        if plist_path.exists():
            subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
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
    elif platform == "windows":
        subprocess.run(["schtasks", "/delete", "/tn", paths["service_name"], "/f"], check=False)
        return True
    return False
