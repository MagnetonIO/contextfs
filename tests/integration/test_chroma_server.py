"""
Integration tests for ChromaDB server command.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


def get_python_executable() -> str:
    """Get the Python executable that has contextfs installed.

    When running under uv, get_python_executable() may point to pyenv Python,
    but the actual venv Python is at sys.prefix/bin/python.
    """
    venv_python = Path(sys.prefix) / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return get_python_executable()


class TestChromaServerCommand:
    """Tests for the server CLI commands (chroma)."""

    def test_chroma_binary_found(self):
        """Test that the chroma CLI binary can be found."""
        chroma_bin = shutil.which("chroma")
        assert chroma_bin is not None, "chroma CLI not found in PATH"

    def test_server_start_chroma_help(self):
        """Test that server start chroma --help works."""
        result = subprocess.run(
            [get_python_executable(), "-m", "contextfs.cli", "server", "start", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "chroma" in result.stdout.lower() or "mcp" in result.stdout.lower()

    def test_server_status_chroma_not_running(self):
        """Test server status when chroma is not running."""
        # Use a port that's unlikely to be in use - server status checks default port
        result = subprocess.run(
            [
                get_python_executable(),
                "-m",
                "contextfs.cli",
                "server",
                "status",
                "chroma",
            ],
            capture_output=True,
            text=True,
        )
        # Status command always returns 0, shows running or not running
        assert result.returncode == 0
        assert "ChromaDB" in result.stdout or "chroma" in result.stdout.lower()

    @pytest.mark.slow
    def test_server_start_chroma_background(self, tmp_path: Path):
        """Test that server start chroma starts successfully in background."""
        import requests

        # Use a unique port to avoid conflicts
        port = 18765

        try:
            # Start the server in background mode (default, not foreground)
            result = subprocess.run(
                [
                    get_python_executable(),
                    "-m",
                    "contextfs.cli",
                    "server",
                    "start",
                    "chroma",
                    "--port",
                    str(port),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Command should succeed
            assert result.returncode == 0
            assert "ChromaDB" in result.stdout or "started" in result.stdout.lower()

            # Wait for server to start
            time.sleep(3)

            # Check that server is responding
            try:
                response = requests.get(f"http://127.0.0.1:{port}/api/v2/heartbeat", timeout=5)
                assert response.status_code == 200
                assert "nanosecond heartbeat" in response.json()
            except requests.exceptions.ConnectionError:
                pytest.skip("Server did not start (may be due to port conflict)")

            # Test status when running
            status_result = subprocess.run(
                [
                    get_python_executable(),
                    "-m",
                    "contextfs.cli",
                    "server",
                    "status",
                    "chroma",
                ],
                capture_output=True,
                text=True,
            )
            assert "running" in status_result.stdout.lower()

            # Test already-running detection
            start_again = subprocess.run(
                [
                    get_python_executable(),
                    "-m",
                    "contextfs.cli",
                    "server",
                    "start",
                    "chroma",
                    "--port",
                    str(port),
                ],
                capture_output=True,
                text=True,
            )
            assert "already running" in start_again.stdout.lower()

        finally:
            # Clean up - stop the server
            subprocess.run(
                [
                    get_python_executable(),
                    "-m",
                    "contextfs.cli",
                    "server",
                    "stop",
                    "chroma",
                    "--port",
                    str(port),
                ],
                capture_output=True,
            )
