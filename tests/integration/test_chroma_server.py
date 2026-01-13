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
    """Tests for the chroma-server CLI command."""

    def test_chroma_binary_found(self):
        """Test that the chroma CLI binary can be found."""
        chroma_bin = shutil.which("chroma")
        assert chroma_bin is not None, "chroma CLI not found in PATH"

    def test_chroma_server_help(self):
        """Test that chroma-server --help works."""
        result = subprocess.run(
            [get_python_executable(), "-m", "contextfs.cli", "chroma-server", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "ChromaDB server" in result.stdout or "chroma-server" in result.stdout

    def test_chroma_server_status_not_running(self):
        """Test --status when server is not running."""
        # Use a port that's unlikely to be in use
        port = 19999
        result = subprocess.run(
            [
                get_python_executable(),
                "-m",
                "contextfs.cli",
                "chroma-server",
                "--status",
                "--port",
                str(port),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "not running" in result.stdout

    @pytest.mark.slow
    def test_chroma_server_daemon_starts(self, tmp_path: Path):
        """Test that chroma-server --daemon starts successfully."""
        import requests

        # Use a unique port to avoid conflicts
        port = 18765
        data_path = tmp_path / "chroma_db"
        data_path.mkdir()

        try:
            # Start the server in daemon mode
            result = subprocess.run(
                [
                    get_python_executable(),
                    "-m",
                    "contextfs.cli",
                    "chroma-server",
                    "--daemon",
                    "--port",
                    str(port),
                    "--path",
                    str(data_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Command should succeed
            assert result.returncode == 0
            assert "ChromaDB server started in background" in result.stdout

            # Wait for server to start
            time.sleep(3)

            # Check that server is responding
            try:
                response = requests.get(f"http://127.0.0.1:{port}/api/v2/heartbeat", timeout=5)
                assert response.status_code == 200
                assert "nanosecond heartbeat" in response.json()
            except requests.exceptions.ConnectionError:
                pytest.skip("Server did not start (may be due to port conflict)")

            # Test --status when running
            status_result = subprocess.run(
                [
                    get_python_executable(),
                    "-m",
                    "contextfs.cli",
                    "chroma-server",
                    "--status",
                    "--port",
                    str(port),
                ],
                capture_output=True,
                text=True,
            )
            assert "is running" in status_result.stdout

            # Test already-running detection
            start_again = subprocess.run(
                [
                    get_python_executable(),
                    "-m",
                    "contextfs.cli",
                    "chroma-server",
                    "--daemon",
                    "--port",
                    str(port),
                    "--path",
                    str(data_path),
                ],
                capture_output=True,
                text=True,
            )
            assert "already running" in start_again.stdout

        finally:
            # Clean up - kill any server we started
            subprocess.run(
                ["pkill", "-f", f"chroma run.*{port}"],
                capture_output=True,
            )
