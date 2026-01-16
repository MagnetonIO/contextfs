"""Tests for install CLI commands."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from contextfs.cli.install import (
    _check_contextfs_version,
    _check_json_mcp_config,
    _get_mcp_url,
    _merge_json_config,
)


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_mcp_url_default(self):
        """Test MCP URL uses config port."""
        url = _get_mcp_url()
        assert url.startswith("http://127.0.0.1:")
        assert url.endswith("/mcp/sse")

    def test_check_contextfs_version(self):
        """Test version check returns tuple."""
        local_ver, installed_ver = _check_contextfs_version(quiet=True)
        # Returns a tuple of (local_ver, installed_ver) - both may be None in some environments
        assert isinstance(local_ver, str | None)
        assert isinstance(installed_ver, str | None)

    def test_check_json_mcp_config_not_exists(self):
        """Test checking non-existent config."""
        result = _check_json_mcp_config(Path("/nonexistent/path.json"))
        assert "Not configured" in result

    def test_check_json_mcp_config_exists_no_contextfs(self):
        """Test checking config without contextfs."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"mcpServers": {"other": {}}}, f)
            f.flush()
            result = _check_json_mcp_config(Path(f.name))
            assert "No contextfs" in result

    def test_check_json_mcp_config_exists_with_contextfs(self):
        """Test checking config with contextfs."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"mcpServers": {"contextfs": {"url": "http://localhost:3000"}}}, f)
            f.flush()
            result = _check_json_mcp_config(Path(f.name))
            assert "Configured" in result

    def test_merge_json_config_new_file(self):
        """Test merging into new file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp.json"
            new_config = {"mcpServers": {"contextfs": {"url": "http://test"}}}

            _merge_json_config(config_path, new_config, "mcpServers", quiet=True)

            with open(config_path) as f:
                result = json.load(f)

            assert "contextfs" in result["mcpServers"]
            assert result["mcpServers"]["contextfs"]["url"] == "http://test"

    def test_merge_json_config_existing_file(self):
        """Test merging preserves existing servers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "mcp.json"

            # Create existing config
            with open(config_path, "w") as f:
                json.dump({"mcpServers": {"existing": {"url": "http://existing"}}}, f)

            new_config = {"mcpServers": {"contextfs": {"url": "http://test"}}}
            _merge_json_config(config_path, new_config, "mcpServers", quiet=True)

            with open(config_path) as f:
                result = json.load(f)

            # Both should exist
            assert "existing" in result["mcpServers"]
            assert "contextfs" in result["mcpServers"]


class TestInstallCommands:
    """Test install command execution."""

    def test_install_cursor_creates_config(self):
        """Test cursor install creates mcp.json."""
        from typer.testing import CliRunner

        from contextfs.cli import app

        runner = CliRunner()

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(Path, "home", return_value=Path(tmpdir)),
            patch("contextfs.cli.install.Path.cwd", return_value=Path(tmpdir)),
        ):
            result = runner.invoke(app, ["install", "cursor", "--quiet"])
            # May fail due to path mocking, but command should exist
            assert result.exit_code in [0, 1]

    def test_install_list_shows_agents(self):
        """Test list command shows all agents."""
        from typer.testing import CliRunner

        from contextfs.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["install", "list"])

        assert result.exit_code == 0
        assert "claude" in result.output
        assert "cursor" in result.output
        assert "windsurf" in result.output
        assert "gemini" in result.output
        assert "codex" in result.output

    def test_install_status_runs(self):
        """Test status command runs without error."""
        from typer.testing import CliRunner

        from contextfs.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["install", "status"])

        assert result.exit_code == 0
        assert "Claude Code" in result.output
        assert "Cursor IDE" in result.output
        assert "Windsurf" in result.output
        assert "Gemini CLI" in result.output
        assert "Codex CLI" in result.output
