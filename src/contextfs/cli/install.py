"""Install CLI commands for agent integrations."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from contextfs.config import get_config

console = Console()

install_app = typer.Typer(
    name="install",
    help="Install ContextFS integrations for coding agents",
    no_args_is_help=True,
)

# Template directory (relative to this file)
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


def _get_template_path(template: str) -> Path:
    """Get the path to a template file."""
    return TEMPLATES_DIR / template


def _ensure_dir(path: Path) -> None:
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def _copy_template(src: Path, dest: Path, quiet: bool = False, force: bool = False) -> bool:
    """Copy a template file, backing up if exists."""
    if dest.exists() and not force:
        # Create backup
        backup = dest.with_suffix(dest.suffix + ".bak")
        shutil.copy2(dest, backup)
        if not quiet:
            console.print(f"  [dim]Backed up: {dest.name} -> {backup.name}[/dim]")

    shutil.copy2(src, dest)
    return True


def _get_mcp_url() -> str:
    """Get the MCP server URL from config."""
    config = get_config()
    return f"http://127.0.0.1:{config.mcp_port}/mcp/sse"


def _check_contextfs_version(quiet: bool = False) -> tuple[str | None, str | None]:
    """Check for installed contextfs version.

    Returns:
        Tuple of (local_dev_version, installed_version)
        - local_dev_version: Version if running from local dev, None otherwise
        - installed_version: pip installed version, None if not installed
    """
    local_version = None
    installed_version = None

    # Check if we're running from local development (editable install)
    try:
        from contextfs import __version__

        # Check if it's an editable install by looking for .egg-link or pyproject.toml nearby
        contextfs_path = Path(__file__).parent.parent
        if (contextfs_path.parent / "pyproject.toml").exists():
            local_version = __version__
    except ImportError:
        pass

    # Check pip installed version
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "contextfs"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if line.startswith("Version:"):
                    installed_version = line.split(":")[1].strip()
                    break
    except Exception:
        pass

    if not quiet and (local_version or installed_version):
        if local_version:
            console.print(f"[dim]Local dev version: {local_version}[/dim]")
        if installed_version and installed_version != local_version:
            console.print(f"[dim]Installed version: {installed_version}[/dim]")

    return local_version, installed_version


def _merge_json_config(config_path: Path, new_config: dict, key: str, quiet: bool = False) -> None:
    """Merge new config into existing JSON config file."""
    if config_path.exists():
        try:
            with open(config_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, Exception):
            existing = {}
        if not quiet:
            console.print(f"  [dim]Merging with existing: {config_path.name}[/dim]")
    else:
        existing = {}

    if key not in existing:
        existing[key] = {}

    existing[key].update(new_config[key])

    with open(config_path, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")


def _write_toml_config(
    config_path: Path, server_name: str, server_config: dict, quiet: bool = False
) -> None:
    """Write or update TOML config for MCP server (Codex uses TOML)."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore

    existing = {}
    if config_path.exists():
        try:
            with open(config_path, "rb") as f:
                existing = tomllib.load(f)
            if not quiet:
                console.print(f"  [dim]Merging with existing: {config_path.name}[/dim]")
        except Exception:
            pass

    if "mcp_servers" not in existing:
        existing["mcp_servers"] = {}

    existing["mcp_servers"][server_name] = server_config

    # Write TOML manually (tomllib is read-only)
    lines = []
    for section, values in existing.items():
        if section == "mcp_servers":
            for srv_name, srv_config in values.items():
                lines.append(f"[mcp_servers.{srv_name}]")
                for k, v in srv_config.items():
                    if isinstance(v, str):
                        lines.append(f'{k} = "{v}"')
                    elif isinstance(v, bool):
                        lines.append(f"{k} = {'true' if v else 'false'}")
                    elif isinstance(v, int | float):
                        lines.append(f"{k} = {v}")
                    elif isinstance(v, list):
                        items = ", ".join(f'"{i}"' if isinstance(i, str) else str(i) for i in v)
                        lines.append(f"{k} = [{items}]")
                lines.append("")
        else:
            # Preserve other sections
            if isinstance(values, dict):
                lines.append(f"[{section}]")
                for k, v in values.items():
                    if isinstance(v, str):
                        lines.append(f'{k} = "{v}"')
                    else:
                        lines.append(f"{k} = {v}")
                lines.append("")

    with open(config_path, "w") as f:
        f.write("\n".join(lines))


@install_app.command("claude")
def install_claude(
    global_only: bool = typer.Option(
        False, "--global", "-g", help="Only install global skills and MCP"
    ),
    project_only: bool = typer.Option(False, "--project", "-p", help="Only install project hooks"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing files without backup"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
):
    """Install ContextFS integration for Claude Code.

    This command installs:
    - Global skills: ~/.claude/commands/remember.md, recall.md, sync.md
    - Project hooks: .claude/settings.local.json
    - Project agents: .claude/agents/memory-extractor.md
    - MCP server configuration in ~/.claude.json

    Examples:
        contextfs install claude           # Full installation
        contextfs install claude --global  # Only global skills + MCP
        contextfs install claude --project # Only project hooks
    """
    # Check version
    _check_contextfs_version(quiet)

    installed = []
    skipped = []

    # Global skills
    if not project_only:
        global_claude_dir = Path.home() / ".claude"
        global_commands_dir = global_claude_dir / "commands"
        _ensure_dir(global_commands_dir)

        # Install remember.md
        remember_src = _get_template_path("commands/remember.md")
        remember_dest = global_commands_dir / "remember.md"
        if remember_src.exists():
            _copy_template(remember_src, remember_dest, quiet, force)
            installed.append(("Global skill", str(remember_dest)))
        else:
            skipped.append(("remember.md", "Template not found"))

        # Install recall.md
        recall_src = _get_template_path("commands/recall.md")
        recall_dest = global_commands_dir / "recall.md"
        if recall_src.exists():
            _copy_template(recall_src, recall_dest, quiet, force)
            installed.append(("Global skill", str(recall_dest)))
        else:
            skipped.append(("recall.md", "Template not found"))

        # Install sync.md
        sync_src = _get_template_path("commands/sync.md")
        sync_dest = global_commands_dir / "sync.md"
        if sync_src.exists():
            _copy_template(sync_src, sync_dest, quiet, force)
            installed.append(("Global skill", str(sync_dest)))
        else:
            skipped.append(("sync.md", "Template not found"))

    # Project hooks and agents
    if not global_only:
        project_claude_dir = Path.cwd() / ".claude"
        _ensure_dir(project_claude_dir)

        # Install hooks (settings.local.json)
        hooks_src = _get_template_path("hooks.json")
        hooks_dest = project_claude_dir / "settings.local.json"

        if hooks_src.exists():
            # Load template and modify structure
            with open(hooks_src) as f:
                hooks_config = json.load(f)

            # Write to project
            if hooks_dest.exists() and not force:
                # Merge with existing
                with open(hooks_dest) as f:
                    existing = json.load(f)
                # Deep merge hooks
                if "hooks" not in existing:
                    existing["hooks"] = {}
                for hook_type, hook_list in hooks_config.get("hooks", {}).items():
                    if hook_type not in existing["hooks"]:
                        existing["hooks"][hook_type] = hook_list
                    # else: keep existing hooks
                with open(hooks_dest, "w") as f:
                    json.dump(existing, f, indent=2)
                    f.write("\n")
                installed.append(("Project hooks", f"{hooks_dest} (merged)"))
            else:
                with open(hooks_dest, "w") as f:
                    json.dump(hooks_config, f, indent=2)
                    f.write("\n")
                installed.append(("Project hooks", str(hooks_dest)))

        # Install agents
        agents_dir = project_claude_dir / "agents"
        _ensure_dir(agents_dir)

        agent_src = _get_template_path("agents/memory-extractor.md")
        agent_dest = agents_dir / "memory-extractor.md"
        if agent_src.exists():
            _copy_template(agent_src, agent_dest, quiet, force)
            installed.append(("Project agent", str(agent_dest)))

        # Install project commands (copy from global)
        commands_dir = project_claude_dir / "commands"
        _ensure_dir(commands_dir)

        for cmd_file in ["remember.md", "recall.md", "sync.md"]:
            cmd_src = _get_template_path(f"commands/{cmd_file}")
            cmd_dest = commands_dir / cmd_file
            if cmd_src.exists():
                _copy_template(cmd_src, cmd_dest, quiet, force)
                installed.append(("Project skill", str(cmd_dest)))

    # MCP server configuration (always installed unless project_only)
    if not project_only:
        mcp_config_path = Path.home() / ".claude.json"
        mcp_url = _get_mcp_url()

        if mcp_config_path.exists():
            with open(mcp_config_path) as f:
                claude_config = json.load(f)
        else:
            claude_config = {}

        if "mcpServers" not in claude_config:
            claude_config["mcpServers"] = {}

        # Add contextfs MCP server (SSE/HTTP mode) - use config port
        claude_config["mcpServers"]["contextfs"] = {
            "type": "sse",
            "url": mcp_url,
        }

        with open(mcp_config_path, "w") as f:
            json.dump(claude_config, f, indent=2)
            f.write("\n")

        installed.append(("MCP server", str(mcp_config_path)))

    # Output summary
    if not quiet:
        if installed:
            console.print("\n[green]ContextFS installed for Claude Code![/green]\n")

            table = Table(title="Installed Components")
            table.add_column("Type", style="cyan")
            table.add_column("Location")

            for item_type, location in installed:
                table.add_row(item_type, location)

            console.print(table)

            console.print("\n[bold]Next steps:[/bold]")
            console.print("  1. Start MCP server: contextfs server")
            console.print("  2. Restart Claude Code to load hooks and MCP server")
            console.print("  3. Use /remember to save memories")
            console.print("  4. Use /recall to search memories")
            console.print(
                "  5. MCP tools (contextfs_save, contextfs_search, etc.) are now available"
            )
        else:
            console.print("[yellow]Nothing installed[/yellow]")

        if skipped:
            console.print("\n[yellow]Skipped:[/yellow]")
            for name, reason in skipped:
                console.print(f"  {name}: {reason}")
    else:
        if installed:
            print(f"Installed {len(installed)} components for Claude Code")


@install_app.command("cursor")
def install_cursor(
    global_only: bool = typer.Option(
        False, "--global", "-g", help="Only install global MCP config"
    ),
    project_only: bool = typer.Option(
        False, "--project", "-p", help="Only install project MCP config"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
):
    """Install ContextFS integration for Cursor IDE.

    This configures the MCP server in Cursor's mcp.json config.

    Config locations:
    - Global: ~/.cursor/mcp.json
    - Project: .cursor/mcp.json

    Reference: https://cursor.com/docs/context/mcp

    Examples:
        contextfs install cursor           # Both global and project
        contextfs install cursor --global  # Only global config
        contextfs install cursor --project # Only project config
    """
    _check_contextfs_version(quiet)

    installed = []
    mcp_url = _get_mcp_url()

    mcp_config = {
        "mcpServers": {
            "contextfs": {
                "url": mcp_url,
            }
        }
    }

    # Global config
    if not project_only:
        global_config_path = Path.home() / ".cursor" / "mcp.json"
        _ensure_dir(global_config_path.parent)
        _merge_json_config(global_config_path, mcp_config, "mcpServers", quiet)
        installed.append(("Global MCP config", str(global_config_path)))

    # Project config
    if not global_only:
        project_config_path = Path.cwd() / ".cursor" / "mcp.json"
        _ensure_dir(project_config_path.parent)
        _merge_json_config(project_config_path, mcp_config, "mcpServers", quiet)
        installed.append(("Project MCP config", str(project_config_path)))

    # Output summary
    if not quiet:
        if installed:
            console.print("\n[green]ContextFS installed for Cursor![/green]\n")

            table = Table(title="Installed Components")
            table.add_column("Type", style="cyan")
            table.add_column("Location")

            for item_type, location in installed:
                table.add_row(item_type, location)

            console.print(table)

            console.print("\n[bold]Next steps:[/bold]")
            console.print("  1. Start MCP server: contextfs server")
            console.print("  2. Restart Cursor to load MCP config")
            console.print("  3. MCP tools available in Cursor AI chat")
    else:
        if installed:
            print(f"Installed {len(installed)} components for Cursor")


@install_app.command("windsurf")
def install_windsurf(
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
):
    """Install ContextFS integration for Windsurf (Codeium Cascade).

    This configures the MCP server in Windsurf's mcp_config.json.

    Config location: ~/.codeium/windsurf/mcp_config.json

    Reference: https://docs.windsurf.com/windsurf/cascade/mcp

    Examples:
        contextfs install windsurf
    """
    _check_contextfs_version(quiet)

    installed = []
    mcp_url = _get_mcp_url()

    # Windsurf uses "serverUrl" instead of "url"
    mcp_config = {
        "mcpServers": {
            "contextfs": {
                "serverUrl": mcp_url,
            }
        }
    }

    config_path = Path.home() / ".codeium" / "windsurf" / "mcp_config.json"
    _ensure_dir(config_path.parent)
    _merge_json_config(config_path, mcp_config, "mcpServers", quiet)
    installed.append(("MCP config", str(config_path)))

    # Output summary
    if not quiet:
        if installed:
            console.print("\n[green]ContextFS installed for Windsurf![/green]\n")

            table = Table(title="Installed Components")
            table.add_column("Type", style="cyan")
            table.add_column("Location")

            for item_type, location in installed:
                table.add_row(item_type, location)

            console.print(table)

            console.print("\n[bold]Next steps:[/bold]")
            console.print("  1. Start MCP server: contextfs server")
            console.print("  2. Completely quit and restart Windsurf")
            console.print("  3. MCP tools available in Cascade AI chat")
    else:
        if installed:
            print(f"Installed {len(installed)} components for Windsurf")


@install_app.command("gemini")
def install_gemini(
    global_only: bool = typer.Option(False, "--global", "-g", help="Only install global config"),
    project_only: bool = typer.Option(False, "--project", "-p", help="Only install project config"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
):
    """Install ContextFS integration for Google Gemini CLI.

    This configures the MCP server in Gemini CLI's settings.json.

    Config locations:
    - Global: ~/.gemini/settings.json
    - Project: .gemini/settings.json

    Reference: https://geminicli.com/docs/tools/mcp-server/

    Examples:
        contextfs install gemini           # Both global and project
        contextfs install gemini --global  # Only global config
        contextfs install gemini --project # Only project config
    """
    _check_contextfs_version(quiet)

    installed = []
    mcp_url = _get_mcp_url()

    # Gemini uses "url" for SSE transport (httpUrl is for Streamable HTTP)
    mcp_config = {
        "mcpServers": {
            "contextfs": {
                "url": mcp_url,
            }
        }
    }

    # Global config
    if not project_only:
        global_config_path = Path.home() / ".gemini" / "settings.json"
        _ensure_dir(global_config_path.parent)
        _merge_json_config(global_config_path, mcp_config, "mcpServers", quiet)
        installed.append(("Global MCP config", str(global_config_path)))

    # Project config
    if not global_only:
        project_config_path = Path.cwd() / ".gemini" / "settings.json"
        _ensure_dir(project_config_path.parent)
        _merge_json_config(project_config_path, mcp_config, "mcpServers", quiet)
        installed.append(("Project MCP config", str(project_config_path)))

    # Output summary
    if not quiet:
        if installed:
            console.print("\n[green]ContextFS installed for Gemini CLI![/green]\n")

            table = Table(title="Installed Components")
            table.add_column("Type", style="cyan")
            table.add_column("Location")

            for item_type, location in installed:
                table.add_row(item_type, location)

            console.print(table)

            console.print("\n[bold]Next steps:[/bold]")
            console.print("  1. Start MCP server: contextfs server")
            console.print("  2. Run: gemini mcp list (to verify)")
            console.print("  3. MCP tools available in Gemini CLI")
    else:
        if installed:
            print(f"Installed {len(installed)} components for Gemini CLI")


@install_app.command("codex")
def install_codex(
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
):
    """Install ContextFS integration for OpenAI Codex CLI.

    This configures the MCP server in Codex's config.toml.

    Config location: ~/.codex/config.toml

    Reference: https://developers.openai.com/codex/mcp/

    Examples:
        contextfs install codex
    """
    _check_contextfs_version(quiet)

    installed = []
    mcp_url = _get_mcp_url()

    # Codex uses TOML format with url field
    server_config = {
        "url": mcp_url,
    }

    config_path = Path.home() / ".codex" / "config.toml"
    _ensure_dir(config_path.parent)
    _write_toml_config(config_path, "contextfs", server_config, quiet)
    installed.append(("MCP config", str(config_path)))

    # Output summary
    if not quiet:
        if installed:
            console.print("\n[green]ContextFS installed for Codex CLI![/green]\n")

            table = Table(title="Installed Components")
            table.add_column("Type", style="cyan")
            table.add_column("Location")

            for item_type, location in installed:
                table.add_row(item_type, location)

            console.print(table)

            console.print("\n[bold]Next steps:[/bold]")
            console.print("  1. Start MCP server: contextfs server")
            console.print("  2. Run: codex mcp list (to verify)")
            console.print("  3. MCP tools available in Codex CLI")
    else:
        if installed:
            print(f"Installed {len(installed)} components for Codex CLI")


@install_app.command("list")
def list_agents():
    """List available agent integrations."""
    console.print("\n[bold]Available Agent Integrations[/bold]\n")

    table = Table()
    table.add_column("Agent", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Config Location")
    table.add_column("Description")

    table.add_row(
        "claude",
        "[green]Available[/green]",
        "~/.claude.json",
        "Claude Code CLI with hooks, skills & MCP",
    )
    table.add_row(
        "cursor",
        "[green]Available[/green]",
        "~/.cursor/mcp.json",
        "Cursor IDE MCP integration",
    )
    table.add_row(
        "windsurf",
        "[green]Available[/green]",
        "~/.codeium/windsurf/mcp_config.json",
        "Windsurf Cascade MCP integration",
    )
    table.add_row(
        "gemini",
        "[green]Available[/green]",
        "~/.gemini/settings.json",
        "Google Gemini CLI MCP integration",
    )
    table.add_row(
        "codex",
        "[green]Available[/green]",
        "~/.codex/config.toml",
        "OpenAI Codex CLI MCP integration",
    )
    table.add_row(
        "vscode",
        "[dim]Planned[/dim]",
        "-",
        "VS Code extension",
    )
    table.add_row(
        "jetbrains",
        "[dim]Planned[/dim]",
        "-",
        "JetBrains IDE plugin",
    )

    console.print(table)
    console.print("\n[dim]Use 'contextfs install <agent>' to install[/dim]")
    console.print("[dim]Use 'contextfs install status' to check installation status[/dim]")


def _check_json_mcp_config(config_path: Path, key: str = "mcpServers") -> str:
    """Check if contextfs is configured in a JSON MCP config file."""
    if not config_path.exists():
        return "[dim]Not configured[/dim]"
    try:
        with open(config_path) as f:
            config = json.load(f)
        if "contextfs" in config.get(key, {}):
            return "[green]Configured[/green]"
        return "[yellow]No contextfs[/yellow]"
    except Exception:
        return "[red]Config error[/red]"


def _check_toml_mcp_config(config_path: Path) -> str:
    """Check if contextfs is configured in a TOML MCP config file."""
    if not config_path.exists():
        return "[dim]Not configured[/dim]"
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        if "contextfs" in config.get("mcp_servers", {}):
            return "[green]Configured[/green]"
        return "[yellow]No contextfs[/yellow]"
    except Exception:
        return "[red]Config error[/red]"


@install_app.command("status")
def install_status():
    """Check installation status for all integrations."""
    console.print("\n[bold]ContextFS Installation Status[/bold]\n")

    # Check version
    _check_contextfs_version(quiet=False)
    console.print("")

    # ========== Claude Code ==========
    global_commands = Path.home() / ".claude" / "commands"
    claude_config = Path.home() / ".claude.json"

    claude_items = []

    # Global skills
    for skill in ["remember.md", "recall.md", "sync.md"]:
        skill_path = global_commands / skill
        if skill_path.exists():
            claude_items.append((f"Global skill: {skill}", "[green]Installed[/green]"))
        else:
            claude_items.append((f"Global skill: {skill}", "[red]Missing[/red]"))

    # Project hooks
    project_hooks = Path.cwd() / ".claude" / "settings.local.json"
    if project_hooks.exists():
        claude_items.append(("Project hooks", "[green]Installed[/green]"))
    else:
        claude_items.append(("Project hooks", "[yellow]Not installed[/yellow]"))

    # MCP server
    claude_items.append(("MCP server", _check_json_mcp_config(claude_config)))

    # Display Claude
    table = Table(title="Claude Code")
    table.add_column("Component", style="cyan")
    table.add_column("Status")

    for component, status in claude_items:
        table.add_row(component, status)

    console.print(table)

    # ========== Cursor ==========
    cursor_global = Path.home() / ".cursor" / "mcp.json"
    cursor_project = Path.cwd() / ".cursor" / "mcp.json"

    cursor_items = [
        ("Global MCP", _check_json_mcp_config(cursor_global)),
        ("Project MCP", _check_json_mcp_config(cursor_project)),
    ]

    table = Table(title="Cursor IDE")
    table.add_column("Component", style="cyan")
    table.add_column("Status")

    for component, status in cursor_items:
        table.add_row(component, status)

    console.print(table)

    # ========== Windsurf ==========
    windsurf_config = Path.home() / ".codeium" / "windsurf" / "mcp_config.json"

    table = Table(title="Windsurf")
    table.add_column("Component", style="cyan")
    table.add_column("Status")
    table.add_row("MCP config", _check_json_mcp_config(windsurf_config))

    console.print(table)

    # ========== Gemini CLI ==========
    gemini_global = Path.home() / ".gemini" / "settings.json"
    gemini_project = Path.cwd() / ".gemini" / "settings.json"

    gemini_items = [
        ("Global MCP", _check_json_mcp_config(gemini_global)),
        ("Project MCP", _check_json_mcp_config(gemini_project)),
    ]

    table = Table(title="Gemini CLI")
    table.add_column("Component", style="cyan")
    table.add_column("Status")

    for component, status in gemini_items:
        table.add_row(component, status)

    console.print(table)

    # ========== Codex CLI ==========
    codex_config = Path.home() / ".codex" / "config.toml"

    table = Table(title="Codex CLI")
    table.add_column("Component", style="cyan")
    table.add_column("Status")
    table.add_row("MCP config", _check_toml_mcp_config(codex_config))

    console.print(table)

    # ========== Suggestions ==========
    all_missing = []

    claude_missing = [
        c
        for c, s in claude_items
        if "Missing" in s or "Not installed" in s or "Not configured" in s
    ]
    if claude_missing:
        all_missing.append(("Claude Code", "contextfs install claude"))

    cursor_missing = [c for c, s in cursor_items if "Not configured" in s]
    if len(cursor_missing) == 2:  # Both missing
        all_missing.append(("Cursor", "contextfs install cursor"))

    if "Not configured" in _check_json_mcp_config(windsurf_config):
        all_missing.append(("Windsurf", "contextfs install windsurf"))

    gemini_missing = [c for c, s in gemini_items if "Not configured" in s]
    if len(gemini_missing) == 2:
        all_missing.append(("Gemini CLI", "contextfs install gemini"))

    if "Not configured" in _check_toml_mcp_config(codex_config):
        all_missing.append(("Codex CLI", "contextfs install codex"))

    if all_missing:
        console.print("\n[yellow]To install missing integrations:[/yellow]")
        for name, cmd in all_missing:
            console.print(f"  {name}: {cmd}")
