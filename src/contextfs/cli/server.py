"""Server-related CLI commands."""

import shutil
import subprocess
from pathlib import Path

import typer

from .services import (
    check_chroma_running,
    check_mcp_running,
    find_chroma_bin,
    get_chroma_pid,
    get_chroma_pid_file,
    get_chroma_service_paths,
    get_mcp_pid,
    get_mcp_pid_file,
    get_mcp_service_paths,
    install_chroma_linux_service,
    install_chroma_macos_service,
    install_chroma_windows_service,
    install_mcp_linux_service,
    install_mcp_macos_service,
    stop_chroma,
    stop_mcp,
    uninstall_chroma_service,
    uninstall_mcp_service,
    write_pid_file,
)
from .utils import console

server_app = typer.Typer(help="Server commands", no_args_is_help=True)


@server_app.command("start")
def start_server(
    service: str = typer.Argument(..., help="Service to start: mcp, chroma"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(None, "--port", "-p", help="Port (default: mcp=8003, chroma=8000)"),
    foreground: bool = typer.Option(False, "--foreground", "-f", help="Run in foreground"),
):
    """Start MCP or ChromaDB server.

    Examples:
        contextfs start mcp               # Start MCP server (background)
        contextfs start chroma            # Start ChromaDB server (background)
        contextfs start mcp -f            # Run MCP in foreground
        contextfs start mcp -p 9000       # Custom port
    """
    if service == "mcp":
        _start_mcp(host, port or 8003, foreground)
    elif service == "chroma":
        _start_chroma(host, port or 8000, foreground)
    else:
        console.print(f"[red]Unknown service: {service}[/red]")
        console.print("Valid services: mcp, chroma")
        raise typer.Exit(1)


def _start_mcp(host: str, port: int, foreground: bool) -> None:
    """Start MCP server."""
    if check_mcp_running(host, port):
        pid = get_mcp_pid(port)
        console.print(f"[yellow]MCP server already running on {host}:{port}[/yellow]")
        if pid:
            console.print(f"   PID: {pid}")
        return

    if foreground:
        console.print(f"[bold]Starting MCP server on {host}:{port}[/bold]")
        console.print("[dim]Press Ctrl+C to stop[/dim]")
        from contextfs.mcp import run_mcp_server

        run_mcp_server(host=host, port=port)
    else:
        import sys
        import time

        cmd = [
            sys.executable,
            "-m",
            "contextfs.mcp.server",
            "--host",
            host,
            "--port",
            str(port),
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        write_pid_file(get_mcp_pid_file(), proc.pid)
        time.sleep(0.5)
        if check_mcp_running(host, port):
            console.print(f"[green]MCP server started on {host}:{port}[/green]")
            console.print(f"   PID: {proc.pid}")
        else:
            console.print(f"[yellow]MCP server starting on {host}:{port}...[/yellow]")
            console.print(f"   PID: {proc.pid}")


def _start_chroma(host: str, port: int, foreground: bool) -> None:
    """Start ChromaDB server."""
    data_path = Path.home() / ".contextfs" / "chroma_db"

    if check_chroma_running(host, port):
        pid = get_chroma_pid(port)
        console.print(f"[yellow]ChromaDB already running on {host}:{port}[/yellow]")
        if pid:
            console.print(f"   PID: {pid}")
        return

    chroma_bin = find_chroma_bin()
    if not chroma_bin:
        console.print("[red]Error: 'chroma' CLI not found.[/red]")
        console.print("Install it with: pip install chromadb")
        raise typer.Exit(1)

    data_path.mkdir(parents=True, exist_ok=True)
    cmd = [chroma_bin, "run", "--path", str(data_path), "--host", host, "--port", str(port)]

    if foreground:
        console.print(f"[bold]Starting ChromaDB server on {host}:{port}[/bold]")
        console.print(f"  Data path: {data_path}")
        console.print("[dim]Press Ctrl+C to stop[/dim]")
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped[/yellow]")
    else:
        import time

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        write_pid_file(get_chroma_pid_file(), proc.pid)
        time.sleep(0.5)
        if check_chroma_running(host, port):
            console.print(f"[green]ChromaDB server started on {host}:{port}[/green]")
            console.print(f"   PID: {proc.pid}")
        else:
            console.print(f"[yellow]ChromaDB server starting on {host}:{port}...[/yellow]")
            console.print(f"   PID: {proc.pid}")


@server_app.command("stop")
def stop_server(
    service: str = typer.Argument(..., help="Service to stop: mcp, chroma, all"),
    port: int = typer.Option(None, "--port", "-p", help="Port (default: mcp=8003, chroma=8000)"),
):
    """Stop MCP or ChromaDB server.

    Note: If a system service is installed (launchd/systemd), it will restart
    the process automatically. Use 'contextfs uninstall-service' to remove it.

    Examples:
        contextfs stop mcp                # Stop MCP server
        contextfs stop chroma             # Stop ChromaDB server
        contextfs stop all                # Stop both servers
    """
    if service == "mcp" or service == "all":
        mcp_port = port or 8003
        paths = get_mcp_service_paths()
        if paths.get("service_file") and paths["service_file"].exists():
            console.print("[yellow]Warning: MCP system service is installed.[/yellow]")
            console.print("   The service will restart MCP automatically.")
            console.print("   To stop permanently: contextfs uninstall-service mcp")
            console.print()

        if stop_mcp(mcp_port):
            console.print(f"[green]MCP server stopped (port {mcp_port})[/green]")
        else:
            console.print(f"[yellow]MCP server not running on port {mcp_port}[/yellow]")

    if service == "chroma" or service == "all":
        chroma_port = port or 8000
        paths = get_chroma_service_paths()
        if paths.get("service_file") and paths["service_file"].exists():
            console.print("[yellow]Warning: ChromaDB system service is installed.[/yellow]")
            console.print("   The service will restart ChromaDB automatically.")
            console.print("   To stop permanently: contextfs uninstall-service chroma")
            console.print()

        if stop_chroma(chroma_port):
            console.print(f"[green]ChromaDB server stopped (port {chroma_port})[/green]")
        else:
            console.print(f"[yellow]ChromaDB server not running on port {chroma_port}[/yellow]")

    if service not in ("mcp", "chroma", "all"):
        console.print(f"[red]Unknown service: {service}[/red]")
        console.print("Valid services: mcp, chroma, all")
        raise typer.Exit(1)


@server_app.command("status")
def server_status(
    service: str = typer.Argument(None, help="Service to check: mcp, chroma (default: all)"),
):
    """Check status of MCP and ChromaDB servers.

    Examples:
        contextfs status                  # Check all servers
        contextfs status mcp              # Check MCP only
        contextfs status chroma           # Check ChromaDB only
    """
    services = [service] if service else ["mcp", "chroma"]

    for svc in services:
        if svc == "mcp":
            _show_mcp_status()
        elif svc == "chroma":
            _show_chroma_status()
        else:
            console.print(f"[red]Unknown service: {svc}[/red]")


def _show_mcp_status() -> None:
    """Show MCP server status."""
    status = check_mcp_running()
    if status:
        pid = get_mcp_pid()
        console.print("[green]MCP server:[/green] running")
        console.print("   URL: http://127.0.0.1:8003/mcp/sse")
        if pid:
            console.print(f"   PID: {pid}")

        paths = get_mcp_service_paths()
        pid_file = get_mcp_pid_file()
        if paths["platform"] == "macos" and paths["service_file"].exists():
            console.print("   Mode: launchd service (auto-start enabled)")
            console.print("   Stop with: contextfs uninstall-service mcp")
        elif paths["platform"] == "linux" and paths["service_file"].exists():
            console.print("   Mode: systemd service (auto-start enabled)")
            console.print("   Stop with: contextfs uninstall-service mcp")
        elif pid_file.exists():
            console.print("   Mode: manual (started with 'contextfs start mcp')")
            console.print("   Stop with: contextfs stop mcp")
        else:
            console.print("   Mode: external (started outside contextfs)")
    else:
        console.print("[red]MCP server:[/red] not running")
        console.print("   Start with: contextfs start mcp")


def _show_chroma_status() -> None:
    """Show ChromaDB server status."""
    status = check_chroma_running("127.0.0.1", 8000)
    if status:
        pid = get_chroma_pid(8000)
        console.print("[green]ChromaDB server:[/green] running")
        console.print("   URL: http://127.0.0.1:8000")
        if pid:
            console.print(f"   PID: {pid}")

        paths = get_chroma_service_paths()
        pid_file = get_chroma_pid_file()
        if (
            paths["platform"] == "macos"
            and paths.get("service_file")
            and paths["service_file"].exists()
        ):
            console.print("   Mode: launchd service (auto-start enabled)")
            console.print("   Stop with: contextfs uninstall-service chroma")
        elif (
            paths["platform"] == "linux"
            and paths.get("service_file")
            and paths["service_file"].exists()
        ):
            console.print("   Mode: systemd service (auto-start enabled)")
            console.print("   Stop with: contextfs uninstall-service chroma")
        elif pid_file.exists():
            console.print("   Mode: manual (started with 'contextfs start chroma')")
            console.print("   Stop with: contextfs stop chroma")
        else:
            console.print("   Mode: external (started outside contextfs)")
    else:
        console.print("[red]ChromaDB server:[/red] not running")
        console.print("   Start with: contextfs start chroma")


@server_app.command("install")
def install(
    target: str = typer.Argument("claude", help="Target: claude, gemini, codex, all"),
    path: Path = typer.Option(None, "--path", "-p", help="Project path for project-level install"),
    user_only: bool = typer.Option(
        False, "--user-only", help="Only install user-level (skip project)"
    ),
    no_service: bool = typer.Option(False, "--no-service", help="Don't install auto-start service"),
    no_start: bool = typer.Option(False, "--no-start", help="Don't start MCP server now"),
    uninstall: bool = typer.Option(False, "--uninstall", help="Remove installation"),
):
    """Install ContextFS for AI coding tools.

    Targets:
        claude  - Claude Code & Desktop (default)
        gemini  - Gemini CLI
        codex   - Codex CLI
        all     - All supported tools

    Examples:
        contextfs install                    # Install for Claude (default)
        contextfs install gemini             # Install for Gemini CLI
        contextfs install all                # Install for all tools
        contextfs install --uninstall        # Remove installation
    """
    targets = [target] if target != "all" else ["claude", "gemini", "codex"]

    for t in targets:
        if t == "claude":
            _install_claude(path, user_only, no_service, no_start, uninstall)
        elif t == "gemini":
            _install_gemini(uninstall)
        elif t == "codex":
            _install_codex(uninstall)
        else:
            console.print(f"[red]Unknown target: {t}[/red]")
            console.print("Valid targets: claude, gemini, codex, all")
            raise typer.Exit(1)


def _install_claude(
    path: Path | None, user_only: bool, no_service: bool, no_start: bool, uninstall: bool
) -> None:
    """Install for Claude Code & Desktop."""
    from contextfs.plugins.claude_code import (
        ClaudeCodePlugin,
        uninstall_claude_code,
        uninstall_claude_code_from_project,
    )

    project_path = path.resolve() if path else Path.cwd()

    if uninstall:
        uninstall_claude_code()
        if not user_only:
            uninstall_claude_code_from_project(project_path)
        console.print("[green]Claude Code installation removed.[/green]")
        return

    plugin = ClaudeCodePlugin(project_path=project_path if not user_only else None)
    plugin.install(include_project=not user_only)


def _install_gemini(uninstall: bool) -> None:
    """Install for Gemini CLI."""
    from contextfs.plugins.gemini import GeminiPlugin

    if uninstall:
        console.print("[yellow]Gemini uninstall not yet implemented.[/yellow]")
        return

    plugin = GeminiPlugin()
    plugin.install()
    console.print("[green]Gemini CLI integration installed.[/green]")


def _install_codex(uninstall: bool) -> None:
    """Install for Codex CLI."""
    from contextfs.plugins.codex import CodexPlugin

    if uninstall:
        console.print("[yellow]Codex uninstall not yet implemented.[/yellow]")
        return

    plugin = CodexPlugin()
    plugin.install()
    console.print("[green]Codex CLI integration installed.[/green]")


@server_app.command("git-hooks")
def git_hooks(
    repo_path: str = typer.Argument(None, help="Repository path (default: current directory)"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing hooks"),
) -> None:
    """Install git hooks for automatic indexing.

    Installs post-commit and post-merge hooks that automatically
    run incremental indexing after commits and pulls.

    Examples:
        contextfs git-hooks              # Install to current repo
        contextfs git-hooks /path/to/repo
        contextfs git-hooks --force      # Overwrite existing hooks
    """
    target = Path(repo_path).resolve() if repo_path else Path.cwd()

    git_dir = target / ".git"
    if not git_dir.exists():
        console.print(f"[red]Error: {target} is not a git repository[/red]")
        raise typer.Exit(1)

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)

    hooks = {
        "post-commit": """#!/bin/bash
# ContextFS Post-Commit Hook - Auto-index on commit
set -e
if command -v contextfs &> /dev/null; then
    CONTEXTFS="contextfs"
elif [ -f "$HOME/.local/bin/contextfs" ]; then
    CONTEXTFS="$HOME/.local/bin/contextfs"
else
    exit 0
fi
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
(cd "$REPO_ROOT" && $CONTEXTFS index --incremental --mode files_only --quiet 2>/dev/null &) &
exit 0
""",
        "post-merge": """#!/bin/bash
# ContextFS Post-Merge Hook - Auto-index on pull/merge
set -e
if command -v contextfs &> /dev/null; then
    CONTEXTFS="contextfs"
elif [ -f "$HOME/.local/bin/contextfs" ]; then
    CONTEXTFS="$HOME/.local/bin/contextfs"
else
    exit 0
fi
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
(cd "$REPO_ROOT" && $CONTEXTFS index --incremental --quiet 2>/dev/null &) &
exit 0
""",
    }

    console.print(f"Installing ContextFS git hooks to: [cyan]{target}[/cyan]\n")

    for hook_name, hook_content in hooks.items():
        hook_path = hooks_dir / hook_name

        if hook_path.exists() and not force:
            console.print(f"  [yellow]{hook_name}:[/yellow] exists (use --force to overwrite)")
            continue

        if hook_path.exists():
            backup_path = hooks_dir / f"{hook_name}.bak"
            shutil.copy(hook_path, backup_path)
            console.print(f"  [dim]{hook_name}: backed up to {hook_name}.bak[/dim]")

        hook_path.write_text(hook_content)
        hook_path.chmod(0o755)
        console.print(f"  [green]{hook_name}:[/green] installed")

    console.print("\n[green]Done![/green] ContextFS will auto-index on:")
    console.print("  - git commit (indexes changed files)")
    console.print("  - git pull/merge (indexes new files and commits)")


@server_app.command("install-service")
def install_service(
    service: str = typer.Argument(..., help="Service to install: mcp, chroma"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(None, "--port", "-p", help="Port (default: mcp=8003, chroma=8000)"),
):
    """Install auto-start service for MCP or ChromaDB.

    Sets up launchd (macOS) or systemd (Linux) to automatically
    start the service at login/boot.

    Examples:
        contextfs install-service mcp       # Install MCP auto-start
        contextfs install-service chroma    # Install ChromaDB auto-start
    """
    import platform

    system = platform.system()

    if service == "mcp":
        default_port = port or 8003
        paths = get_mcp_service_paths()

        if paths["platform"] == "unknown":
            console.print(f"[red]Unsupported platform: {system}[/red]")
            raise typer.Exit(1)

        if paths["service_file"].exists():
            console.print("[yellow]MCP service already installed.[/yellow]")
            console.print("   Use 'contextfs uninstall-service mcp' to remove it first.")
            return

        console.print(f"Installing MCP service for {paths['platform']}...")
        try:
            if paths["platform"] == "macos":
                install_mcp_macos_service(host, default_port)
            else:
                install_mcp_linux_service(host, default_port)

            console.print("[green]MCP service installed and started[/green]")
            console.print(f"   URL: http://{host}:{default_port}/mcp/sse")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to install service: {e}[/red]")
            raise typer.Exit(1)

    elif service == "chroma":
        default_port = port or 8000
        paths = get_chroma_service_paths()
        data_path = Path.home() / ".contextfs" / "chroma_db"

        if paths["platform"] == "unknown":
            console.print(f"[red]Unsupported platform: {system}[/red]")
            raise typer.Exit(1)

        if paths.get("service_file") and paths["service_file"].exists():
            console.print("[yellow]ChromaDB service already installed.[/yellow]")
            console.print("   Use 'contextfs uninstall-service chroma' to remove it first.")
            return

        chroma_bin = find_chroma_bin()
        if not chroma_bin:
            console.print("[red]Error: 'chroma' CLI not found.[/red]")
            console.print("Install it with: pip install chromadb")
            raise typer.Exit(1)

        data_path.mkdir(parents=True, exist_ok=True)

        console.print(f"Installing ChromaDB service for {paths['platform']}...")
        try:
            if paths["platform"] == "macos":
                install_chroma_macos_service(host, default_port, data_path, chroma_bin)
            elif paths["platform"] == "linux":
                install_chroma_linux_service(host, default_port, data_path, chroma_bin)
            elif paths["platform"] == "windows":
                install_chroma_windows_service(host, default_port, data_path, chroma_bin)

            console.print("[green]ChromaDB service installed and started[/green]")
            console.print(f"   URL: http://{host}:{default_port}")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to install service: {e}[/red]")
            raise typer.Exit(1)

    else:
        console.print(f"[red]Unknown service: {service}[/red]")
        console.print("Valid services: mcp, chroma")
        raise typer.Exit(1)


@server_app.command("uninstall-service")
def uninstall_service_cmd(
    service: str = typer.Argument(..., help="Service to uninstall: mcp, chroma, all"),
):
    """Remove auto-start service for MCP or ChromaDB.

    Stops and removes the launchd (macOS) or systemd (Linux) service.

    Examples:
        contextfs uninstall-service mcp     # Remove MCP auto-start
        contextfs uninstall-service chroma  # Remove ChromaDB auto-start
        contextfs uninstall-service all     # Remove all services
    """
    if service == "mcp" or service == "all":
        paths = get_mcp_service_paths()
        if paths.get("service_file") and paths["service_file"].exists():
            console.print("Removing MCP service...")
            if uninstall_mcp_service():
                console.print("[green]MCP service removed[/green]")
            else:
                console.print("[yellow]Failed to remove MCP service[/yellow]")
        else:
            console.print("[yellow]MCP service not installed[/yellow]")

    if service == "chroma" or service == "all":
        paths = get_chroma_service_paths()
        if paths.get("service_file") and paths["service_file"].exists():
            console.print("Removing ChromaDB service...")
            if uninstall_chroma_service():
                console.print("[green]ChromaDB service removed[/green]")
            else:
                console.print("[yellow]Failed to remove ChromaDB service[/yellow]")
        else:
            console.print("[yellow]ChromaDB service not installed[/yellow]")

    if service not in ("mcp", "chroma", "all"):
        console.print(f"[red]Unknown service: {service}[/red]")
        console.print("Valid services: mcp, chroma, all")
        raise typer.Exit(1)


@server_app.command("completions")
def completions(
    shell: str = typer.Argument(
        None, help="Shell to install completions for: bash, zsh (default: auto-detect)"
    ),
    uninstall: bool = typer.Option(False, "--uninstall", help="Remove completions"),
):
    """Install shell completions for contextfs CLI.

    Adds tab-completion for commands, options, and arguments.

    Examples:
        contextfs completions           # Auto-detect shell and install
        contextfs completions bash      # Install for bash
        contextfs completions zsh       # Install for zsh
        contextfs completions --uninstall  # Remove completions
    """
    import os

    from typer.completion import get_completion_script

    if not shell:
        shell_path = os.environ.get("SHELL", "")
        if "zsh" in shell_path:
            shell = "zsh"
        elif "bash" in shell_path:
            shell = "bash"
        else:
            console.print("[red]Could not detect shell. Please specify: bash or zsh[/red]")
            raise typer.Exit(1)

    home = Path.home()

    if shell == "zsh":
        completion_dir = home / ".zfunc"
        completion_file = completion_dir / "_contextfs"
        rc_file = home / ".zshrc"

        if uninstall:
            if completion_file.exists():
                completion_file.unlink()
                console.print("[green]Zsh completions removed[/green]")
            else:
                console.print("[yellow]Zsh completions not installed[/yellow]")
            return

        completion_dir.mkdir(parents=True, exist_ok=True)
        script = get_completion_script(
            prog_name="contextfs", complete_var="_CONTEXTFS_COMPLETE", shell="zsh"
        )
        completion_file.write_text(script)
        console.print("[green]Zsh completions installed[/green]")
        console.print(f"   File: {completion_file}")

        rc_content = rc_file.read_text() if rc_file.exists() else ""
        if ".zfunc" not in rc_content:
            console.print()
            console.print("[yellow]Add to ~/.zshrc:[/yellow]")
            console.print("   fpath=(~/.zfunc $fpath)")
            console.print("   autoload -Uz compinit && compinit")
        console.print()
        console.print("[dim]Restart shell or: source ~/.zshrc[/dim]")

    elif shell == "bash":
        completion_dir = home / ".local" / "share" / "bash-completion" / "completions"
        completion_file = completion_dir / "contextfs"

        if uninstall:
            if completion_file.exists():
                completion_file.unlink()
                console.print("[green]Bash completions removed[/green]")
            else:
                console.print("[yellow]Bash completions not installed[/yellow]")
            return

        completion_dir.mkdir(parents=True, exist_ok=True)
        script = get_completion_script(
            prog_name="contextfs", complete_var="_CONTEXTFS_COMPLETE", shell="bash"
        )
        completion_file.write_text(script)
        console.print("[green]Bash completions installed[/green]")
        console.print(f"   File: {completion_file}")
        console.print()
        console.print("[dim]Restart shell or: source ~/.bashrc[/dim]")

    else:
        console.print(f"[red]Unsupported shell: {shell}[/red]")
        console.print("Supported shells: bash, zsh")
        raise typer.Exit(1)
