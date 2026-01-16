"""CLI commands for agent orchestration."""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from contextfs.orchestrator import AgentRole, Orchestrator

orchestrator_app = typer.Typer(
    name="orchestrator",
    help="Multi-agent orchestration commands",
    no_args_is_help=True,
)

console = Console()


@orchestrator_app.command("spawn")
def spawn_agent(
    role: str = typer.Argument(..., help="Agent role: test, code, review, docs, debug"),
    directory: str = typer.Argument(..., help="Working directory for the agent"),
    task: str = typer.Argument(..., help="Task description for the agent"),
    project: str = typer.Option("default", "--project", "-p", help="Project name for grouping"),
    tool: str = typer.Option("claude", "--tool", "-t", help="CLI tool to use"),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for completion"),
):
    """Spawn a single agent in a directory."""
    try:
        agent_role = AgentRole(role)
    except ValueError:
        console.print(f"[red]Invalid role: {role}[/red]")
        console.print(f"Valid roles: {', '.join(r.value for r in AgentRole)}")
        raise typer.Exit(1)

    directory = str(Path(directory).expanduser().resolve())

    orch = Orchestrator(project=project, tool=tool)
    agent = orch.spawn_agent(role=agent_role, directory=directory, task=task)

    console.print(f"[green]Spawned agent {agent.id}[/green]")
    console.print(f"  Role: {agent.role.value}")
    console.print(f"  Directory: {agent.directory}")
    console.print(f"  Task: {agent.task}")

    if wait:
        console.print("\n[dim]Waiting for completion...[/dim]")
        agent.wait()
        orch.print_status()


@orchestrator_app.command("fleet")
def spawn_fleet(
    config_file: str = typer.Argument(..., help="JSON file with agent configurations"),
    project: str = typer.Option("default", "--project", "-p", help="Project name for grouping"),
    tool: str = typer.Option("claude", "--tool", "-t", help="CLI tool to use"),
    wait: bool = typer.Option(True, "--wait/--no-wait", "-w", help="Wait for completion"),
    monitor: bool = typer.Option(False, "--monitor", "-m", help="Monitor activity in real-time"),
):
    """
    Spawn a fleet of agents from a config file.

    Config file format:
    [
      {"role": "test", "directory": "./tests", "task": "Write tests"},
      {"role": "code", "directory": "./src", "task": "Implement feature"}
    ]
    """
    config_path = Path(config_file).expanduser()
    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_file}[/red]")
        raise typer.Exit(1)

    try:
        agents_config = json.loads(config_path.read_text())
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON: {e}[/red]")
        raise typer.Exit(1)

    # Resolve directories relative to config file
    for config in agents_config:
        dir_path = Path(config["directory"])
        if not dir_path.is_absolute():
            config["directory"] = str((config_path.parent / dir_path).resolve())

    from contextfs.orchestrator import run_fleet as _run_fleet

    console.print(f"[green]Spawning {len(agents_config)} agents...[/green]")
    _run_fleet(
        agents=agents_config,
        project=project,
        tool=tool,
        wait=wait,
        monitor=monitor,
    )

    if not wait:
        console.print(
            "\n[dim]Agents running in background. Use 'orchestrator status' to check.[/dim]"
        )


@orchestrator_app.command("run")
def run_async(
    config_file: str = typer.Argument(..., help="JSON file with agent configurations"),
    project: str = typer.Option("default", "--project", "-p", help="Project name for grouping"),
    tool: str = typer.Option("claude", "--tool", "-t", help="CLI tool to use"),
    workers: int = typer.Option(3, "--workers", "-w", help="Max concurrent agents"),
    timeout: float | None = typer.Option(None, "--timeout", help="Per-task timeout in seconds"),
):
    """
    Run agents concurrently using async worker pool.

    This is the recommended way to run multiple agents - uses asyncio for
    efficient concurrent execution without external dependencies.

    Config file format:
    [
      {"role": "test", "directory": "./tests", "task": "Write tests"},
      {"role": "code", "directory": "./src", "task": "Implement feature"}
    ]
    """
    import asyncio

    config_path = Path(config_file).expanduser()
    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_file}[/red]")
        raise typer.Exit(1)

    try:
        agents_config = json.loads(config_path.read_text())
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON: {e}[/red]")
        raise typer.Exit(1)

    # Resolve directories relative to config file
    for config in agents_config:
        dir_path = Path(config["directory"])
        if not dir_path.is_absolute():
            config["directory"] = str((config_path.parent / dir_path).resolve())

    from contextfs.orchestrator import run_agents

    console.print(
        f"[green]Running {len(agents_config)} agents (max {workers} concurrent)...[/green]"
    )

    def on_complete(result):
        status_icon = "✅" if result.status.value == "completed" else "❌"
        console.print(
            f"{status_icon} [{result.agent_id}] {result.role}: {result.status.value} ({result.duration:.1f}s)"
        )
        if result.error:
            console.print(f"   [red]Error: {result.error[:100]}[/red]")

    results = asyncio.run(
        run_agents(
            agents=agents_config,
            project=project,
            tool=tool,
            num_workers=workers,
            timeout=timeout,
            on_complete=on_complete,
        )
    )

    # Summary
    completed = sum(1 for r in results if r.status.value == "completed")
    failed = len(results) - completed
    total_duration = sum(r.duration for r in results)

    console.print(
        f"\n[bold]Summary:[/bold] {completed} completed, {failed} failed, {total_duration:.1f}s total"
    )

    # Show results
    for result in results:
        if result.result:
            console.print(f"\n[cyan]{result.role} ({result.agent_id}):[/cyan]")
            console.print(
                result.result[:500] + "..." if len(result.result) > 500 else result.result
            )


@orchestrator_app.command("status")
def show_status(
    project: str = typer.Option("default", "--project", "-p", help="Project name"),
):
    """Show status of orchestrated agents."""
    try:
        orch = Orchestrator.load(project)
        orch.print_status()
    except FileNotFoundError:
        console.print(f"[yellow]No orchestrator state found for project: {project}[/yellow]")
        raise typer.Exit(1)


@orchestrator_app.command("monitor")
def monitor_activity(
    project: str = typer.Option("default", "--project", "-p", help="Project name"),
    interval: float = typer.Option(5.0, "--interval", "-i", help="Seconds between checks"),
    duration: float | None = typer.Option(
        None, "--duration", "-d", help="Total seconds to monitor"
    ),
):
    """Monitor agent activity via shared memory."""
    from contextfs.orchestrator import MemoryMonitor

    console.print(f"[dim]Monitoring agent activity for project: {project}[/dim]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    monitor = MemoryMonitor(project=project)

    def on_activity(activity):
        console.print(
            f"[cyan]{activity.timestamp.strftime('%H:%M:%S')}[/cyan] "
            f"[yellow]{activity.agent_role}[/yellow]: {activity.summary}"
        )

    try:
        monitor.watch(interval=interval, duration=duration, on_activity=on_activity)
    except KeyboardInterrupt:
        console.print("\n[dim]Monitoring stopped.[/dim]")


@orchestrator_app.command("findings")
def show_findings(
    project: str = typer.Option("default", "--project", "-p", help="Project name"),
    role: str | None = typer.Option(None, "--role", "-r", help="Filter by agent role"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum results"),
):
    """Show findings from agents via shared memory."""
    from contextfs.orchestrator import MemoryMonitor

    monitor = MemoryMonitor(project=project)
    findings = monitor.get_agent_findings(agent_role=role)[:limit]

    if not findings:
        console.print("[yellow]No findings yet.[/yellow]")
        return

    table = Table(title="Agent Findings")
    table.add_column("ID", style="dim")
    table.add_column("Type", style="cyan")
    table.add_column("Summary")
    table.add_column("Tags", style="dim")

    for finding in findings:
        table.add_row(
            finding["id"][:8],
            finding["type"],
            (finding["summary"] or "")[:60],
            ", ".join(finding.get("tags", [])[:3]),
        )

    console.print(table)


@orchestrator_app.command("errors")
def show_errors(
    project: str = typer.Option("default", "--project", "-p", help="Project name"),
):
    """Show errors reported by agents."""
    from contextfs.orchestrator import MemoryMonitor

    monitor = MemoryMonitor(project=project)
    errors = monitor.get_errors()

    if not errors:
        console.print("[green]No errors reported.[/green]")
        return

    for error in errors:
        console.print(f"\n[red]Error ({error['id'][:8]}):[/red]")
        console.print(f"  Source: {error.get('source', 'unknown')}")
        console.print(f"  Summary: {error.get('summary', 'No summary')}")


@orchestrator_app.command("roles")
def list_roles():
    """List available agent roles and their descriptions."""

    table = Table(title="Agent Roles")
    table.add_column("Role", style="cyan")
    table.add_column("Description")

    role_descriptions = {
        "orchestrator": "Monitors and coordinates other agents",
        "test": "Writes and runs tests, saves failures as errors",
        "code": "Implements features, saves decisions and patterns",
        "review": "Reviews code quality, links related findings",
        "docs": "Writes documentation",
        "debug": "Investigates and fixes bugs",
        "custom": "Generic agent for custom tasks",
    }

    for role in AgentRole:
        table.add_row(role.value, role_descriptions.get(role.value, ""))

    console.print(table)


@orchestrator_app.command("init-config")
def init_config(
    output: str = typer.Option("agents.json", "--output", "-o", help="Output file path"),
):
    """Generate a sample agent fleet configuration file."""
    sample_config = [
        {
            "role": "test",
            "directory": "./tests",
            "task": "Write comprehensive unit tests for the auth module. Search memory for existing test patterns first.",
        },
        {
            "role": "code",
            "directory": "./src",
            "task": "Implement the auth module. Save all architecture decisions to memory.",
        },
        {
            "role": "review",
            "directory": ".",
            "task": "Review the auth implementation for security issues. Link findings to related decisions.",
        },
    ]

    output_path = Path(output)
    output_path.write_text(json.dumps(sample_config, indent=2))
    console.print(f"[green]Created sample config: {output_path}[/green]")
    console.print("\nEdit the file, then run:")
    console.print(f"  contextfs orchestrator fleet {output}")
