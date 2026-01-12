"""Cloud sync commands for ContextFS commercial platform."""

import asyncio
from pathlib import Path

import typer
from rich.table import Table

from .utils import console, get_ctx

cloud_app = typer.Typer(
    name="cloud",
    help="Cloud sync commands for ContextFS commercial platform.",
    no_args_is_help=True,
)


def _get_cloud_config() -> dict:
    """Get cloud configuration from config file."""
    import yaml

    config_path = Path.home() / ".contextfs" / "config.yaml"
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    return config.get("cloud", {})


def _save_cloud_config(cloud_config: dict) -> None:
    """Save cloud configuration to config file."""
    import yaml

    config_path = Path.home() / ".contextfs" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    config["cloud"] = cloud_config

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


@cloud_app.command()
def login(
    provider: str = typer.Option(
        "google", "--provider", "-p", help="OAuth provider (google, github)"
    ),
):
    """Login to ContextFS Cloud (opens browser for OAuth)."""
    import webbrowser

    cloud_config = _get_cloud_config()
    server_url = cloud_config.get("server_url", "https://api.contextfs.ai")

    # Open browser for OAuth
    oauth_url = f"{server_url}/auth/login?provider={provider}"
    console.print(f"Opening browser for {provider} login...")
    console.print(f"URL: {oauth_url}")
    webbrowser.open(oauth_url)

    console.print("\n[yellow]After login, copy your API key and run:[/yellow]")
    console.print("  contextfs cloud configure --api-key YOUR_KEY")


@cloud_app.command()
def configure(
    api_key: str = typer.Option(None, "--api-key", "-k", help="API key from dashboard"),
    server_url: str = typer.Option(
        "https://api.contextfs.ai", "--server", "-s", help="Cloud server URL"
    ),
    enabled: bool = typer.Option(True, "--enabled/--disabled", help="Enable/disable cloud sync"),
):
    """Configure cloud sync settings.

    E2EE is automatic - encryption key is derived from your API key.
    No separate encryption key needed.
    """
    cloud_config = _get_cloud_config()

    if api_key:
        cloud_config["api_key"] = api_key
    if server_url:
        cloud_config["server_url"] = server_url
    cloud_config["enabled"] = enabled

    _save_cloud_config(cloud_config)
    console.print("[green]Cloud configuration saved![/green]")
    console.print("[dim]E2EE is automatic - encryption key derived from API key[/dim]")

    # Display current config (hide secrets)
    display_config = cloud_config.copy()
    if "api_key" in display_config:
        display_config["api_key"] = display_config["api_key"][:12] + "..."

    console.print(display_config)


@cloud_app.command()
def status():
    """Show cloud sync status."""
    import httpx

    cloud_config = _get_cloud_config()

    if not cloud_config.get("enabled"):
        console.print("[yellow]Cloud sync is disabled[/yellow]")
        return

    if not cloud_config.get("api_key"):
        console.print("[red]No API key configured. Run: contextfs cloud login[/red]")
        return

    server_url = cloud_config.get("server_url", "https://api.contextfs.ai")
    api_key = cloud_config.get("api_key")

    async def check_status():
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{server_url}/api/billing/subscription",
                    headers={"X-API-Key": api_key},
                    timeout=10.0,
                )
                if response.status_code == 200:
                    data = response.json()
                    console.print("[green]Connected to ContextFS Cloud[/green]")
                    console.print(f"  Tier: {data.get('tier', 'free')}")
                    console.print(f"  Status: {data.get('status', 'unknown')}")
                    console.print(f"  Device Limit: {data.get('device_limit', 3)}")
                    console.print(f"  Memory Limit: {data.get('memory_limit', 10000)}")
                elif response.status_code == 401:
                    console.print("[red]Invalid API key[/red]")
                else:
                    console.print(f"[red]Error: {response.status_code}[/red]")
            except Exception as e:
                console.print(f"[red]Connection failed: {e}[/red]")

    asyncio.run(check_status())


@cloud_app.command(name="api-key")
def api_key_cmd(
    action: str = typer.Argument(..., help="Action: create, list, or revoke"),
    name: str = typer.Option(None, "--name", "-n", help="Name for new API key"),
    key_id: str = typer.Option(None, "--id", help="Key ID for revoke action"),
):
    """Manage API keys (create, list, revoke)."""
    import httpx

    cloud_config = _get_cloud_config()
    if not cloud_config.get("api_key"):
        console.print("[red]Not logged in. Run: contextfs cloud login[/red]")
        return

    server_url = cloud_config.get("server_url", "https://api.contextfs.ai")
    api_key = cloud_config.get("api_key")

    async def manage_keys():
        async with httpx.AsyncClient() as client:
            headers = {"X-API-Key": api_key}

            if action == "list":
                response = await client.get(f"{server_url}/api/auth/api-keys", headers=headers)
                if response.status_code == 200:
                    keys = response.json().get("keys", [])
                    table = Table(title="API Keys")
                    table.add_column("ID", style="dim")
                    table.add_column("Name")
                    table.add_column("Prefix")
                    table.add_column("Active")
                    table.add_column("Last Used")

                    for key in keys:
                        table.add_row(
                            key["id"][:8] + "...",
                            key["name"],
                            f"ctxfs_{key['key_prefix']}...",
                            "Yes" if key["is_active"] else "No",
                            key.get("last_used_at", "Never"),
                        )
                    console.print(table)
                else:
                    console.print(f"[red]Error: {response.status_code}[/red]")

            elif action == "create":
                if not name:
                    console.print("[red]--name is required for create[/red]")
                    return

                response = await client.post(
                    f"{server_url}/api/auth/api-keys",
                    headers=headers,
                    json={"name": name, "with_encryption": True},
                )
                if response.status_code == 200:
                    data = response.json()
                    console.print("[green]API key created successfully![/green]")
                    console.print(
                        "\n[yellow]IMPORTANT: Save this value - it won't be shown again![/yellow]\n"
                    )
                    console.print(f"API Key: {data['api_key']}")
                    console.print(
                        "[dim]E2EE is automatic - encryption key derived from API key[/dim]"
                    )
                    console.print("\nConfig snippet:")
                    console.print(data.get("config_snippet", ""))
                else:
                    console.print(f"[red]Error: {response.status_code}[/red]")

            elif action == "revoke":
                if not key_id:
                    console.print("[red]--id is required for revoke[/red]")
                    return

                response = await client.post(
                    f"{server_url}/api/auth/api-keys/revoke",
                    headers=headers,
                    json={"key_id": key_id},
                )
                if response.status_code == 200:
                    console.print("[green]API key revoked[/green]")
                else:
                    console.print(f"[red]Error: {response.status_code}[/red]")

            else:
                console.print(f"[red]Unknown action: {action}. Use: create, list, or revoke[/red]")

    asyncio.run(manage_keys())


@cloud_app.command()
def upgrade():
    """Open browser to upgrade subscription."""
    import webbrowser

    cloud_config = _get_cloud_config()
    server_url = cloud_config.get("server_url", "https://contextfs.ai")

    # Open billing page
    billing_url = f"{server_url}/dashboard/billing"
    console.print(f"Opening billing page: {billing_url}")
    webbrowser.open(billing_url)


@cloud_app.command()
def sync(
    push_all: bool = typer.Option(
        False, "--all", "-a", help="Push all memories (not just changed)"
    ),
):
    """Sync memories with cloud (authenticated sync)."""
    cloud_config = _get_cloud_config()

    if not cloud_config.get("enabled"):
        console.print(
            "[yellow]Cloud sync is disabled. Run: contextfs cloud configure --enabled[/yellow]"
        )
        return

    if not cloud_config.get("api_key"):
        console.print("[red]No API key configured. Run: contextfs cloud login[/red]")
        return

    server_url = cloud_config.get("server_url", "https://api.contextfs.ai")
    api_key = cloud_config.get("api_key")

    async def do_sync():
        from contextfs.sync.client import SyncClient

        ctx = get_ctx()
        async with SyncClient(
            server_url=server_url,
            ctx=ctx,
            api_key=api_key,
        ) as client:
            console.print(f"[dim]Syncing with {server_url}...[/dim]")
            console.print("[dim]E2EE: automatic[/dim]")

            result = await client.sync_all()

            console.print("[green]Sync complete![/green]")
            console.print(f"  Pushed: {result.pushed.accepted} memories")
            console.print(f"  Pulled: {len(result.pulled.memories)} memories")
            console.print(f"  Duration: {result.duration_ms:.0f}ms")

            if result.errors:
                for error in result.errors:
                    console.print(f"  [red]Error: {error}[/red]")

    asyncio.run(do_sync())


@cloud_app.command("create-admin")
def create_admin():
    """Create admin user for local development/testing.

    Admin user bypasses email verification and has unlimited usage limits.
    Use for local testing only - not for production.

    The API key is printed once. Save it for future use.
    """

    async def do_create_admin():
        from contextfs.auth.storage import create_auth_storage
        from contextfs.auth.storage.factory import create_admin_user

        storage = create_auth_storage()
        try:
            user_id, api_key = await create_admin_user(storage)

            console.print("\n[green]Admin user created/retrieved![/green]")
            console.print(f"  User ID: {user_id}")
            console.print("  Email: admin@contextfs.local")
            console.print()
            console.print(
                "[yellow]IMPORTANT: Save this API key - it won't be shown again![/yellow]"
            )
            console.print(f"\n  API Key: {api_key}\n")
            console.print("To use this key, add to your config:")
            console.print("  contextfs cloud configure --api-key " + api_key[:12] + "...")
        finally:
            await storage.close()

    asyncio.run(do_create_admin())
