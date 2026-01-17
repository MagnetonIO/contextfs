"""Cloud sync commands for ContextFS commercial platform."""

import asyncio
import os
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


def _get_device_id() -> str:
    """Get or create device ID - uses same logic as SyncClient for consistency."""
    import socket
    import uuid

    config_path = Path.home() / ".contextfs" / "device_id"
    if config_path.exists():
        return config_path.read_text().strip()

    # Generate new device ID using hostname and MAC address
    hostname = socket.gethostname()
    mac = uuid.getnode()
    device_id = f"{hostname}-{mac:012x}"[:32]

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(device_id)
    return device_id


# CLI OAuth credentials (from environment)
# Client IDs are public and can be hardcoded in releases
# Client secrets are NOT needed for Device Code (GitHub) and PKCE (Google) flows
CLI_GITHUB_CLIENT_ID = os.environ.get("CONTEXTFS_CLI_GITHUB_CLIENT_ID", "Ov23liqwVX5FABE0q7LR")
CLI_GOOGLE_CLIENT_ID = os.environ.get(
    "CONTEXTFS_CLI_GOOGLE_CLIENT_ID",
    "165667507258-9qlsa7c5kejsj5rq90at3ksvl3moeqds.apps.googleusercontent.com",
)
CLI_GOOGLE_CLIENT_SECRET = os.environ.get(
    "CONTEXTFS_CLI_GOOGLE_CLIENT_SECRET", ""
)  # Optional for PKCE
CLI_OAUTH_PORT = int(os.environ.get("CONTEXTFS_CLI_OAUTH_PORT", "8400"))


@cloud_app.command()
def login(
    provider: str = typer.Option(
        "github", "--provider", "-p", help="Login provider (github, google, email)"
    ),
    email: str = typer.Option(None, "--email", "-e", help="Email for email login"),
    password: str = typer.Option(None, "--password", help="Password for email login"),
):
    """Login to ContextFS Cloud.

    Supports:
    - github: OAuth via browser (default)
    - google: Google OAuth via browser
    - email: Email/password login
    """

    cloud_config = _get_cloud_config()
    server_url = cloud_config.get("server_url", "https://api.contextfs.ai")

    if provider == "email":
        _login_email(cloud_config, server_url, email, password)
    elif provider == "github":
        _login_github(cloud_config, server_url)
    elif provider == "google":
        _login_google(cloud_config, server_url)
    else:
        console.print(
            f"[red]Unknown provider: {provider}. Use 'github', 'google', or 'email'[/red]"
        )


def _login_email(cloud_config: dict, server_url: str, email: str | None, password: str | None):
    """Login with email/password."""
    import platform
    import socket

    import httpx

    # Prompt for credentials if not provided
    if not email:
        email = typer.prompt("Email")
    if not password:
        password = typer.prompt("Password", hide_input=True)

    console.print("[dim]Logging in...[/dim]")

    try:
        with httpx.Client() as client:
            resp = client.post(
                f"{server_url}/api/auth/login",
                json={"email": email, "password": password, "session_type": "CLI Session"},
            )

            if resp.status_code == 401:
                console.print("[red]Invalid email or password[/red]")
                return

            resp.raise_for_status()
            data = resp.json()

            api_key = data["apiKey"]
            encryption_key = data.get("encryptionKey")
            user = data["user"]

            # Save to config
            cloud_config["api_key"] = api_key
            cloud_config["enabled"] = True
            if encryption_key:
                cloud_config["encryption_key"] = encryption_key
            else:
                cloud_config.pop("encryption_key", None)  # Remove if E2EE disabled
            _save_cloud_config(cloud_config)

            console.print(
                f"[green]Login successful! Welcome {user.get('name') or user['email']}[/green]"
            )
            console.print("[dim]API key saved to ~/.contextfs/config.yaml[/dim]")
            if encryption_key:
                console.print("[dim]E2EE encryption key configured[/dim]")

            # Auto-register device
            console.print("[dim]Registering device...[/dim]")
            device_id = _get_device_id()
            device_resp = client.post(
                f"{server_url}/api/sync/register",
                json={
                    "device_id": device_id,
                    "device_name": socket.gethostname(),
                    "platform": platform.system().lower(),
                    "client_version": "0.2.0",
                },
                headers={"X-API-Key": api_key},
            )
            if device_resp.status_code == 200:
                device_info = device_resp.json()
                console.print(f"[green]Device registered: {device_info['device_name']}[/green]")
            else:
                console.print(f"[yellow]Device registration failed: {device_resp.text}[/yellow]")

    except httpx.HTTPStatusError as e:
        console.print(f"[red]Login failed: {e.response.text}[/red]")
    except Exception as e:
        console.print(f"[red]Login failed: {e}[/red]")


def _login_github(cloud_config: dict, server_url: str):
    """Login with GitHub OAuth using Device Code Flow.

    This flow doesn't require a client secret, making it safe for distributed CLIs.
    User gets a code, enters it at github.com/login/device, and CLI polls for completion.
    """
    import platform
    import socket
    import time
    import webbrowser

    import httpx

    if not CLI_GITHUB_CLIENT_ID:
        console.print(
            "[red]GitHub OAuth not configured. Set CONTEXTFS_CLI_GITHUB_CLIENT_ID env var.[/red]"
        )
        return

    try:
        with httpx.Client() as client:
            # Step 1: Request device code from GitHub
            device_resp = client.post(
                "https://github.com/login/device/code",
                data={
                    "client_id": CLI_GITHUB_CLIENT_ID,
                    "scope": "user:email",
                },
                headers={"Accept": "application/json"},
            )
            device_resp.raise_for_status()
            device_data = device_resp.json()

            if "error" in device_data:
                console.print(
                    f"[red]GitHub error: {device_data.get('error_description', device_data['error'])}[/red]"
                )
                return

            user_code = device_data["user_code"]
            device_code = device_data["device_code"]
            verification_uri = device_data["verification_uri"]
            expires_in = device_data.get("expires_in", 900)
            interval = device_data.get("interval", 5)

            # Step 2: Display code and open browser
            console.print()
            console.print(
                f"[bold yellow]! First, copy your one-time code: {user_code}[/bold yellow]"
            )
            console.print()

            if typer.confirm("Press Enter to open github.com in your browser", default=True):
                webbrowser.open(verification_uri)
            else:
                console.print(f"[dim]Open this URL manually: {verification_uri}[/dim]")

            console.print()
            console.print("[dim]Waiting for authorization...[/dim]")

            # Step 3: Poll for access token
            start_time = time.time()
            access_token = None

            while time.time() - start_time < expires_in:
                time.sleep(interval)

                token_resp = client.post(
                    "https://github.com/login/oauth/access_token",
                    data={
                        "client_id": CLI_GITHUB_CLIENT_ID,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                    headers={"Accept": "application/json"},
                )
                token_data = token_resp.json()

                if "access_token" in token_data:
                    access_token = token_data["access_token"]
                    break
                elif token_data.get("error") == "authorization_pending":
                    continue  # User hasn't completed authorization yet
                elif token_data.get("error") == "slow_down":
                    interval += 5  # GitHub wants us to slow down
                elif token_data.get("error") == "expired_token":
                    console.print("[red]Authorization expired. Please try again.[/red]")
                    return
                elif token_data.get("error") == "access_denied":
                    console.print("[red]Authorization denied.[/red]")
                    return
                elif "error" in token_data:
                    console.print(
                        f"[red]GitHub error: {token_data.get('error_description', token_data['error'])}[/red]"
                    )
                    return

            if not access_token:
                console.print("[red]Authorization timed out. Please try again.[/red]")
                return

            # Step 4: Exchange GitHub access token for ContextFS API key
            console.print("[dim]Getting ContextFS API key...[/dim]")
            resp = client.post(
                f"{server_url}/api/auth/oauth/token",
                json={"provider": "github", "access_token": access_token},
            )
            resp.raise_for_status()
            data = resp.json()

            api_key = data["api_key"]
            encryption_key = data.get("encryption_key")

            # Save to config
            cloud_config["api_key"] = api_key
            cloud_config["enabled"] = True
            if encryption_key:
                cloud_config["encryption_key"] = encryption_key
            _save_cloud_config(cloud_config)

            console.print("[green]✓ Authentication complete![/green]")
            console.print("[dim]API key saved to ~/.contextfs/config.yaml[/dim]")

            # Auto-register device
            console.print("[dim]Registering device...[/dim]")
            device_id = _get_device_id()
            device_resp = client.post(
                f"{server_url}/api/sync/register",
                json={
                    "device_id": device_id,
                    "device_name": socket.gethostname(),
                    "platform": platform.system().lower(),
                    "client_version": "0.2.0",
                },
                headers={"X-API-Key": api_key},
            )
            if device_resp.status_code == 200:
                device_info = device_resp.json()
                console.print(f"[green]✓ Device registered: {device_info['device_name']}[/green]")
            else:
                console.print(
                    f"[yellow]Device registration failed (non-critical): {device_resp.text}[/yellow]"
                )

    except httpx.HTTPStatusError as e:
        console.print(f"[red]Failed to complete login: {e.response.text}[/red]")
    except Exception as e:
        console.print(f"[red]Failed to complete login: {e}[/red]")


def _login_google(cloud_config: dict, server_url: str):
    """Login with Google OAuth."""
    import base64
    import hashlib
    import http.server
    import platform
    import secrets
    import socket
    import socketserver
    import threading
    import urllib.parse
    import webbrowser

    import httpx

    if not CLI_GOOGLE_CLIENT_ID:
        console.print(
            "[red]Google OAuth not configured. Set CONTEXTFS_CLI_GOOGLE_CLIENT_ID env var.[/red]"
        )
        return

    redirect_uri = f"http://localhost:{CLI_OAUTH_PORT}/callback"
    auth_code = None
    state_token = secrets.token_urlsafe(32)

    # PKCE: Generate code verifier and challenge
    code_verifier = secrets.token_urlsafe(64)[:128]
    code_challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )

    class CallbackHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress logging

        def do_GET(self):
            nonlocal auth_code
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)

            if "code" in params:
                auth_code = params["code"][0]
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(b"""
                    <html><body style="font-family: sans-serif; text-align: center; padding: 50px;">
                    <h1>Login Successful!</h1>
                    <p>You can close this window and return to the terminal.</p>
                    </body></html>
                """)
            else:
                self.send_response(400)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                error = params.get("error", ["Unknown error"])[0]
                self.wfile.write(
                    f"<html><body><h1>Login Failed</h1><p>{error}</p></body></html>".encode()
                )

    # Start local server
    try:
        server = socketserver.TCPServer(("localhost", CLI_OAUTH_PORT), CallbackHandler)
    except OSError:
        console.print(
            f"[red]Port {CLI_OAUTH_PORT} is in use. Close other applications and try again.[/red]"
        )
        return

    server_thread = threading.Thread(target=server.handle_request)
    server_thread.start()

    # Build Google OAuth URL with PKCE
    oauth_params = {
        "client_id": CLI_GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state_token,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",
    }
    oauth_url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(oauth_params)}"
    )

    console.print("Opening browser for Google login...")
    webbrowser.open(oauth_url)
    console.print("[dim]Waiting for authentication...[/dim]")

    # Wait for callback
    server_thread.join(timeout=120)
    server.server_close()

    if not auth_code:
        console.print("[red]Login timed out or was cancelled[/red]")
        return

    # Exchange code for Google access token
    console.print("[dim]Exchanging code for access token...[/dim]")
    try:
        with httpx.Client() as client:
            # Exchange code with Google using PKCE (no client secret needed for desktop apps)
            token_request = {
                "client_id": CLI_GOOGLE_CLIENT_ID,
                "code": auth_code,
                "code_verifier": code_verifier,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            }
            # Include client_secret only if provided (optional for PKCE)
            if CLI_GOOGLE_CLIENT_SECRET:
                token_request["client_secret"] = CLI_GOOGLE_CLIENT_SECRET

            token_resp = client.post(
                "https://oauth2.googleapis.com/token",
                data=token_request,
            )
            token_resp.raise_for_status()
            token_data = token_resp.json()

            if "error" in token_data:
                console.print(
                    f"[red]Google error: {token_data.get('error_description', token_data['error'])}[/red]"
                )
                return

            access_token = token_data["access_token"]

            # Exchange Google access token for ContextFS API key
            console.print("[dim]Getting ContextFS API key...[/dim]")
            resp = client.post(
                f"{server_url}/api/auth/oauth/token",
                json={"provider": "google", "access_token": access_token},
            )
            resp.raise_for_status()
            data = resp.json()

            api_key = data["api_key"]
            encryption_key = data.get("encryption_key")

            # Save to config
            cloud_config["api_key"] = api_key
            cloud_config["enabled"] = True
            if encryption_key:
                cloud_config["encryption_key"] = encryption_key
            _save_cloud_config(cloud_config)

            console.print("[green]Login successful![/green]")
            console.print("[dim]API key saved to ~/.contextfs/config.yaml[/dim]")

            # Auto-register device
            console.print("[dim]Registering device...[/dim]")
            device_id = _get_device_id()
            device_resp = client.post(
                f"{server_url}/api/sync/register",
                json={
                    "device_id": device_id,
                    "device_name": socket.gethostname(),
                    "platform": platform.system().lower(),
                    "client_version": "0.2.0",
                },
                headers={"X-API-Key": api_key},
            )
            if device_resp.status_code == 200:
                device_info = device_resp.json()
                console.print(f"[green]Device registered: {device_info['device_name']}[/green]")
            else:
                console.print(
                    f"[yellow]Device registration failed (non-critical): {device_resp.text}[/yellow]"
                )

    except httpx.HTTPStatusError as e:
        console.print(f"[red]Failed to complete login: {e.response.text}[/red]")
    except Exception as e:
        console.print(f"[red]Failed to complete login: {e}[/red]")


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
    force: bool = typer.Option(
        False, "--force", "-f", help="Force overwrite server data regardless of vector clock state"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed sync breakdown"),
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
            if force:
                console.print("[dim]Force mode: overwriting stale data[/dim]")

            result = await client.sync_all(force=force)

            console.print("[green]Sync complete![/green]")
            # Use memory-specific counts (not sessions/edges)
            pushed_count = getattr(result.pushed, "accepted_memories", result.pushed.accepted)
            console.print(f"  Pushed: {pushed_count} memories")
            rejected_count = getattr(result.pushed, "rejected_memories", result.pushed.rejected)
            if rejected_count:
                console.print(f"  Rejected: {rejected_count}")
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
