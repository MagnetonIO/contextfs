"""Shared utilities for CLI commands."""

from pathlib import Path

from rich.console import Console

from contextfs.core import ContextFS

console = Console()


def get_ctx() -> ContextFS:
    """Get ContextFS instance."""
    return ContextFS(auto_load=True)


def find_git_root(start_path: Path) -> Path | None:
    """Find the root of a git repository."""
    current = start_path.resolve()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return None


def get_contextfs_config_path(repo_path: Path) -> Path:
    """Get the path to the contextfs config file."""
    return repo_path / ".contextfs" / "config.yaml"


def is_repo_initialized(repo_path: Path) -> bool:
    """Check if a repo has been initialized for ContextFS."""
    config_path = get_contextfs_config_path(repo_path)
    return config_path.exists()


def get_repo_config(repo_path: Path) -> dict | None:
    """Get the contextfs config for a repo."""
    config_path = get_contextfs_config_path(repo_path)
    if not config_path.exists():
        return None

    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def create_repo_config(
    repo_path: Path,
    auto_index: bool = True,
    created_by: str = "cli",
    max_commits: int = 100,
    project: str | None = None,
) -> Path:
    """Create .contextfs/config.yaml for a repository."""
    from datetime import datetime, timezone

    import yaml

    config_dir = repo_path / ".contextfs"
    config_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "version": 1,
        "auto_index": auto_index,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": created_by,
        "max_commits": max_commits,
    }
    if project:
        config["project"] = project

    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))

    # Add .contextfs to .gitignore if not already there
    gitignore_path = repo_path / ".gitignore"
    gitignore_entry = ".contextfs/"
    if gitignore_path.exists():
        content = gitignore_path.read_text()
        if gitignore_entry not in content:
            with open(gitignore_path, "a") as f:
                if not content.endswith("\n"):
                    f.write("\n")
                f.write(f"\n# ContextFS local config\n{gitignore_entry}\n")
    else:
        gitignore_path.write_text(f"# ContextFS local config\n{gitignore_entry}\n")

    return config_path


def _get_cloud_config() -> dict:
    """Get cloud configuration from local config file."""
    import yaml

    config_path = Path.home() / ".contextfs" / "cloud.yaml"
    if not config_path.exists():
        return {"enabled": False}

    with open(config_path) as f:
        return yaml.safe_load(f) or {"enabled": False}


def _save_cloud_config(cloud_config: dict) -> None:
    """Save cloud configuration to local config file."""
    import yaml

    config_path = Path.home() / ".contextfs" / "cloud.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(cloud_config, f, default_flow_style=False)
