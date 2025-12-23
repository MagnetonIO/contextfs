"""Path normalization and repo registry for cross-machine sync.

This module handles the conversion between absolute local paths
and portable references (repo_url + relative_path) that can be
synced across machines with different directory structures.

Key concepts:
- RepoRegistry: Maps repo URLs to local paths
- PathResolver: Converts between absolute and portable paths
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Portable Path Reference
# =============================================================================


class PortablePath(BaseModel):
    """Portable path reference for cross-machine sync.

    Uses repo URL as canonical identifier and relative path from repo root.
    This allows the same memory to be resolved on different machines
    with different local directory structures.
    """

    repo_url: str | None = None  # git@github.com:user/repo.git
    repo_name: str | None = None  # Human-readable name
    relative_path: str | None = None  # Path from repo root

    def is_valid(self) -> bool:
        """Check if this is a valid portable path."""
        return bool(self.repo_url and self.relative_path)

    def to_dict(self) -> dict[str, str | None]:
        """Export as dict."""
        return {
            "repo_url": self.repo_url,
            "repo_name": self.repo_name,
            "relative_path": self.relative_path,
        }


# =============================================================================
# Repo Registry
# =============================================================================


class RepoRegistry(BaseModel):
    """Registry mapping repo URLs to local paths.

    Stored at ~/.contextfs/repo_registry.json and updated
    automatically when repos are discovered or indexed.
    """

    repos: dict[str, str] = Field(default_factory=dict)  # repo_url -> local_path
    _registry_path: Path | None = None

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def get_default_path(cls) -> Path:
        """Get default registry file path."""
        return Path.home() / ".contextfs" / "repo_registry.json"

    @classmethod
    def load(cls, path: Path | None = None) -> RepoRegistry:
        """Load registry from file."""
        registry_path = path or cls.get_default_path()

        if not registry_path.exists():
            registry = cls()
            registry._registry_path = registry_path
            return registry

        try:
            with open(registry_path) as f:
                data = json.load(f)
            registry = cls(repos=data.get("repos", {}))
            registry._registry_path = registry_path
            return registry
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load repo registry: {e}")
            registry = cls()
            registry._registry_path = registry_path
            return registry

    def save(self, path: Path | None = None) -> None:
        """Save registry to file."""
        registry_path = path or self._registry_path or self.get_default_path()
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        with open(registry_path, "w") as f:
            json.dump({"repos": self.repos}, f, indent=2)

    def register(self, repo_url: str, local_path: str | Path) -> None:
        """Register a repo URL to local path mapping."""
        normalized_url = self._normalize_url(repo_url)
        self.repos[normalized_url] = str(local_path)
        self.save()

    def unregister(self, repo_url: str) -> bool:
        """Unregister a repo URL."""
        normalized_url = self._normalize_url(repo_url)
        if normalized_url in self.repos:
            del self.repos[normalized_url]
            self.save()
            return True
        return False

    def get_local_path(self, repo_url: str) -> Path | None:
        """Get local path for a repo URL."""
        normalized_url = self._normalize_url(repo_url)
        local_path = self.repos.get(normalized_url)
        if local_path:
            path = Path(local_path)
            if path.exists():
                return path
        return None

    def get_repo_url(self, local_path: str | Path) -> str | None:
        """Get repo URL for a local path."""
        local_path = Path(local_path).resolve()
        for repo_url, registered_path in self.repos.items():
            if Path(registered_path).resolve() == local_path:
                return repo_url
        return None

    def find_containing_repo(self, file_path: str | Path) -> tuple[str, Path] | None:
        """Find the repo that contains a file path.

        Returns:
            Tuple of (repo_url, repo_local_path) or None if not found
        """
        file_path = Path(file_path).resolve()

        for repo_url, local_path in self.repos.items():
            repo_path = Path(local_path).resolve()
            try:
                file_path.relative_to(repo_path)
                return repo_url, repo_path
            except ValueError:
                continue

        return None

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize repo URL for consistent lookup.

        Handles variations like:
        - git@github.com:user/repo.git
        - https://github.com/user/repo.git
        - https://github.com/user/repo
        """
        url = url.strip()

        # Remove trailing .git if present
        if url.endswith(".git"):
            url = url[:-4]

        # Convert HTTPS to SSH format for consistency
        if url.startswith("https://github.com/"):
            path = url.replace("https://github.com/", "")
            url = f"git@github.com:{path}"
        elif url.startswith("https://gitlab.com/"):
            path = url.replace("https://gitlab.com/", "")
            url = f"git@gitlab.com:{path}"

        # Add .git back for storage
        if not url.endswith(".git"):
            url = url + ".git"

        return url


# =============================================================================
# Path Resolver
# =============================================================================


class PathResolver:
    """Resolves between absolute local paths and portable references.

    Uses git to determine repo URLs and relative paths, with
    fallback to the repo registry for non-git directories.
    """

    def __init__(self, registry: RepoRegistry | None = None):
        """Initialize path resolver.

        Args:
            registry: Repo registry to use (loads default if not provided)
        """
        self.registry = registry or RepoRegistry.load()

    def normalize(self, abs_path: str | Path) -> PortablePath:
        """Convert absolute path to portable reference.

        Args:
            abs_path: Absolute file path

        Returns:
            PortablePath with repo_url and relative_path
        """
        abs_path = Path(abs_path).resolve()

        # Try to get repo info from git
        repo_root = self._get_git_root(abs_path)
        if repo_root:
            repo_url = self._get_git_remote_url(repo_root)
            repo_name = repo_root.name

            if repo_url:
                relative_path = str(abs_path.relative_to(repo_root))

                # Register in registry for future lookups
                self.registry.register(repo_url, repo_root)

                return PortablePath(
                    repo_url=repo_url,
                    repo_name=repo_name,
                    relative_path=relative_path,
                )

        # Fallback: check registry for containing repo
        result = self.registry.find_containing_repo(abs_path)
        if result:
            repo_url, repo_path = result
            relative_path = str(abs_path.relative_to(repo_path))
            return PortablePath(
                repo_url=repo_url,
                repo_name=repo_path.name,
                relative_path=relative_path,
            )

        # No repo found - return with just the filename
        return PortablePath(
            repo_url=None,
            repo_name=None,
            relative_path=abs_path.name,
        )

    def resolve(self, portable: PortablePath) -> Path | None:
        """Resolve portable reference to local path.

        Args:
            portable: Portable path reference

        Returns:
            Absolute local path or None if repo not found locally
        """
        if not portable.repo_url or not portable.relative_path:
            return None

        local_repo = self.registry.get_local_path(portable.repo_url)
        if not local_repo:
            return None

        return local_repo / portable.relative_path

    def resolve_to_string(self, portable: PortablePath) -> str | None:
        """Resolve portable reference to local path string."""
        path = self.resolve(portable)
        return str(path) if path else None

    def auto_discover_repos(self, search_paths: list[Path] | None = None) -> int:
        """Auto-discover git repos and register them.

        Args:
            search_paths: Paths to search (defaults to common locations)

        Returns:
            Number of repos discovered
        """
        if search_paths is None:
            home = Path.home()
            search_paths = [
                home / "Documents",
                home / "Projects",
                home / "code",
                home / "dev",
                home / "Development",
                home / "workspace",
            ]

        discovered = 0
        for search_path in search_paths:
            if not search_path.exists():
                continue

            for git_dir in search_path.rglob(".git"):
                if git_dir.is_dir():
                    repo_root = git_dir.parent
                    repo_url = self._get_git_remote_url(repo_root)
                    if repo_url:
                        self.registry.register(repo_url, repo_root)
                        discovered += 1
                        logger.debug(f"Discovered repo: {repo_url} at {repo_root}")

        return discovered

    @staticmethod
    def _get_git_root(path: Path) -> Path | None:
        """Get git repository root for a path."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=path if path.is_dir() else path.parent,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except (subprocess.SubprocessError, OSError):
            pass
        return None

    @staticmethod
    def _get_git_remote_url(repo_path: Path) -> str | None:
        """Get git remote URL (origin) for a repository."""
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            pass
        return None


# =============================================================================
# Convenience Functions
# =============================================================================


def get_path_resolver() -> PathResolver:
    """Get a path resolver with the default registry."""
    return PathResolver()


def normalize_path(abs_path: str | Path) -> PortablePath:
    """Convenience function to normalize a path."""
    return get_path_resolver().normalize(abs_path)


def resolve_path(portable: PortablePath | dict[str, Any]) -> Path | None:
    """Convenience function to resolve a portable path."""
    if isinstance(portable, dict):
        portable = PortablePath(**portable)
    return get_path_resolver().resolve(portable)
