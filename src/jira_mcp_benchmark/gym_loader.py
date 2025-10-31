"""Gym configuration loader for multi-environment support."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class GymConfig:
    """Configuration for a specific gym (MCP server environment)."""

    name: str
    description: str
    mcp_url: str
    mcp_transport: str
    sql_runner_url: str
    headers: Dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"GymConfig(name={self.name!r}, mcp_url={self.mcp_url!r})"
    
    def get_headers_for_run(self, database_id: str) -> Dict[str, str]:
        """Get headers for a run, merging custom headers with the database ID.
        
        Args:
            database_id: The x-database-id to use for this run.
            
        Returns:
            Dictionary of headers with x-database-id and any custom headers.
        """
        headers = dict(self.headers)
        headers["x-database-id"] = database_id
        return headers


class GymRegistry:
    """Registry for managing gym configurations."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """Initialize the gym registry.

        Args:
            config_path: Path to the gyms.json file. If None, uses default location.
        """
        self.config_path = config_path or self._find_default_config()
        self._gyms: dict[str, GymConfig] = {}
        self._default_gym: Optional[str] = None
        self._load_config()

    @staticmethod
    def _find_default_config() -> Path:
        """Find the default gyms.json configuration file."""
        # Try current directory first
        current_dir = Path.cwd()
        candidate = current_dir / "gyms.json"
        if candidate.exists():
            return candidate

        # Try project root (parent of src directory)
        try:
            # Get the directory where this module is located
            module_dir = Path(__file__).parent
            project_root = module_dir.parent.parent
            candidate = project_root / "gyms.json"
            if candidate.exists():
                return candidate
        except Exception:
            pass

        # Return default path even if it doesn't exist yet
        return current_dir / "gyms.json"

    def _load_config(self) -> None:
        """Load gym configurations from the JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Gym configuration file not found: {self.config_path}\n"
                "Please create a gyms.json file with your gym configurations."
            )

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in gym configuration file: {self.config_path}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError("Gym configuration must be a JSON object")

        gyms_data = data.get("gyms", {})
        if not isinstance(gyms_data, dict):
            raise ValueError("'gyms' must be a JSON object")

        for gym_name, gym_data in gyms_data.items():
            if not isinstance(gym_data, dict):
                raise ValueError(f"Gym '{gym_name}' configuration must be a JSON object")

            # Extract transport type from nested structure or direct field
            transport_config = gym_data.get("transport", {})
            if isinstance(transport_config, dict):
                mcp_transport = transport_config.get("type", "streamable_http")
            else:
                mcp_transport = gym_data.get("mcp_transport", "streamable_http")

            # Extract custom headers (excluding x-database-id which is set per-run)
            custom_headers = gym_data.get("headers", {})
            if isinstance(custom_headers, dict):
                # Filter out x-database-id if present, as it will be set per-run
                custom_headers = {
                    k: v for k, v in custom_headers.items() 
                    if k.lower() != "x-database-id"
                }
            else:
                custom_headers = {}

            # Support both 'url' and 'mcp_url' field names
            mcp_url = gym_data.get("url") or gym_data.get("mcp_url")
            if not mcp_url:
                raise ValueError(f"Gym '{gym_name}' must have 'url' or 'mcp_url' field")

            # sql_runner_url is optional, derive from mcp_url if not provided
            sql_runner_url = gym_data.get("sql_runner_url")
            if not sql_runner_url:
                # Derive from mcp_url by replacing /mcp with /api/sql-runner
                base_url = mcp_url.rstrip("/")
                if base_url.endswith("/mcp"):
                    base_url = base_url[:-4]
                sql_runner_url = f"{base_url}/api/sql-runner"

            self._gyms[gym_name] = GymConfig(
                name=gym_data.get("name", gym_name),
                description=gym_data.get("description", ""),
                mcp_url=mcp_url,
                mcp_transport=mcp_transport,
                sql_runner_url=sql_runner_url,
                headers=custom_headers,
            )

        self._default_gym = data.get("default")
        if self._default_gym and self._default_gym not in self._gyms:
            raise ValueError(
                f"Default gym '{self._default_gym}' not found in gym configurations"
            )

    def get_gym(self, name: Optional[str] = None) -> GymConfig:
        """Get a gym configuration by name.

        Args:
            name: Name of the gym. If None, returns the default gym.

        Returns:
            GymConfig for the requested gym.

        Raises:
            ValueError: If the gym name is not found or no default is set.
        """
        if name is None:
            if self._default_gym is None:
                raise ValueError(
                    "No gym name provided and no default gym configured in gyms.json"
                )
            name = self._default_gym

        if name not in self._gyms:
            available = ", ".join(sorted(self._gyms.keys()))
            raise ValueError(
                f"Gym '{name}' not found. Available gyms: {available}"
            )

        return self._gyms[name]

    def list_gyms(self) -> list[GymConfig]:
        """List all available gym configurations.

        Returns:
            List of all registered gym configurations.
        """
        return list(self._gyms.values())

    def get_default_gym_name(self) -> Optional[str]:
        """Get the name of the default gym.

        Returns:
            Name of the default gym, or None if no default is set.
        """
        return self._default_gym


# Global registry instance
_registry: Optional[GymRegistry] = None


def get_gym_registry(config_path: Optional[Path] = None) -> GymRegistry:
    """Get or create the global gym registry instance.

    Args:
        config_path: Optional path to gyms.json. Only used on first call.

    Returns:
        The global GymRegistry instance.
    """
    global _registry
    if _registry is None:
        _registry = GymRegistry(config_path)
    return _registry


def reset_gym_registry() -> None:
    """Reset the global gym registry. Useful for testing."""
    global _registry
    _registry = None

