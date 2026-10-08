"""Configuration loading for Forge.

Loads YAML scenario and default configuration files.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Resolve project root relative to this file (forge_core/ -> forge/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_scenario(scenario_name: str) -> dict[str, Any]:
    """Load a scenario configuration by name.

    Looks in ``simulation/scenarios/<scenario_name>.yaml``.
    """
    scenario_path = PROJECT_ROOT / "simulation" / "scenarios" / f"{scenario_name}.yaml"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario not found: {scenario_path}")
    logger.info("scenario_loaded: %s", scenario_name)
    return load_yaml(scenario_path)


def load_default_config() -> dict[str, Any]:
    """Load the default Forge configuration."""
    return load_yaml(PROJECT_ROOT / "config" / "default.yaml")


def resolve_model_path(relative_path: str) -> Path:
    """Resolve a model path relative to the project root."""
    full = PROJECT_ROOT / relative_path
    if not full.exists():
        raise FileNotFoundError(f"Model not found: {full}")
    return full
