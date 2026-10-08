"""Unit tests for configuration loading."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.config import load_scenario, load_default_config, resolve_model_path


class TestConfigLoading:
    def test_load_default_config(self):
        """Default config loads successfully."""
        cfg = load_default_config()
        assert "robot" in cfg
        assert "simulation" in cfg

    def test_load_basic_workspace_scenario(self):
        """Scenario basic_workspace loads successfully."""
        scenario = load_scenario("basic_workspace")
        assert scenario["scenario"]["name"] == "basic_workspace"
        assert "robot" in scenario
        assert "objects" in scenario

    def test_scenario_has_required_objects(self):
        """Scenario includes red_cube, blue_cube, target_bin."""
        scenario = load_scenario("basic_workspace")
        objects = scenario["objects"]
        assert "red_cube" in objects
        assert "blue_cube" in objects
        assert "target_bin" in objects

    def test_scenario_has_robot_config(self):
        """Scenario specifies robot model and home position."""
        scenario = load_scenario("basic_workspace")
        robot = scenario["robot"]
        assert robot["model"] == "panda"
        assert "home_qpos" in robot

    def test_missing_scenario_raises(self):
        """Loading a non-existent scenario raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_scenario("nonexistent_scenario")

    def test_resolve_model_path(self):
        """Model paths resolve correctly."""
        path = resolve_model_path("simulation/models/panda.xml")
        assert path.exists()
