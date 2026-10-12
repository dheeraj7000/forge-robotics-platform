"""Unit tests for the safety monitoring system.

Tests cover SafetyConfig, SafetyViolation, SafetyMonitor check/send/stop,
and configuration loading from YAML.

Phase 11 — FEAT-002.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.hardware import RobotCommand, RobotState, SimulatedRobot
from forge_core.safety import (
    SafetyConfig,
    SafetyMonitor,
    SafetyViolation,
    load_safety_config,
)


# ── TestSafetyConfig ───────────────────────────────────────


class TestSafetyConfig:
    def test_defaults(self):
        config = SafetyConfig()
        assert len(config.joint_limits_lower) == 7
        assert len(config.joint_limits_upper) == 7
        assert len(config.velocity_limits) == 7
        assert len(config.workspace_min) == 3
        assert len(config.workspace_max) == 3
        assert config.max_gripper_force == 70.0

    def test_from_dict_with_custom_values(self):
        data = {
            "joint_limits_lower": [-1.0] * 7,
            "joint_limits_upper": [1.0] * 7,
            "velocity_limits": [3.0] * 7,
            "workspace_min": [-2.0, -2.0, -1.0],
            "workspace_max": [2.0, 2.0, 2.0],
            "max_gripper_force": 50.0,
        }
        config = SafetyConfig.from_dict(data)
        assert config.joint_limits_lower == [-1.0] * 7
        assert config.joint_limits_upper == [1.0] * 7
        assert config.velocity_limits == [3.0] * 7
        assert config.workspace_min == [-2.0, -2.0, -1.0]
        assert config.max_gripper_force == 50.0

    def test_load_safety_config_missing_file_returns_defaults(self):
        config = load_safety_config(path=None)
        # Should return config (either from file or defaults)
        assert isinstance(config, SafetyConfig)
        assert len(config.joint_limits_lower) == 7

    def test_load_safety_config_from_real_file(self):
        from forge_core.config import PROJECT_ROOT
        config_path = PROJECT_ROOT / "config" / "safety.yaml"
        if config_path.exists():
            config = load_safety_config(str(config_path))
            assert isinstance(config, SafetyConfig)
            assert len(config.joint_limits_lower) == 7

    def test_load_safety_config_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_safety_config(path="/nonexistent/safety.yaml")


# ── TestSafetyViolation ────────────────────────────────────


class TestSafetyViolation:
    def test_creation(self):
        v = SafetyViolation(
            violation_type="joint_limit",
            message="Joint 0 target below limit",
            joint_index=0,
            value=-3.5,
            limit=-2.9,
            timestamp=time.time(),
        )
        assert v.violation_type == "joint_limit"
        assert v.joint_index == 0
        assert v.value == -3.5

    def test_to_dict(self):
        v = SafetyViolation(
            violation_type="velocity_limit",
            message="Joint 2 velocity exceeded",
            joint_index=2,
            value=5.0,
            limit=2.2,
            timestamp=100.0,
        )
        d = v.to_dict()
        assert d["violation_type"] == "velocity_limit"
        assert d["joint_index"] == 2
        assert d["value"] == 5.0
        assert d["limit"] == 2.2
        assert d["timestamp"] == 100.0


# ── TestSafetyMonitor ──────────────────────────────────────


class TestSafetyMonitor:
    def _make_safe_state(self) -> RobotState:
        """Return a robot state well within safety bounds."""
        return RobotState(
            joint_positions=np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0]),
            joint_velocities=np.zeros(7),
            ee_position=np.array([0.3, 0.0, 0.5]),
            ee_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper_state=0.0,
            timestamp=time.time(),
        )

    def _make_safe_targets(self) -> np.ndarray:
        """Return joint targets within all safety limits."""
        # Panda joint 3 limits are [-3.1, -0.07], so use the home position
        return np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0])

    def test_check_command_safe(self):
        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        monitor = SafetyMonitor(robot)
        state = self._make_safe_state()
        cmd = RobotCommand(
            joint_targets=self._make_safe_targets(),
            gripper_command=0.0,
            timestamp=time.time(),
        )
        violations = monitor.check_command(cmd, state)
        assert violations == []
        robot.disconnect()

    def test_check_command_joint_limit_violation(self):
        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        monitor = SafetyMonitor(robot)
        state = self._make_safe_state()
        # Joint 0 lower limit is -2.9 — send target far below
        targets = np.zeros(7)
        targets[0] = -5.0
        cmd = RobotCommand(
            joint_targets=targets,
            gripper_command=0.0,
            timestamp=time.time(),
        )
        violations = monitor.check_command(cmd, state)
        joint_violations = [v for v in violations if v.violation_type == "joint_limit"]
        assert len(joint_violations) >= 1
        assert joint_violations[0].joint_index == 0
        robot.disconnect()

    def test_check_command_velocity_violation(self):
        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        monitor = SafetyMonitor(robot)
        state = self._make_safe_state()
        # Large position delta interpreted as velocity > limit
        targets = np.zeros(7)
        targets[0] = 5.0  # delta of 5.0, velocity limit is 2.2
        cmd = RobotCommand(
            joint_targets=targets,
            gripper_command=0.0,
            timestamp=time.time(),
        )
        violations = monitor.check_command(cmd, state)
        vel_violations = [v for v in violations if v.violation_type == "velocity_limit"]
        assert len(vel_violations) >= 1
        robot.disconnect()

    def test_check_command_workspace_boundary_violation(self):
        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        monitor = SafetyMonitor(robot)
        # State with EE outside workspace bounds
        state = RobotState(
            joint_positions=np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0]),
            joint_velocities=np.zeros(7),
            ee_position=np.array([5.0, 5.0, 5.0]),  # Far outside workspace
            ee_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper_state=0.0,
            timestamp=time.time(),
        )
        cmd = RobotCommand(
            joint_targets=self._make_safe_targets(),
            gripper_command=0.0,
            timestamp=time.time(),
        )
        violations = monitor.check_command(cmd, state)
        ws_violations = [v for v in violations if v.violation_type == "workspace_boundary"]
        assert len(ws_violations) >= 1
        robot.disconnect()

    def test_is_safe_returns_true(self):
        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        monitor = SafetyMonitor(robot)
        state = self._make_safe_state()
        cmd = RobotCommand(
            joint_targets=self._make_safe_targets(),
            gripper_command=0.0,
            timestamp=time.time(),
        )
        assert monitor.is_safe(cmd, state) is True
        robot.disconnect()

    def test_is_safe_returns_false(self):
        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        monitor = SafetyMonitor(robot)
        state = self._make_safe_state()
        targets = np.zeros(7)
        targets[0] = -5.0  # Violates joint limit
        cmd = RobotCommand(
            joint_targets=targets,
            gripper_command=0.0,
            timestamp=time.time(),
        )
        assert monitor.is_safe(cmd, state) is False
        robot.disconnect()

    def test_safe_send_command_sends_when_safe(self):
        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        monitor = SafetyMonitor(robot)
        cmd = RobotCommand(
            joint_targets=self._make_safe_targets(),
            gripper_command=0.0,
            timestamp=time.time(),
        )
        sent, violations = monitor.safe_send_command(cmd)
        assert sent is True
        assert violations == []
        robot.disconnect()

    def test_safe_send_command_blocks_when_unsafe(self):
        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        # Use tight limits so the default robot state's EE is outside workspace
        config = SafetyConfig(
            joint_limits_lower=[-0.001] * 7,
            joint_limits_upper=[0.001] * 7,
            velocity_limits=[0.001] * 7,
            workspace_min=[-0.001, -0.001, -0.001],
            workspace_max=[0.001, 0.001, 0.001],
        )
        monitor = SafetyMonitor(robot, config=config)
        # Any non-zero target should violate the tight limits
        cmd = RobotCommand(
            joint_targets=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            gripper_command=0.0,
            timestamp=time.time(),
        )
        sent, violations = monitor.safe_send_command(cmd)
        assert sent is False
        assert len(violations) > 0
        robot.disconnect()

    def test_emergency_stop_delegates_to_robot(self):
        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        monitor = SafetyMonitor(robot)
        monitor.emergency_stop()
        status = robot.get_status()
        assert status.status == "e_stopped"
        robot.disconnect()

    def test_get_violations_log_accumulates(self):
        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        config = SafetyConfig(
            joint_limits_lower=[-0.001] * 7,
            joint_limits_upper=[0.001] * 7,
        )
        monitor = SafetyMonitor(robot, config=config)
        # Send an unsafe command
        cmd = RobotCommand(
            joint_targets=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            gripper_command=0.0,
            timestamp=time.time(),
        )
        monitor.safe_send_command(cmd)
        log = monitor.get_violations_log()
        assert len(log) > 0
        assert all(isinstance(v, dict) for v in log)
        robot.disconnect()

    def test_safe_send_with_custom_config(self):
        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        # Wide limits — everything should pass
        config = SafetyConfig(
            joint_limits_lower=[-10.0] * 7,
            joint_limits_upper=[10.0] * 7,
            velocity_limits=[100.0] * 7,
            workspace_min=[-100.0, -100.0, -100.0],
            workspace_max=[100.0, 100.0, 100.0],
        )
        monitor = SafetyMonitor(robot, config=config)
        cmd = RobotCommand(
            joint_targets=np.array([2.0, 1.0, 0.5, -2.0, 1.5, 3.0, 1.0]),
            gripper_command=0.0,
            timestamp=time.time(),
        )
        sent, violations = monitor.safe_send_command(cmd)
        assert sent is True
        assert violations == []
        robot.disconnect()
