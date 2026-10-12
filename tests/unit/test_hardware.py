"""Unit tests for the hardware abstraction layer.

Tests cover RobotState, RobotCommand, RobotStatus dataclasses,
SimulatedRobot lifecycle, RobotInterface ABC enforcement, and
run_policy_on_robot integration.

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

from forge_core.hardware import (
    RobotCommand,
    RobotInterface,
    RobotState,
    RobotStatus,
    SimulatedRobot,
    run_policy_on_robot,
)


# ── TestRobotStateDataclass ────────────────────────────────


class TestRobotStateDataclass:
    def test_creation(self):
        state = RobotState(
            joint_positions=np.zeros(7),
            joint_velocities=np.zeros(7),
            ee_position=np.array([0.3, 0.0, 0.5]),
            ee_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper_state=0.0,
            timestamp=1.0,
        )
        assert state.gripper_state == 0.0
        assert state.timestamp == 1.0
        assert len(state.joint_positions) == 7

    def test_to_dict(self):
        state = RobotState(
            joint_positions=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            joint_velocities=np.zeros(7),
            ee_position=np.array([0.3, 0.0, 0.5]),
            ee_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper_state=0.5,
            timestamp=2.0,
        )
        d = state.to_dict()
        assert d["gripper_state"] == 0.5
        assert d["timestamp"] == 2.0
        assert isinstance(d["joint_positions"], list)
        assert len(d["joint_positions"]) == 7

    def test_from_dict_roundtrip(self):
        original = RobotState(
            joint_positions=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            joint_velocities=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            ee_position=np.array([0.3, 0.1, 0.5]),
            ee_orientation=np.array([0.707, 0.0, 0.707, 0.0]),
            gripper_state=0.75,
            timestamp=3.0,
        )
        d = original.to_dict()
        restored = RobotState.from_dict(d)
        np.testing.assert_allclose(restored.joint_positions, original.joint_positions)
        np.testing.assert_allclose(restored.joint_velocities, original.joint_velocities)
        np.testing.assert_allclose(restored.ee_position, original.ee_position)
        np.testing.assert_allclose(restored.ee_orientation, original.ee_orientation)
        assert restored.gripper_state == original.gripper_state
        assert restored.timestamp == original.timestamp


# ── TestRobotCommandDataclass ──────────────────────────────


class TestRobotCommandDataclass:
    def test_creation(self):
        cmd = RobotCommand(
            joint_targets=np.zeros(7),
            gripper_command=0.0,
            timestamp=1.0,
        )
        assert cmd.gripper_command == 0.0
        assert len(cmd.joint_targets) == 7

    def test_to_dict(self):
        cmd = RobotCommand(
            joint_targets=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            gripper_command=128.0,
            timestamp=5.0,
        )
        d = cmd.to_dict()
        assert d["gripper_command"] == 128.0
        assert d["timestamp"] == 5.0
        assert isinstance(d["joint_targets"], list)

    def test_from_dict_roundtrip(self):
        original = RobotCommand(
            joint_targets=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            gripper_command=255.0,
            timestamp=10.0,
        )
        d = original.to_dict()
        restored = RobotCommand.from_dict(d)
        np.testing.assert_allclose(restored.joint_targets, original.joint_targets)
        assert restored.gripper_command == original.gripper_command
        assert restored.timestamp == original.timestamp


# ── TestRobotStatusDataclass ───────────────────────────────


class TestRobotStatusDataclass:
    def test_creation(self):
        status = RobotStatus(
            connected=True,
            status="running",
            error_message=None,
            uptime=10.5,
            last_heartbeat=time.time(),
        )
        assert status.connected is True
        assert status.status == "running"
        assert status.error_message is None

    def test_to_dict(self):
        status = RobotStatus(
            connected=False,
            status="e_stopped",
            error_message="Emergency stop activated",
            uptime=0.0,
            last_heartbeat=0.0,
        )
        d = status.to_dict()
        assert d["connected"] is False
        assert d["status"] == "e_stopped"
        assert d["error_message"] == "Emergency stop activated"


# ── TestSimulatedRobot ─────────────────────────────────────


class TestSimulatedRobot:
    def test_connect_disconnect_lifecycle(self):
        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        assert robot.get_status().connected is True
        robot.disconnect()
        assert robot.get_status().connected is False

    def test_get_state_returns_valid_robot_state(self):
        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        state = robot.get_state()
        assert isinstance(state, RobotState)
        assert len(state.joint_positions) == 7
        assert len(state.joint_velocities) == 7
        assert len(state.ee_position) == 3
        assert len(state.ee_orientation) == 4
        robot.disconnect()

    def test_send_command_changes_state(self):
        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        state_before = robot.get_state()
        cmd = RobotCommand(
            joint_targets=np.array([0.5, 0.3, 0.0, -1.5, 0.0, 1.5, 0.0]),
            gripper_command=0.0,
            timestamp=time.time(),
        )
        robot.send_command(cmd)
        state_after = robot.get_state()
        # State should have changed after sending command
        assert not np.allclose(state_before.joint_positions, state_after.joint_positions) or \
               not np.allclose(state_before.joint_velocities, state_after.joint_velocities)
        robot.disconnect()

    def test_emergency_stop_sets_e_stopped(self):
        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        robot.emergency_stop()
        status = robot.get_status()
        assert status.status == "e_stopped"
        assert status.error_message is not None
        robot.disconnect()

    def test_get_status_connected_after_connect(self):
        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        status = robot.get_status()
        assert status.connected is True
        assert status.status == "running"
        robot.disconnect()

    def test_double_connect_is_safe(self):
        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        robot.connect()  # Should not raise
        assert robot.get_status().connected is True
        robot.disconnect()

    def test_disconnect_sets_connected_false(self):
        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        robot.disconnect()
        status = robot.get_status()
        assert status.connected is False
        assert status.status == "idle"

    def test_send_command_while_disconnected_raises(self):
        robot = SimulatedRobot("basic_workspace")
        cmd = RobotCommand(
            joint_targets=np.zeros(7),
            gripper_command=0.0,
            timestamp=time.time(),
        )
        with pytest.raises(AssertionError):
            robot.send_command(cmd)


# ── TestRobotInterfaceABC ──────────────────────────────────


class TestRobotInterfaceABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            RobotInterface()  # type: ignore[abstract]


# ── TestRunPolicyOnRobot ───────────────────────────────────


class TestRunPolicyOnRobot:
    def test_with_null_policy(self):
        from forge_core.policy import NullPolicy

        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        policy = NullPolicy()
        summary = run_policy_on_robot(robot, policy)
        assert "steps" in summary
        assert "completed" in summary
        assert "safety_violations" in summary
        assert "e_stopped" in summary
        assert summary["completed"] is True
        assert summary["steps"] > 0
        robot.disconnect()

    def test_with_safety_monitor(self):
        from forge_core.policy import NullPolicy
        from forge_core.safety import SafetyMonitor

        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        policy = NullPolicy()
        monitor = SafetyMonitor(robot)
        summary = run_policy_on_robot(robot, policy, safety_monitor=monitor)
        assert summary["completed"] is True
        assert isinstance(summary["safety_violations"], int)
        robot.disconnect()

    def test_with_recorder(self):
        from forge_core.policy import NullPolicy
        from forge_core.recorder import TrajectoryBuffer

        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        policy = NullPolicy()
        recorder = TrajectoryBuffer()
        summary = run_policy_on_robot(robot, policy, recorder=recorder)
        assert summary["completed"] is True
        assert recorder.length > 0
        robot.disconnect()

    def test_max_steps_limits_execution(self):
        from forge_core.policy import NullPolicy

        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        policy = NullPolicy()
        policy._max_steps = 10000  # Set high so it doesn't finish early
        summary = run_policy_on_robot(robot, policy, max_steps=5)
        assert summary["steps"] <= 5
        robot.disconnect()

    def test_summary_dict_keys(self):
        from forge_core.policy import NullPolicy

        robot = SimulatedRobot("basic_workspace")
        robot.connect()
        policy = NullPolicy()
        summary = run_policy_on_robot(robot, policy)
        expected_keys = {"steps", "completed", "safety_violations", "e_stopped"}
        assert set(summary.keys()) == expected_keys
        robot.disconnect()
