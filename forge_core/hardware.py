"""Hardware abstraction layer for Forge.

Provides a uniform interface (RobotInterface ABC) for interacting with
robot hardware — real or simulated.  SimulatedRobot wraps ForgeSimulation
to expose the same API that a physical robot driver would implement.

This module has NO ROS dependency — it can be used standalone for testing.

Phase 11 — FEAT-001.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from forge_core.simulation import ForgeSimulation, RobotState as SimRobotState

logger = logging.getLogger(__name__)


# ── Dataclasses ─────────────────────────────────────────────


@dataclass
class RobotState:
    """Hardware-abstracted robot state with gripper information."""

    joint_positions: np.ndarray   # (n_joints,)
    joint_velocities: np.ndarray  # (n_joints,)
    ee_position: np.ndarray       # (3,) xyz
    ee_orientation: np.ndarray    # (4,) quaternion (w, x, y, z)
    gripper_state: float          # 0.0 = open, 1.0 = closed
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_positions": self.joint_positions.tolist(),
            "joint_velocities": self.joint_velocities.tolist(),
            "ee_position": self.ee_position.tolist(),
            "ee_orientation": self.ee_orientation.tolist(),
            "gripper_state": self.gripper_state,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotState:
        return cls(
            joint_positions=np.array(data["joint_positions"]),
            joint_velocities=np.array(data["joint_velocities"]),
            ee_position=np.array(data["ee_position"]),
            ee_orientation=np.array(data["ee_orientation"]),
            gripper_state=data["gripper_state"],
            timestamp=data["timestamp"],
        )


@dataclass
class RobotCommand:
    """Command to send to a robot."""

    joint_targets: np.ndarray  # (n_joints,) position targets
    gripper_command: float     # 0 = open, 255 = closed
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_targets": self.joint_targets.tolist(),
            "gripper_command": self.gripper_command,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotCommand:
        return cls(
            joint_targets=np.array(data["joint_targets"]),
            gripper_command=data["gripper_command"],
            timestamp=data["timestamp"],
        )


@dataclass
class RobotStatus:
    """Operational status of a robot."""

    connected: bool
    status: str              # 'idle', 'running', 'error', 'e_stopped'
    error_message: str | None
    uptime: float
    last_heartbeat: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "status": self.status,
            "error_message": self.error_message,
            "uptime": self.uptime,
            "last_heartbeat": self.last_heartbeat,
        }


# ── Robot interface ─────────────────────────────────────────


class RobotInterface(ABC):
    """Abstract base class for robot hardware drivers."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the robot."""

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the robot."""

    @abstractmethod
    def get_state(self) -> RobotState:
        """Return the current robot state."""

    @abstractmethod
    def send_command(self, cmd: RobotCommand) -> None:
        """Send a command to the robot."""

    @abstractmethod
    def emergency_stop(self) -> None:
        """Trigger an emergency stop."""

    @abstractmethod
    def get_status(self) -> RobotStatus:
        """Return the current operational status."""


# ── Simulated robot ─────────────────────────────────────────


class SimulatedRobot(RobotInterface):
    """RobotInterface backed by ForgeSimulation.

    Wraps a MuJoCo simulation to provide the same API that a physical
    robot driver would expose.
    """

    def __init__(
        self,
        scenario_name: str = "basic_workspace",
        steps_per_command: int = 10,
    ) -> None:
        self._scenario_name = scenario_name
        self._steps_per_command = steps_per_command
        self._sim: ForgeSimulation | None = None
        self._connected = False
        self._status = "idle"
        self._error_message: str | None = None
        self._connect_time: float = 0.0
        self._last_heartbeat: float = 0.0

    def connect(self) -> None:
        """Create the simulation and reset it."""
        self._sim = ForgeSimulation(self._scenario_name)
        self._sim.reset()
        self._connected = True
        self._status = "running"
        self._error_message = None
        self._connect_time = time.time()
        self._last_heartbeat = time.time()
        logger.info("simulated_robot_connected: scenario=%s", self._scenario_name)

    def disconnect(self) -> None:
        """Shut down the simulation."""
        if self._sim is not None:
            self._sim.shutdown()
            self._sim = None
        self._connected = False
        self._status = "idle"
        logger.info("simulated_robot_disconnected")

    def get_state(self) -> RobotState:
        """Map simulation RobotState to hardware RobotState."""
        assert self._sim is not None, "Robot not connected"
        sim_state: SimRobotState = self._sim.get_robot_state()
        self._last_heartbeat = time.time()

        # Derive gripper_state from finger joint positions.
        # Panda fingers: qpos indices 7 and 8 (0.04 = open, 0.0 = closed).
        finger_positions = sim_state.joint_positions[7:9]
        max_finger_gap = 0.04
        avg_gap = float(np.mean(finger_positions))
        gripper_state = 1.0 - min(avg_gap / max_finger_gap, 1.0)

        return RobotState(
            joint_positions=sim_state.joint_positions[:7].copy(),
            joint_velocities=sim_state.joint_velocities[:7].copy(),
            ee_position=sim_state.ee_position.copy(),
            ee_orientation=sim_state.ee_orientation.copy(),
            gripper_state=gripper_state,
            timestamp=sim_state.timestamp,
        )

    def send_command(self, cmd: RobotCommand) -> None:
        """Set joint targets and step the simulation."""
        assert self._sim is not None, "Robot not connected"
        if self._status == "e_stopped":
            logger.warning("command_rejected: robot is e-stopped")
            return

        targets = np.zeros(8)
        targets[:len(cmd.joint_targets)] = cmd.joint_targets
        targets[7] = cmd.gripper_command
        self._sim.set_joint_targets(targets)
        self._sim.step(self._steps_per_command)
        self._last_heartbeat = time.time()

    def emergency_stop(self) -> None:
        """Send zero-velocity targets and set status to e_stopped."""
        if self._sim is not None:
            zero_targets = np.zeros(8)
            self._sim.set_joint_targets(zero_targets)
        self._status = "e_stopped"
        self._error_message = "Emergency stop activated"
        logger.warning("emergency_stop_activated")

    def get_status(self) -> RobotStatus:
        """Return current operational status."""
        uptime = time.time() - self._connect_time if self._connected else 0.0
        return RobotStatus(
            connected=self._connected,
            status=self._status,
            error_message=self._error_message,
            uptime=uptime,
            last_heartbeat=self._last_heartbeat,
        )
