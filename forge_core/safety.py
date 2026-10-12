"""Safety monitoring system for Forge.

Provides real-time safety checking for robot commands against configured
limits (joint positions, velocities, workspace boundaries).  The
SafetyMonitor wraps a RobotInterface and can intercept unsafe commands
before they reach the hardware.

This module has NO ROS dependency — it can be used standalone for testing.

Phase 11 — FEAT-001.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from forge_core.config import PROJECT_ROOT, load_yaml
from forge_core.hardware import RobotCommand, RobotInterface, RobotState

logger = logging.getLogger(__name__)


# ── Dataclasses ─────────────────────────────────────────────


@dataclass
class SafetyConfig:
    """Safety limits configuration for robot operation."""

    joint_limits_lower: list[float] = field(
        default_factory=lambda: [-2.9, -1.8, -2.9, -3.1, -2.9, -0.02, -2.9]
    )
    joint_limits_upper: list[float] = field(
        default_factory=lambda: [2.9, 1.8, 2.9, -0.07, 2.9, 3.8, 2.9]
    )
    velocity_limits: list[float] = field(
        default_factory=lambda: [2.2, 2.2, 2.2, 2.2, 2.6, 2.6, 2.6]
    )
    workspace_min: list[float] = field(
        default_factory=lambda: [-1.0, -1.0, 0.0]
    )
    workspace_max: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.5]
    )
    max_gripper_force: float = 70.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SafetyConfig:
        return cls(
            joint_limits_lower=data.get(
                "joint_limits_lower",
                [-2.9, -1.8, -2.9, -3.1, -2.9, -0.02, -2.9],
            ),
            joint_limits_upper=data.get(
                "joint_limits_upper",
                [2.9, 1.8, 2.9, -0.07, 2.9, 3.8, 2.9],
            ),
            velocity_limits=data.get(
                "velocity_limits",
                [2.2, 2.2, 2.2, 2.2, 2.6, 2.6, 2.6],
            ),
            workspace_min=data.get("workspace_min", [-1.0, -1.0, 0.0]),
            workspace_max=data.get("workspace_max", [1.0, 1.0, 1.5]),
            max_gripper_force=data.get("max_gripper_force", 70.0),
        )


@dataclass
class SafetyViolation:
    """A single safety violation detected during command checking."""

    violation_type: str       # 'joint_limit', 'velocity_limit', 'workspace_boundary'
    message: str
    joint_index: int | None
    value: float | None
    limit: float | None
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "violation_type": self.violation_type,
            "message": self.message,
            "joint_index": self.joint_index,
            "value": self.value,
            "limit": self.limit,
            "timestamp": self.timestamp,
        }


# ── Config loader ───────────────────────────────────────────


def load_safety_config(path: str | Path | None = None) -> SafetyConfig:
    """Load safety configuration from a YAML file.

    When *path* is given, loads that file (raises FileNotFoundError if
    missing).  When *path* is None, tries the default
    ``config/safety.yaml``; returns defaults if the default file is missing.
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Safety config not found: {p}")
        data = load_yaml(p)
    else:
        default_path = PROJECT_ROOT / "config" / "safety.yaml"
        if not default_path.exists():
            logger.debug("no default safety config found, using defaults")
            return SafetyConfig()
        data = load_yaml(default_path)

    # Flatten nested YAML structure into the dataclass fields
    joint_limits = data.get("joint_limits", {})
    workspace = data.get("workspace", {})
    return SafetyConfig.from_dict({
        "joint_limits_lower": joint_limits.get("lower", SafetyConfig.joint_limits_lower),
        "joint_limits_upper": joint_limits.get("upper", SafetyConfig.joint_limits_upper),
        "velocity_limits": data.get("velocity_limits", SafetyConfig.velocity_limits),
        "workspace_min": workspace.get("min", SafetyConfig.workspace_min),
        "workspace_max": workspace.get("max", SafetyConfig.workspace_max),
        "max_gripper_force": data.get("max_gripper_force", SafetyConfig.max_gripper_force),
    })


# ── Safety monitor ──────────────────────────────────────────


class SafetyMonitor:
    """Monitor robot commands for safety violations.

    Checks joint limits, velocity limits, and workspace boundaries before
    commands are sent to the robot.
    """

    def __init__(
        self,
        robot: RobotInterface,
        config: SafetyConfig | None = None,
    ) -> None:
        self._robot = robot
        self._config = config if config is not None else load_safety_config()
        self._violations_log: list[dict[str, Any]] = []

    def check_command(
        self,
        cmd: RobotCommand,
        current_state: RobotState,
    ) -> list[SafetyViolation]:
        """Check a command against safety limits.

        Returns a list of violations (empty means safe).
        """
        violations: list[SafetyViolation] = []
        now = time.time()

        # Check joint position limits
        n_joints = min(
            len(cmd.joint_targets),
            len(self._config.joint_limits_lower),
            len(self._config.joint_limits_upper),
        )
        for i in range(n_joints):
            target = cmd.joint_targets[i]
            lower = self._config.joint_limits_lower[i]
            upper = self._config.joint_limits_upper[i]
            if target < lower:
                v = SafetyViolation(
                    violation_type="joint_limit",
                    message=f"Joint {i} target {target:.4f} below lower limit {lower:.4f}",
                    joint_index=i,
                    value=float(target),
                    limit=float(lower),
                    timestamp=now,
                )
                violations.append(v)
            elif target > upper:
                v = SafetyViolation(
                    violation_type="joint_limit",
                    message=f"Joint {i} target {target:.4f} above upper limit {upper:.4f}",
                    joint_index=i,
                    value=float(target),
                    limit=float(upper),
                    timestamp=now,
                )
                violations.append(v)

        # Estimate velocity from position delta and check limits
        n_vel = min(
            len(cmd.joint_targets),
            len(current_state.joint_positions),
            len(self._config.velocity_limits),
        )
        for i in range(n_vel):
            delta = abs(cmd.joint_targets[i] - current_state.joint_positions[i])
            vel_limit = self._config.velocity_limits[i]
            if delta > vel_limit:
                v = SafetyViolation(
                    violation_type="velocity_limit",
                    message=(
                        f"Joint {i} estimated velocity {delta:.4f} "
                        f"exceeds limit {vel_limit:.4f}"
                    ),
                    joint_index=i,
                    value=float(delta),
                    limit=float(vel_limit),
                    timestamp=now,
                )
                violations.append(v)

        # Check workspace boundaries of current EE position
        ee = current_state.ee_position
        ws_min = self._config.workspace_min
        ws_max = self._config.workspace_max
        for axis, label in enumerate(["x", "y", "z"]):
            if axis < len(ee):
                if ee[axis] < ws_min[axis]:
                    v = SafetyViolation(
                        violation_type="workspace_boundary",
                        message=(
                            f"EE {label} position {ee[axis]:.4f} "
                            f"below workspace min {ws_min[axis]:.4f}"
                        ),
                        joint_index=None,
                        value=float(ee[axis]),
                        limit=float(ws_min[axis]),
                        timestamp=now,
                    )
                    violations.append(v)
                elif ee[axis] > ws_max[axis]:
                    v = SafetyViolation(
                        violation_type="workspace_boundary",
                        message=(
                            f"EE {label} position {ee[axis]:.4f} "
                            f"above workspace max {ws_max[axis]:.4f}"
                        ),
                        joint_index=None,
                        value=float(ee[axis]),
                        limit=float(ws_max[axis]),
                        timestamp=now,
                    )
                    violations.append(v)

        return violations

    def is_safe(
        self,
        cmd: RobotCommand,
        current_state: RobotState,
    ) -> bool:
        """Return True if the command passes all safety checks."""
        return len(self.check_command(cmd, current_state)) == 0

    def safe_send_command(
        self,
        cmd: RobotCommand,
    ) -> tuple[bool, list[SafetyViolation]]:
        """Check command safety and send only if safe.

        Returns (sent, violations) — sent is True if the command was
        forwarded to the robot.
        """
        current_state = self._robot.get_state()
        violations = self.check_command(cmd, current_state)

        if violations:
            for v in violations:
                logger.warning("safety_violation: %s", v.message)
                self._violations_log.append(v.to_dict())
            return False, violations

        self._robot.send_command(cmd)
        return True, []

    def emergency_stop(self) -> None:
        """Delegate emergency stop to the robot."""
        self._robot.emergency_stop()

    def get_violations_log(self) -> list[dict[str, Any]]:
        """Return all recorded violations as dicts."""
        return list(self._violations_log)
