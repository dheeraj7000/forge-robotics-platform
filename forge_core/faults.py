"""Failure injection (chaos testing) for Forge.

Faults are composable perturbations applied during task execution.
Each fault activates at a configured time and modifies simulation
state, sensor readings, or command delivery.

Supported fault types:
    object_moved        — teleport an object to a new position mid-run
    sensor_noise        — add Gaussian noise to reported joint positions
    sensor_delay        — return stale robot state for N cycles
    actuator_stuck      — freeze one or more actuators at current value
    controller_dropout  — drop commands for N cycles (zero torque)
    gravity_shift       — change gravity during the run

Faults are defined in YAML and loaded alongside tasks.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml

from forge_core.config import PROJECT_ROOT
from forge_core.simulation import ForgeSimulation, RobotState, ObjectState

logger = logging.getLogger(__name__)


@dataclass
class FaultConfig:
    """Configuration for a single fault."""

    type: str
    trigger_time: float  # sim time to activate (seconds)
    duration: float = 0.0  # 0 = instantaneous
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""


class Fault(ABC):
    """Base class for injectable faults."""

    def __init__(self, config: FaultConfig):
        self.config = config
        self._activated = False
        self._deactivated = False

    @property
    def name(self) -> str:
        return self.config.type

    def is_active(self, sim_time: float) -> bool:
        """Check if fault is currently active."""
        t = self.config.trigger_time
        d = self.config.duration
        if d <= 0:
            return not self._activated and sim_time >= t
        return t <= sim_time < t + d

    def should_trigger(self, sim_time: float) -> bool:
        """Check if fault should fire (one-shot or continuous)."""
        if self.config.duration <= 0:
            # One-shot
            return not self._activated and sim_time >= self.config.trigger_time
        # Duration-based
        return self.is_active(sim_time)

    def apply(self, sim: ForgeSimulation, sim_time: float) -> None:
        """Apply the fault if conditions are met."""
        if self.should_trigger(sim_time):
            self._do_apply(sim, sim_time)
            if self.config.duration <= 0:
                self._activated = True
                logger.info("fault_triggered: %s at t=%.3f", self.name, sim_time)

    @abstractmethod
    def _do_apply(self, sim: ForgeSimulation, sim_time: float) -> None:
        """Implement the fault effect."""

    def filter_robot_state(self, state: RobotState, sim_time: float) -> RobotState:
        """Optionally modify robot state before it reaches the policy."""
        return state

    def filter_targets(self, targets: np.ndarray, sim_time: float) -> np.ndarray:
        """Optionally modify joint targets before they reach the simulation."""
        return targets


class ObjectMovedFault(Fault):
    """Teleport an object to a new position mid-run.

    Params:
        object_id: which object to move
        new_position: [x, y, z] target position
    """

    def _do_apply(self, sim: ForgeSimulation, sim_time: float) -> None:
        object_id = self.config.params["object_id"]
        new_pos = self.config.params["new_position"]

        import mujoco
        body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, object_id)
        if body_id < 0:
            logger.error("fault: object '%s' not found", object_id)
            return

        # Find the freejoint for this object
        joint_name = f"{object_id}_joint"
        joint_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            logger.error("fault: joint '%s' not found", joint_name)
            return

        # Freejoint qpos: [x, y, z, qw, qx, qy, qz]
        qpos_addr = sim.model.jnt_qposadr[joint_id]
        sim.data.qpos[qpos_addr:qpos_addr + 3] = new_pos
        mujoco.mj_forward(sim.model, sim.data)

        logger.info("fault_object_moved: %s -> %s", object_id, new_pos)


class SensorNoiseFault(Fault):
    """Add Gaussian noise to reported joint positions.

    Params:
        std: noise standard deviation (radians, default 0.05)
    """

    def _do_apply(self, sim: ForgeSimulation, sim_time: float) -> None:
        pass  # Continuous — applied in filter

    def filter_robot_state(self, state: RobotState, sim_time: float) -> RobotState:
        if not self.is_active(sim_time):
            return state
        std = self.config.params.get("std", 0.05)
        noise = np.random.normal(0, std, size=state.joint_positions.shape)
        state.joint_positions = state.joint_positions + noise
        return state


class SensorDelayFault(Fault):
    """Return stale robot state for a number of cycles.

    Params:
        delay_steps: how many cycles to hold stale data (default 50)
    """

    def __init__(self, config: FaultConfig):
        super().__init__(config)
        self._stale_state: RobotState | None = None
        self._steps_remaining = 0

    def _do_apply(self, sim: ForgeSimulation, sim_time: float) -> None:
        delay = self.config.params.get("delay_steps", 50)
        self._steps_remaining = delay
        self._stale_state = sim.get_robot_state()
        logger.info("fault_sensor_delay: freezing state for %d steps", delay)

    def filter_robot_state(self, state: RobotState, sim_time: float) -> RobotState:
        if self._steps_remaining > 0 and self._stale_state is not None:
            self._steps_remaining -= 1
            return self._stale_state
        return state


class ActuatorStuckFault(Fault):
    """Freeze one or more actuators at their current value.

    Params:
        joints: list of joint indices to freeze (0-indexed), default [0]
    """

    def __init__(self, config: FaultConfig):
        super().__init__(config)
        self._frozen_values: dict[int, float] = {}

    def _do_apply(self, sim: ForgeSimulation, sim_time: float) -> None:
        joints = self.config.params.get("joints", [0])
        for j in joints:
            if j < len(sim.data.ctrl):
                self._frozen_values[j] = float(sim.data.ctrl[j])
        logger.info("fault_actuator_stuck: joints %s frozen", list(self._frozen_values.keys()))

    def filter_targets(self, targets: np.ndarray, sim_time: float) -> np.ndarray:
        if not self.is_active(sim_time) and self._frozen_values:
            return targets
        result = targets.copy()
        for j, val in self._frozen_values.items():
            if j < len(result):
                result[j] = val
        return result


class ControllerDropoutFault(Fault):
    """Drop all commands for a duration — robot goes limp.

    Params: (uses duration from config)
    """

    def _do_apply(self, sim: ForgeSimulation, sim_time: float) -> None:
        logger.info("fault_controller_dropout: commands dropped for %.2fs",
                     self.config.duration)

    def filter_targets(self, targets: np.ndarray, sim_time: float) -> np.ndarray:
        if self.is_active(sim_time):
            return np.zeros_like(targets)  # zero torque
        return targets


class GravityShiftFault(Fault):
    """Change gravity during the run.

    Params:
        gravity: [gx, gy, gz] new gravity vector (default [0, 0, -5])
    """

    def __init__(self, config: FaultConfig):
        super().__init__(config)
        self._original_gravity: np.ndarray | None = None

    def _do_apply(self, sim: ForgeSimulation, sim_time: float) -> None:
        new_grav = self.config.params.get("gravity", [0, 0, -5.0])
        self._original_gravity = sim.model.opt.gravity.copy()
        sim.model.opt.gravity[:] = new_grav
        logger.info("fault_gravity_shift: %s -> %s",
                     self._original_gravity.tolist(), new_grav)


# ── Fault registry ──────────────────────────────────────────

FAULT_REGISTRY: dict[str, type[Fault]] = {
    "object_moved": ObjectMovedFault,
    "sensor_noise": SensorNoiseFault,
    "sensor_delay": SensorDelayFault,
    "actuator_stuck": ActuatorStuckFault,
    "controller_dropout": ControllerDropoutFault,
    "gravity_shift": GravityShiftFault,
}


def create_fault(config: FaultConfig) -> Fault:
    """Create a fault from a config."""
    cls = FAULT_REGISTRY.get(config.type)
    if cls is None:
        available = ", ".join(FAULT_REGISTRY.keys())
        raise ValueError(f"Unknown fault type '{config.type}'. Available: {available}")
    return cls(config)


def load_fault_profile(name: str) -> list[FaultConfig]:
    """Load a fault profile (set of faults) from YAML.

    Profiles live in faults/<name>.yaml.
    """
    path = PROJECT_ROOT / "faults" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Fault profile not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    configs = []
    for item in data.get("faults", []):
        configs.append(FaultConfig(
            type=item["type"],
            trigger_time=item.get("trigger_time", 5.0),
            duration=item.get("duration", 0.0),
            params=item.get("params", {}),
            description=item.get("description", ""),
        ))
    return configs


def list_fault_profiles() -> list[str]:
    """List available fault profile names."""
    faults_dir = PROJECT_ROOT / "faults"
    if not faults_dir.exists():
        return []
    return sorted(p.stem for p in faults_dir.glob("*.yaml"))


class FaultInjector:
    """Manages a set of faults during a task run.

    Integrates with the runner loop:
    - pre_step: apply faults, filter targets
    - post_step: filter robot state for policy
    """

    def __init__(self, faults: list[Fault] | None = None):
        self.faults = faults or []
        self._events: list[dict[str, Any]] = []

    @classmethod
    def from_profile(cls, profile_name: str) -> FaultInjector:
        """Create from a named fault profile."""
        configs = load_fault_profile(profile_name)
        faults = [create_fault(c) for c in configs]
        return cls(faults)

    @classmethod
    def from_configs(cls, configs: list[FaultConfig]) -> FaultInjector:
        """Create from a list of fault configs."""
        faults = [create_fault(c) for c in configs]
        return cls(faults)

    def pre_step(self, sim: ForgeSimulation, targets: np.ndarray) -> np.ndarray:
        """Called before sim.step — apply faults and filter targets."""
        sim_time = sim.sim_time
        for fault in self.faults:
            fault.apply(sim, sim_time)
            targets = fault.filter_targets(targets, sim_time)
        return targets

    def post_step(self, sim: ForgeSimulation) -> None:
        """Called after sim.step — for bookkeeping."""
        pass

    def filter_state(self, state: RobotState, sim_time: float) -> RobotState:
        """Filter robot state before it reaches the policy."""
        for fault in self.faults:
            state = fault.filter_robot_state(state, sim_time)
        return state

    @property
    def fault_names(self) -> list[str]:
        return [f.name for f in self.faults]
