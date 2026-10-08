"""Policy interface for Forge.

A policy reads robot/object state and produces joint commands.
Phase 2 policies are deterministic/scripted — no ML.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from forge_core.simulation import ForgeSimulation, RobotState, ObjectState

logger = logging.getLogger(__name__)


class Policy(ABC):
    """Base interface for robot control policies.

    A policy is called each control cycle to produce joint targets.
    It receives the simulation handle for state queries.
    """

    @abstractmethod
    def reset(self, sim: ForgeSimulation) -> None:
        """Reset policy state for a new episode."""

    @abstractmethod
    def act(self, sim: ForgeSimulation) -> np.ndarray:
        """Compute joint targets from current state.

        Returns:
            Joint position targets (7 for arm, or 9 for arm+fingers).
        """

    @property
    @abstractmethod
    def done(self) -> bool:
        """Whether the policy has finished its intended sequence."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


class ScriptedPickAndPlace(Policy):
    """Deterministic scripted policy that moves a cube to the target zone.

    Strategy: push the target object toward the target bin using the
    robot's end-effector. The robot lowers to table height behind the
    object and sweeps forward through it.

    This is a joint-space waypoint policy with configurations tuned for
    the basic_workspace scenario. In a real system, IK would compute
    these. For Phase 2, they are hand-tuned.

    The waypoints are calibrated against the simplified Panda MJCF model
    and the basic_workspace object layout.
    """

    def __init__(self, target_object: str = "red_cube"):
        self._target_object = target_object
        self._waypoints: list[tuple[np.ndarray, int]] = []
        self._current_wp: int = 0
        self._steps_at_wp: int = 0
        self._is_done: bool = False

    def reset(self, sim: ForgeSimulation) -> None:
        """Build waypoint sequence based on current scenario."""
        self._current_wp = 0
        self._steps_at_wp = 0
        self._is_done = False

        obj = sim.get_object_state(self._target_object)
        if obj is None:
            logger.error("Target object '%s' not found", self._target_object)
            self._is_done = True
            return

        open_grip = [0.04, 0.04]

        # Determine side: if object y > 0, approach from positive y side
        obj_y = obj.position[1]
        j1_start = 0.30 if obj_y >= 0 else -0.30

        # Waypoint sequence: (9-element joint+finger targets, hold_steps)
        # Strategy: lower arm near cube, sweep j1 through 0 to push cube toward target
        # Each hold step = one policy call = 10 sim steps = 0.02s sim time
        self._waypoints = [
            # 1. Transit: safe height, rotated toward object
            (np.array([j1_start, 0.0, 0.0, -1.8, 0.0, 1.8, 0.785] + open_grip), 150),
            # 2. Lower to cube height while extending
            (np.array([j1_start, 1.0, 0.0, -0.4, 0.0, 1.0, 0.785] + open_grip), 300),
            # 3. Sweep j1 through center to push cube toward y=0
            (np.array([0.0, 1.1, 0.0, -0.3, 0.0, 0.95, 0.785] + open_grip), 300),
            # 4. Continue sweep past center + extend more
            (np.array([-j1_start * 0.3, 1.15, 0.0, -0.25, 0.0, 0.9, 0.785] + open_grip), 250),
            # 5. Retract to home
            (np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785] + open_grip), 200),
        ]

        logger.info("policy_reset: ScriptedPickAndPlace for '%s' with %d waypoints",
                     self._target_object, len(self._waypoints))

    def act(self, sim: ForgeSimulation) -> np.ndarray:
        """Return the current waypoint's joint targets."""
        if self._is_done or self._current_wp >= len(self._waypoints):
            self._is_done = True
            home = np.array(sim._scenario["robot"]["home_qpos"])
            return home

        targets, hold = self._waypoints[self._current_wp]
        self._steps_at_wp += 1

        if self._steps_at_wp >= hold:
            logger.debug("policy_waypoint_reached: %d/%d",
                         self._current_wp + 1, len(self._waypoints))
            self._current_wp += 1
            self._steps_at_wp = 0
            if self._current_wp >= len(self._waypoints):
                self._is_done = True

        return targets

    @property
    def done(self) -> bool:
        return self._is_done


class NullPolicy(Policy):
    """Policy that does nothing — useful for testing evaluation without movement."""

    def __init__(self, **kwargs):
        self._is_done = False
        self._steps = 0
        self._max_steps = 100

    def reset(self, sim: ForgeSimulation) -> None:
        self._is_done = False
        self._steps = 0

    def act(self, sim: ForgeSimulation) -> np.ndarray:
        self._steps += 1
        if self._steps >= self._max_steps:
            self._is_done = True
        home = np.array(sim._scenario["robot"]["home_qpos"])
        return home

    @property
    def done(self) -> bool:
        return self._is_done


# Registry of available policies
POLICY_REGISTRY: dict[str, type[Policy]] = {
    "scripted_pick_and_place": ScriptedPickAndPlace,
    "null": NullPolicy,
}


def create_policy(name: str, **kwargs) -> Policy:
    """Create a policy by registered name."""
    cls = POLICY_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(POLICY_REGISTRY.keys())
        raise ValueError(f"Unknown policy '{name}'. Available: {available}")
    return cls(**kwargs)
