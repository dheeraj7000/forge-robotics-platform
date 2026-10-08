"""Policy interface for Forge.

A policy reads robot/object state and produces joint commands.
Phase 2 policies are deterministic/scripted — no ML.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from forge_core.simulation import ForgeSimulation

logger = logging.getLogger(__name__)


class Policy(ABC):
    """Base interface for robot control policies."""

    @abstractmethod
    def reset(self, sim: ForgeSimulation) -> None:
        """Reset policy state for a new episode."""

    @abstractmethod
    def act(self, sim: ForgeSimulation) -> np.ndarray:
        """Compute joint targets from current state.

        Returns:
            Joint position targets (7 arm + finger control).
        """

    @property
    @abstractmethod
    def done(self) -> bool:
        """Whether the policy has finished its intended sequence."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


class ScriptedPickAndPlace(Policy):
    """Deterministic scripted pick-and-place policy.

    Executes a waypoint sequence using the Menagerie Panda model:
    1. Move above target object
    2. Lower to grasp height
    3. Close gripper
    4. Lift object
    5. Move above target zone
    6. Lower to place height
    7. Open gripper
    8. Retract home

    Joint targets are 8-element: 7 arm joints + 1 finger actuator (0-255).
    The Menagerie Panda's general actuators accept desired joint positions
    as ctrl values. Finger actuator: 0=open (0.04m gap), 255=closed.
    """

    def __init__(self, target_object: str = "red_cube", **kwargs):
        self._target_object = target_object
        self._waypoints: list[tuple[np.ndarray, int]] = []
        self._current_wp: int = 0
        self._steps_at_wp: int = 0
        self._is_done: bool = False

    def reset(self, sim: ForgeSimulation) -> None:
        """Build waypoint sequence based on scenario."""
        self._current_wp = 0
        self._steps_at_wp = 0
        self._is_done = False

        obj = sim.get_object_state(self._target_object)
        if obj is None:
            logger.error("Target object '%s' not found", self._target_object)
            self._is_done = True
            return

        obj_y = obj.position[1]
        # j1 rotates toward the object's y side
        j1_obj = 0.3 if obj_y >= 0 else -0.3
        # j1 for target zone (y~0)
        j1_tgt = 0.0

        OPEN = 0      # finger actuator open
        CLOSED = 255   # finger actuator closed

        # Waypoint: (8-element ctrl targets, hold_steps)
        # hold_steps × 10 sim_steps × 0.002s = sim time per waypoint
        self._waypoints = [
            # 1. Above object — safe height
            (np.array([j1_obj, 0.3, 0.0, -2.5, 0.0, 2.0, 0.785, OPEN]), 200),
            # 2. Lower to grasp height — EE near z=0.26
            (np.array([j1_obj, 1.2, 0.0, -2.0, 0.0, 1.2, 0.785, OPEN]), 200),
            # 3. Close gripper
            (np.array([j1_obj, 1.2, 0.0, -2.0, 0.0, 1.2, 0.785, CLOSED]), 150),
            # 4. Lift
            (np.array([j1_obj, 0.3, 0.0, -2.5, 0.0, 2.0, 0.785, CLOSED]), 200),
            # 5. Move above target zone
            (np.array([j1_tgt, 0.3, 0.0, -2.5, 0.0, 2.0, 0.785, CLOSED]), 200),
            # 6. Lower to place
            (np.array([j1_tgt, 1.2, 0.0, -2.0, 0.0, 1.2, 0.785, CLOSED]), 200),
            # 7. Release
            (np.array([j1_tgt, 1.2, 0.0, -2.0, 0.0, 1.2, 0.785, OPEN]), 150),
            # 8. Retract home
            (np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, OPEN]), 200),
        ]

        logger.info(
            "policy_reset: ScriptedPickAndPlace for '%s' with %d waypoints",
            self._target_object, len(self._waypoints),
        )

    def act(self, sim: ForgeSimulation) -> np.ndarray:
        """Return current waypoint targets."""
        if self._is_done or self._current_wp >= len(self._waypoints):
            self._is_done = True
            return np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0])

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
    """Policy that does nothing — useful for testing evaluation."""

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
        return np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0])

    @property
    def done(self) -> bool:
        return self._is_done


POLICY_REGISTRY: dict[str, type[Policy]] = {
    "scripted_pick_and_place": ScriptedPickAndPlace,
    "null": NullPolicy,
}


def create_policy(name: str, **kwargs) -> Policy:
    """Create a policy by registered name."""
    # Lazy import learned policy to avoid PyTorch dependency when not needed
    if name == "learned":
        from forge_core.learned_policy import LearnedPolicy
        return LearnedPolicy(**kwargs)

    cls = POLICY_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(list(POLICY_REGISTRY.keys()) + ["learned"])
        raise ValueError(f"Unknown policy '{name}'. Available: {available}")
    return cls(**kwargs)
