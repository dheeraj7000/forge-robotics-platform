"""MuJoCo simulation engine for Forge.

Owns the simulation lifecycle: load, step, reset, query state.
Uses the MuJoCo Menagerie Franka Emika Panda model with mesh-based
collision geometry for realistic manipulation.

This module has NO ROS dependency — it can be used standalone for testing.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from forge_core.config import PROJECT_ROOT, load_scenario, resolve_model_path
from forge_core.menagerie import get_menagerie_panda_dir

logger = logging.getLogger(__name__)

# Joint names for the Menagerie Panda (7 DOF arm + 2 finger)
PANDA_ARM_JOINTS = [
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7",
]
PANDA_FINGER_JOINTS = ["finger_joint1", "finger_joint2"]
PANDA_ALL_JOINTS = PANDA_ARM_JOINTS + PANDA_FINGER_JOINTS

# Number of actuators: 7 arm + 1 finger (tendon-coupled)
N_ARM_ACTUATORS = 7
N_ACTUATORS = 8  # 7 arm + 1 finger actuator


@dataclass
class ObjectState:
    """Ground-truth state of a scene object."""

    id: str
    position: np.ndarray  # (3,) xyz
    orientation: np.ndarray  # (4,) quaternion (w, x, y, z)
    timestamp: float = 0.0


@dataclass
class RobotState:
    """State of the robot."""

    joint_positions: np.ndarray  # (n_joints,)
    joint_velocities: np.ndarray  # (n_joints,)
    joint_names: list[str] = field(default_factory=lambda: list(PANDA_ALL_JOINTS))
    ee_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ee_orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    timestamp: float = 0.0


class ForgeSimulation:
    """MuJoCo simulation wrapper for the Forge platform.

    Uses the MuJoCo Menagerie Franka Emika Panda model with proper mesh
    collision geometry. Builds a combined scene with table, objects, and
    robot from scenario YAML configuration.
    """

    def __init__(self, scenario_name: str = "basic_workspace"):
        self._scenario_name = scenario_name
        self._scenario: dict[str, Any] = {}
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._initial_qpos: np.ndarray | None = None
        self._initial_qvel: np.ndarray | None = None
        self._object_names: list[str] = []
        self._object_body_ids: dict[str, int] = {}
        self._hand_body_id: int = -1
        self._menagerie_dir: Path | None = None

        self._load_scenario(scenario_name)

    def _load_scenario(self, scenario_name: str) -> None:
        """Build a combined MJCF model from scenario configuration."""
        self._scenario = load_scenario(scenario_name)
        self._menagerie_dir = get_menagerie_panda_dir()

        robot_cfg = self._scenario["robot"]
        sim_cfg = self._scenario["simulation"]
        objects_cfg = self._scenario.get("objects", {})

        scene_xml = self._build_scene_xml(robot_cfg, objects_cfg, sim_cfg)

        # Write to temp file in Menagerie dir so <include> resolves meshes
        tmp_path = self._menagerie_dir / "_forge_scene_tmp.xml"
        try:
            tmp_path.write_text(scene_xml)
            self._model = mujoco.MjModel.from_xml_path(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)

        self._data = mujoco.MjData(self._model)

        # Set home configuration
        home_qpos = robot_cfg.get("home_qpos")
        if home_qpos is not None:
            n_robot_joints = len(home_qpos)
            self._data.qpos[:n_robot_joints] = home_qpos

        # Set home ctrl
        home_ctrl = robot_cfg.get("home_ctrl")
        if home_ctrl is not None:
            self._data.ctrl[:len(home_ctrl)] = home_ctrl
        else:
            # Default: arm ctrl = home joint positions, finger open
            if home_qpos is not None:
                self._data.ctrl[:N_ARM_ACTUATORS] = home_qpos[:N_ARM_ACTUATORS]
                self._data.ctrl[N_ARM_ACTUATORS] = 0  # finger open

        mujoco.mj_forward(self._model, self._data)

        # Cache initial state for reset
        self._initial_qpos = self._data.qpos.copy()
        self._initial_qvel = self._data.qvel.copy()
        self._initial_ctrl = self._data.ctrl.copy()

        # Cache object body IDs
        for obj_name in objects_cfg:
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            if body_id >= 0:
                self._object_body_ids[obj_name] = body_id
                self._object_names.append(obj_name)

        # Cache hand body ID for EE tracking
        self._hand_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "hand"
        )

        logger.info(
            "simulation_started: scenario=%s, objects=%s",
            scenario_name, self._object_names,
        )

    def _build_scene_xml(
        self,
        robot_cfg: dict,
        objects_cfg: dict,
        sim_cfg: dict,
    ) -> str:
        """Build combined MJCF XML that includes the Menagerie Panda."""
        base_pos = robot_cfg.get("base_position", [0, 0, 0])
        timestep = sim_cfg.get("timestep", 0.002)

        # Build object XML
        object_bodies = []
        for obj_name, obj_cfg in objects_cfg.items():
            pos = obj_cfg["position"]
            size = obj_cfg.get("size", [0.025, 0.025, 0.025])
            obj_type = obj_cfg.get("type", "box")
            mass = obj_cfg.get("mass", 0.05)
            rgba = self._material_to_rgba(obj_cfg.get("material", ""))
            is_static = obj_cfg.get("static", False)

            if is_static:
                body_xml = (
                    f'    <body name="{obj_name}" pos="{pos[0]} {pos[1]} {pos[2]}">\n'
                    f'      <geom name="{obj_name}_geom" type="{obj_type}" '
                    f'size="{size[0]} {size[1]} {size[2]}" rgba="{rgba}" '
                    f'mass="{mass}"/>\n'
                    f"    </body>\n"
                )
            else:
                body_xml = (
                    f'    <body name="{obj_name}" pos="{pos[0]} {pos[1]} {pos[2]}">\n'
                    f'      <freejoint name="{obj_name}_joint"/>\n'
                    f'      <geom name="{obj_name}_geom" type="{obj_type}" '
                    f'size="{size[0]} {size[1]} {size[2]}" rgba="{rgba}" '
                    f'mass="{mass}" condim="4" friction="1 0.5 0.01"/>\n'
                    f"    </body>\n"
                )
            object_bodies.append(body_xml)

        objects_xml = "\n".join(object_bodies)

        # Base position offset — wrap robot in a body if non-zero
        bx, by, bz = base_pos
        robot_include = '<include file="panda.xml"/>'
        if bx != 0 or by != 0 or bz != 0:
            # Menagerie panda.xml puts link0 directly in worldbody.
            # We can't easily offset it with <include>. For non-zero base,
            # we'd need to modify the model. For now, only z-offset is common
            # (raising robot above table). We handle this by noting that the
            # Menagerie model already has link0 at origin. A table at z=0.4
            # means the robot is on a pedestal or the table is lower.
            # For simplicity, keep robot at origin and adjust table height.
            pass

        scene_xml = f"""\
<mujoco model="forge_scene">
  <include file="panda.xml"/>

  <option timestep="{timestep}" gravity="0 0 -9.81"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.8 0.8 0.8"
             rgb2="0.6 0.6 0.6" width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.1"/>
  </asset>

  <worldbody>
    <!-- Ground plane -->
    <geom name="floor" type="plane" size="2 2 0.01" material="grid_mat" condim="3"/>

    <!-- Lighting -->
    <light pos="0.5 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.5 0.5 1.5" dir="0.5 -0.5 -1" diffuse="0.4 0.4 0.4"/>

    <!-- Table -->
    <body name="table" pos="0.5 0.0 0.2">
      <geom name="table_top" type="box" size="0.4 0.5 0.02"
            rgba="0.6 0.45 0.3 1" mass="20" condim="4"/>
      <geom type="cylinder" pos="-0.35 -0.45 -0.1" size="0.025 0.1" rgba="0.5 0.4 0.3 1"/>
      <geom type="cylinder" pos="0.35 -0.45 -0.1" size="0.025 0.1" rgba="0.5 0.4 0.3 1"/>
      <geom type="cylinder" pos="-0.35 0.45 -0.1" size="0.025 0.1" rgba="0.5 0.4 0.3 1"/>
      <geom type="cylinder" pos="0.35 0.45 -0.1" size="0.025 0.1" rgba="0.5 0.4 0.3 1"/>
    </body>

    <!-- Scene objects -->
{objects_xml}
  </worldbody>
</mujoco>"""
        return scene_xml

    @staticmethod
    def _material_to_rgba(material: str) -> str:
        """Convert material name to rgba string."""
        materials = {
            "red_mat": "0.9 0.1 0.1 1",
            "blue_mat": "0.1 0.1 0.9 1",
            "green_mat": "0.1 0.7 0.1 0.3",
        }
        return materials.get(material, "0.5 0.5 0.5 1")

    @property
    def model(self) -> mujoco.MjModel:
        assert self._model is not None
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        assert self._data is not None
        return self._data

    @property
    def timestep(self) -> float:
        return self._model.opt.timestep

    @property
    def sim_time(self) -> float:
        return self._data.time

    @property
    def n_arm_joints(self) -> int:
        return len(PANDA_ARM_JOINTS)

    @property
    def n_all_joints(self) -> int:
        return len(PANDA_ALL_JOINTS)

    def step(self, n_steps: int = 1) -> None:
        """Advance the simulation by n_steps."""
        for _ in range(n_steps):
            mujoco.mj_step(self._model, self._data)

    def set_joint_targets(self, targets: np.ndarray) -> None:
        """Set position targets for the robot.

        The Menagerie Panda uses general actuators with affine bias that
        behave like PD controllers. Setting ctrl[i] = desired_position for
        arm joints. For fingers: ctrl[7] controls both fingers via a tendon
        (0 = open, 255 = closed).

        Args:
            targets: Joint position targets.
                - Length 7: arm only, fingers unchanged
                - Length 8: arm + finger actuator (0-255)
                - Length 9: arm + individual finger positions (converted to actuator8)
        """
        n = len(targets)
        if n >= 7:
            self._data.ctrl[:N_ARM_ACTUATORS] = targets[:N_ARM_ACTUATORS]
        if n == 8:
            self._data.ctrl[N_ARM_ACTUATORS] = targets[7]
        elif n == 9:
            # Convert finger gap (0.04=open, 0=closed) to actuator8 (0=open, 255=closed)
            finger_gap = targets[7]  # use first finger value
            finger_ctrl = max(0, min(255, (0.04 - finger_gap) / 0.04 * 255))
            self._data.ctrl[N_ARM_ACTUATORS] = finger_ctrl

        logger.debug("robot_command_received: %d targets", n)

    def get_robot_state(self) -> RobotState:
        """Return current robot state."""
        n_joints = self.n_all_joints

        qpos = self._data.qpos[:n_joints].copy()
        qvel = self._data.qvel[:n_joints].copy()

        # End-effector pose from hand body
        ee_pos = np.zeros(3)
        ee_quat = np.array([1.0, 0.0, 0.0, 0.0])
        if self._hand_body_id >= 0:
            ee_pos = self._data.xpos[self._hand_body_id].copy()
            ee_quat = self._data.xquat[self._hand_body_id].copy()

        return RobotState(
            joint_positions=qpos,
            joint_velocities=qvel,
            joint_names=list(PANDA_ALL_JOINTS),
            ee_position=ee_pos,
            ee_orientation=ee_quat,
            timestamp=self._data.time,
        )

    def get_object_states(self) -> list[ObjectState]:
        """Return ground-truth state of all scene objects."""
        states = []
        for obj_name, body_id in self._object_body_ids.items():
            pos = self._data.xpos[body_id].copy()
            quat = self._data.xquat[body_id].copy()
            states.append(
                ObjectState(
                    id=obj_name,
                    position=pos,
                    orientation=quat,
                    timestamp=self._data.time,
                )
            )
        return states

    def get_object_state(self, object_id: str) -> ObjectState | None:
        """Return state of a specific object, or None if not found."""
        body_id = self._object_body_ids.get(object_id)
        if body_id is None:
            return None
        pos = self._data.xpos[body_id].copy()
        quat = self._data.xquat[body_id].copy()
        return ObjectState(
            id=object_id,
            position=pos,
            orientation=quat,
            timestamp=self._data.time,
        )

    def reset(self) -> None:
        """Reset simulation to initial scenario state."""
        logger.info("reset_requested")
        assert self._initial_qpos is not None

        self._data.qpos[:] = self._initial_qpos
        self._data.qvel[:] = self._initial_qvel
        self._data.ctrl[:] = self._initial_ctrl
        self._data.time = 0.0
        mujoco.mj_forward(self._model, self._data)
        logger.info("reset_completed")

    def shutdown(self) -> None:
        """Clean up simulation resources."""
        logger.info("simulation_stopped")
        self._model = None
        self._data = None
