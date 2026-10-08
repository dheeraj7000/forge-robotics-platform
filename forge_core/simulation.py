"""MuJoCo simulation engine for Forge.

Owns the simulation lifecycle: load, step, reset, query state.
This module has NO ROS dependency — it can be used standalone for testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from forge_core.config import PROJECT_ROOT, load_scenario, resolve_model_path

logger = logging.getLogger(__name__)

# Joint names for the Panda arm (7 DOF) + 2 finger joints
PANDA_ARM_JOINTS = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]
PANDA_FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
PANDA_ALL_JOINTS = PANDA_ARM_JOINTS + PANDA_FINGER_JOINTS


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

    Responsibilities:
    - Load robot + scene from scenario config
    - Step the physics
    - Provide robot and object state queries
    - Reset to initial scenario state
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
        self._ee_site_id: int = -1

        self._load_scenario(scenario_name)

    def _load_scenario(self, scenario_name: str) -> None:
        """Build a combined MJCF model from scenario configuration."""
        self._scenario = load_scenario(scenario_name)

        robot_cfg = self._scenario["robot"]
        sim_cfg = self._scenario["simulation"]
        objects_cfg = self._scenario.get("objects", {})

        # Load the world XML as the base
        world_path = resolve_model_path(sim_cfg["world_path"])
        robot_path = resolve_model_path(robot_cfg["model_path"])

        # Build a combined scene XML that includes world + robot + objects
        scene_xml = self._build_scene_xml(
            world_path, robot_path, robot_cfg, objects_cfg, sim_cfg
        )

        self._model = mujoco.MjModel.from_xml_string(scene_xml)
        self._data = mujoco.MjData(self._model)

        # Set home configuration
        home_qpos = robot_cfg.get("home_qpos")
        if home_qpos is not None:
            n_robot_joints = len(home_qpos)
            self._data.qpos[:n_robot_joints] = home_qpos

        # Forward kinematics to get consistent initial state
        mujoco.mj_forward(self._model, self._data)

        # Cache initial state for reset
        self._initial_qpos = self._data.qpos.copy()
        self._initial_qvel = self._data.qvel.copy()

        # Cache object body IDs
        for obj_name in objects_cfg:
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
            if body_id >= 0:
                self._object_body_ids[obj_name] = body_id
                self._object_names.append(obj_name)

        # Cache end-effector site ID
        self._ee_site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, "end_effector"
        )

        logger.info("simulation_started: scenario=%s, objects=%s", scenario_name, self._object_names)

    def _build_scene_xml(
        self,
        world_path: Path,
        robot_path: Path,
        robot_cfg: dict,
        objects_cfg: dict,
        sim_cfg: dict,
    ) -> str:
        """Programmatically build a combined MJCF scene XML string."""
        base_pos = robot_cfg.get("base_position", [0, 0, 0.32])
        timestep = sim_cfg.get("timestep", 0.002)

        # Read robot XML to include it
        robot_xml_content = robot_path.read_text()

        # Build object XML fragments
        object_bodies = []
        for obj_name, obj_cfg in objects_cfg.items():
            pos = obj_cfg["position"]
            size = obj_cfg.get("size", [0.025, 0.025, 0.025])
            obj_type = obj_cfg.get("type", "box")
            material = obj_cfg.get("material", "")
            mass = obj_cfg.get("mass", 0.05)
            is_static = obj_cfg.get("static", False)

            if is_static:
                body_xml = (
                    f'    <body name="{obj_name}" pos="{pos[0]} {pos[1]} {pos[2]}">\n'
                    f'      <geom name="{obj_name}_geom" type="{obj_type}" '
                    f'size="{size[0]} {size[1]} {size[2]}" material="{material}" '
                    f'mass="{mass}"/>\n'
                    f"    </body>\n"
                )
            else:
                body_xml = (
                    f'    <body name="{obj_name}" pos="{pos[0]} {pos[1]} {pos[2]}">\n'
                    f'      <freejoint name="{obj_name}_joint"/>\n'
                    f'      <geom name="{obj_name}_geom" type="{obj_type}" '
                    f'size="{size[0]} {size[1]} {size[2]}" material="{material}" '
                    f'mass="{mass}" condim="4"/>\n'
                    f"    </body>\n"
                )
            object_bodies.append(body_xml)

        objects_xml = "\n".join(object_bodies)

        # Combine into a single scene
        scene_xml = f"""\
<mujoco model="forge_scene">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep}" gravity="0 0 -9.81" integrator="implicit"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.8 0.8 0.8"
             rgb2="0.6 0.6 0.6" width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <material name="table_mat" rgba="0.6 0.45 0.3 1" reflectance="0.05"/>
    <material name="red_mat" rgba="0.9 0.1 0.1 1"/>
    <material name="blue_mat" rgba="0.1 0.1 0.9 1"/>
    <material name="green_mat" rgba="0.1 0.7 0.1 0.3"/>
  </asset>

  <default>
    <joint damping="1.0" armature="0.1"/>
    <geom condim="4" friction="1 0.5 0.01" margin="0.001"/>
    <position kp="100" kv="20"/>
  </default>

  <worldbody>
    <!-- Ground plane -->
    <geom name="floor" type="plane" size="2 2 0.01" material="grid_mat" condim="3"/>

    <!-- Lighting -->
    <light pos="0.5 0 1.5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="-0.5 0.5 1.5" dir="0.5 -0.5 -1" diffuse="0.4 0.4 0.4"/>

    <!-- Table -->
    <body name="table" pos="0.5 0.0 0.3">
      <geom name="table_top" type="box" size="0.4 0.5 0.02" material="table_mat"
            mass="20" condim="4"/>
      <geom type="cylinder" pos="-0.35 -0.45 -0.15" size="0.025 0.15" rgba="0.5 0.4 0.3 1"/>
      <geom type="cylinder" pos="0.35 -0.45 -0.15" size="0.025 0.15" rgba="0.5 0.4 0.3 1"/>
      <geom type="cylinder" pos="-0.35 0.45 -0.15" size="0.025 0.15" rgba="0.5 0.4 0.3 1"/>
      <geom type="cylinder" pos="0.35 0.45 -0.15" size="0.025 0.15" rgba="0.5 0.4 0.3 1"/>
    </body>

    <!-- Robot base -->
    <body name="robot_base" pos="{base_pos[0]} {base_pos[1]} {base_pos[2]}">
      <body name="panda_link0" pos="0 0 0">
        <geom type="cylinder" size="0.06 0.03" rgba="0.9 0.9 0.9 1" mass="2.0"/>

        <body name="panda_link1" pos="0 0 0.333">
          <joint name="panda_joint1" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.06" rgba="0.9 0.9 0.9 1" mass="2.0"/>

          <body name="panda_link2" pos="0 0 0" quat="0.707107 -0.707107 0 0">
            <joint name="panda_joint2" type="hinge" axis="0 0 1" range="-1.7628 1.7628"/>
            <geom type="capsule" fromto="0 0 0 0 -0.316 0" size="0.06" rgba="0.9 0.9 0.9 1" mass="2.0"/>

            <body name="panda_link3" pos="0 -0.316 0" quat="0.707107 0.707107 0 0">
              <joint name="panda_joint3" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
              <geom type="capsule" fromto="0 0 0 0.0825 0 0" size="0.05" rgba="0.9 0.9 0.9 1" mass="1.5"/>

              <body name="panda_link4" pos="0.0825 0 0" quat="0.707107 0.707107 0 0">
                <joint name="panda_joint4" type="hinge" axis="0 0 1" range="-3.0718 -0.0698"/>
                <geom type="capsule" fromto="0 0 0 -0.0825 0.384 0" size="0.05" rgba="0.9 0.9 0.9 1" mass="1.5"/>

                <body name="panda_link5" pos="-0.0825 0.384 0" quat="0.707107 -0.707107 0 0">
                  <joint name="panda_joint5" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                  <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.04" rgba="0.9 0.9 0.9 1" mass="1.0"/>

                  <body name="panda_link6" pos="0 0 0" quat="0.707107 0.707107 0 0">
                    <joint name="panda_joint6" type="hinge" axis="0 0 1" range="-0.0175 3.7525"/>
                    <geom type="capsule" fromto="0 0 0 0.088 0 0" size="0.04" rgba="0.9 0.9 0.9 1" mass="1.0"/>

                    <body name="panda_link7" pos="0.088 0 0" quat="0.707107 0.707107 0 0">
                      <joint name="panda_joint7" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                      <geom type="cylinder" size="0.04 0.02" rgba="0.5 0.5 0.5 1" mass="0.5"/>

                      <body name="panda_hand" pos="0 0 0.107" quat="0.924 0 0 0.383">
                        <geom type="box" size="0.03 0.06 0.01" rgba="0.3 0.3 0.3 1" mass="0.5"/>

                        <body name="panda_leftfinger" pos="0 0.02 0.05">
                          <joint name="panda_finger_joint1" type="slide" axis="0 1 0" range="0 0.04"/>
                          <geom type="box" size="0.01 0.005 0.03" rgba="0.5 0.5 0.5 1" mass="0.1"/>
                        </body>

                        <body name="panda_rightfinger" pos="0 -0.02 0.05">
                          <joint name="panda_finger_joint2" type="slide" axis="0 -1 0" range="0 0.04"/>
                          <geom type="box" size="0.01 0.005 0.03" rgba="0.5 0.5 0.5 1" mass="0.1"/>
                        </body>

                        <site name="end_effector" pos="0 0 0.07" size="0.01" rgba="1 0 0 1"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Scene objects -->
{objects_xml}
  </worldbody>

  <actuator>
    <position joint="panda_joint1" name="actuator1" ctrlrange="-2.8973 2.8973"/>
    <position joint="panda_joint2" name="actuator2" ctrlrange="-1.7628 1.7628"/>
    <position joint="panda_joint3" name="actuator3" ctrlrange="-2.8973 2.8973"/>
    <position joint="panda_joint4" name="actuator4" ctrlrange="-3.0718 -0.0698"/>
    <position joint="panda_joint5" name="actuator5" ctrlrange="-2.8973 2.8973"/>
    <position joint="panda_joint6" name="actuator6" ctrlrange="-0.0175 3.7525"/>
    <position joint="panda_joint7" name="actuator7" ctrlrange="-2.8973 2.8973"/>
    <position joint="panda_finger_joint1" name="actuator_finger1" ctrlrange="0 0.04"/>
    <position joint="panda_finger_joint2" name="actuator_finger2" ctrlrange="0 0.04"/>
  </actuator>
</mujoco>"""
        return scene_xml

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
        """Set position targets for all arm actuators.

        Args:
            targets: Array of joint targets. Length 7 for arm only,
                     or 9 for arm + fingers.
        """
        n = len(targets)
        if n > self._model.nu:
            raise ValueError(
                f"Expected at most {self._model.nu} targets, got {n}"
            )
        self._data.ctrl[:n] = targets
        logger.debug("robot_command_received: %d joint targets", n)

    def get_robot_state(self) -> RobotState:
        """Return current robot state (joint positions, velocities, EE pose)."""
        n_joints = self.n_all_joints

        # Joint positions and velocities (arm + fingers occupy first n_joints slots)
        qpos = self._data.qpos[:n_joints].copy()
        qvel = self._data.qvel[:n_joints].copy()

        # End-effector pose from site
        ee_pos = np.zeros(3)
        ee_quat = np.array([1.0, 0.0, 0.0, 0.0])
        if self._ee_site_id >= 0:
            ee_pos = self._data.site_xpos[self._ee_site_id].copy()
            # MuJoCo stores site rotation as 3x3 matrix; convert to quaternion
            ee_mat = self._data.site_xmat[self._ee_site_id].reshape(3, 3)
            ee_quat = np.zeros(4)
            mujoco.mju_mat2Quat(ee_quat, ee_mat.flatten())

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
        assert self._initial_qvel is not None

        self._data.qpos[:] = self._initial_qpos
        self._data.qvel[:] = self._initial_qvel
        self._data.ctrl[:] = 0.0
        self._data.time = 0.0
        mujoco.mj_forward(self._model, self._data)
        logger.info("reset_completed")

    def shutdown(self) -> None:
        """Clean up simulation resources."""
        logger.info("simulation_stopped")
        self._model = None
        self._data = None
