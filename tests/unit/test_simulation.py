"""Unit tests for the Forge simulation engine.

These tests exercise the MuJoCo simulation directly — no ROS required.
They correspond to the Phase 1 testing requirements:
  Test 1 — Simulation starts
  Test 2 — Robot state exists
  Test 3 — Object state exists
  Test 4 — Command changes state
  Test 5 — Reset
  Test 6 — Scenario reproducibility
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure forge_core is importable
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.simulation import ForgeSimulation, PANDA_ARM_JOINTS, PANDA_ALL_JOINTS


@pytest.fixture
def sim():
    """Create a fresh simulation for each test."""
    s = ForgeSimulation("basic_workspace")
    yield s
    s.shutdown()


class TestSimulationStarts:
    """Test 1 — Simulator initializes successfully."""

    def test_simulation_loads(self, sim: ForgeSimulation):
        """Simulation creates a valid MuJoCo model and data."""
        assert sim.model is not None
        assert sim.data is not None

    def test_simulation_has_correct_timestep(self, sim: ForgeSimulation):
        """Timestep matches scenario configuration (0.002)."""
        assert sim.timestep == pytest.approx(0.002)

    def test_simulation_can_step(self, sim: ForgeSimulation):
        """Simulation advances without error."""
        initial_time = sim.sim_time
        sim.step(100)
        assert sim.sim_time > initial_time


class TestRobotStateExists:
    """Test 2 — Joint state is available."""

    def test_robot_state_returned(self, sim: ForgeSimulation):
        """get_robot_state returns a valid RobotState."""
        state = sim.get_robot_state()
        assert state is not None

    def test_joint_positions_shape(self, sim: ForgeSimulation):
        """Joint positions has correct number of elements (9: 7 arm + 2 finger)."""
        state = sim.get_robot_state()
        assert len(state.joint_positions) == len(PANDA_ALL_JOINTS)

    def test_joint_velocities_shape(self, sim: ForgeSimulation):
        """Joint velocities has correct number of elements."""
        state = sim.get_robot_state()
        assert len(state.joint_velocities) == len(PANDA_ALL_JOINTS)

    def test_joint_names(self, sim: ForgeSimulation):
        """Joint names match expected Panda configuration."""
        state = sim.get_robot_state()
        assert state.joint_names == list(PANDA_ALL_JOINTS)

    def test_ee_position_exists(self, sim: ForgeSimulation):
        """End-effector position is a 3D vector."""
        state = sim.get_robot_state()
        assert state.ee_position.shape == (3,)

    def test_ee_orientation_exists(self, sim: ForgeSimulation):
        """End-effector orientation is a quaternion (4 elements)."""
        state = sim.get_robot_state()
        assert state.ee_orientation.shape == (4,)

    def test_timestamp_exists(self, sim: ForgeSimulation):
        """Robot state has a simulation timestamp."""
        state = sim.get_robot_state()
        assert isinstance(state.timestamp, float)


class TestObjectStateExists:
    """Test 3 — Expected objects are present."""

    def test_objects_returned(self, sim: ForgeSimulation):
        """get_object_states returns a non-empty list."""
        objects = sim.get_object_states()
        assert len(objects) > 0

    def test_expected_objects_present(self, sim: ForgeSimulation):
        """red_cube and blue_cube are in the scene."""
        objects = sim.get_object_states()
        obj_ids = {o.id for o in objects}
        assert "red_cube" in obj_ids
        assert "blue_cube" in obj_ids

    def test_target_bin_present(self, sim: ForgeSimulation):
        """target_bin is in the scene."""
        obj = sim.get_object_state("target_bin")
        assert obj is not None

    def test_object_has_position(self, sim: ForgeSimulation):
        """Each object has a 3D position."""
        for obj in sim.get_object_states():
            assert obj.position.shape == (3,)

    def test_object_has_orientation(self, sim: ForgeSimulation):
        """Each object has a quaternion orientation."""
        for obj in sim.get_object_states():
            assert obj.orientation.shape == (4,)

    def test_object_has_timestamp(self, sim: ForgeSimulation):
        """Each object state has a timestamp."""
        for obj in sim.get_object_states():
            assert isinstance(obj.timestamp, float)

    def test_red_cube_near_expected_position(self, sim: ForgeSimulation):
        """Red cube starts near its configured position."""
        obj = sim.get_object_state("red_cube")
        assert obj is not None
        # Should be near [0.45, 0.15, 0.345] — allow tolerance for physics settling
        assert abs(obj.position[0] - 0.5) < 0.1
        assert abs(obj.position[1] - 0.15) < 0.1


class TestCommandChangesState:
    """Test 4 — Send robot command, verify robot state changes."""

    def test_joint_command_changes_position(self, sim: ForgeSimulation):
        """Sending a joint target and stepping changes joint positions."""
        state_before = sim.get_robot_state()
        initial_pos = state_before.joint_positions.copy()

        # Send a different joint configuration (move joint 1)
        targets = initial_pos[:7].copy()
        targets[0] += 0.5  # rotate first joint
        sim.set_joint_targets(targets)

        # Step enough for the controller to move
        sim.step(500)

        state_after = sim.get_robot_state()
        # Joint 1 should have moved toward the target
        assert abs(state_after.joint_positions[0] - initial_pos[0]) > 0.01

    def test_ee_position_changes_with_command(self, sim: ForgeSimulation):
        """End-effector position changes when joints move."""
        state_before = sim.get_robot_state()
        ee_before = state_before.ee_position.copy()

        # Move multiple joints
        targets = np.array([0.5, -0.3, 0.2, -1.5, 0.1, 1.2, 0.3])
        sim.set_joint_targets(targets)
        sim.step(1000)

        state_after = sim.get_robot_state()
        ee_after = state_after.ee_position.copy()

        # EE should have moved
        distance = np.linalg.norm(ee_after - ee_before)
        assert distance > 0.01


class TestReset:
    """Test 5 — Modify state, reset, verify initial state restored within tolerance."""

    def test_reset_restores_joint_positions(self, sim: ForgeSimulation):
        """After reset, joint positions match initial state."""
        initial = sim.get_robot_state().joint_positions.copy()

        # Move the robot
        sim.set_joint_targets(np.array([0.5, -0.3, 0.2, -1.5, 0.1, 1.2, 0.3]))
        sim.step(500)

        # Verify it moved
        moved = sim.get_robot_state().joint_positions.copy()
        assert not np.allclose(moved[:7], initial[:7], atol=0.01)

        # Reset
        sim.reset()
        restored = sim.get_robot_state().joint_positions.copy()

        np.testing.assert_allclose(restored, initial, atol=1e-6)

    def test_reset_restores_simulation_time(self, sim: ForgeSimulation):
        """After reset, simulation time is zero."""
        sim.step(100)
        assert sim.sim_time > 0

        sim.reset()
        assert sim.sim_time == 0.0

    def test_reset_restores_object_positions(self, sim: ForgeSimulation):
        """After reset, objects return to initial positions."""
        initial_objects = {o.id: o.position.copy() for o in sim.get_object_states()}

        # Step simulation to let physics evolve
        sim.step(500)

        # Reset
        sim.reset()

        restored_objects = {o.id: o.position.copy() for o in sim.get_object_states()}

        for obj_id in initial_objects:
            np.testing.assert_allclose(
                restored_objects[obj_id], initial_objects[obj_id], atol=1e-6,
                err_msg=f"Object {obj_id} position not restored"
            )


class TestScenarioReproducibility:
    """Test 6 — Start Scenario 001 twice, compare initial state."""

    def test_two_simulations_identical_initial_state(self):
        """Two independent simulation loads produce identical initial state."""
        sim1 = ForgeSimulation("basic_workspace")
        sim2 = ForgeSimulation("basic_workspace")

        state1 = sim1.get_robot_state()
        state2 = sim2.get_robot_state()

        np.testing.assert_allclose(
            state1.joint_positions, state2.joint_positions, atol=1e-10
        )
        np.testing.assert_allclose(
            state1.ee_position, state2.ee_position, atol=1e-10
        )

        objects1 = {o.id: o.position for o in sim1.get_object_states()}
        objects2 = {o.id: o.position for o in sim2.get_object_states()}

        for obj_id in objects1:
            np.testing.assert_allclose(
                objects1[obj_id], objects2[obj_id], atol=1e-10,
                err_msg=f"Object {obj_id} position differs between runs"
            )

        sim1.shutdown()
        sim2.shutdown()

    def test_reset_produces_same_state_as_fresh_load(self):
        """A reset simulation matches a freshly loaded one."""
        sim = ForgeSimulation("basic_workspace")
        initial = sim.get_robot_state().joint_positions.copy()
        initial_objects = {o.id: o.position.copy() for o in sim.get_object_states()}

        # Mutate state
        sim.set_joint_targets(np.array([0.3, -0.5, 0.1, -2.0, 0.4, 1.0, -0.2]))
        sim.step(300)

        # Reset
        sim.reset()
        after_reset = sim.get_robot_state().joint_positions.copy()
        after_objects = {o.id: o.position.copy() for o in sim.get_object_states()}

        np.testing.assert_allclose(after_reset, initial, atol=1e-6)
        for obj_id in initial_objects:
            np.testing.assert_allclose(
                after_objects[obj_id], initial_objects[obj_id], atol=1e-6
            )

        sim.shutdown()
