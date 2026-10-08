"""Phase 1 acceptance test.

Executes the full acceptance flow without ROS:
  launch simulation → read initial robot state → send robot command →
  verify robot moved → read object state → reset simulation →
  verify initial state restored

This test uses the simulation engine directly. A full ROS integration
test requires a running ROS 2 environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.simulation import ForgeSimulation


class TestAcceptance:
    """Primary Phase 1 acceptance test."""

    def test_full_acceptance_flow(self):
        """Execute the complete acceptance test sequence."""
        # 1. Launch simulation
        sim = ForgeSimulation("basic_workspace")
        assert sim.model is not None, "Simulation failed to launch"

        # 2. Read initial robot state
        initial_state = sim.get_robot_state()
        assert len(initial_state.joint_positions) == 9
        initial_joints = initial_state.joint_positions.copy()
        initial_ee = initial_state.ee_position.copy()

        # 3. Send robot command
        targets = initial_joints[:7].copy()
        targets[0] += 0.4
        targets[1] += 0.2
        sim.set_joint_targets(targets)

        # 4. Verify robot moved
        sim.step(1000)
        moved_state = sim.get_robot_state()
        joint_delta = np.linalg.norm(
            moved_state.joint_positions[:7] - initial_joints[:7]
        )
        assert joint_delta > 0.01, "Robot did not move after command"

        ee_delta = np.linalg.norm(moved_state.ee_position - initial_ee)
        assert ee_delta > 0.001, "End-effector did not move"

        # 5. Read object state
        objects = sim.get_object_states()
        assert len(objects) >= 2, "Expected at least 2 objects"
        obj_ids = {o.id for o in objects}
        assert "red_cube" in obj_ids, "red_cube missing"
        assert "blue_cube" in obj_ids, "blue_cube missing"

        for obj in objects:
            assert obj.position.shape == (3,)
            assert obj.orientation.shape == (4,)

        # 6. Reset simulation
        sim.reset()

        # 7. Verify initial state restored
        restored_state = sim.get_robot_state()
        np.testing.assert_allclose(
            restored_state.joint_positions, initial_joints, atol=1e-6,
            err_msg="Joint positions not restored after reset"
        )
        assert sim.sim_time == 0.0, "Simulation time not reset"

        sim.shutdown()
