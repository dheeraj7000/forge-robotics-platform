"""Unit tests for the trajectory recorder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.recorder import TrajectoryBuffer, save_run, get_next_run_id


class TestTrajectoryBuffer:
    def test_record_step(self):
        buf = TrajectoryBuffer()
        buf.record_step(
            timestamp=0.1,
            joint_pos=np.zeros(9),
            joint_vel=np.zeros(9),
            ee_pos=np.array([0.5, 0.0, 0.6]),
            ee_quat=np.array([1.0, 0, 0, 0]),
            action=np.zeros(8),
            object_states={"cube": (np.array([0.5, 0.1, 0.2]), np.array([1, 0, 0, 0]))},
        )
        assert buf.length == 1

    def test_multiple_steps(self):
        buf = TrajectoryBuffer()
        for i in range(10):
            buf.record_step(
                timestamp=i * 0.02,
                joint_pos=np.ones(9) * i,
                joint_vel=np.zeros(9),
                ee_pos=np.array([0.5, 0.0, 0.6]),
                ee_quat=np.array([1.0, 0, 0, 0]),
                action=np.zeros(8),
                object_states={},
            )
        assert buf.length == 10

    def test_to_npz_dict(self):
        buf = TrajectoryBuffer()
        buf.record_step(
            timestamp=0.0,
            joint_pos=np.zeros(9),
            joint_vel=np.zeros(9),
            ee_pos=np.zeros(3),
            ee_quat=np.array([1, 0, 0, 0]),
            action=np.zeros(8),
            object_states={"red_cube": (np.array([0.5, 0.1, 0.2]), np.array([1, 0, 0, 0]))},
        )
        data = buf.to_npz_dict()
        assert "timestamps" in data
        assert "joint_positions" in data
        assert "obj_red_cube_pos" in data
        assert data["timestamps"].shape == (1,)


class TestSaveRun:
    def test_save_and_load(self, tmp_path):
        buf = TrajectoryBuffer()
        for i in range(5):
            buf.record_step(
                timestamp=i * 0.02,
                joint_pos=np.ones(9) * i,
                joint_vel=np.zeros(9),
                ee_pos=np.array([0.5, 0.0, 0.6]),
                ee_quat=np.array([1, 0, 0, 0]),
                action=np.zeros(8),
                object_states={"cube": (np.array([0.5, 0.1, 0.2]), np.array([1, 0, 0, 0]))},
            )

        # Save to tmp_path manually
        run_dir = tmp_path / "run_00001"
        run_dir.mkdir()

        # Save trajectory
        npz_data = buf.to_npz_dict()
        np.savez_compressed(run_dir / "trajectory.npz", **npz_data)

        # Verify files
        assert (run_dir / "trajectory.npz").exists()

        # Load back
        loaded = np.load(run_dir / "trajectory.npz")
        assert loaded["timestamps"].shape == (5,)
        assert loaded["joint_positions"].shape == (5, 9)
