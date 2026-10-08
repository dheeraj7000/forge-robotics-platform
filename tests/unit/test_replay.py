"""Unit tests for run replay."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.replay import load_run, RunRecord


@pytest.fixture
def sample_run(tmp_path):
    """Create a minimal run directory for testing."""
    run_dir = tmp_path / "run_00099"
    run_dir.mkdir()

    # metadata
    meta = {
        "run_id": 99,
        "task_name": "test_task",
        "scenario": "basic_workspace",
        "policy": "NullPolicy",
        "success": True,
        "sim_time": 2.0,
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta))

    # result
    result = {"task_name": "test_task", "success": True, "criteria": []}
    (run_dir / "result.json").write_text(json.dumps(result))

    # config
    config = {"scenario": {"name": "basic_workspace"}}
    (run_dir / "config.yaml").write_text(yaml.dump(config))

    # trajectory
    n = 10
    np.savez_compressed(
        run_dir / "trajectory.npz",
        timestamps=np.arange(n) * 0.02,
        joint_positions=np.random.randn(n, 9),
        joint_velocities=np.zeros((n, 9)),
        ee_positions=np.random.randn(n, 3),
        ee_orientations=np.tile([1, 0, 0, 0], (n, 1)),
        actions=np.zeros((n, 8)),
        object_names=np.array(["red_cube"]),
        obj_red_cube_pos=np.random.randn(n, 3),
        obj_red_cube_quat=np.tile([1, 0, 0, 0], (n, 1)),
    )
    return run_dir


class TestLoadRun:
    def test_load_run(self, sample_run):
        record = load_run(sample_run)
        assert isinstance(record, RunRecord)
        assert record.task_name == "test_task"
        assert record.n_steps == 10
        assert record.success is True

    def test_state_at_step(self, sample_run):
        record = load_run(sample_run)
        state = record.state_at_step(5)
        assert state["step"] == 5
        assert "joint_positions" in state
        assert "ee_position" in state
        assert "objects" in state
        assert "red_cube" in state["objects"]

    def test_state_at_time(self, sample_run):
        record = load_run(sample_run)
        state = record.state_at_time(0.05)  # nearest to step 2-3
        assert "timestamp" in state

    def test_summary(self, sample_run):
        record = load_run(sample_run)
        s = record.summary()
        assert "test_task" in s
        assert "SUCCESS" in s

    def test_duration(self, sample_run):
        record = load_run(sample_run)
        assert record.duration > 0

    def test_missing_run_raises(self):
        with pytest.raises(FileNotFoundError):
            load_run("/nonexistent/run_00000")

    def test_object_names(self, sample_run):
        record = load_run(sample_run)
        assert "red_cube" in record.object_names
        assert "red_cube" in record.object_positions
