"""Unit tests for sim-to-real transfer utilities.

Tests cover DomainRandomizationConfig, apply_domain_randomization,
trajectory comparison, calibration, and config loading.

Phase 11 — FEAT-002.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.sim2real import (
    CalibrationResult,
    DomainRandomizationConfig,
    TrajectoryComparison,
    apply_domain_randomization,
    calibrate_joint_offsets,
    calibrate_workspace,
    compare_trajectories,
    load_sim2real_config,
)
from forge_core.simulation import ForgeSimulation


# ── TestDomainRandomizationConfig ──────────────────────────


class TestDomainRandomizationConfig:
    def test_defaults(self):
        config = DomainRandomizationConfig()
        assert config.gravity_range == (-10.5, -9.0)
        assert config.friction_range == (0.5, 1.5)
        assert config.mass_scale_range == (0.8, 1.2)
        assert config.sensor_noise_std == 0.01
        assert config.actuator_noise_std == 0.005

    def test_from_dict(self):
        data = {
            "gravity_range": [-11.0, -8.0],
            "friction_range": [0.3, 2.0],
            "mass_scale_range": [0.5, 1.5],
            "sensor_noise_std": 0.02,
            "actuator_noise_std": 0.01,
        }
        config = DomainRandomizationConfig.from_dict(data)
        assert config.gravity_range == (-11.0, -8.0)
        assert config.friction_range == (0.3, 2.0)
        assert config.mass_scale_range == (0.5, 1.5)
        assert config.sensor_noise_std == 0.02
        assert config.actuator_noise_std == 0.01

    def test_from_dict_defaults_for_missing_keys(self):
        config = DomainRandomizationConfig.from_dict({})
        assert config.gravity_range == (-10.5, -9.0)
        assert config.sensor_noise_std == 0.01


# ── TestApplyDomainRandomization ───────────────────────────


class TestApplyDomainRandomization:
    def test_returns_dict_of_applied_values(self):
        sim = ForgeSimulation("basic_workspace")
        sim.reset()
        config = DomainRandomizationConfig()
        rng = np.random.default_rng(42)
        applied = apply_domain_randomization(sim, config, rng=rng)
        assert "gravity_z" in applied
        assert "friction_scale" in applied
        assert "mass_scale" in applied
        assert "sensor_noise_std" in applied
        assert "actuator_noise_std" in applied
        sim.shutdown()

    def test_with_custom_config(self):
        sim = ForgeSimulation("basic_workspace")
        sim.reset()
        config = DomainRandomizationConfig(
            gravity_range=(-9.5, -9.5),  # Fixed value
            friction_range=(1.0, 1.0),   # Fixed value
            mass_scale_range=(1.0, 1.0), # Fixed value
        )
        rng = np.random.default_rng(0)
        applied = apply_domain_randomization(sim, config, rng=rng)
        assert applied["gravity_z"] == pytest.approx(-9.5)
        assert applied["friction_scale"] == pytest.approx(1.0)
        assert applied["mass_scale"] == pytest.approx(1.0)
        sim.shutdown()


# ── TestTrajectoryComparison ───────────────────────────────


class TestTrajectoryComparison:
    def test_identical_trajectories_zero_rmse(self):
        traj = {
            "joint_positions": np.ones((10, 7)),
            "joint_velocities": np.zeros((10, 7)),
            "ee_positions": np.ones((10, 3)) * 0.5,
        }
        result = compare_trajectories(traj, traj)
        assert result.position_rmse == pytest.approx(0.0, abs=1e-10)
        assert result.velocity_rmse == pytest.approx(0.0, abs=1e-10)
        assert result.ee_position_rmse == pytest.approx(0.0, abs=1e-10)
        assert result.max_position_error == pytest.approx(0.0, abs=1e-10)

    def test_different_trajectories_nonzero_rmse(self):
        traj1 = {
            "joint_positions": np.zeros((10, 7)),
            "joint_velocities": np.zeros((10, 7)),
            "ee_positions": np.zeros((10, 3)),
        }
        traj2 = {
            "joint_positions": np.ones((10, 7)),
            "joint_velocities": np.ones((10, 7)),
            "ee_positions": np.ones((10, 3)),
        }
        result = compare_trajectories(traj1, traj2)
        assert result.position_rmse > 0
        assert result.velocity_rmse > 0
        assert result.ee_position_rmse > 0
        assert result.max_position_error > 0

    def test_different_lengths_truncation(self):
        traj1 = {
            "joint_positions": np.zeros((20, 7)),
            "joint_velocities": np.zeros((20, 7)),
            "ee_positions": np.zeros((20, 3)),
        }
        traj2 = {
            "joint_positions": np.zeros((10, 7)),
            "joint_velocities": np.zeros((10, 7)),
            "ee_positions": np.zeros((10, 3)),
        }
        result = compare_trajectories(traj1, traj2)
        # Should truncate to 10 steps and produce zero rmse
        assert result.position_rmse == pytest.approx(0.0, abs=1e-10)

    def test_to_dict(self):
        result = TrajectoryComparison(
            position_rmse=0.1,
            velocity_rmse=0.2,
            ee_position_rmse=0.15,
            max_position_error=0.3,
            max_ee_error=0.25,
            correlation=0.95,
            summary="test",
        )
        d = result.to_dict()
        assert d["position_rmse"] == 0.1
        assert d["correlation"] == 0.95
        assert d["summary"] == "test"


# ── TestCalibration ────────────────────────────────────────


class TestCalibration:
    def test_calibrate_joint_offsets_known_offset(self):
        rng = np.random.default_rng(42)
        sim_pos = rng.uniform(-1, 1, size=(50, 7))
        known_offset = np.array([0.1, -0.05, 0.02, -0.03, 0.04, -0.01, 0.06])
        real_pos = sim_pos + known_offset
        result = calibrate_joint_offsets(sim_pos, real_pos)
        np.testing.assert_allclose(result.joint_offsets, known_offset, atol=1e-10)
        assert result.residual_error < 1e-10
        assert result.n_samples == 50

    def test_calibrate_workspace_known_translation(self):
        rng = np.random.default_rng(42)
        sim_points = rng.uniform(-1, 1, size=(30, 3))
        translation = np.array([0.5, -0.3, 0.1])
        real_points = sim_points + translation
        result = calibrate_workspace(sim_points, real_points)
        np.testing.assert_allclose(
            result.workspace_transform[:3, 3], translation, atol=1e-10,
        )
        assert result.residual_error < 1e-10
        assert result.n_samples == 30
        # Rotation should be identity
        np.testing.assert_allclose(
            result.workspace_transform[:3, :3], np.eye(3), atol=1e-10,
        )

    def test_residual_error_small_after_calibration(self):
        rng = np.random.default_rng(123)
        sim_pos = rng.uniform(-1, 1, size=(100, 7))
        # Add small noise to create imperfect offset
        noise = rng.normal(0, 0.001, size=(100, 7))
        real_pos = sim_pos + np.array([0.05] * 7) + noise
        result = calibrate_joint_offsets(sim_pos, real_pos)
        # Residual should be on the order of noise std
        assert result.residual_error < 0.01

    def test_calibration_result_to_dict(self):
        result = CalibrationResult(
            joint_offsets=np.array([0.1, 0.2, 0.3]),
            workspace_transform=np.eye(4),
            residual_error=0.001,
            n_samples=10,
        )
        d = result.to_dict()
        assert d["residual_error"] == 0.001
        assert d["n_samples"] == 10
        assert isinstance(d["joint_offsets"], list)
        assert isinstance(d["workspace_transform"], list)


# ── TestLoadSim2realConfig ─────────────────────────────────


class TestLoadSim2realConfig:
    def test_load_missing_file_returns_defaults(self):
        config = load_sim2real_config(path=None)
        assert isinstance(config, DomainRandomizationConfig)
        assert config.gravity_range == (-10.5, -9.0)

    def test_load_from_robot_yaml(self):
        from forge_core.config import PROJECT_ROOT
        config_path = PROJECT_ROOT / "config" / "robot.yaml"
        if config_path.exists():
            config = load_sim2real_config(str(config_path))
            assert isinstance(config, DomainRandomizationConfig)

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_sim2real_config(path="/nonexistent/robot.yaml")
