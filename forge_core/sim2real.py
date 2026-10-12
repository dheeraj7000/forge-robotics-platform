"""Sim-to-real transfer utilities for Forge.

Provides domain randomization, trajectory comparison, and calibration
tools for bridging the gap between simulated and real robot behaviour.
All functions operate on numpy arrays and ForgeSimulation — no ROS
dependency.

Phase 11 — FEAT-001.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forge_core.config import PROJECT_ROOT, load_yaml
from forge_core.simulation import ForgeSimulation

logger = logging.getLogger(__name__)


# ── Domain randomization ────────────────────────────────────


@dataclass
class DomainRandomizationConfig:
    """Configuration for physics domain randomization."""

    gravity_range: tuple[float, float] = (-10.5, -9.0)
    friction_range: tuple[float, float] = (0.5, 1.5)
    mass_scale_range: tuple[float, float] = (0.8, 1.2)
    sensor_noise_std: float = 0.01
    actuator_noise_std: float = 0.005

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainRandomizationConfig:
        gravity = data.get("gravity_range", [-10.5, -9.0])
        friction = data.get("friction_range", [0.5, 1.5])
        mass_scale = data.get("mass_scale_range", [0.8, 1.2])
        return cls(
            gravity_range=tuple(gravity),  # type: ignore[arg-type]
            friction_range=tuple(friction),  # type: ignore[arg-type]
            mass_scale_range=tuple(mass_scale),  # type: ignore[arg-type]
            sensor_noise_std=data.get("sensor_noise_std", 0.01),
            actuator_noise_std=data.get("actuator_noise_std", 0.005),
        )


def apply_domain_randomization(
    sim: ForgeSimulation,
    config: DomainRandomizationConfig,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Apply randomized physics parameters to a simulation.

    Modifies gravity, geom friction, and body masses on the MuJoCo model
    and returns a dict of the applied values.
    """
    if rng is None:
        rng = np.random.default_rng()

    model = sim.model
    applied: dict[str, float] = {}

    # Randomize gravity (z component)
    gravity_z = rng.uniform(config.gravity_range[0], config.gravity_range[1])
    model.opt.gravity[2] = gravity_z
    applied["gravity_z"] = float(gravity_z)

    # Randomize geom friction (first friction coefficient for all geoms)
    friction_scale = rng.uniform(config.friction_range[0], config.friction_range[1])
    for i in range(model.ngeom):
        model.geom_friction[i, 0] *= friction_scale
    applied["friction_scale"] = float(friction_scale)

    # Randomize body masses
    mass_scale = rng.uniform(config.mass_scale_range[0], config.mass_scale_range[1])
    for i in range(model.nbody):
        model.body_mass[i] *= mass_scale
    applied["mass_scale"] = float(mass_scale)

    applied["sensor_noise_std"] = config.sensor_noise_std
    applied["actuator_noise_std"] = config.actuator_noise_std

    logger.info(
        "domain_randomization_applied: gravity_z=%.3f, friction=%.3f, mass=%.3f",
        gravity_z, friction_scale, mass_scale,
    )
    return applied


# ── Trajectory comparison ───────────────────────────────────


@dataclass
class TrajectoryComparison:
    """Result of comparing simulated and real trajectories."""

    position_rmse: float
    velocity_rmse: float
    ee_position_rmse: float
    max_position_error: float
    max_ee_error: float
    correlation: float
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_rmse": self.position_rmse,
            "velocity_rmse": self.velocity_rmse,
            "ee_position_rmse": self.ee_position_rmse,
            "max_position_error": self.max_position_error,
            "max_ee_error": self.max_ee_error,
            "correlation": self.correlation,
            "summary": self.summary,
        }


def compare_trajectories(
    sim_trajectory: dict[str, np.ndarray],
    real_trajectory: dict[str, np.ndarray],
) -> TrajectoryComparison:
    """Compare simulated and real trajectories.

    Both trajectories are dicts with keys ``'joint_positions'``,
    ``'joint_velocities'``, ``'ee_positions'`` containing 2-D numpy
    arrays shaped ``(timesteps, dims)``.  Truncates to the shorter
    trajectory length.
    """
    sim_pos = sim_trajectory["joint_positions"]
    real_pos = real_trajectory["joint_positions"]
    sim_vel = sim_trajectory["joint_velocities"]
    real_vel = real_trajectory["joint_velocities"]
    sim_ee = sim_trajectory["ee_positions"]
    real_ee = real_trajectory["ee_positions"]

    # Truncate to shorter length
    n = min(len(sim_pos), len(real_pos))
    sim_pos = sim_pos[:n]
    real_pos = real_pos[:n]
    sim_vel = sim_vel[:n]
    real_vel = real_vel[:n]
    sim_ee = sim_ee[:n]
    real_ee = real_ee[:n]

    # Compute RMSE values
    position_rmse = float(np.sqrt(np.mean((sim_pos - real_pos) ** 2)))
    velocity_rmse = float(np.sqrt(np.mean((sim_vel - real_vel) ** 2)))
    ee_position_rmse = float(np.sqrt(np.mean((sim_ee - real_ee) ** 2)))

    # Max errors
    max_position_error = float(np.max(np.abs(sim_pos - real_pos)))
    max_ee_error = float(np.max(np.abs(sim_ee - real_ee)))

    # Correlation (flatten and compute Pearson r)
    sim_flat = sim_pos.flatten()
    real_flat = real_pos.flatten()
    if np.std(sim_flat) > 0 and np.std(real_flat) > 0:
        correlation = float(np.corrcoef(sim_flat, real_flat)[0, 1])
    else:
        correlation = 0.0

    summary = (
        f"Trajectory comparison ({n} steps): "
        f"pos_rmse={position_rmse:.4f}, "
        f"vel_rmse={velocity_rmse:.4f}, "
        f"ee_rmse={ee_position_rmse:.4f}, "
        f"corr={correlation:.4f}"
    )
    logger.info("trajectory_comparison: %s", summary)

    return TrajectoryComparison(
        position_rmse=position_rmse,
        velocity_rmse=velocity_rmse,
        ee_position_rmse=ee_position_rmse,
        max_position_error=max_position_error,
        max_ee_error=max_ee_error,
        correlation=correlation,
        summary=summary,
    )


# ── Calibration ─────────────────────────────────────────────


@dataclass
class CalibrationResult:
    """Result of a sim-to-real calibration procedure."""

    joint_offsets: np.ndarray       # (n_joints,)
    workspace_transform: np.ndarray  # (4, 4) homogeneous
    residual_error: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_offsets": self.joint_offsets.tolist(),
            "workspace_transform": self.workspace_transform.tolist(),
            "residual_error": self.residual_error,
            "n_samples": self.n_samples,
        }


def calibrate_joint_offsets(
    sim_positions: np.ndarray,
    real_positions: np.ndarray,
) -> CalibrationResult:
    """Compute per-joint offset between sim and real positions.

    Args:
        sim_positions: (n_samples, n_joints) simulated joint positions.
        real_positions: (n_samples, n_joints) real joint positions.

    Returns:
        CalibrationResult with mean offset per joint and residual RMSE
        after correction.
    """
    offsets = np.mean(real_positions - sim_positions, axis=0)
    corrected = sim_positions + offsets
    residual = float(np.sqrt(np.mean((corrected - real_positions) ** 2)))

    return CalibrationResult(
        joint_offsets=offsets,
        workspace_transform=np.eye(4),
        residual_error=residual,
        n_samples=len(sim_positions),
    )


def calibrate_workspace(
    sim_points: np.ndarray,
    real_points: np.ndarray,
) -> CalibrationResult:
    """Compute workspace transformation via centroid alignment.

    Args:
        sim_points: (n_samples, 3) simulated 3-D points.
        real_points: (n_samples, 3) real 3-D points.

    Returns:
        CalibrationResult with identity rotation and translation offset
        as a 4×4 homogeneous transform.
    """
    sim_centroid = np.mean(sim_points, axis=0)
    real_centroid = np.mean(real_points, axis=0)
    translation = real_centroid - sim_centroid

    transform = np.eye(4)
    transform[:3, 3] = translation

    corrected = sim_points + translation
    residual = float(np.sqrt(np.mean((corrected - real_points) ** 2)))

    return CalibrationResult(
        joint_offsets=translation,
        workspace_transform=transform,
        residual_error=residual,
        n_samples=len(sim_points),
    )


# ── Config loader ───────────────────────────────────────────


def load_sim2real_config(
    path: str | Path | None = None,
) -> DomainRandomizationConfig:
    """Load sim-to-real configuration from a YAML file.

    When *path* is given, loads that file (raises FileNotFoundError if
    missing).  When *path* is None, tries the default
    ``config/robot.yaml``; returns defaults if the file is missing or
    the ``sim2real.domain_randomization`` key is absent.
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Sim2real config not found: {p}")
        data = load_yaml(p)
    else:
        default_path = PROJECT_ROOT / "config" / "robot.yaml"
        if not default_path.exists():
            logger.debug("no default robot config found, using defaults")
            return DomainRandomizationConfig()
        data = load_yaml(default_path)

    sim2real = data.get("sim2real", {})
    dr_data = sim2real.get("domain_randomization", {})
    if not dr_data:
        return DomainRandomizationConfig()
    return DomainRandomizationConfig.from_dict(dr_data)
