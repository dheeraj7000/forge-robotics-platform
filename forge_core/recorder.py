"""Episode recorder for Forge.

Captures robot state, object state, and actions at each control cycle.
Saves trajectories as numpy .npz files alongside JSON metadata.

Stored run format:
    runs/run_XXXXX/
        metadata.json   — run ID, task, scenario, policy, timestamp, success
        trajectory.npz  — numpy arrays (timestamps, joint_pos, joint_vel,
                          ee_pos, ee_quat, object_pos, object_quat, actions)
        config.yaml     — scenario config snapshot
        result.json     — evaluation result
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from forge_core.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryBuffer:
    """In-memory buffer for recording a single episode."""

    timestamps: list[float] = field(default_factory=list)
    joint_positions: list[np.ndarray] = field(default_factory=list)
    joint_velocities: list[np.ndarray] = field(default_factory=list)
    ee_positions: list[np.ndarray] = field(default_factory=list)
    ee_orientations: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    object_positions: list[dict[str, np.ndarray]] = field(default_factory=list)
    object_orientations: list[dict[str, np.ndarray]] = field(default_factory=list)
    frames: list[np.ndarray] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.timestamps)

    def record_step(
        self,
        timestamp: float,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
        action: np.ndarray,
        object_states: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Record one timestep of data."""
        self.timestamps.append(timestamp)
        self.joint_positions.append(joint_pos.copy())
        self.joint_velocities.append(joint_vel.copy())
        self.ee_positions.append(ee_pos.copy())
        self.ee_orientations.append(ee_quat.copy())
        self.actions.append(action.copy())
        self.object_positions.append(
            {k: v[0].copy() for k, v in object_states.items()}
        )
        self.object_orientations.append(
            {k: v[1].copy() for k, v in object_states.items()}
        )

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        """Convert buffer to dict of numpy arrays for saving."""
        data: dict[str, np.ndarray] = {
            "timestamps": np.array(self.timestamps),
            "joint_positions": np.array(self.joint_positions),
            "joint_velocities": np.array(self.joint_velocities),
            "ee_positions": np.array(self.ee_positions),
            "ee_orientations": np.array(self.ee_orientations),
            "actions": np.array(self.actions),
        }

        # Flatten object states into per-object arrays
        if self.object_positions:
            obj_names = sorted(self.object_positions[0].keys())
            for obj_name in obj_names:
                data[f"obj_{obj_name}_pos"] = np.array(
                    [step[obj_name] for step in self.object_positions]
                )
                data[f"obj_{obj_name}_quat"] = np.array(
                    [step[obj_name] for step in self.object_orientations]
                )
            data["object_names"] = np.array(obj_names)

        return data


def get_next_run_id(output_dir: str = "runs") -> int:
    """Find the next available run ID."""
    runs_dir = PROJECT_ROOT / output_dir
    if not runs_dir.exists():
        return 1
    existing = [
        d.name for d in runs_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]
    if not existing:
        return 1
    ids = []
    for name in existing:
        try:
            ids.append(int(name.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return max(ids, default=0) + 1


def save_run(
    run_id: int,
    trajectory: TrajectoryBuffer,
    metadata: dict[str, Any],
    scenario_config: dict[str, Any],
    result: dict[str, Any],
    output_dir: str = "runs",
) -> Path:
    """Save a complete run to disk.

    Creates:
        runs/run_XXXXX/
            metadata.json
            trajectory.npz
            config.yaml
            result.json
    """
    run_dir = PROJECT_ROOT / output_dir / f"run_{run_id:05d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Metadata
    metadata["run_id"] = run_id
    metadata["run_dir"] = str(run_dir)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Trajectory
    npz_data = trajectory.to_npz_dict()
    np.savez_compressed(run_dir / "trajectory.npz", **npz_data)

    # Perception frames (optional)
    if trajectory.frames:
        frames_array = np.stack(trajectory.frames)
        np.savez_compressed(run_dir / "frames.npz", frames=frames_array)

    # Config snapshot
    (run_dir / "config.yaml").write_text(yaml.dump(scenario_config, default_flow_style=False))

    # Result
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))

    logger.info("run_saved: %s (%d steps)", run_dir, trajectory.length)
    return run_dir
