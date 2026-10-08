"""Replay recorded Forge runs.

Load a run directory and inspect trajectory data, metadata, and results.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """A loaded run record with trajectory and metadata."""

    run_dir: Path
    metadata: dict[str, Any]
    result: dict[str, Any]
    config: dict[str, Any]
    timestamps: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    ee_positions: np.ndarray
    ee_orientations: np.ndarray
    actions: np.ndarray
    object_names: list[str]
    object_positions: dict[str, np.ndarray]  # name -> (T, 3)
    object_orientations: dict[str, np.ndarray]  # name -> (T, 4)

    @property
    def n_steps(self) -> int:
        return len(self.timestamps)

    @property
    def duration(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def task_name(self) -> str:
        return self.metadata.get("task_name", "unknown")

    @property
    def success(self) -> bool:
        return self.result.get("success", False)

    def state_at_step(self, step: int) -> dict[str, Any]:
        """Return all state data at a specific step index."""
        step = max(0, min(step, self.n_steps - 1))
        state = {
            "step": step,
            "timestamp": float(self.timestamps[step]),
            "joint_positions": self.joint_positions[step].tolist(),
            "joint_velocities": self.joint_velocities[step].tolist(),
            "ee_position": self.ee_positions[step].tolist(),
            "ee_orientation": self.ee_orientations[step].tolist(),
            "action": self.actions[step].tolist(),
            "objects": {},
        }
        for name in self.object_names:
            state["objects"][name] = {
                "position": self.object_positions[name][step].tolist(),
                "orientation": self.object_orientations[name][step].tolist(),
            }
        return state

    def state_at_time(self, t: float) -> dict[str, Any]:
        """Return state at the step closest to time t."""
        idx = int(np.argmin(np.abs(self.timestamps - t)))
        return self.state_at_step(idx)

    def summary(self) -> str:
        """Human-readable summary of the run."""
        status = "SUCCESS" if self.success else "FAILURE"
        lines = [
            f"Run: {self.run_dir.name}",
            f"Task: {self.task_name}",
            f"Result: {status}",
            f"Steps: {self.n_steps}",
            f"Duration: {self.duration:.3f}s",
            f"Policy: {self.metadata.get('policy', 'unknown')}",
            f"Scenario: {self.metadata.get('scenario', 'unknown')}",
        ]
        if self.object_names:
            lines.append(f"Objects: {', '.join(self.object_names)}")

            # Show initial and final object positions
            for name in self.object_names:
                pos_i = self.object_positions[name][0]
                pos_f = self.object_positions[name][-1]
                dist = np.linalg.norm(pos_f - pos_i)
                lines.append(
                    f"  {name}: [{pos_i[0]:.3f},{pos_i[1]:.3f},{pos_i[2]:.3f}] "
                    f"-> [{pos_f[0]:.3f},{pos_f[1]:.3f},{pos_f[2]:.3f}] "
                    f"(moved {dist:.4f}m)"
                )
        return "\n".join(lines)


def load_run(run_dir: str | Path) -> RunRecord:
    """Load a recorded run from disk."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Load metadata
    metadata = json.loads((run_dir / "metadata.json").read_text())

    # Load result
    result = json.loads((run_dir / "result.json").read_text())

    # Load config
    config = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}

    # Load trajectory
    traj_path = run_dir / "trajectory.npz"
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")

    data = np.load(traj_path, allow_pickle=True)

    # Extract object data
    object_names = list(data["object_names"]) if "object_names" in data else []
    object_positions = {}
    object_orientations = {}
    for name in object_names:
        object_positions[name] = data[f"obj_{name}_pos"]
        object_orientations[name] = data[f"obj_{name}_quat"]

    return RunRecord(
        run_dir=run_dir,
        metadata=metadata,
        result=result,
        config=config,
        timestamps=data["timestamps"],
        joint_positions=data["joint_positions"],
        joint_velocities=data["joint_velocities"],
        ee_positions=data["ee_positions"],
        ee_orientations=data["ee_orientations"],
        actions=data["actions"],
        object_names=object_names,
        object_positions=object_positions,
        object_orientations=object_orientations,
    )


def list_runs(output_dir: str = "runs") -> list[Path]:
    """List all recorded run directories."""
    from forge_core.config import PROJECT_ROOT
    runs_dir = PROJECT_ROOT / output_dir
    if not runs_dir.exists():
        return []
    return sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    )
