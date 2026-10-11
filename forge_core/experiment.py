"""Experiment tracking for Forge.

File-based experiment tracker with zero external dependencies.
Each experiment is stored as a JSON file under experiments/.

Stored experiment format:
    experiments/exp_XXXXX.json — experiment ID, config, metrics, status
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forge_core.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


def _get_git_commit() -> str | None:
    """Return the current git HEAD commit hash, or None on any failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


class ExperimentTracker:
    """Track ML experiments with JSON files on disk.

    Each experiment gets its own ``exp_XXXXX.json`` file under *experiments_dir*.
    """

    def __init__(self, experiments_dir: Path | None = None) -> None:
        self.experiments_dir = experiments_dir or PROJECT_ROOT / "experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    # ── ID generation ───────────────────────────────────────

    def _get_next_id(self) -> str:
        """Return the next sequential experiment ID (``exp_XXXXX``)."""
        existing = [
            f.stem for f in self.experiments_dir.iterdir()
            if f.is_file() and f.name.startswith("exp_") and f.suffix == ".json"
        ]
        if not existing:
            return "exp_00001"
        ids: list[int] = []
        for name in existing:
            try:
                ids.append(int(name.split("_")[1]))
            except (IndexError, ValueError):
                pass
        next_id = max(ids, default=0) + 1
        return f"exp_{next_id:05d}"

    # ── Persistence helpers ─────────────────────────────────

    def _load_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Load an experiment record from disk."""
        path = self.experiments_dir / f"{experiment_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Experiment not found: {experiment_id}")
        with open(path, "r") as f:
            return json.load(f)

    def _save_experiment(self, record: dict[str, Any]) -> None:
        """Write an experiment record to disk."""
        path = self.experiments_dir / f"{record['experiment_id']}.json"
        with open(path, "w") as f:
            json.dump(record, f, indent=2)

    # ── Public API ──────────────────────────────────────────

    def create_experiment(
        self,
        config: dict[str, Any],
        dataset_info: dict[str, Any],
        name: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Create a new experiment and return its ID."""
        experiment_id = self._get_next_id()
        record: dict[str, Any] = {
            "experiment_id": experiment_id,
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "git_commit": _get_git_commit(),
            "config": config,
            "dataset_info": dataset_info,
            "metrics": {},
            "model_path": None,
            "duration_seconds": None,
            "tags": tags or [],
            "error": None,
        }
        self._save_experiment(record)
        logger.info("experiment_created: %s", experiment_id)
        return experiment_id

    def complete_experiment(
        self,
        experiment_id: str,
        model_path: str,
        metrics: dict[str, Any],
        duration_seconds: float,
    ) -> dict[str, Any]:
        """Mark an experiment as completed with results."""
        record = self._load_experiment(experiment_id)
        record["status"] = "completed"
        record["model_path"] = model_path
        record["metrics"] = metrics
        record["duration_seconds"] = duration_seconds
        self._save_experiment(record)
        logger.info("experiment_completed: %s", experiment_id)
        return record

    def fail_experiment(self, experiment_id: str, error: str) -> None:
        """Mark an experiment as failed with an error message."""
        record = self._load_experiment(experiment_id)
        record["status"] = "failed"
        record["error"] = error
        self._save_experiment(record)
        logger.info("experiment_failed: %s", experiment_id)

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Return the full experiment record."""
        return self._load_experiment(experiment_id)

    def list_experiments(
        self,
        status: str | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        """List experiments, optionally filtered by status and/or tag.

        Returns experiments sorted by timestamp descending.
        """
        experiments: list[dict[str, Any]] = []
        for path in self.experiments_dir.iterdir():
            if not (path.is_file() and path.name.startswith("exp_") and path.suffix == ".json"):
                continue
            with open(path, "r") as f:
                record = json.load(f)
            if status is not None and record.get("status") != status:
                continue
            if tag is not None and tag not in record.get("tags", []):
                continue
            experiments.append(record)
        experiments.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
        return experiments

    def compare_experiments(self, experiment_ids: list[str]) -> dict[str, Any]:
        """Compare multiple experiments side by side.

        Returns a dict with an ``experiments`` key containing a list of
        summary dicts suitable for tabular display.
        """
        summaries: list[dict[str, Any]] = []
        for eid in experiment_ids:
            record = self._load_experiment(eid)
            metrics = record.get("metrics", {})
            summaries.append({
                "experiment_id": record["experiment_id"],
                "name": record.get("name"),
                "status": record.get("status"),
                "final_loss": metrics.get("final_loss"),
                "final_val_loss": metrics.get("final_val_loss"),
                "epochs": record.get("config", {}).get("epochs"),
                "duration_seconds": record.get("duration_seconds"),
            })
        return {"experiments": summaries}

    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment's JSON file from disk."""
        path = self.experiments_dir / f"{experiment_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Experiment not found: {experiment_id}")
        path.unlink()
        logger.info("experiment_deleted: %s", experiment_id)
