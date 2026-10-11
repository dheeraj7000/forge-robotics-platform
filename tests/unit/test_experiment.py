"""Unit tests for the experiment tracking system."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.experiment import ExperimentTracker, _get_git_commit


# ── Helpers ─────────────────────────────────────────────────

def _sample_config() -> dict:
    return {"epochs": 10, "batch_size": 32, "lr": 0.001}


def _sample_dataset_info() -> dict:
    return {
        "n_samples": 200,
        "source": "collected",
        "source_runs": [],
        "target_object": "red_cube",
        "task": "pick_red_cube",
    }


def _sample_metrics() -> dict:
    return {
        "final_loss": 0.05,
        "final_val_loss": 0.07,
        "loss_history": [0.5, 0.3, 0.1, 0.05],
        "val_history": [0.6, 0.4, 0.15, 0.07],
        "best_epoch": 4,
        "best_loss": 0.05,
    }


# ── TestExperimentCreation ──────────────────────────────────

class TestExperimentCreation:
    def test_create_experiment_returns_id(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        eid = tracker.create_experiment(_sample_config(), _sample_dataset_info())
        assert eid.startswith("exp_")

    def test_create_writes_json_file(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        eid = tracker.create_experiment(_sample_config(), _sample_dataset_info())
        path = tmp_path / f"{eid}.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["experiment_id"] == eid

    def test_sequential_ids(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        id1 = tracker.create_experiment(_sample_config(), _sample_dataset_info())
        id2 = tracker.create_experiment(_sample_config(), _sample_dataset_info())
        assert id1 == "exp_00001"
        assert id2 == "exp_00002"

    def test_create_with_name_and_tags(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        eid = tracker.create_experiment(
            _sample_config(),
            _sample_dataset_info(),
            name="baseline_v1",
            tags=["baseline", "lr-search"],
        )
        record = tracker.get_experiment(eid)
        assert record["name"] == "baseline_v1"
        assert "baseline" in record["tags"]
        assert "lr-search" in record["tags"]

    def test_initial_status_is_running(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        eid = tracker.create_experiment(_sample_config(), _sample_dataset_info())
        record = tracker.get_experiment(eid)
        assert record["status"] == "running"

    def test_git_commit_captured(self, tmp_path, monkeypatch):
        """Monkeypatch subprocess to verify git commit capture."""
        fake_hash = "abc123def456"

        def mock_run(*args, **kwargs):
            class FakeResult:
                returncode = 0
                stdout = fake_hash + "\n"
            return FakeResult()

        import forge_core.experiment as exp_mod
        monkeypatch.setattr(subprocess, "run", mock_run)

        tracker = ExperimentTracker(experiments_dir=tmp_path)
        eid = tracker.create_experiment(_sample_config(), _sample_dataset_info())
        record = tracker.get_experiment(eid)
        assert record["git_commit"] == fake_hash


# ── TestExperimentLifecycle ─────────────────────────────────

class TestExperimentLifecycle:
    def test_complete_experiment(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        eid = tracker.create_experiment(_sample_config(), _sample_dataset_info())
        result = tracker.complete_experiment(
            eid, model_path="models/bc_policy.pt",
            metrics=_sample_metrics(), duration_seconds=42.5,
        )
        assert result["status"] == "completed"

    def test_complete_sets_model_path_and_metrics(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        eid = tracker.create_experiment(_sample_config(), _sample_dataset_info())
        tracker.complete_experiment(
            eid, model_path="models/bc_policy.pt",
            metrics=_sample_metrics(), duration_seconds=10.0,
        )
        record = tracker.get_experiment(eid)
        assert record["model_path"] == "models/bc_policy.pt"
        assert record["metrics"]["final_loss"] == 0.05
        assert record["metrics"]["final_val_loss"] == 0.07
        assert record["duration_seconds"] == 10.0

    def test_fail_experiment(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        eid = tracker.create_experiment(_sample_config(), _sample_dataset_info())
        tracker.fail_experiment(eid, error="CUDA out of memory")
        record = tracker.get_experiment(eid)
        assert record["status"] == "failed"

    def test_fail_sets_error_message(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        eid = tracker.create_experiment(_sample_config(), _sample_dataset_info())
        tracker.fail_experiment(eid, error="NaN in loss at epoch 3")
        record = tracker.get_experiment(eid)
        assert record["error"] == "NaN in loss at epoch 3"

    def test_get_experiment(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        eid = tracker.create_experiment(_sample_config(), _sample_dataset_info())
        record = tracker.get_experiment(eid)
        assert record["experiment_id"] == eid
        assert record["config"] == _sample_config()
        assert record["dataset_info"] == _sample_dataset_info()

    def test_get_nonexistent_raises(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            tracker.get_experiment("exp_99999")


# ── TestExperimentListing ───────────────────────────────────

class TestExperimentListing:
    def test_list_empty(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        assert tracker.list_experiments() == []

    def test_list_all(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        tracker.create_experiment(_sample_config(), _sample_dataset_info())
        tracker.create_experiment(_sample_config(), _sample_dataset_info())
        experiments = tracker.list_experiments()
        assert len(experiments) == 2

    def test_list_filter_by_status(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        eid1 = tracker.create_experiment(_sample_config(), _sample_dataset_info())
        eid2 = tracker.create_experiment(_sample_config(), _sample_dataset_info())
        tracker.complete_experiment(
            eid1, model_path="m.pt", metrics=_sample_metrics(), duration_seconds=1.0,
        )
        running = tracker.list_experiments(status="running")
        completed = tracker.list_experiments(status="completed")
        assert len(running) == 1
        assert len(completed) == 1
        assert running[0]["experiment_id"] == eid2
        assert completed[0]["experiment_id"] == eid1

    def test_list_filter_by_tag(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        tracker.create_experiment(
            _sample_config(), _sample_dataset_info(), tags=["baseline"],
        )
        tracker.create_experiment(
            _sample_config(), _sample_dataset_info(), tags=["lr-search"],
        )
        baseline = tracker.list_experiments(tag="baseline")
        assert len(baseline) == 1
        assert "baseline" in baseline[0]["tags"]

    def test_list_sorted_by_timestamp(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        eid1 = tracker.create_experiment(_sample_config(), _sample_dataset_info())
        # Tiny sleep so timestamps differ
        time.sleep(0.01)
        eid2 = tracker.create_experiment(_sample_config(), _sample_dataset_info())
        experiments = tracker.list_experiments()
        # Most recent first
        assert experiments[0]["experiment_id"] == eid2
        assert experiments[1]["experiment_id"] == eid1


# ── TestExperimentComparison ────────────────────────────────

class TestExperimentComparison:
    def test_compare_two_experiments(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        eid1 = tracker.create_experiment(
            {"epochs": 10}, _sample_dataset_info(), name="run_a",
        )
        eid2 = tracker.create_experiment(
            {"epochs": 20}, _sample_dataset_info(), name="run_b",
        )
        tracker.complete_experiment(
            eid1, model_path="m1.pt", metrics=_sample_metrics(), duration_seconds=10.0,
        )
        tracker.complete_experiment(
            eid2, model_path="m2.pt",
            metrics={"final_loss": 0.03, "final_val_loss": 0.04},
            duration_seconds=20.0,
        )
        result = tracker.compare_experiments([eid1, eid2])
        assert "experiments" in result
        assert len(result["experiments"]) == 2
        assert result["experiments"][0]["experiment_id"] == eid1
        assert result["experiments"][0]["final_loss"] == 0.05
        assert result["experiments"][1]["final_loss"] == 0.03

    def test_compare_nonexistent_raises(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            tracker.compare_experiments(["exp_99999"])


# ── TestExperimentDeletion ──────────────────────────────────

class TestExperimentDeletion:
    def test_delete_experiment(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        eid = tracker.create_experiment(_sample_config(), _sample_dataset_info())
        tracker.delete_experiment(eid)
        assert not (tmp_path / f"{eid}.json").exists()

    def test_delete_nonexistent_raises(self, tmp_path):
        tracker = ExperimentTracker(experiments_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            tracker.delete_experiment("exp_99999")
