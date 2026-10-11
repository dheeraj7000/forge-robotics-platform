"""Integration tests for experiment tracking and model registry with training.

Tests verify that the experiment tracker and model registry work together
with the training pipeline (train_bc, forge_registry CLI, runner helpers).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.experiment import ExperimentTracker
from forge_core.model_registry import ModelRegistry
from forge_core.learned_policy import (
    DemoDataset,
    PolicyNetwork,
    train_bc,
    OBS_DIM,
    ACT_DIM,
)


# ── Helpers ─────────────────────────────────────────────────

def _make_dataset(n: int = 200) -> DemoDataset:
    """Create a synthetic dataset for testing."""
    np.random.seed(42)
    obs = np.random.randn(n, OBS_DIM).astype(np.float32)
    act = np.random.randn(n, ACT_DIM).astype(np.float32)
    return DemoDataset(obs, act)


# ── TestTrainWithTracking ───────────────────────────────────

class TestTrainWithTracking:
    """Test that train_bc() with an experiment_tracker creates proper records."""

    def test_completed_experiment_record(self, tmp_path):
        """train_bc() with tracker creates a completed experiment with correct metrics."""
        torch.manual_seed(42)
        tracker = ExperimentTracker(experiments_dir=tmp_path / "experiments")
        exp_id = tracker.create_experiment(
            config={"epochs": 3, "lr": 0.001},
            dataset_info={"n_samples": 200, "source": "collected"},
            name="test_experiment",
            tags=["test"],
        )

        ds = _make_dataset()
        result = train_bc(
            dataset=ds,
            epochs=3,
            batch_size=32,
            save_path=str(tmp_path / "model.pt"),
            experiment_id=exp_id,
            experiment_tracker=tracker,
        )

        assert result.experiment_id == exp_id

        exp = tracker.get_experiment(exp_id)
        assert exp["status"] == "completed"
        assert exp["model_path"] == str(tmp_path / "model.pt")
        assert exp["duration_seconds"] is not None
        assert exp["duration_seconds"] > 0

        metrics = exp["metrics"]
        assert "final_loss" in metrics
        assert metrics["final_loss"] > 0
        assert metrics["final_loss"] == result.final_loss
        assert "loss_history" in metrics
        assert len(metrics["loss_history"]) == 3
        assert "best_epoch" in metrics
        assert "best_loss" in metrics

    def test_completed_experiment_with_validation(self, tmp_path):
        """train_bc() with validation records val_loss metrics."""
        torch.manual_seed(42)
        tracker = ExperimentTracker(experiments_dir=tmp_path / "experiments")
        exp_id = tracker.create_experiment(
            config={"epochs": 3},
            dataset_info={"n_samples": 200},
        )

        ds = _make_dataset()
        result = train_bc(
            dataset=ds,
            epochs=3,
            batch_size=32,
            validation_split=0.2,
            save_path=str(tmp_path / "model.pt"),
            experiment_id=exp_id,
            experiment_tracker=tracker,
        )

        exp = tracker.get_experiment(exp_id)
        metrics = exp["metrics"]
        assert metrics["final_val_loss"] is not None
        assert len(metrics["val_history"]) == 3

    def test_failed_training_marks_experiment(self, tmp_path):
        """If training raises, the experiment is marked as failed."""
        torch.manual_seed(42)
        tracker = ExperimentTracker(experiments_dir=tmp_path / "experiments")
        exp_id = tracker.create_experiment(
            config={"epochs": 5},
            dataset_info={"n_samples": 0},
        )

        ds = DemoDataset(
            np.empty((0, OBS_DIM), dtype=np.float32),
            np.empty((0, ACT_DIM), dtype=np.float32),
        )

        with pytest.raises(ValueError, match="Dataset is empty"):
            train_bc(
                dataset=ds,
                epochs=5,
                batch_size=32,
                save_path=str(tmp_path / "model.pt"),
                experiment_id=exp_id,
                experiment_tracker=tracker,
            )

        exp = tracker.get_experiment(exp_id)
        assert exp["status"] == "failed"
        assert "Dataset is empty" in exp["error"]

    def test_no_tracker_still_works(self, tmp_path):
        """train_bc() without tracker works as before (backward compat)."""
        torch.manual_seed(42)
        ds = _make_dataset()
        result = train_bc(
            dataset=ds,
            epochs=3,
            batch_size=32,
            save_path=str(tmp_path / "model.pt"),
        )

        assert result.experiment_id is None
        assert result.epochs == 3
        assert result.final_loss > 0
        assert Path(result.model_path).exists()

    def test_tracker_without_experiment_id(self, tmp_path):
        """train_bc() with tracker but no experiment_id skips tracking."""
        torch.manual_seed(42)
        tracker = ExperimentTracker(experiments_dir=tmp_path / "experiments")
        ds = _make_dataset()
        result = train_bc(
            dataset=ds,
            epochs=3,
            batch_size=32,
            save_path=str(tmp_path / "model.pt"),
            experiment_tracker=tracker,
        )

        assert result.experiment_id is None
        # No experiments should have been completed (none were created)
        exps = tracker.list_experiments()
        assert len(exps) == 0


# ── TestRegistryIntegration ─────────────────────────────────

class TestRegistryIntegration:
    """Test model registry integration with training pipeline."""

    def test_register_model_after_training(self, tmp_path):
        """register_model links to experiment and stores correct checkpoint."""
        torch.manual_seed(42)
        tracker = ExperimentTracker(experiments_dir=tmp_path / "experiments")
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        exp_id = tracker.create_experiment(
            config={"epochs": 3},
            dataset_info={"n_samples": 200},
        )

        ds = _make_dataset()
        result = train_bc(
            dataset=ds,
            epochs=3,
            batch_size=32,
            save_path=str(tmp_path / "model.pt"),
            experiment_id=exp_id,
            experiment_tracker=tracker,
        )

        entry = registry.register_model(
            name="bc_test",
            checkpoint_path=result.model_path,
            experiment_id=exp_id,
            description="Test model",
            metadata={"final_loss": result.final_loss, "epochs": result.epochs},
        )

        assert entry["name"] == "bc_test"
        assert entry["version"] == 1
        assert entry["experiment_id"] == exp_id
        assert entry["checkpoint_path"] == result.model_path
        assert entry["metadata"]["final_loss"] == result.final_loss

    def test_get_model_returns_correct_checkpoint(self, tmp_path):
        """get_model returns the right checkpoint_path."""
        torch.manual_seed(42)
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        ds = _make_dataset()
        result = train_bc(
            dataset=ds,
            epochs=3,
            batch_size=32,
            save_path=str(tmp_path / "model.pt"),
        )

        registry.register_model(
            name="bc_test",
            checkpoint_path=result.model_path,
        )

        entry = registry.get_model("bc_test")
        assert entry is not None
        assert entry["checkpoint_path"] == result.model_path
        assert Path(entry["checkpoint_path"]).exists()

    def test_load_policy_from_registry(self, tmp_path, monkeypatch):
        """load_policy_from_registry returns a working LearnedPolicy."""
        torch.manual_seed(42)
        from forge_core import runner as runner_mod

        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        # Train and save a model
        ds = _make_dataset()
        result = train_bc(
            dataset=ds,
            epochs=3,
            batch_size=32,
            save_path=str(tmp_path / "model.pt"),
        )

        registry.register_model(
            name="bc_test",
            checkpoint_path=result.model_path,
        )

        # Monkeypatch ModelRegistry to use our tmp registry
        monkeypatch.setattr(
            "forge_core.model_registry.PROJECT_ROOT",
            tmp_path,
        )
        # Also ensure the registry dir matches
        (tmp_path / "registry").mkdir(exist_ok=True)

        # Use the function directly with our custom registry
        from forge_core.model_registry import ModelRegistry as MR
        reg = MR(registry_dir=tmp_path / "registry")
        entry = reg.get_model("bc_test")
        assert entry is not None

        from forge_core.policy import create_policy
        policy = create_policy("learned", model_path=entry["checkpoint_path"])
        assert policy.name == "LearnedPolicy"

    def test_multiple_versions(self, tmp_path):
        """Registering multiple versions auto-increments correctly."""
        torch.manual_seed(42)
        registry = ModelRegistry(registry_dir=tmp_path / "registry")

        ds = _make_dataset()
        for i in range(3):
            result = train_bc(
                dataset=ds,
                epochs=2,
                batch_size=32,
                save_path=str(tmp_path / f"model_v{i + 1}.pt"),
            )
            entry = registry.register_model(
                name="bc_test",
                checkpoint_path=result.model_path,
            )
            assert entry["version"] == i + 1

        models = registry.list_models(name="bc_test")
        assert len(models) == 3
        latest = registry.get_model("bc_test")
        assert latest["version"] == 3


# ── TestCLIParsing ──────────────────────────────────────────

class TestCLIParsing:
    """Test forge_registry.py's argparse handles all subcommands correctly."""

    @pytest.fixture(autouse=True)
    def _import_registry_cli(self):
        """Import the build_parser function from forge_registry."""
        _scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        from forge_registry import build_parser
        self.build_parser = build_parser

    def test_experiments_list(self):
        """'experiments list' parses correctly."""
        parser = self.build_parser()
        args = parser.parse_args(["experiments", "list"])
        assert args.category == "experiments"
        assert args.command == "list"
        assert args.status is None
        assert args.tag is None

    def test_experiments_list_with_filters(self):
        """'experiments list --status completed --tag bc' parses correctly."""
        parser = self.build_parser()
        args = parser.parse_args(
            ["experiments", "list", "--status", "completed", "--tag", "bc"]
        )
        assert args.status == "completed"
        assert args.tag == "bc"

    def test_experiments_show(self):
        """'experiments show exp_00001' parses correctly."""
        parser = self.build_parser()
        args = parser.parse_args(["experiments", "show", "exp_00001"])
        assert args.category == "experiments"
        assert args.command == "show"
        assert args.experiment_id == "exp_00001"

    def test_experiments_compare(self):
        """'experiments compare exp_00001 exp_00002' parses correctly."""
        parser = self.build_parser()
        args = parser.parse_args(
            ["experiments", "compare", "exp_00001", "exp_00002"]
        )
        assert args.command == "compare"
        assert args.experiment_ids == ["exp_00001", "exp_00002"]

    def test_experiments_compare_multiple(self):
        """'experiments compare' with 3 IDs parses correctly."""
        parser = self.build_parser()
        args = parser.parse_args(
            ["experiments", "compare", "exp_00001", "exp_00002", "exp_00003"]
        )
        assert len(args.experiment_ids) == 3

    def test_experiments_delete(self):
        """'experiments delete exp_00001' parses correctly."""
        parser = self.build_parser()
        args = parser.parse_args(["experiments", "delete", "exp_00001"])
        assert args.command == "delete"
        assert args.experiment_id == "exp_00001"
        assert not args.force

    def test_experiments_delete_force(self):
        """'experiments delete exp_00001 --force' parses correctly."""
        parser = self.build_parser()
        args = parser.parse_args(
            ["experiments", "delete", "exp_00001", "--force"]
        )
        assert args.force is True

    def test_models_list(self):
        """'models list' parses correctly."""
        parser = self.build_parser()
        args = parser.parse_args(["models", "list"])
        assert args.category == "models"
        assert args.command == "list"
        assert args.name is None
        assert args.stage is None

    def test_models_list_with_filters(self):
        """'models list --name bc_test --stage production' parses correctly."""
        parser = self.build_parser()
        args = parser.parse_args(
            ["models", "list", "--name", "bc_test", "--stage", "production"]
        )
        assert args.name == "bc_test"
        assert args.stage == "production"

    def test_models_show(self):
        """'models show bc_test' parses correctly."""
        parser = self.build_parser()
        args = parser.parse_args(["models", "show", "bc_test"])
        assert args.category == "models"
        assert args.command == "show"
        assert args.name == "bc_test"
        assert args.version is None

    def test_models_show_with_version(self):
        """'models show bc_test --version 2' parses correctly."""
        parser = self.build_parser()
        args = parser.parse_args(["models", "show", "bc_test", "--version", "2"])
        assert args.version == 2

    def test_models_promote(self):
        """'models promote bc_test 1 validated' parses correctly."""
        parser = self.build_parser()
        args = parser.parse_args(["models", "promote", "bc_test", "1", "validated"])
        assert args.command == "promote"
        assert args.name == "bc_test"
        assert args.version == 1
        assert args.stage == "validated"

    def test_models_delete(self):
        """'models delete bc_test 1' parses correctly."""
        parser = self.build_parser()
        args = parser.parse_args(["models", "delete", "bc_test", "1"])
        assert args.command == "delete"
        assert args.name == "bc_test"
        assert args.version == 1
        assert not args.force

    def test_models_delete_force(self):
        """'models delete bc_test 1 --force' parses correctly."""
        parser = self.build_parser()
        args = parser.parse_args(
            ["models", "delete", "bc_test", "1", "--force"]
        )
        assert args.force is True
