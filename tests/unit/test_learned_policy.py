"""Unit tests for the learned policy system."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Add scripts/ to sys.path for TestConfigMerge
_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from forge_core.simulation import ForgeSimulation
from forge_core.learned_policy import (
    extract_observation,
    PolicyNetwork,
    LearnedPolicy,
    DemoDataset,
    train_bc,
    collect_demos,
    load_training_config,
    OBS_DIM,
    ACT_DIM,
)
from forge_core.policy import create_policy


@pytest.fixture
def sim():
    s = ForgeSimulation("basic_workspace")
    s.step(100)
    yield s
    s.shutdown()


class TestObservation:
    def test_observation_shape(self, sim):
        obs = extract_observation(sim)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32

    def test_observation_is_finite(self, sim):
        obs = extract_observation(sim)
        assert np.all(np.isfinite(obs))

    def test_observation_changes_after_step(self, sim):
        obs1 = extract_observation(sim)
        sim.set_joint_targets(np.array([0.5, 0, 0, -1.5, 0, 1.5, 0.7, 0]))
        sim.step(200)
        obs2 = extract_observation(sim)
        assert not np.allclose(obs1, obs2)


class TestPolicyNetwork:
    def test_forward_shape(self):
        net = PolicyNetwork()
        obs = torch.randn(1, OBS_DIM)
        action = net(obs)
        assert action.shape == (1, ACT_DIM)

    def test_batch_forward(self):
        net = PolicyNetwork()
        obs = torch.randn(16, OBS_DIM)
        actions = net(obs)
        assert actions.shape == (16, ACT_DIM)

    def test_custom_hidden(self):
        net = PolicyNetwork(hidden=[64, 64])
        obs = torch.randn(1, OBS_DIM)
        action = net(obs)
        assert action.shape == (1, ACT_DIM)


class TestLearnedPolicy:
    def test_policy_interface(self, sim):
        policy = LearnedPolicy(max_steps=10)
        policy.reset(sim)
        assert not policy.done

        action = policy.act(sim)
        assert len(action) == ACT_DIM

    def test_policy_terminates(self, sim):
        policy = LearnedPolicy(max_steps=5)
        policy.reset(sim)
        for _ in range(10):
            policy.act(sim)
        assert policy.done

    def test_create_learned_policy(self):
        policy = create_policy("learned", max_steps=10)
        assert policy.name == "LearnedPolicy"

    def test_save_and_load(self, sim, tmp_path):
        # Save
        net = PolicyNetwork()
        path = str(tmp_path / "test_model.pt")
        torch.save(net.state_dict(), path)

        # Load and run
        policy = LearnedPolicy(model_path=path, max_steps=5)
        policy.reset(sim)
        action = policy.act(sim)
        assert len(action) == ACT_DIM


class TestDemoDataset:
    def test_from_arrays(self):
        obs = np.random.randn(100, OBS_DIM).astype(np.float32)
        act = np.random.randn(100, ACT_DIM).astype(np.float32)
        ds = DemoDataset(obs, act)
        assert len(ds) == 100
        o, a = ds[0]
        assert o.shape == (OBS_DIM,)
        assert a.shape == (ACT_DIM,)


class TestBehavioralCloning:
    def test_train_bc(self, tmp_path):
        # Synthetic dataset
        obs = np.random.randn(200, OBS_DIM).astype(np.float32)
        act = np.random.randn(200, ACT_DIM).astype(np.float32)
        ds = DemoDataset(obs, act)

        result = train_bc(
            dataset=ds,
            epochs=5,
            batch_size=32,
            lr=1e-3,
            save_path=str(tmp_path / "test_bc.pt"),
        )

        assert result.epochs == 5
        assert result.final_loss > 0
        assert Path(result.model_path).exists()
        assert len(result.history) == 5

    def test_collect_demos(self):
        from forge_core.policy import NullPolicy
        ds = collect_demos("pick_red_cube", NullPolicy(), n_episodes=1)
        assert len(ds) > 0
        o, a = ds[0]
        assert o.shape == (OBS_DIM,)


# ── Shared fixtures for new test classes ────────────────────

@pytest.fixture
def synthetic_run_dir(tmp_path):
    """Create a temporary run directory with synthetic trajectory data."""
    T = 50
    np.random.seed(42)
    data = {
        "joint_positions": np.random.randn(T, 9).astype(np.float32),
        "joint_velocities": np.random.randn(T, 9).astype(np.float32),
        "ee_positions": np.random.randn(T, 3).astype(np.float32),
        "ee_orientations": np.random.randn(T, 4).astype(np.float32),
        "actions": np.random.randn(T, 8).astype(np.float32),
        "object_names": np.array(["red_cube", "target_bin"]),
        "obj_red_cube_pos": np.random.randn(T, 3).astype(np.float32),
        "obj_target_bin_pos": np.random.randn(T, 3).astype(np.float32),
    }
    np.savez_compressed(tmp_path / "trajectory.npz", **data)
    return tmp_path


# ── TestTrainingConfig ──────────────────────────────────────

class TestTrainingConfig:
    def test_load_default_config(self):
        """load_training_config() returns a dict with expected keys."""
        torch.manual_seed(42)
        np.random.seed(42)
        config = load_training_config()
        assert isinstance(config, dict)
        assert "training" in config
        assert "model" in config
        assert "data" in config
        assert config["training"]["epochs"] == 100

    def test_load_custom_config(self, tmp_path):
        """load_training_config(path) loads values from a custom file."""
        torch.manual_seed(42)
        np.random.seed(42)
        custom = tmp_path / "custom.yaml"
        custom.write_text(
            "training:\n  epochs: 50\n  batch_size: 32\n  learning_rate: 0.01\n"
            "  validation_split: 0.2\n  checkpoint_interval: 10\n"
            "model:\n  hidden_layers: [128, 128]\n  obs_dim: 31\n  act_dim: 8\n"
        )
        config = load_training_config(custom)
        assert config["training"]["epochs"] == 50
        assert config["training"]["batch_size"] == 32
        assert config["model"]["hidden_layers"] == [128, 128]

    def test_load_missing_explicit_raises(self):
        """load_training_config(nonexistent) raises FileNotFoundError."""
        torch.manual_seed(42)
        np.random.seed(42)
        with pytest.raises(FileNotFoundError):
            load_training_config("/nonexistent/training.yaml")

    def test_load_missing_default_returns_empty(self, tmp_path, monkeypatch):
        """When default config is absent, returns {}."""
        torch.manual_seed(42)
        np.random.seed(42)
        import forge_core.learned_policy as lp
        monkeypatch.setattr(lp, "PROJECT_ROOT", tmp_path)
        config = load_training_config()
        assert config == {}

    def test_invalid_epochs_raises(self, tmp_path):
        """Config with epochs: -1 raises ValueError."""
        torch.manual_seed(42)
        np.random.seed(42)
        bad = tmp_path / "bad.yaml"
        bad.write_text("training:\n  epochs: -1\n")
        with pytest.raises(ValueError, match="training.epochs"):
            load_training_config(bad)

    def test_invalid_validation_split_raises(self, tmp_path):
        """Config with validation_split: 0.8 raises ValueError."""
        torch.manual_seed(42)
        np.random.seed(42)
        bad = tmp_path / "bad.yaml"
        bad.write_text("training:\n  validation_split: 0.8\n")
        with pytest.raises(ValueError, match="training.validation_split"):
            load_training_config(bad)

    def test_invalid_lr_raises(self, tmp_path):
        """Config with learning_rate: -0.01 raises ValueError."""
        torch.manual_seed(42)
        np.random.seed(42)
        bad = tmp_path / "bad.yaml"
        bad.write_text("training:\n  learning_rate: -0.01\n")
        with pytest.raises(ValueError, match="training.learning_rate"):
            load_training_config(bad)


# ── TestConfigMerge ─────────────────────────────────────────

class TestConfigMerge:
    def test_code_defaults_used_when_no_yaml_no_cli(self):
        """merge_training_config(defaults, {}, {}) returns defaults."""
        torch.manual_seed(42)
        np.random.seed(42)
        from forge_train import merge_training_config, CODE_DEFAULTS
        result = merge_training_config(CODE_DEFAULTS, {}, {})
        assert result == CODE_DEFAULTS

    def test_yaml_overrides_defaults(self):
        """YAML values override code defaults."""
        torch.manual_seed(42)
        np.random.seed(42)
        from forge_train import merge_training_config, CODE_DEFAULTS
        result = merge_training_config(CODE_DEFAULTS, {"epochs": 50}, {})
        assert result["epochs"] == 50
        assert result["batch_size"] == CODE_DEFAULTS["batch_size"]

    def test_cli_overrides_yaml(self):
        """CLI values override YAML values."""
        torch.manual_seed(42)
        np.random.seed(42)
        from forge_train import merge_training_config, CODE_DEFAULTS
        result = merge_training_config(CODE_DEFAULTS, {"epochs": 50}, {"epochs": 25})
        assert result["epochs"] == 25

    def test_cli_overrides_defaults_without_yaml(self):
        """CLI values override code defaults even without YAML."""
        torch.manual_seed(42)
        np.random.seed(42)
        from forge_train import merge_training_config, CODE_DEFAULTS
        result = merge_training_config(CODE_DEFAULTS, {}, {"lr": 0.0005})
        assert result["lr"] == 0.0005

    def test_partial_yaml_partial_cli(self):
        """YAML sets epochs/batch_size, CLI sets lr — all reflected."""
        torch.manual_seed(42)
        np.random.seed(42)
        from forge_train import merge_training_config, CODE_DEFAULTS
        result = merge_training_config(
            CODE_DEFAULTS,
            {"epochs": 50, "batch_size": 128},
            {"lr": 0.0005},
        )
        assert result["epochs"] == 50
        assert result["batch_size"] == 128
        assert result["lr"] == 0.0005
        assert result["validation_split"] == CODE_DEFAULTS["validation_split"]


# ── TestValidationMetrics ───────────────────────────────────

class TestValidationMetrics:
    def _make_dataset(self, n: int = 200) -> DemoDataset:
        np.random.seed(42)
        obs = np.random.randn(n, OBS_DIM).astype(np.float32)
        act = np.random.randn(n, ACT_DIM).astype(np.float32)
        return DemoDataset(obs, act)

    def test_train_bc_with_validation(self, tmp_path):
        """train_bc with validation_split returns val_history of correct length."""
        torch.manual_seed(42)
        np.random.seed(42)
        ds = self._make_dataset(200)
        result = train_bc(
            dataset=ds,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            save_path=str(tmp_path / "model.pt"),
        )
        assert len(result.val_history) == 5
        assert all(v > 0 for v in result.val_history)

    def test_train_bc_no_validation(self, tmp_path):
        """train_bc without validation returns empty val_history."""
        torch.manual_seed(42)
        np.random.seed(42)
        ds = self._make_dataset(200)
        result = train_bc(
            dataset=ds,
            epochs=5,
            batch_size=32,
            save_path=str(tmp_path / "model.pt"),
        )
        assert result.val_history == []

    def test_summary_includes_val_loss(self, tmp_path):
        """summary() contains 'val_loss=' when val_history is non-empty."""
        torch.manual_seed(42)
        np.random.seed(42)
        ds = self._make_dataset(200)
        result = train_bc(
            dataset=ds,
            epochs=3,
            batch_size=32,
            validation_split=0.2,
            save_path=str(tmp_path / "model.pt"),
        )
        assert "val_loss=" in result.summary()

    def test_summary_without_val_loss(self, tmp_path):
        """summary() does not contain 'val_loss=' when val_history is empty."""
        torch.manual_seed(42)
        np.random.seed(42)
        ds = self._make_dataset(200)
        result = train_bc(
            dataset=ds,
            epochs=3,
            batch_size=32,
            save_path=str(tmp_path / "model.pt"),
        )
        assert "val_loss=" not in result.summary()

    def test_validation_split_too_large_raises(self, tmp_path):
        """train_bc on a 3-sample dataset with validation_split=0.2 raises ValueError."""
        torch.manual_seed(42)
        np.random.seed(42)
        ds = self._make_dataset(3)
        # 3 * 0.2 = 0.6 -> max(1, 0) = 1; train = 3 - 1 = 2 -> should work
        # We need a truly too-small dataset: 1 sample with split=0.5
        ds_tiny = self._make_dataset(1)
        with pytest.raises(ValueError, match="Dataset too small"):
            train_bc(
                dataset=ds_tiny,
                epochs=2,
                validation_split=0.5,
                save_path=str(tmp_path / "model.pt"),
            )

    def test_validation_split_out_of_range_raises(self, tmp_path):
        """train_bc with validation_split=0.8 raises ValueError."""
        torch.manual_seed(42)
        np.random.seed(42)
        ds = self._make_dataset(200)
        with pytest.raises(ValueError, match="validation_split must be in"):
            train_bc(
                dataset=ds,
                epochs=2,
                validation_split=0.8,
                save_path=str(tmp_path / "model.pt"),
            )

    def test_small_dataset_gets_at_least_one_val_sample(self, tmp_path):
        """train_bc on 10-sample dataset with validation_split=0.05 succeeds."""
        torch.manual_seed(42)
        np.random.seed(42)
        ds = self._make_dataset(10)
        result = train_bc(
            dataset=ds,
            epochs=2,
            batch_size=4,
            validation_split=0.05,
            save_path=str(tmp_path / "model.pt"),
        )
        # 10 * 0.05 = 0.5 -> max(1, 0) = 1 val sample
        assert len(result.val_history) == 2
        assert all(v > 0 for v in result.val_history)


# ── TestCheckpointing ──────────────────────────────────────

class TestCheckpointing:
    def _make_dataset(self, n: int = 200) -> DemoDataset:
        np.random.seed(42)
        obs = np.random.randn(n, OBS_DIM).astype(np.float32)
        act = np.random.randn(n, ACT_DIM).astype(np.float32)
        return DemoDataset(obs, act)

    def test_checkpoints_created(self, tmp_path):
        """checkpoint_interval=5 with epochs=10 creates epoch 5 and 10 files."""
        torch.manual_seed(42)
        np.random.seed(42)
        ds = self._make_dataset()
        ckpt_dir = tmp_path / "checkpoints"
        train_bc(
            dataset=ds,
            epochs=10,
            batch_size=32,
            checkpoint_interval=5,
            checkpoint_dir=str(ckpt_dir),
            save_path=str(tmp_path / "model.pt"),
        )
        assert (ckpt_dir / "bc_policy_epoch_005.pt").exists()
        assert (ckpt_dir / "bc_policy_epoch_010.pt").exists()

    def test_final_always_checkpointed(self, tmp_path):
        """Final epoch is always checkpointed even if not aligned with interval."""
        torch.manual_seed(42)
        np.random.seed(42)
        ds = self._make_dataset()
        ckpt_dir = tmp_path / "checkpoints"
        train_bc(
            dataset=ds,
            epochs=7,
            batch_size=32,
            checkpoint_interval=5,
            checkpoint_dir=str(ckpt_dir),
            save_path=str(tmp_path / "model.pt"),
        )
        assert (ckpt_dir / "bc_policy_epoch_005.pt").exists()
        assert (ckpt_dir / "bc_policy_epoch_007.pt").exists()

    def test_checkpoint_loadable(self, tmp_path):
        """A checkpoint file can be loaded by LearnedPolicy."""
        torch.manual_seed(42)
        np.random.seed(42)
        ds = self._make_dataset()
        ckpt_dir = tmp_path / "checkpoints"
        train_bc(
            dataset=ds,
            epochs=5,
            batch_size=32,
            checkpoint_interval=5,
            checkpoint_dir=str(ckpt_dir),
            save_path=str(tmp_path / "model.pt"),
        )
        ckpt_path = str(ckpt_dir / "bc_policy_epoch_005.pt")
        policy = LearnedPolicy(model_path=ckpt_path, max_steps=5)
        # Verify the network produces actions of the right shape
        obs = torch.randn(1, OBS_DIM)
        with torch.no_grad():
            action = policy._network(obs)
        assert action.shape == (1, ACT_DIM)

    def test_no_checkpoints_by_default(self, tmp_path):
        """train_bc with no checkpoint_interval does not create epoch-numbered files."""
        torch.manual_seed(42)
        np.random.seed(42)
        ds = self._make_dataset()
        train_bc(
            dataset=ds,
            epochs=5,
            batch_size=32,
            save_path=str(tmp_path / "model.pt"),
        )
        # No epoch-numbered files should exist
        epoch_files = list(tmp_path.glob("bc_policy_epoch_*.pt"))
        assert len(epoch_files) == 0

    def test_checkpoint_interval_zero_raises(self, tmp_path):
        """checkpoint_interval=0 raises ValueError."""
        torch.manual_seed(42)
        np.random.seed(42)
        ds = self._make_dataset()
        with pytest.raises(ValueError, match="checkpoint_interval must be >= 1"):
            train_bc(
                dataset=ds,
                epochs=5,
                checkpoint_interval=0,
                save_path=str(tmp_path / "model.pt"),
            )

    def test_checkpoint_interval_negative_raises(self, tmp_path):
        """checkpoint_interval=-1 raises ValueError."""
        torch.manual_seed(42)
        np.random.seed(42)
        ds = self._make_dataset()
        with pytest.raises(ValueError, match="checkpoint_interval must be >= 1"):
            train_bc(
                dataset=ds,
                epochs=5,
                checkpoint_interval=-1,
                save_path=str(tmp_path / "model.pt"),
            )


# ── TestMultiRunDataset ─────────────────────────────────────

class TestMultiRunDataset:
    def test_from_run_dirs_merges(self, tmp_path):
        """from_run_dirs merges multiple run directories."""
        torch.manual_seed(42)
        np.random.seed(42)
        # Create two synthetic run dirs
        for i in range(2):
            run_dir = tmp_path / f"run_{i}"
            run_dir.mkdir()
            T = 30 + i * 20  # 30 and 50 samples
            data = {
                "joint_positions": np.random.randn(T, 9).astype(np.float32),
                "joint_velocities": np.random.randn(T, 9).astype(np.float32),
                "ee_positions": np.random.randn(T, 3).astype(np.float32),
                "ee_orientations": np.random.randn(T, 4).astype(np.float32),
                "actions": np.random.randn(T, 8).astype(np.float32),
                "object_names": np.array(["red_cube", "target_bin"]),
                "obj_red_cube_pos": np.random.randn(T, 3).astype(np.float32),
                "obj_target_bin_pos": np.random.randn(T, 3).astype(np.float32),
            }
            np.savez_compressed(run_dir / "trajectory.npz", **data)

        dirs = [tmp_path / "run_0", tmp_path / "run_1"]
        ds = DemoDataset.from_run_dirs(dirs)
        assert len(ds) == 80  # 30 + 50

    def test_from_run_dirs_empty_raises(self):
        """from_run_dirs([]) raises ValueError."""
        torch.manual_seed(42)
        np.random.seed(42)
        with pytest.raises(ValueError, match="No run directories provided"):
            DemoDataset.from_run_dirs([])

    def test_from_run_dir_missing_object_warns(self, tmp_path, caplog):
        """A run dir without target object data logs a warning."""
        torch.manual_seed(42)
        np.random.seed(42)
        T = 20
        data = {
            "joint_positions": np.random.randn(T, 9).astype(np.float32),
            "joint_velocities": np.random.randn(T, 9).astype(np.float32),
            "ee_positions": np.random.randn(T, 3).astype(np.float32),
            "ee_orientations": np.random.randn(T, 4).astype(np.float32),
            "actions": np.random.randn(T, 8).astype(np.float32),
            # No obj_red_cube_pos or obj_target_bin_pos
        }
        np.savez_compressed(tmp_path / "trajectory.npz", **data)

        with caplog.at_level(logging.WARNING, logger="forge_core.learned_policy"):
            ds = DemoDataset.from_run_dir(tmp_path)
        assert "missing_object_data" in caplog.text
        assert len(ds) == T

    def test_from_run_dirs_logs_aggregate_stats(self, tmp_path, caplog):
        """After loading 2 dirs, caplog contains aggregate stats."""
        torch.manual_seed(42)
        np.random.seed(42)
        for i in range(2):
            run_dir = tmp_path / f"run_{i}"
            run_dir.mkdir()
            T = 25
            data = {
                "joint_positions": np.random.randn(T, 9).astype(np.float32),
                "joint_velocities": np.random.randn(T, 9).astype(np.float32),
                "ee_positions": np.random.randn(T, 3).astype(np.float32),
                "ee_orientations": np.random.randn(T, 4).astype(np.float32),
                "actions": np.random.randn(T, 8).astype(np.float32),
                "object_names": np.array(["red_cube", "target_bin"]),
                "obj_red_cube_pos": np.random.randn(T, 3).astype(np.float32),
                "obj_target_bin_pos": np.random.randn(T, 3).astype(np.float32),
            }
            np.savez_compressed(run_dir / "trajectory.npz", **data)

        dirs = [tmp_path / "run_0", tmp_path / "run_1"]
        with caplog.at_level(logging.INFO, logger="forge_core.learned_policy"):
            DemoDataset.from_run_dirs(dirs)
        assert "loaded_run_dirs: 2 dirs" in caplog.text


# ── TestObsMode ─────────────────────────────────────────────

class TestObsMode:
    def test_default_obs_mode(self):
        """LearnedPolicy default obs_mode is 'state'."""
        torch.manual_seed(42)
        np.random.seed(42)
        policy = LearnedPolicy()
        assert policy._obs_mode == "state"

    def test_invalid_obs_mode_raises(self):
        """LearnedPolicy with invalid obs_mode raises ValueError."""
        torch.manual_seed(42)
        np.random.seed(42)
        with pytest.raises(ValueError, match="Unknown obs_mode"):
            LearnedPolicy(obs_mode="lidar")

    def test_vision_obs_mode_not_implemented(self):
        """LearnedPolicy with obs_mode='vision' raises NotImplementedError."""
        torch.manual_seed(42)
        np.random.seed(42)
        with pytest.raises(NotImplementedError, match="Vision-based"):
            LearnedPolicy(obs_mode="vision")
