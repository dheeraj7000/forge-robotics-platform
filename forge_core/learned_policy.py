"""Learned policy for Forge using PyTorch.

Provides a neural network policy that can be trained via behavioral
cloning (from scripted policy demonstrations) or reinforcement learning.
The learned policy implements the same Policy interface as the scripted
policies — it's a drop-in replacement.

Phase 6 includes:
- Observation extraction from simulation state
- MLP policy network
- Behavioral cloning trainer (learn from demonstrations)
- Model save/load for reproducibility
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from forge_core.config import PROJECT_ROOT
from forge_core.policy import Policy
from forge_core.simulation import ForgeSimulation

logger = logging.getLogger(__name__)


# ── Observation / Action helpers ────────────────────────────

def extract_observation(sim: ForgeSimulation, target_object: str = "red_cube") -> np.ndarray:
    """Extract a flat observation vector from simulation state.

    Observation (31-dim):
        [0:9]   joint positions
        [9:18]  joint velocities
        [18:21] EE position (xyz)
        [21:25] EE orientation (quaternion wxyz)
        [25:28] target object position (xyz)
        [28:31] goal position (target_bin xyz)
    """
    robot = sim.get_robot_state()
    target = sim.get_object_state(target_object)
    goal = sim.get_object_state("target_bin")

    obs = np.concatenate([
        robot.joint_positions,       # 9
        robot.joint_velocities,      # 9
        robot.ee_position,           # 3
        robot.ee_orientation,        # 4
        target.position if target else np.zeros(3),  # 3
        goal.position if goal else np.zeros(3),      # 3
    ])
    return obs.astype(np.float32)


OBS_DIM = 31
ACT_DIM = 8  # 7 arm joints + 1 finger actuator


# ── Neural Network ──────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """MLP policy network: observation → action.

    Architecture: obs → 256 → 256 → 128 → action
    Uses tanh activation. Output is raw (no bounds clipping here —
    the simulation handles actuator limits).
    """

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: list[int] | None = None):
        super().__init__()
        hidden = hidden or [256, 256, 128]

        layers: list[nn.Module] = []
        prev_dim = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, act_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ── Learned Policy (Policy interface) ───────────────────────

class LearnedPolicy(Policy):
    """PyTorch-based learned policy.

    Loads a trained PolicyNetwork and uses it for inference.
    Implements the same Policy interface as ScriptedPickAndPlace.
    """

    # Observation mode: "state" (MLP, Phase 6) or "vision" (CNN, future)
    OBS_MODES = ("state", "vision")

    def __init__(self, model_path: str | None = None,
                 target_object: str = "red_cube",
                 max_steps: int = 1500,
                 obs_mode: str = "state",
                 **kwargs):
        if obs_mode not in self.OBS_MODES:
            raise ValueError(f"Unknown obs_mode '{obs_mode}'. Must be one of {self.OBS_MODES}")
        if obs_mode == "vision":
            raise NotImplementedError(
                "Vision-based policy input is planned for a future phase. "
                "Use obs_mode='state' for MLP policies."
            )
        self._obs_mode = obs_mode
        self._target_object = target_object
        self._max_steps = max_steps
        self._steps = 0
        self._is_done = False

        self._network = PolicyNetwork()
        if model_path is not None:
            self._load_model(model_path)
            logger.info("learned_policy_loaded: %s", model_path)

        self._network.eval()

    def _load_model(self, path: str) -> None:
        """Load trained weights."""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self._network.load_state_dict(state_dict)

    def reset(self, sim: ForgeSimulation) -> None:
        self._steps = 0
        self._is_done = False

    def act(self, sim: ForgeSimulation) -> np.ndarray:
        self._steps += 1
        if self._steps >= self._max_steps:
            self._is_done = True
            return np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0])

        obs = extract_observation(sim, self._target_object)
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action = self._network(obs_tensor).squeeze(0).numpy()
        return action

    @property
    def done(self) -> bool:
        return self._is_done

    @property
    def name(self) -> str:
        return "LearnedPolicy"


# ── Demonstration Dataset ───────────────────────────────────

class DemoDataset(Dataset):
    """Dataset of (observation, action) pairs from recorded runs."""

    def __init__(self, observations: np.ndarray, actions: np.ndarray):
        assert len(observations) == len(actions)
        self.observations = torch.from_numpy(observations.astype(np.float32))
        self.actions = torch.from_numpy(actions.astype(np.float32))

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.observations[idx], self.actions[idx]

    @classmethod
    def from_run_dir(cls, run_dir: str | Path,
                     target_object: str = "red_cube") -> DemoDataset:
        """Build dataset from a recorded run's trajectory.

        Loads trajectory.npz and extracts (obs, action) pairs.
        """
        run_dir = Path(run_dir)
        data = np.load(run_dir / "trajectory.npz", allow_pickle=True)

        joint_pos = data["joint_positions"]      # (T, 9)
        joint_vel = data["joint_velocities"]     # (T, 9)
        ee_pos = data["ee_positions"]            # (T, 3)
        ee_quat = data["ee_orientations"]        # (T, 4)
        actions = data["actions"]                # (T, 8)

        obj_names = list(data["object_names"]) if "object_names" in data else []
        if f"obj_{target_object}_pos" not in data:
            logger.warning("missing_object_data: %s not found in %s, using zeros",
                           target_object, run_dir)
        target_pos = (data[f"obj_{target_object}_pos"]
                      if f"obj_{target_object}_pos" in data
                      else np.zeros((len(joint_pos), 3)))
        if "obj_target_bin_pos" not in data:
            logger.warning("missing_object_data: target_bin not found in %s, using zeros",
                           run_dir)
        goal_pos = (data["obj_target_bin_pos"]
                    if "obj_target_bin_pos" in data
                    else np.zeros((len(joint_pos), 3)))

        # Build observation array
        observations = np.concatenate([
            joint_pos, joint_vel, ee_pos, ee_quat, target_pos, goal_pos,
        ], axis=1).astype(np.float32)

        return cls(observations, actions.astype(np.float32))

    @classmethod
    def from_run_dirs(cls, run_dirs: list[str | Path],
                      target_object: str = "red_cube") -> DemoDataset:
        """Build dataset from multiple recorded run directories.

        Loads each run directory and concatenates the data.
        """
        if not run_dirs:
            raise ValueError("No run directories provided")
        datasets = [cls.from_run_dir(d, target_object) for d in run_dirs]
        all_obs = np.concatenate([ds.observations.numpy() for ds in datasets])
        all_act = np.concatenate([ds.actions.numpy() for ds in datasets])
        logger.info("loaded_run_dirs: %d dirs, %d total samples", len(run_dirs), len(all_obs))
        return cls(all_obs, all_act)


# ── Training Configuration ──────────────────────────────────

def _validate_training_config(config: dict[str, Any]) -> None:
    """Validate training configuration values.

    Raises ValueError with a descriptive message on invalid values.
    Unrecognized keys are silently ignored.
    """
    training = config.get("training", {})
    model = config.get("model", {})

    # training section
    epochs = training.get("epochs")
    if epochs is not None:
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError(f"training.epochs must be an int >= 1, got {epochs!r}")

    batch_size = training.get("batch_size")
    if batch_size is not None:
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"training.batch_size must be an int >= 1, got {batch_size!r}")

    lr = training.get("learning_rate")
    if lr is not None:
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError(f"training.learning_rate must be a float > 0, got {lr!r}")

    val_split = training.get("validation_split")
    if val_split is not None:
        if not isinstance(val_split, (int, float)) or not (0.0 <= val_split <= 0.5):
            raise ValueError(
                f"training.validation_split must be a float in [0.0, 0.5], got {val_split!r}"
            )

    ckpt_interval = training.get("checkpoint_interval")
    if ckpt_interval is not None:
        if not isinstance(ckpt_interval, int) or ckpt_interval < 1:
            raise ValueError(
                f"training.checkpoint_interval must be an int >= 1, got {ckpt_interval!r}"
            )

    # model section
    hidden = model.get("hidden_layers")
    if hidden is not None:
        if not isinstance(hidden, list) or not all(isinstance(h, int) and h >= 1 for h in hidden):
            raise ValueError(
                f"model.hidden_layers must be a list of ints each >= 1, got {hidden!r}"
            )

    obs_dim = model.get("obs_dim")
    if obs_dim is not None:
        if not isinstance(obs_dim, int) or obs_dim < 1:
            raise ValueError(f"model.obs_dim must be an int >= 1, got {obs_dim!r}")

    act_dim = model.get("act_dim")
    if act_dim is not None:
        if not isinstance(act_dim, int) or act_dim < 1:
            raise ValueError(f"model.act_dim must be an int >= 1, got {act_dim!r}")


def load_training_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load training configuration from a YAML file.

    When path is given, loads that file (raises FileNotFoundError if missing).
    When path is None, tries the default config/training.yaml; returns {}
    if the default file is missing.
    """
    from forge_core.config import load_yaml

    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Training config not found: {p}")
        config = load_yaml(p)
    else:
        default_path = PROJECT_ROOT / "config" / "training.yaml"
        if not default_path.exists():
            logger.debug("no default training config found")
            return {}
        config = load_yaml(default_path)

    _validate_training_config(config)
    return config


# ── Behavioral Cloning Trainer ──────────────────────────────

@dataclass
class TrainResult:
    """Result of a training run."""
    epochs: int
    final_loss: float
    model_path: str
    history: list[float]
    val_history: list[float] = field(default_factory=list)

    def summary(self) -> str:
        s = (f"Training complete: {self.epochs} epochs, "
             f"final_loss={self.final_loss:.6f}")
        if self.val_history:
            s += f", val_loss={self.val_history[-1]:.6f}"
        s += f", saved={self.model_path}"
        return s


def train_bc(
    dataset: DemoDataset,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    save_path: str | None = None,
    network: PolicyNetwork | None = None,
    validation_split: float = 0.0,
    checkpoint_interval: int | None = None,
    checkpoint_dir: str | None = None,
) -> TrainResult:
    """Train a policy via behavioral cloning.

    Minimizes MSE between network predictions and demonstration actions.

    Args:
        dataset: (obs, action) pairs from demonstrations.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate.
        save_path: Where to save the trained model. If None, auto-generates.
        network: Optional pre-initialized network.
        validation_split: Fraction of data for validation (0.0–0.5).
        checkpoint_interval: Save a checkpoint every N epochs. None = no checkpoints.
        checkpoint_dir: Directory for checkpoint files. Defaults to models/.

    Returns:
        TrainResult with training history and model path.
    """
    import time
    from torch.utils.data import random_split

    # Input guards
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    if not (0.0 <= validation_split <= 0.5):
        raise ValueError(f"validation_split must be in [0.0, 0.5], got {validation_split}")
    if checkpoint_interval is not None and checkpoint_interval < 1:
        raise ValueError(f"checkpoint_interval must be >= 1, got {checkpoint_interval}")

    if network is None:
        network = PolicyNetwork()

    # Train/val split
    val_loader = None
    train_size = len(dataset)
    val_size = 0
    if validation_split > 0:
        val_size = max(1, int(len(dataset) * validation_split))
        train_size = len(dataset) - val_size
        if train_size < 1:
            raise ValueError(
                f"Dataset too small for requested validation split: "
                f"{len(dataset)} samples with validation_split={validation_split} "
                f"leaves {train_size} training samples"
            )
        train_subset, val_subset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(network.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history: list[float] = []
    val_history: list[float] = []
    network.train()

    logger.info("bc_training_start: %d samples (train=%d, val=%d), epochs=%d, lr=%s, batch=%d",
                len(dataset), train_size, val_size, epochs, lr, batch_size)

    t_start = time.monotonic()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        network.train()
        for obs_batch, act_batch in loader:
            pred = network(obs_batch)
            loss = loss_fn(pred, act_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history.append(avg_loss)

        # Validation pass
        if val_loader is not None:
            network.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for obs_batch, act_batch in val_loader:
                    pred = network(obs_batch)
                    val_loss_sum += loss_fn(pred, act_batch).item()
                    val_batches += 1
            avg_val_loss = val_loss_sum / max(val_batches, 1)
            val_history.append(avg_val_loss)
            network.train()
            logger.info("bc_epoch %d/%d: loss=%.6f val_loss=%.6f",
                        epoch + 1, epochs, avg_loss, avg_val_loss)
        else:
            logger.info("bc_epoch %d/%d: loss=%.6f", epoch + 1, epochs, avg_loss)

        # Checkpointing
        if checkpoint_interval is not None:
            if (epoch + 1) % checkpoint_interval == 0 or epoch == epochs - 1:
                ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else PROJECT_ROOT / "models"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"bc_policy_epoch_{epoch + 1:03d}.pt"
                torch.save(network.state_dict(), ckpt_path)
                logger.info("checkpoint_saved: %s", ckpt_path)

    elapsed = time.monotonic() - t_start

    # Save model
    if save_path is None:
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        save_path = str(models_dir / "bc_policy.pt")

    torch.save(network.state_dict(), save_path)
    logger.info("model_saved: %s", save_path)

    if val_history:
        logger.info("bc_training_complete: final_loss=%.6f, val_loss=%.6f, model=%s, time=%.1fs",
                     history[-1] if history else 0.0, val_history[-1], save_path, elapsed)
    else:
        logger.info("bc_training_complete: final_loss=%.6f, model=%s, time=%.1fs",
                     history[-1] if history else 0.0, save_path, elapsed)

    return TrainResult(
        epochs=epochs,
        final_loss=history[-1] if history else 0.0,
        model_path=save_path,
        history=history,
        val_history=val_history,
    )


def collect_demos(
    task_name: str,
    policy: Policy,
    n_episodes: int = 1,
    target_object: str = "red_cube",
) -> DemoDataset:
    """Collect demonstration data by running a policy.

    Returns a DemoDataset of (observation, action) pairs.
    """
    from forge_core.task import load_task
    from forge_core.runner import STEPS_PER_CYCLE

    all_obs: list[np.ndarray] = []
    all_act: list[np.ndarray] = []

    for ep in range(n_episodes):
        task = load_task(task_name)
        sim = ForgeSimulation(scenario_name=task.scenario)
        sim.reset()
        policy.reset(sim)

        while not policy.done and sim.sim_time < task.time_limit:
            obs = extract_observation(sim, target_object)
            action = policy.act(sim)
            all_obs.append(obs)
            all_act.append(action)
            sim.set_joint_targets(action)
            sim.step(STEPS_PER_CYCLE)

        sim.shutdown()
        logger.info("demo_collected: episode %d/%d, %d steps",
                     ep + 1, n_episodes, len(all_obs))

    return DemoDataset(
        np.array(all_obs, dtype=np.float32),
        np.array(all_act, dtype=np.float32),
    )
