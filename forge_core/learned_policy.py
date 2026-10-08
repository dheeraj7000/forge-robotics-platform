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
from dataclasses import dataclass
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

    def __init__(self, model_path: str | None = None,
                 target_object: str = "red_cube",
                 max_steps: int = 1500, **kwargs):
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
        target_pos = (data[f"obj_{target_object}_pos"]
                      if f"obj_{target_object}_pos" in data
                      else np.zeros((len(joint_pos), 3)))
        goal_pos = (data["obj_target_bin_pos"]
                    if "obj_target_bin_pos" in data
                    else np.zeros((len(joint_pos), 3)))

        # Build observation array
        observations = np.concatenate([
            joint_pos, joint_vel, ee_pos, ee_quat, target_pos, goal_pos,
        ], axis=1).astype(np.float32)

        return cls(observations, actions.astype(np.float32))


# ── Behavioral Cloning Trainer ──────────────────────────────

@dataclass
class TrainResult:
    """Result of a training run."""
    epochs: int
    final_loss: float
    model_path: str
    history: list[float]

    def summary(self) -> str:
        return (f"Training complete: {self.epochs} epochs, "
                f"final_loss={self.final_loss:.6f}, saved={self.model_path}")


def train_bc(
    dataset: DemoDataset,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    save_path: str | None = None,
    network: PolicyNetwork | None = None,
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

    Returns:
        TrainResult with training history and model path.
    """
    if network is None:
        network = PolicyNetwork()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(network.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history: list[float] = []
    network.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
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

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info("bc_epoch %d/%d: loss=%.6f", epoch + 1, epochs, avg_loss)

    # Save model
    if save_path is None:
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        save_path = str(models_dir / "bc_policy.pt")

    torch.save(network.state_dict(), save_path)
    logger.info("model_saved: %s", save_path)

    return TrainResult(
        epochs=epochs,
        final_loss=history[-1] if history else 0.0,
        model_path=save_path,
        history=history,
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
