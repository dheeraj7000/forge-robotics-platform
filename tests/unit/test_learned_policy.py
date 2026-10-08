"""Unit tests for the learned policy system."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.simulation import ForgeSimulation
from forge_core.learned_policy import (
    extract_observation,
    PolicyNetwork,
    LearnedPolicy,
    DemoDataset,
    train_bc,
    collect_demos,
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
