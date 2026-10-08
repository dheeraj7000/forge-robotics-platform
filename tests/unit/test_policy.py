"""Unit tests for policies."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.policy import (
    ScriptedPickAndPlace,
    NullPolicy,
    create_policy,
    POLICY_REGISTRY,
)
from forge_core.simulation import ForgeSimulation


@pytest.fixture
def sim():
    s = ForgeSimulation("basic_workspace")
    yield s
    s.shutdown()


class TestPolicyInterface:
    def test_null_policy_runs(self, sim):
        policy = NullPolicy()
        policy.reset(sim)
        assert not policy.done
        targets = policy.act(sim)
        assert len(targets) == 9  # 7 arm + 2 finger

    def test_null_policy_eventually_done(self, sim):
        policy = NullPolicy()
        policy.reset(sim)
        for _ in range(200):
            policy.act(sim)
        assert policy.done

    def test_scripted_policy_produces_targets(self, sim):
        policy = ScriptedPickAndPlace(target_object="red_cube")
        policy.reset(sim)
        targets = policy.act(sim)
        assert len(targets) == 9

    def test_scripted_policy_eventually_done(self, sim):
        policy = ScriptedPickAndPlace(target_object="red_cube")
        policy.reset(sim)
        for _ in range(10000):
            if policy.done:
                break
            policy.act(sim)
        assert policy.done

    def test_scripted_policy_reset(self, sim):
        policy = ScriptedPickAndPlace(target_object="red_cube")
        policy.reset(sim)
        for _ in range(100):
            policy.act(sim)
        # Reset should allow running again
        policy.reset(sim)
        assert not policy.done

    def test_create_policy_by_name(self):
        policy = create_policy("null")
        assert isinstance(policy, NullPolicy)

    def test_create_policy_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown policy"):
            create_policy("nonexistent_policy")

    def test_policy_registry_has_expected(self):
        assert "scripted_pick_and_place" in POLICY_REGISTRY
        assert "null" in POLICY_REGISTRY

    def test_scripted_policy_name(self):
        policy = ScriptedPickAndPlace()
        assert policy.name == "ScriptedPickAndPlace"
