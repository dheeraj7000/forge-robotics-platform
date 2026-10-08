"""Unit tests for the fault injection system."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.faults import (
    FaultConfig,
    FaultInjector,
    ObjectMovedFault,
    SensorNoiseFault,
    SensorDelayFault,
    ActuatorStuckFault,
    ControllerDropoutFault,
    GravityShiftFault,
    create_fault,
    load_fault_profile,
    list_fault_profiles,
    FAULT_REGISTRY,
)
from forge_core.simulation import ForgeSimulation


@pytest.fixture
def sim():
    s = ForgeSimulation("basic_workspace")
    yield s
    s.shutdown()


class TestFaultRegistry:
    def test_all_types_registered(self):
        expected = {"object_moved", "sensor_noise", "sensor_delay",
                    "actuator_stuck", "controller_dropout", "gravity_shift"}
        assert expected == set(FAULT_REGISTRY.keys())

    def test_create_fault(self):
        cfg = FaultConfig(type="object_moved", trigger_time=5.0,
                          params={"object_id": "red_cube", "new_position": [0.6, 0, 0.245]})
        fault = create_fault(cfg)
        assert isinstance(fault, ObjectMovedFault)

    def test_create_unknown_fault_raises(self):
        cfg = FaultConfig(type="nonexistent", trigger_time=0)
        with pytest.raises(ValueError, match="Unknown fault"):
            create_fault(cfg)


class TestObjectMovedFault:
    def test_object_teleported(self, sim):
        initial = sim.get_object_state("red_cube").position.copy()
        new_pos = [0.6, -0.1, 0.245]

        cfg = FaultConfig(type="object_moved", trigger_time=0.0,
                          params={"object_id": "red_cube", "new_position": new_pos})
        fault = ObjectMovedFault(cfg)
        fault.apply(sim, 0.0)

        moved = sim.get_object_state("red_cube").position
        np.testing.assert_allclose(moved, new_pos, atol=0.01)

    def test_one_shot_only(self, sim):
        cfg = FaultConfig(type="object_moved", trigger_time=0.0,
                          params={"object_id": "red_cube", "new_position": [0.6, 0, 0.245]})
        fault = ObjectMovedFault(cfg)
        fault.apply(sim, 0.0)
        assert fault._activated

        # Second apply should not trigger
        sim.data.qpos[:] = sim._initial_qpos  # reset positions
        fault.apply(sim, 1.0)
        # Object should still be at reset position (fault didn't fire again)


class TestSensorNoiseFault:
    def test_noise_added(self, sim):
        cfg = FaultConfig(type="sensor_noise", trigger_time=0.0, duration=10.0,
                          params={"std": 0.1})
        fault = SensorNoiseFault(cfg)
        state = sim.get_robot_state()
        original = state.joint_positions.copy()

        np.random.seed(42)
        noisy = fault.filter_robot_state(state, 1.0)
        # Should differ due to noise
        assert not np.allclose(noisy.joint_positions, original)

    def test_no_noise_outside_window(self, sim):
        cfg = FaultConfig(type="sensor_noise", trigger_time=5.0, duration=2.0,
                          params={"std": 0.1})
        fault = SensorNoiseFault(cfg)
        state = sim.get_robot_state()
        original = state.joint_positions.copy()

        filtered = fault.filter_robot_state(state, 1.0)  # before window
        np.testing.assert_array_equal(filtered.joint_positions, original)


class TestControllerDropoutFault:
    def test_targets_zeroed(self):
        cfg = FaultConfig(type="controller_dropout", trigger_time=2.0, duration=3.0)
        fault = ControllerDropoutFault(cfg)
        targets = np.array([0.5, -0.3, 0.2, -1.5, 0.1, 1.2, 0.3, 0])

        # During dropout window
        filtered = fault.filter_targets(targets, 3.0)
        np.testing.assert_array_equal(filtered, np.zeros(8))

    def test_targets_unchanged_outside(self):
        cfg = FaultConfig(type="controller_dropout", trigger_time=2.0, duration=3.0)
        fault = ControllerDropoutFault(cfg)
        targets = np.array([0.5, -0.3, 0.2, -1.5, 0.1, 1.2, 0.3, 0])

        # Before dropout
        filtered = fault.filter_targets(targets, 1.0)
        np.testing.assert_array_equal(filtered, targets)

        # After dropout
        filtered = fault.filter_targets(targets, 6.0)
        np.testing.assert_array_equal(filtered, targets)


class TestActuatorStuckFault:
    def test_joint_frozen(self, sim):
        cfg = FaultConfig(type="actuator_stuck", trigger_time=0.0, duration=10.0,
                          params={"joints": [1]})
        fault = ActuatorStuckFault(cfg)
        fault.apply(sim, 0.0)

        targets = np.array([0.5, 0.8, 0.2, -1.5, 0.1, 1.2, 0.3, 0])
        filtered = fault.filter_targets(targets, 1.0)
        # Joint 1 should be frozen at its initial ctrl value
        assert filtered[1] != 0.8  # should be frozen
        assert filtered[0] == 0.5  # other joints unchanged


class TestGravityShiftFault:
    def test_gravity_changed(self, sim):
        original = sim.model.opt.gravity.copy()
        cfg = FaultConfig(type="gravity_shift", trigger_time=0.0,
                          params={"gravity": [0, 0, -5.0]})
        fault = GravityShiftFault(cfg)
        fault.apply(sim, 0.0)
        np.testing.assert_allclose(sim.model.opt.gravity, [0, 0, -5.0])


class TestFaultInjector:
    def test_from_profile(self):
        injector = FaultInjector.from_profile("object_moved")
        assert len(injector.faults) == 1
        assert injector.fault_names == ["object_moved"]

    def test_from_cascading_profile(self):
        injector = FaultInjector.from_profile("cascading")
        assert len(injector.faults) == 3

    def test_pre_step_applies_faults(self, sim):
        cfg = FaultConfig(type="controller_dropout", trigger_time=0.0, duration=5.0)
        injector = FaultInjector.from_configs([cfg])
        targets = np.ones(8)
        filtered = injector.pre_step(sim, targets)
        np.testing.assert_array_equal(filtered, np.zeros(8))


class TestFaultProfiles:
    def test_list_profiles(self):
        profiles = list_fault_profiles()
        assert "object_moved" in profiles
        assert "cascading" in profiles
        assert "controller_dropout" in profiles

    def test_load_missing_profile_raises(self):
        with pytest.raises(FileNotFoundError):
            load_fault_profile("nonexistent_profile")


class TestRunnerWithFaults:
    def test_run_with_fault_injector(self):
        """Task runs to completion with faults active."""
        from forge_core.policy import NullPolicy
        from forge_core.runner import run_task_by_name

        injector = FaultInjector.from_profile("object_moved")
        result = run_task_by_name("pick_red_cube", NullPolicy(),
                                   fault_injector=injector)
        assert result is not None
        assert "faults" in result.metrics
        assert "object_moved" in result.metrics["faults"]
