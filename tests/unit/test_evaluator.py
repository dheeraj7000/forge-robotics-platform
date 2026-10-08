"""Unit tests for the task evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.evaluator import Evaluator, EvalResult
from forge_core.simulation import ForgeSimulation
from forge_core.task import load_task, SuccessCriterion, TaskSpec


@pytest.fixture
def sim():
    s = ForgeSimulation("basic_workspace")
    yield s
    s.shutdown()


class TestEvaluator:
    def test_evaluate_returns_result(self, sim):
        task = load_task("pick_red_cube")
        evaluator = Evaluator(sim, task)
        result = evaluator.evaluate()
        assert isinstance(result, EvalResult)
        assert result.task_name == "pick_red_cube"

    def test_initial_state_fails_task(self, sim):
        """Red cube starts at its original position, not in the target zone."""
        task = load_task("pick_red_cube")
        evaluator = Evaluator(sim, task)
        result = evaluator.evaluate()
        # Red cube is at [0.45, 0.15] not at target [0.65, 0.0]
        assert result.success is False

    def test_criteria_results_populated(self, sim):
        task = load_task("pick_red_cube")
        evaluator = Evaluator(sim, task)
        result = evaluator.evaluate()
        assert len(result.criteria_results) == len(task.success_criteria)

    def test_object_above_height_passes_initially(self, sim):
        """Red cube starts on table, above min height."""
        task = load_task("pick_red_cube")
        evaluator = Evaluator(sim, task)
        result = evaluator.evaluate()
        # Find the height criterion result
        height_results = [
            cr for cr in result.criteria_results
            if cr.criterion.type == "object_above_height"
        ]
        assert len(height_results) == 1
        # Cube starts at z=0.345, min=0.30 -> should pass
        assert height_results[0].passed is True

    def test_result_has_metrics(self, sim):
        task = load_task("pick_red_cube")
        evaluator = Evaluator(sim, task)
        result = evaluator.evaluate()
        assert "target_object_final_pos" in result.metrics
        assert "ee_final_pos" in result.metrics

    def test_result_to_dict(self, sim):
        task = load_task("pick_red_cube")
        evaluator = Evaluator(sim, task)
        result = evaluator.evaluate()
        d = result.to_dict()
        assert d["task_name"] == "pick_red_cube"
        assert isinstance(d["criteria"], list)
        assert isinstance(d["success"], bool)

    def test_result_summary_string(self, sim):
        task = load_task("pick_red_cube")
        evaluator = Evaluator(sim, task)
        result = evaluator.evaluate()
        s = result.summary()
        assert "pick_red_cube" in s
        assert "FAILURE" in s or "SUCCESS" in s

    def test_unknown_criterion_type_fails(self, sim):
        task = TaskSpec(
            name="test", description="test", scenario="basic_workspace",
            target_object="red_cube",
            success_criteria=[SuccessCriterion(type="unknown_type", params={})],
        )
        evaluator = Evaluator(sim, task)
        result = evaluator.evaluate()
        assert result.success is False
        assert "Unknown" in result.criteria_results[0].detail
