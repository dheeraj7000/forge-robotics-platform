"""Unit tests for the task runner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.evaluator import EvalResult
from forge_core.policy import NullPolicy, ScriptedPickAndPlace
from forge_core.runner import run_task, run_task_by_name, format_suite_summary
from forge_core.task import load_task


class TestRunner:
    def test_run_task_returns_result(self):
        """Runner produces an EvalResult."""
        task = load_task("pick_red_cube")
        policy = NullPolicy()
        result = run_task(task, policy)
        assert isinstance(result, EvalResult)
        assert result.task_name == "pick_red_cube"

    def test_null_policy_fails_task(self):
        """A do-nothing policy should fail the pick task."""
        result = run_task_by_name("pick_red_cube", NullPolicy())
        assert result.success is False

    def test_result_has_timing(self):
        """Result includes simulation and wall time."""
        result = run_task_by_name("pick_red_cube", NullPolicy())
        assert result.sim_time >= 0
        assert result.wall_time > 0
        assert result.steps > 0

    def test_run_with_scripted_policy(self):
        """Scripted policy runs to completion without error."""
        policy = ScriptedPickAndPlace(target_object="red_cube")
        result = run_task_by_name("pick_red_cube", policy)
        assert isinstance(result, EvalResult)
        # The scripted policy may or may not succeed — it depends on
        # whether the waypoints actually achieve the task.
        # This test just verifies it runs without crashing.
        assert result.steps > 0

    def test_format_suite_summary(self):
        """Suite summary formatter produces readable output."""
        results = [
            EvalResult(task_name="task_a", success=True, sim_time=1.5, steps=500),
            EvalResult(task_name="task_b", success=False, sim_time=2.0, steps=1000),
        ]
        summary = format_suite_summary(results)
        assert "task_a" in summary
        assert "PASS" in summary
        assert "FAIL" in summary
        assert "1/2 passed" in summary
