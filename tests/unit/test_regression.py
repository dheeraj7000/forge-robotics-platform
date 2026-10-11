"""Unit tests for the regression evaluation system."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.evaluator import EvalResult
from forge_core.regression import (
    Baseline,
    RegressionReport,
    RegressionResult,
    check_regression,
    load_baseline,
    load_regression_config,
    save_baseline,
)


# ── Helpers ─────────────────────────────────────────────────

def _make_eval_result(
    task_name: str = "pick_red_cube",
    success: bool = True,
    sim_time: float = 5.0,
    metrics: dict | None = None,
) -> EvalResult:
    """Create a mock EvalResult without needing MuJoCo."""
    return EvalResult(
        task_name=task_name,
        success=success,
        sim_time=sim_time,
        metrics=metrics or {},
    )


def _make_baseline(
    suite_name: str = "test-suite",
    tasks: dict | None = None,
) -> Baseline:
    """Create a Baseline object for testing."""
    if tasks is None:
        tasks = {
            "pick_red_cube": {"success": True, "sim_time": 5.0, "metrics": {}},
            "pick_blue_cube": {"success": True, "sim_time": 6.0, "metrics": {}},
        }
    return Baseline(
        suite_name=suite_name,
        timestamp="2025-01-28T00:00:00+00:00",
        git_commit="abc123",
        tasks=tasks,
    )


def _sample_results() -> list[EvalResult]:
    """Two passing eval results matching the default baseline."""
    return [
        _make_eval_result("pick_red_cube", success=True, sim_time=5.0),
        _make_eval_result("pick_blue_cube", success=True, sim_time=6.0),
    ]


# ── TestBaselineSaveLoad ────────────────────────────────────

class TestBaselineSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path):
        results = [
            _make_eval_result("pick_red_cube", success=True, sim_time=5.1),
            _make_eval_result("pick_blue_cube", success=False, sim_time=8.2),
        ]
        save_baseline(results, "my-suite", baselines_dir=tmp_path)
        baseline = load_baseline("my-suite", baselines_dir=tmp_path)
        assert baseline.suite_name == "my-suite"
        assert "pick_red_cube" in baseline.tasks
        assert "pick_blue_cube" in baseline.tasks
        assert baseline.tasks["pick_red_cube"]["success"] is True
        assert baseline.tasks["pick_blue_cube"]["success"] is False

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_baseline("nonexistent", baselines_dir=tmp_path)

    def test_baseline_has_expected_fields(self, tmp_path):
        results = [_make_eval_result("task_a", success=True, sim_time=3.5)]
        save_baseline(results, "field-check", baselines_dir=tmp_path)
        raw = json.loads((tmp_path / "field-check.json").read_text())
        assert "suite_name" in raw
        assert "timestamp" in raw
        assert "git_commit" in raw
        assert "tasks" in raw
        task_data = raw["tasks"]["task_a"]
        assert "success" in task_data
        assert "sim_time" in task_data
        assert "metrics" in task_data

    def test_save_overwrites_existing(self, tmp_path):
        results_v1 = [_make_eval_result("task_a", success=False, sim_time=1.0)]
        results_v2 = [_make_eval_result("task_a", success=True, sim_time=2.0)]
        save_baseline(results_v1, "overwrite", baselines_dir=tmp_path)
        save_baseline(results_v2, "overwrite", baselines_dir=tmp_path)
        baseline = load_baseline("overwrite", baselines_dir=tmp_path)
        assert baseline.tasks["task_a"]["success"] is True
        assert baseline.tasks["task_a"]["sim_time"] == 2.0


# ── TestRegressionCheck ─────────────────────────────────────

class TestRegressionCheck:
    def test_no_regression_matching_results(self):
        baseline = _make_baseline()
        results = _sample_results()
        report = check_regression(results, baseline)
        assert report.overall_passed is True
        assert all(r.passed for r in report.results)

    def test_detect_newly_failing_task(self):
        baseline = _make_baseline()
        results = [
            _make_eval_result("pick_red_cube", success=False, sim_time=5.0),
            _make_eval_result("pick_blue_cube", success=True, sim_time=6.0),
        ]
        report = check_regression(results, baseline)
        assert report.overall_passed is False
        red_result = next(r for r in report.results if r.task_name == "pick_red_cube")
        assert red_result.newly_failing is True
        assert red_result.passed is False

    def test_detect_metric_degradation(self):
        baseline = _make_baseline()
        # sim_time 20% worse (beyond 10% tolerance)
        results = [
            _make_eval_result("pick_red_cube", success=True, sim_time=6.0),
            _make_eval_result("pick_blue_cube", success=True, sim_time=6.0),
        ]
        report = check_regression(results, baseline)
        assert report.overall_passed is False
        red_result = next(r for r in report.results if r.task_name == "pick_red_cube")
        assert red_result.passed is False
        assert red_result.metric_deltas["sim_time"] > 0.1

    def test_within_tolerance_passes(self):
        baseline = _make_baseline()
        # sim_time 5% worse — within 10% tolerance
        results = [
            _make_eval_result("pick_red_cube", success=True, sim_time=5.25),
            _make_eval_result("pick_blue_cube", success=True, sim_time=6.3),
        ]
        report = check_regression(results, baseline)
        assert report.overall_passed is True

    def test_missing_task_in_baseline(self):
        baseline = _make_baseline()
        results = [
            _make_eval_result("pick_red_cube", success=True, sim_time=5.0),
            _make_eval_result("pick_blue_cube", success=True, sim_time=6.0),
            _make_eval_result("new_task", success=True, sim_time=3.0),
        ]
        report = check_regression(results, baseline)
        assert report.overall_passed is True
        new_result = next(r for r in report.results if r.task_name == "new_task")
        assert new_result.passed is True
        assert "new task" in new_result.details

    def test_empty_results(self):
        baseline = _make_baseline()
        report = check_regression([], baseline)
        assert report.overall_passed is True
        assert len(report.results) == 0

    def test_allow_new_failures_flag(self):
        baseline = _make_baseline()
        results = [
            _make_eval_result("pick_red_cube", success=False, sim_time=5.0),
            _make_eval_result("pick_blue_cube", success=True, sim_time=6.0),
        ]
        report = check_regression(results, baseline, allow_new_failures=True)
        red_result = next(r for r in report.results if r.task_name == "pick_red_cube")
        assert red_result.newly_failing is True
        # With allow_new_failures, newly_failing alone should not fail
        assert red_result.passed is True

    def test_multiple_regressions(self):
        baseline = _make_baseline()
        results = [
            _make_eval_result("pick_red_cube", success=False, sim_time=10.0),
            _make_eval_result("pick_blue_cube", success=False, sim_time=12.0),
        ]
        report = check_regression(results, baseline)
        assert report.overall_passed is False
        assert all(not r.passed for r in report.results)


# ── TestRegressionReport ────────────────────────────────────

class TestRegressionReport:
    def test_summary_format(self):
        report = RegressionReport(
            overall_passed=True,
            results=[
                RegressionResult(
                    task_name="pick_red_cube", passed=True,
                    baseline_success=True, current_success=True, details="ok",
                ),
            ],
            timestamp="2025-01-28T00:00:00+00:00",
        )
        text = report.summary()
        assert "PASS" in text
        assert "pick_red_cube" in text

    def test_to_dict_serialization(self):
        report = RegressionReport(
            overall_passed=False,
            results=[
                RegressionResult(
                    task_name="task_a", passed=False,
                    baseline_success=True, current_success=False,
                    newly_failing=True, metric_deltas={"sim_time": 0.25},
                    details="newly failing",
                ),
            ],
            timestamp="2025-01-28T00:00:00+00:00",
        )
        d = report.to_dict()
        assert d["overall_passed"] is False
        assert len(d["results"]) == 1
        assert d["results"][0]["newly_failing"] is True
        assert d["results"][0]["metric_deltas"]["sim_time"] == 0.25
        # Must be JSON-serializable
        json.dumps(d)

    def test_overall_pass_fail_logic(self):
        # All pass → overall pass
        report_pass = RegressionReport(
            overall_passed=True,
            results=[
                RegressionResult("a", passed=True, baseline_success=True, current_success=True),
                RegressionResult("b", passed=True, baseline_success=True, current_success=True),
            ],
        )
        assert report_pass.overall_passed is True

        # Any fail → overall fail
        report_fail = RegressionReport(
            overall_passed=False,
            results=[
                RegressionResult("a", passed=True, baseline_success=True, current_success=True),
                RegressionResult("b", passed=False, baseline_success=True, current_success=False),
            ],
        )
        assert report_fail.overall_passed is False


# ── TestThresholds ──────────────────────────────────────────

class TestThresholds:
    def test_custom_metric_tolerance(self):
        baseline = _make_baseline()
        # 15% slower — fails at default 10%, passes at 20%
        results = [
            _make_eval_result("pick_red_cube", success=True, sim_time=5.75),
            _make_eval_result("pick_blue_cube", success=True, sim_time=6.0),
        ]
        report_strict = check_regression(results, baseline, metric_tolerance=0.1)
        assert report_strict.overall_passed is False

        report_relaxed = check_regression(results, baseline, metric_tolerance=0.2)
        assert report_relaxed.overall_passed is True

    def test_load_config_from_yaml(self, tmp_path):
        config_path = tmp_path / "regression.yaml"
        config_path.write_text(
            "metric_tolerance: 0.2\n"
            "allow_new_failures: true\n"
            "suite: custom-suite\n"
        )
        config = load_regression_config(config_path)
        assert config["metric_tolerance"] == 0.2
        assert config["allow_new_failures"] is True
        assert config["suite"] == "custom-suite"

    def test_default_config(self, tmp_path):
        # Passing a nonexistent path should still work via the file-based lookup
        # but we test the code path by using the function with an explicit path
        # that exists and has defaults
        config_path = tmp_path / "defaults.yaml"
        config_path.write_text(
            "metric_tolerance: 0.1\n"
            "allow_new_failures: false\n"
            "suite: manipulation-v1\n"
        )
        config = load_regression_config(config_path)
        assert config["metric_tolerance"] == 0.1
        assert config["allow_new_failures"] is False
