"""Regression evaluation for Forge.

Compares current evaluation results against saved baselines to detect
regressions. File-based JSON storage following the ExperimentTracker
and ModelRegistry patterns.

Baseline format (stored in baselines/<suite_name>.json):
    {
        "suite_name": "manipulation-v1",
        "timestamp": "2025-01-28T12:00:00+00:00",
        "git_commit": "abc123...",
        "tasks": {
            "pick_red_cube": {"success": true, "sim_time": 5.1, "metrics": {...}},
            ...
        }
    }
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forge_core.config import PROJECT_ROOT, load_yaml
from forge_core.evaluator import EvalResult
from forge_core.experiment import _get_git_commit

logger = logging.getLogger(__name__)


# ── Dataclasses ─────────────────────────────────────────────

@dataclass
class Baseline:
    """Snapshot of evaluation results used as a regression reference."""
    suite_name: str
    timestamp: str
    git_commit: str | None
    tasks: dict[str, dict[str, Any]]


@dataclass
class RegressionResult:
    """Result of comparing a single task against its baseline."""
    task_name: str
    passed: bool
    baseline_success: bool
    current_success: bool
    newly_failing: bool = False
    metric_deltas: dict[str, float] = field(default_factory=dict)
    details: str = ""


@dataclass
class RegressionReport:
    """Aggregate report from a full regression check."""
    overall_passed: bool
    results: list[RegressionResult] = field(default_factory=list)
    timestamp: str = ""

    def summary(self) -> str:
        """Human-readable summary of the regression report."""
        status = "PASS" if self.overall_passed else "REGRESSION DETECTED"
        lines = [
            "",
            "Regression Report",
            "=" * 60,
            f"Status: {status}",
            f"Timestamp: {self.timestamp}",
            f"{'Task':<30} {'Result':<10} {'Details'}",
            "-" * 60,
        ]
        for r in self.results:
            mark = "PASS" if r.passed else "FAIL"
            lines.append(f"{r.task_name:<30} {mark:<10} {r.details}")
        passed = sum(1 for r in self.results if r.passed)
        lines.append("-" * 60)
        lines.append(f"Total: {passed}/{len(self.results)} passed")
        lines.append("=" * 60)
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serializable dict for JSON output."""
        return {
            "overall_passed": self.overall_passed,
            "timestamp": self.timestamp,
            "results": [
                {
                    "task_name": r.task_name,
                    "passed": r.passed,
                    "baseline_success": r.baseline_success,
                    "current_success": r.current_success,
                    "newly_failing": r.newly_failing,
                    "metric_deltas": r.metric_deltas,
                    "details": r.details,
                }
                for r in self.results
            ],
        }


# ── Baseline persistence ───────────────────────────────────

def _default_baselines_dir() -> Path:
    return PROJECT_ROOT / "baselines"


def save_baseline(
    results: list[EvalResult],
    suite_name: str,
    baselines_dir: Path | None = None,
) -> Path:
    """Convert EvalResult list to a baseline JSON file.

    Writes to ``baselines/<suite_name>.json``.
    """
    baselines_dir = baselines_dir or _default_baselines_dir()
    baselines_dir.mkdir(parents=True, exist_ok=True)

    tasks: dict[str, dict[str, Any]] = {}
    for r in results:
        result_dict = r.to_dict()
        tasks[r.task_name] = {
            "success": r.success,
            "sim_time": r.sim_time,
            "metrics": result_dict.get("metrics", {}),
        }

    data = {
        "suite_name": suite_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit(),
        "tasks": tasks,
    }

    path = baselines_dir / f"{suite_name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("baseline_saved: %s (%d tasks)", suite_name, len(tasks))
    return path


def load_baseline(
    suite_name: str,
    baselines_dir: Path | None = None,
) -> Baseline:
    """Load a baseline from disk.

    Raises FileNotFoundError if the baseline does not exist.
    """
    baselines_dir = baselines_dir or _default_baselines_dir()
    path = baselines_dir / f"{suite_name}.json"

    if not path.exists():
        raise FileNotFoundError(f"Baseline not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    return Baseline(
        suite_name=data["suite_name"],
        timestamp=data["timestamp"],
        git_commit=data.get("git_commit"),
        tasks=data["tasks"],
    )


# ── Regression checking ────────────────────────────────────

def check_regression(
    current_results: list[EvalResult],
    baseline: Baseline,
    metric_tolerance: float = 0.1,
    allow_new_failures: bool = False,
) -> RegressionReport:
    """Compare current results against a baseline and report regressions.

    A regression is detected when:
    - A previously passing task now fails (newly_failing).
    - A metric (e.g. sim_time) degrades beyond *metric_tolerance*
      (relative to the baseline value).

    Args:
        current_results: Fresh evaluation results.
        baseline: Previously saved baseline to compare against.
        metric_tolerance: Maximum allowed relative degradation (0.1 = 10%).
        allow_new_failures: If True, newly failing tasks do not cause failure.

    Returns:
        RegressionReport with per-task results and overall pass/fail.
    """
    results: list[RegressionResult] = []

    for current in current_results:
        task_baseline = baseline.tasks.get(current.task_name)

        if task_baseline is None:
            # New task not in baseline — pass by default
            results.append(RegressionResult(
                task_name=current.task_name,
                passed=True,
                baseline_success=False,
                current_success=current.success,
                details="new task (no baseline)",
            ))
            continue

        baseline_success = task_baseline["success"]
        newly_failing = baseline_success and not current.success
        metric_deltas: dict[str, float] = {}
        metric_regression = False

        # Compare sim_time — higher is worse
        baseline_sim = task_baseline.get("sim_time", 0.0)
        if baseline_sim > 0:
            delta = (current.sim_time - baseline_sim) / baseline_sim
            metric_deltas["sim_time"] = round(delta, 4)
            if delta > metric_tolerance:
                metric_regression = True

        # Determine pass/fail
        task_passed = True
        details_parts: list[str] = []

        if newly_failing:
            if not allow_new_failures:
                task_passed = False
            details_parts.append("newly failing")

        if metric_regression:
            task_passed = False
            details_parts.append(
                f"sim_time degraded {metric_deltas['sim_time']:.1%}"
            )

        if not details_parts:
            details_parts.append("ok")

        results.append(RegressionResult(
            task_name=current.task_name,
            passed=task_passed,
            baseline_success=baseline_success,
            current_success=current.success,
            newly_failing=newly_failing,
            metric_deltas=metric_deltas,
            details=", ".join(details_parts),
        ))

    overall = all(r.passed for r in results)
    timestamp = datetime.now(timezone.utc).isoformat()

    report = RegressionReport(
        overall_passed=overall,
        results=results,
        timestamp=timestamp,
    )
    logger.info("regression_check: %s (%d tasks)",
                "PASS" if overall else "FAIL", len(results))
    return report


# ── Configuration loading ──────────────────────────────────

_DEFAULT_CONFIG: dict[str, Any] = {
    "metric_tolerance": 0.1,
    "allow_new_failures": False,
    "suite": "manipulation-v1",
}


def load_regression_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load regression configuration from YAML or return defaults.

    Looks for ``config/regression.yaml`` under PROJECT_ROOT when no
    explicit path is given.  Returns built-in defaults if the file
    does not exist.
    """
    if config_path is not None:
        return load_yaml(config_path)

    default_path = PROJECT_ROOT / "config" / "regression.yaml"
    if default_path.exists():
        return load_yaml(default_path)

    return dict(_DEFAULT_CONFIG)
