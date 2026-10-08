"""Task runner for Forge.

Orchestrates: load scenario → reset → run policy → evaluate → report.
This is the central execution engine for Phase 2.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge_core.config import PROJECT_ROOT
from forge_core.evaluator import Evaluator, EvalResult
from forge_core.policy import Policy
from forge_core.simulation import ForgeSimulation
from forge_core.task import TaskSpec, load_task

logger = logging.getLogger(__name__)

# Steps per control cycle — policy is called once per cycle
STEPS_PER_CYCLE = 10


@dataclass
class RunConfig:
    """Configuration for a single task run."""

    task_name: str
    policy: Policy
    render: bool = False
    seed: int = 0
    save_result: bool = False
    output_dir: str = "runs"


def run_task(task: TaskSpec, policy: Policy, verbose: bool = False) -> EvalResult:
    """Execute a task with a policy and return evaluation results.

    Flow:
        1. Load scenario into simulation
        2. Reset simulation
        3. Reset policy
        4. Loop: policy.act() → sim.set_joint_targets() → sim.step()
        5. Evaluate success criteria
        6. Return result

    Args:
        task: Task specification with scenario and success criteria.
        policy: Policy that produces joint commands.
        verbose: If True, log progress.

    Returns:
        EvalResult with success/failure and metrics.
    """
    logger.info("task_run_started: %s with policy %s", task.name, policy.name)
    wall_start = time.monotonic()

    # 1. Load scenario
    sim = ForgeSimulation(scenario_name=task.scenario)

    # 2. Reset
    sim.reset()

    # 3. Reset policy
    policy.reset(sim)

    # 4. Run loop
    steps = 0
    max_sim_time = task.time_limit
    timed_out = False

    while not policy.done:
        if sim.sim_time >= max_sim_time:
            timed_out = True
            logger.info("task_timed_out: %s at %.3fs", task.name, sim.sim_time)
            break

        targets = policy.act(sim)
        sim.set_joint_targets(targets)
        sim.step(STEPS_PER_CYCLE)
        steps += STEPS_PER_CYCLE

        if verbose and steps % 1000 == 0:
            logger.info("  step %d, sim_time=%.3f", steps, sim.sim_time)

    # Let simulation settle after policy finishes
    sim.step(200)
    steps += 200

    wall_end = time.monotonic()

    # 5. Evaluate
    evaluator = Evaluator(sim, task)
    result = evaluator.evaluate()
    result.wall_time = wall_end - wall_start
    result.steps = steps
    result.timed_out = timed_out

    logger.info("task_run_completed: %s -> %s (sim=%.3fs, wall=%.3fs)",
                task.name, "SUCCESS" if result.success else "FAILURE",
                result.sim_time, result.wall_time)

    sim.shutdown()
    return result


def run_task_by_name(task_name: str, policy: Policy,
                     verbose: bool = False) -> EvalResult:
    """Load a task by name and run it."""
    task = load_task(task_name)
    return run_task(task, policy, verbose=verbose)


def run_suite(task_names: list[str], policy: Policy,
              verbose: bool = False) -> list[EvalResult]:
    """Run multiple tasks with the same policy."""
    results = []
    for i, name in enumerate(task_names, 1):
        logger.info("suite_progress: %d/%d — %s", i, len(task_names), name)
        result = run_task_by_name(name, policy, verbose=verbose)
        results.append(result)
    return results


def save_result(result: EvalResult, output_dir: str = "runs") -> Path:
    """Save an evaluation result to a JSON file."""
    out = PROJECT_ROOT / output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Generate run ID based on timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{result.task_name}_{timestamp}.json"
    path = out / filename

    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info("result_saved: %s", path)
    return path


def format_suite_summary(results: list[EvalResult]) -> str:
    """Format a summary table for multiple task results."""
    lines = [
        "",
        "Forge Evaluation Summary",
        "=" * 60,
        f"{'Task':<30} {'Result':<10} {'Time':>8} {'Steps':>8}",
        "-" * 60,
    ]

    passed = 0
    total = len(results)

    for r in results:
        status = "PASS" if r.success else "FAIL"
        if r.success:
            passed += 1
        timeout_mark = " [T]" if r.timed_out else ""
        lines.append(
            f"{r.task_name:<30} {status:<10} {r.sim_time:>7.3f}s {r.steps:>8}{timeout_mark}"
        )

    lines.append("-" * 60)
    lines.append(f"Total: {passed}/{total} passed")
    lines.append("=" * 60)
    lines.append("")

    return "\n".join(lines)
