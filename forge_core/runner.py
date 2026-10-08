"""Task runner for Forge.

Orchestrates: load scenario → reset → run policy → evaluate → report.
Optionally records trajectory for replay (Phase 3).
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
from forge_core.recorder import TrajectoryBuffer, get_next_run_id, save_run
from forge_core.simulation import ForgeSimulation
from forge_core.task import TaskSpec, load_task

logger = logging.getLogger(__name__)

STEPS_PER_CYCLE = 10


def run_task(
    task: TaskSpec,
    policy: Policy,
    verbose: bool = False,
    record: bool = False,
) -> EvalResult:
    """Execute a task with a policy and return evaluation results.

    Args:
        task: Task specification.
        policy: Policy that produces joint commands.
        verbose: Log progress.
        record: If True, save trajectory to runs/ directory.

    Returns:
        EvalResult with success/failure and metrics.
    """
    logger.info("task_run_started: %s with policy %s", task.name, policy.name)
    wall_start = time.monotonic()

    sim = ForgeSimulation(scenario_name=task.scenario)
    sim.reset()
    policy.reset(sim)

    # Recording buffer
    trajectory = TrajectoryBuffer() if record else None

    steps = 0
    max_sim_time = task.time_limit
    timed_out = False
    last_action = None

    while not policy.done:
        if sim.sim_time >= max_sim_time:
            timed_out = True
            logger.info("task_timed_out: %s at %.3fs", task.name, sim.sim_time)
            break

        targets = policy.act(sim)
        sim.set_joint_targets(targets)
        sim.step(STEPS_PER_CYCLE)
        steps += STEPS_PER_CYCLE
        last_action = targets

        # Record state
        if trajectory is not None:
            robot = sim.get_robot_state()
            obj_states = {
                o.id: (o.position, o.orientation)
                for o in sim.get_object_states()
            }
            trajectory.record_step(
                timestamp=sim.sim_time,
                joint_pos=robot.joint_positions,
                joint_vel=robot.joint_velocities,
                ee_pos=robot.ee_position,
                ee_quat=robot.ee_orientation,
                action=targets,
                object_states=obj_states,
            )

        if verbose and steps % 1000 == 0:
            logger.info("  step %d, sim_time=%.3f", steps, sim.sim_time)

    # Settle
    sim.step(200)
    steps += 200

    wall_end = time.monotonic()

    # Evaluate
    evaluator = Evaluator(sim, task)
    result = evaluator.evaluate()
    result.wall_time = wall_end - wall_start
    result.steps = steps
    result.timed_out = timed_out

    # Save recorded run
    if trajectory is not None:
        run_id = get_next_run_id()
        metadata = {
            "task_name": task.name,
            "scenario": task.scenario,
            "policy": policy.name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "success": result.success,
            "sim_time": result.sim_time,
            "wall_time": result.wall_time,
            "steps": steps,
            "timed_out": timed_out,
        }
        run_dir = save_run(
            run_id=run_id,
            trajectory=trajectory,
            metadata=metadata,
            scenario_config=sim._scenario,
            result=result.to_dict(),
        )
        result.metrics["run_id"] = run_id
        result.metrics["run_dir"] = str(run_dir)

    logger.info(
        "task_run_completed: %s -> %s (sim=%.3fs, wall=%.3fs)",
        task.name, "SUCCESS" if result.success else "FAILURE",
        result.sim_time, result.wall_time,
    )

    sim.shutdown()
    return result


def run_task_by_name(
    task_name: str, policy: Policy, verbose: bool = False, record: bool = False
) -> EvalResult:
    """Load a task by name and run it."""
    task = load_task(task_name)
    return run_task(task, policy, verbose=verbose, record=record)


def run_suite(
    task_names: list[str], policy: Policy, verbose: bool = False, record: bool = False
) -> list[EvalResult]:
    """Run multiple tasks with the same policy."""
    results = []
    for i, name in enumerate(task_names, 1):
        logger.info("suite_progress: %d/%d — %s", i, len(task_names), name)
        result = run_task_by_name(name, policy, verbose=verbose, record=record)
        results.append(result)
    return results


def save_result(result: EvalResult, output_dir: str = "runs") -> Path:
    """Save an evaluation result to a standalone JSON file."""
    out = PROJECT_ROOT / output_dir
    out.mkdir(parents=True, exist_ok=True)
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
    passed = sum(1 for r in results if r.success)
    for r in results:
        status = "PASS" if r.success else "FAIL"
        t_mark = " [T]" if r.timed_out else ""
        lines.append(
            f"{r.task_name:<30} {status:<10} {r.sim_time:>7.3f}s {r.steps:>8}{t_mark}"
        )
    lines.append("-" * 60)
    lines.append(f"Total: {passed}/{len(results)} passed")
    lines.append("=" * 60)
    lines.append("")
    return "\n".join(lines)
