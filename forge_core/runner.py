"""Task runner for Forge.

Orchestrates: load scenario → reset → run policy → evaluate → report.
Supports trajectory recording (Phase 3), fault injection (Phase 4),
and optional perception wrapping (Phase 5).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from forge_core.config import PROJECT_ROOT
from forge_core.evaluator import Evaluator, EvalResult
from forge_core.faults import FaultInjector
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
    fault_injector: FaultInjector | None = None,
    perception: bool = False,
) -> EvalResult:
    """Execute a task with a policy and return evaluation results.

    Args:
        task: Task specification.
        policy: Policy that produces joint commands.
        verbose: Log progress.
        record: If True, save trajectory to runs/ directory.
        fault_injector: Optional fault injector for chaos testing.
        perception: If True, wrap policy in PerceptionPolicy.

    Returns:
        EvalResult with success/failure and metrics.
    """
    logger.info("task_run_started: %s with policy %s", task.name, policy.name)
    if fault_injector and fault_injector.faults:
        logger.info("faults_active: %s", fault_injector.fault_names)

    wall_start = time.monotonic()

    sim = ForgeSimulation(scenario_name=task.scenario)
    sim.reset()
    policy.reset(sim)

    trajectory = TrajectoryBuffer() if record else None

    steps = 0
    max_sim_time = task.time_limit
    timed_out = False

    # Wrap policy in PerceptionPolicy if requested
    perception_active = False
    if perception:
        from forge_core.perception import PerceptionPolicy
        policy = PerceptionPolicy(policy, sim)
        perception_active = True

    try:
        while not policy.done:
            if sim.sim_time >= max_sim_time:
                timed_out = True
                logger.info("task_timed_out: %s at %.3fs", task.name, sim.sim_time)
                break

            targets = policy.act(sim)

            # Fault injection: filter targets before applying
            if fault_injector:
                targets = fault_injector.pre_step(sim, targets)

            sim.set_joint_targets(targets)
            sim.step(STEPS_PER_CYCLE)
            steps += STEPS_PER_CYCLE

            if fault_injector:
                fault_injector.post_step(sim)

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

            # Record perception frame inside while loop after record_step
            if trajectory is not None and perception_active:
                perc = policy.last_perception
                if perc is not None:
                    trajectory.frames.append(perc.image)

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

        # Add fault info to metrics
        if fault_injector and fault_injector.faults:
            result.metrics["faults"] = fault_injector.fault_names

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
            if fault_injector and fault_injector.faults:
                metadata["faults"] = fault_injector.fault_names
            run_dir = save_run(
                run_id=run_id,
                trajectory=trajectory,
                metadata=metadata,
                scenario_config=sim._scenario,
                result=result.to_dict(),
            )
            result.metrics["run_id"] = run_id
            result.metrics["run_dir"] = str(run_dir)
    finally:
        # Clean up perception resources if active
        if perception_active:
            policy.close()

    logger.info(
        "task_run_completed: %s -> %s (sim=%.3fs, wall=%.3fs)",
        task.name, "SUCCESS" if result.success else "FAILURE",
        result.sim_time, result.wall_time,
    )

    sim.shutdown()
    return result


def run_task_by_name(
    task_name: str,
    policy: Policy,
    verbose: bool = False,
    record: bool = False,
    fault_injector: FaultInjector | None = None,
    perception: bool = False,
) -> EvalResult:
    """Load a task by name and run it."""
    task = load_task(task_name)
    return run_task(task, policy, verbose=verbose, record=record,
                    fault_injector=fault_injector, perception=perception)


def run_suite(
    task_names: list[str],
    policy: Policy,
    verbose: bool = False,
    record: bool = False,
    fault_injector: FaultInjector | None = None,
    perception: bool = False,
) -> list[EvalResult]:
    """Run multiple tasks with the same policy."""
    results = []
    for i, name in enumerate(task_names, 1):
        logger.info("suite_progress: %d/%d — %s", i, len(task_names), name)
        result = run_task_by_name(name, policy, verbose=verbose, record=record,
                                   fault_injector=fault_injector,
                                   perception=perception)
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


def load_edge_policy_from_registry(
    name: str,
    version: int | None = None,
    stage: str | None = None,
    output_dir: str | Path | None = None,
    **policy_kwargs,
) -> Policy:
    """Load a model from the registry, export to ONNX, and return an OnnxPolicy.

    Lazily imports OnnxExporter and OnnxPolicy from edge_deploy, looks up the
    model in the registry, exports the checkpoint to ONNX, and returns an
    OnnxPolicy wrapping the exported file.

    Args:
        name: Registered model name.
        version: Exact version to load. If None, loads the latest.
        stage: Filter by stage (e.g. 'production'). Ignored if version given.
        output_dir: Directory for the exported ONNX file. Defaults to models/.
        **policy_kwargs: Extra keyword arguments passed to OnnxPolicy.

    Returns:
        An OnnxPolicy instance wrapping the exported ONNX model.

    Raises:
        ValueError: If the model is not found in the registry.
    """
    from forge_core.edge_deploy import OnnxExporter, OnnxPolicy
    from forge_core.model_registry import ModelRegistry

    registry = ModelRegistry()
    entry = registry.get_model(name, version=version, stage=stage)
    if entry is None:
        version_str = f" v{version}" if version else ""
        stage_str = f" stage={stage}" if stage else ""
        raise ValueError(
            f"Model not found in registry: {name}{version_str}{stage_str}"
        )

    checkpoint_path = entry["checkpoint_path"]
    v = entry["version"]

    if output_dir is None:
        out = PROJECT_ROOT / "models"
    else:
        out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    onnx_path = out / f"{name}_v{v}.onnx"
    exporter = OnnxExporter()
    exporter.export_from_checkpoint(checkpoint_path, onnx_path)
    logger.info("edge_registry_export: %s v%d -> %s", name, v, onnx_path)

    return OnnxPolicy(onnx_path=onnx_path, **policy_kwargs)


def load_policy_from_registry(
    name: str,
    version: int | None = None,
    stage: str | None = None,
    **policy_kwargs,
) -> Policy:
    """Load a learned policy from the model registry.

    Lazily imports ModelRegistry, looks up the model, resolves the
    checkpoint path, and returns a LearnedPolicy instance.

    Args:
        name: Registered model name.
        version: Exact version to load. If None, loads the latest.
        stage: Filter by stage (e.g. 'production'). Ignored if version given.
        **policy_kwargs: Extra keyword arguments passed to create_policy().

    Returns:
        A Policy instance loaded from the registered checkpoint.

    Raises:
        ValueError: If the model is not found in the registry.
    """
    from forge_core.model_registry import ModelRegistry
    from forge_core.policy import create_policy

    registry = ModelRegistry()
    entry = registry.get_model(name, version=version, stage=stage)
    if entry is None:
        version_str = f" v{version}" if version else ""
        stage_str = f" stage={stage}" if stage else ""
        raise ValueError(
            f"Model not found in registry: {name}{version_str}{stage_str}"
        )

    checkpoint_path = entry["checkpoint_path"]
    logger.info("loading_policy_from_registry: %s v%d (%s)",
                entry["name"], entry["version"], checkpoint_path)
    return create_policy("learned", model_path=checkpoint_path, **policy_kwargs)
