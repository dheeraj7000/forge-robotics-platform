"""Task evaluator for Forge.

Evaluates whether a task's success criteria are met based on simulation state.
Returns structured results with metrics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from forge_core.simulation import ForgeSimulation, ObjectState
from forge_core.task import TaskSpec, SuccessCriterion

logger = logging.getLogger(__name__)


@dataclass
class CriterionResult:
    """Result of evaluating a single success criterion."""

    criterion: SuccessCriterion
    passed: bool
    detail: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Complete evaluation result for a task run."""

    task_name: str
    success: bool
    criteria_results: list[CriterionResult] = field(default_factory=list)
    sim_time: float = 0.0
    wall_time: float = 0.0
    steps: int = 0
    timed_out: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary."""
        status = "SUCCESS" if self.success else "FAILURE"
        lines = [
            f"Task: {self.task_name}",
            f"Result: {status}",
            f"Sim time: {self.sim_time:.3f}s",
            f"Wall time: {self.wall_time:.3f}s",
            f"Steps: {self.steps}",
        ]
        if self.timed_out:
            lines.append("TIMED OUT")
        lines.append("Criteria:")
        for cr in self.criteria_results:
            mark = "PASS" if cr.passed else "FAIL"
            desc = cr.criterion.description or cr.criterion.type
            lines.append(f"  [{mark}] {desc}")
            if cr.detail:
                lines.append(f"         {cr.detail}")
        if self.metrics:
            lines.append("Metrics:")
            for k, v in self.metrics.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serializable dict for JSON output."""
        return {
            "task_name": self.task_name,
            "success": self.success,
            "sim_time": self.sim_time,
            "wall_time": self.wall_time,
            "steps": self.steps,
            "timed_out": self.timed_out,
            "criteria": [
                {
                    "type": cr.criterion.type,
                    "passed": cr.passed,
                    "detail": cr.detail,
                    "metrics": cr.metrics,
                }
                for cr in self.criteria_results
            ],
            "metrics": self.metrics,
        }


class Evaluator:
    """Evaluates task success criteria against simulation state."""

    def __init__(self, sim: ForgeSimulation, task: TaskSpec):
        self.sim = sim
        self.task = task

    def evaluate(self) -> EvalResult:
        """Evaluate all success criteria for the current simulation state."""
        results = []
        for criterion in self.task.success_criteria:
            result = self._evaluate_criterion(criterion)
            results.append(result)

        all_passed = all(r.passed for r in results)

        # Collect aggregate metrics
        metrics = self._compute_metrics()

        return EvalResult(
            task_name=self.task.name,
            success=all_passed,
            criteria_results=results,
            sim_time=self.sim.sim_time,
            metrics=metrics,
        )

    def _evaluate_criterion(self, criterion: SuccessCriterion) -> CriterionResult:
        """Evaluate a single criterion."""
        evaluators = {
            "object_in_zone": self._eval_object_in_zone,
            "object_above_height": self._eval_object_above_height,
            "object_distance_to": self._eval_object_distance_to,
            "robot_at_home": self._eval_robot_at_home,
        }

        evaluator_fn = evaluators.get(criterion.type)
        if evaluator_fn is None:
            return CriterionResult(
                criterion=criterion,
                passed=False,
                detail=f"Unknown criterion type: {criterion.type}",
            )

        return evaluator_fn(criterion)

    def _eval_object_in_zone(self, criterion: SuccessCriterion) -> CriterionResult:
        """Check if an object is within a specified zone.

        Params:
            object_id: which object to check
            zone_center: [x, y, z] center of the target zone
            zone_radius: maximum distance from center (default: uses xy only)
            use_3d: if true, measure 3D distance; otherwise xy only
        """
        params = criterion.params
        object_id = params.get("object_id", self.task.target_object)
        zone_center = np.array(params["zone_center"])
        zone_radius = params["zone_radius"]
        use_3d = params.get("use_3d", False)

        obj = self.sim.get_object_state(object_id)
        if obj is None:
            return CriterionResult(
                criterion=criterion, passed=False,
                detail=f"Object '{object_id}' not found",
            )

        if use_3d:
            distance = np.linalg.norm(obj.position - zone_center)
        else:
            distance = np.linalg.norm(obj.position[:2] - zone_center[:2])

        passed = bool(distance <= zone_radius)
        return CriterionResult(
            criterion=criterion,
            passed=passed,
            detail=f"distance={distance:.4f}, threshold={zone_radius}",
            metrics={"distance_to_zone": float(distance)},
        )

    def _eval_object_above_height(self, criterion: SuccessCriterion) -> CriterionResult:
        """Check if an object is above a minimum height (z).

        Params:
            object_id: which object
            min_height: minimum z value
        """
        params = criterion.params
        object_id = params.get("object_id", self.task.target_object)
        min_height = params["min_height"]

        obj = self.sim.get_object_state(object_id)
        if obj is None:
            return CriterionResult(
                criterion=criterion, passed=False,
                detail=f"Object '{object_id}' not found",
            )

        passed = bool(obj.position[2] >= min_height)
        return CriterionResult(
            criterion=criterion,
            passed=passed,
            detail=f"height={obj.position[2]:.4f}, min={min_height}",
            metrics={"object_height": float(obj.position[2])},
        )

    def _eval_object_distance_to(self, criterion: SuccessCriterion) -> CriterionResult:
        """Check if an object is within a distance of a target point.

        Params:
            object_id: which object
            target: [x, y, z] target position
            max_distance: maximum allowed distance
        """
        params = criterion.params
        object_id = params.get("object_id", self.task.target_object)
        target = np.array(params["target"])
        max_dist = params["max_distance"]

        obj = self.sim.get_object_state(object_id)
        if obj is None:
            return CriterionResult(
                criterion=criterion, passed=False,
                detail=f"Object '{object_id}' not found",
            )

        distance = np.linalg.norm(obj.position - target)
        passed = bool(distance <= max_dist)
        return CriterionResult(
            criterion=criterion,
            passed=passed,
            detail=f"distance={distance:.4f}, max={max_dist}",
            metrics={"distance": float(distance)},
        )

    def _eval_robot_at_home(self, criterion: SuccessCriterion) -> CriterionResult:
        """Check if robot is near home configuration.

        Params:
            tolerance: max joint position error (default 0.1 rad)
        """
        params = criterion.params
        tolerance = params.get("tolerance", 0.1)

        robot = self.sim.get_robot_state()
        home = np.array(self.sim._scenario["robot"]["home_qpos"])
        n = min(len(robot.joint_positions), len(home))
        error = np.max(np.abs(robot.joint_positions[:n] - home[:n]))

        passed = bool(error <= tolerance)
        return CriterionResult(
            criterion=criterion,
            passed=passed,
            detail=f"max_joint_error={error:.4f}, tolerance={tolerance}",
            metrics={"max_joint_error": float(error)},
        )

    def _compute_metrics(self) -> dict[str, Any]:
        """Compute aggregate task metrics."""
        metrics: dict[str, Any] = {}

        # Target object final position
        target_obj = self.sim.get_object_state(self.task.target_object)
        if target_obj is not None:
            metrics["target_object_final_pos"] = target_obj.position.tolist()

        # EE final position
        robot = self.sim.get_robot_state()
        metrics["ee_final_pos"] = robot.ee_position.tolist()

        # Sim time used
        metrics["sim_time"] = self.sim.sim_time

        return metrics
