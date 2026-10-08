"""Task specification and loading for Forge.

A task defines WHAT the robot should accomplish and HOW to measure success.
Tasks are described in YAML and loaded at runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from forge_core.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


@dataclass
class SuccessCriterion:
    """A single measurable condition for task success."""

    type: str  # e.g. "object_in_zone", "object_grasped"
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class TaskSpec:
    """Complete specification of a manipulation task."""

    name: str
    description: str
    scenario: str  # scenario name to load
    target_object: str  # primary object to manipulate
    success_criteria: list[SuccessCriterion] = field(default_factory=list)
    time_limit: float = 30.0  # seconds of sim time
    metadata: dict[str, Any] = field(default_factory=dict)


def load_task(task_name: str) -> TaskSpec:
    """Load a task specification by name from tasks/ directory."""
    task_path = PROJECT_ROOT / "tasks" / f"{task_name}.yaml"
    if not task_path.exists():
        raise FileNotFoundError(f"Task not found: {task_path}")

    with open(task_path, "r") as f:
        data = yaml.safe_load(f)

    task_data = data["task"]

    criteria = []
    for c in task_data.get("success_criteria", []):
        criteria.append(SuccessCriterion(
            type=c["type"],
            params=c.get("params", {}),
            description=c.get("description", ""),
        ))

    spec = TaskSpec(
        name=task_data["name"],
        description=task_data["description"],
        scenario=task_data["scenario"],
        target_object=task_data["target_object"],
        success_criteria=criteria,
        time_limit=task_data.get("time_limit", 30.0),
        metadata=task_data.get("metadata", {}),
    )

    logger.info("task_loaded: %s (scenario=%s, criteria=%d)",
                spec.name, spec.scenario, len(spec.success_criteria))
    return spec


def list_tasks() -> list[str]:
    """Return names of all available tasks."""
    tasks_dir = PROJECT_ROOT / "tasks"
    if not tasks_dir.exists():
        return []
    return sorted(p.stem for p in tasks_dir.glob("*.yaml"))


def load_suite(suite_name: str) -> list[str]:
    """Load a test suite — a list of task names.

    Suites live in tasks/suites/<suite_name>.yaml.
    """
    suite_path = PROJECT_ROOT / "tasks" / "suites" / f"{suite_name}.yaml"
    if not suite_path.exists():
        raise FileNotFoundError(f"Suite not found: {suite_path}")

    with open(suite_path, "r") as f:
        data = yaml.safe_load(f)

    return data.get("tasks", [])
