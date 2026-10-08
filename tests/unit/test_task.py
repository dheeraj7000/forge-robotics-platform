"""Unit tests for task specification loading."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.task import load_task, list_tasks, load_suite, TaskSpec


class TestTaskLoading:
    def test_load_pick_red_cube(self):
        task = load_task("pick_red_cube")
        assert task.name == "pick_red_cube"
        assert task.scenario == "basic_workspace"
        assert task.target_object == "red_cube"
        assert len(task.success_criteria) >= 1

    def test_load_pick_blue_cube(self):
        task = load_task("pick_blue_cube")
        assert task.target_object == "blue_cube"

    def test_task_has_time_limit(self):
        task = load_task("pick_red_cube")
        assert task.time_limit > 0

    def test_task_has_success_criteria(self):
        task = load_task("pick_red_cube")
        for c in task.success_criteria:
            assert c.type in ("object_in_zone", "object_above_height",
                              "object_distance_to", "robot_at_home")
            assert len(c.params) > 0

    def test_missing_task_raises(self):
        with pytest.raises(FileNotFoundError):
            load_task("nonexistent_task")

    def test_list_tasks(self):
        tasks = list_tasks()
        assert "pick_red_cube" in tasks
        assert "pick_blue_cube" in tasks

    def test_load_suite(self):
        suite = load_suite("manipulation-v1")
        assert isinstance(suite, list)
        assert "pick_red_cube" in suite
        assert len(suite) >= 2
