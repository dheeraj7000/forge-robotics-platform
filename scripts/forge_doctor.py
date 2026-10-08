#!/usr/bin/env python3
"""Forge health check utility.

Verifies that all components of the Forge system are operational:
- ROS 2 available
- MuJoCo simulation running (via topic check)
- Controller responding (via topic check)
- Joint state topic active
- Object state topic active

Usage:
    python3 scripts/forge_doctor.py

Requires the Forge system to be running (scripts/run_forge.py or ros2 launch).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# Ensure forge_core is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class HealthCheck:
    """Run system health checks and report results."""

    def __init__(self):
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, passed: bool, detail: str = ""):
        self.results.append((name, passed, detail))

    def check_ros_available(self):
        """Check that ROS 2 CLI tools are available."""
        try:
            result = subprocess.run(
                ["ros2", "topic", "list"],
                capture_output=True, text=True, timeout=10
            )
            topics = result.stdout.strip().split("\n") if result.returncode == 0 else []
            self.check("ROS", result.returncode == 0,
                        f"{len(topics)} topics found" if topics else "ros2 command failed")
            return topics
        except FileNotFoundError:
            self.check("ROS", False, "ros2 command not found — is ROS 2 sourced?")
            return []
        except subprocess.TimeoutExpired:
            self.check("ROS", False, "ros2 command timed out")
            return []

    def check_topic_active(self, name: str, topic: str, topics: list[str]):
        """Check if a specific topic exists in the topic list."""
        found = topic in topics
        self.check(name, found, topic if found else f"{topic} not found")
        return found

    def check_node_running(self, node_name: str):
        """Check if a ROS 2 node is running."""
        try:
            result = subprocess.run(
                ["ros2", "node", "list"],
                capture_output=True, text=True, timeout=10
            )
            nodes = result.stdout.strip().split("\n") if result.returncode == 0 else []
            found = any(node_name in n for n in nodes)
            self.check(f"Node {node_name}", found,
                        "running" if found else "not found")
            return found
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.check(f"Node {node_name}", False, "could not query nodes")
            return False

    def check_mujoco_importable(self):
        """Check that MuJoCo Python bindings are installed."""
        try:
            import mujoco
            self.check("MuJoCo", True, f"version {mujoco.__version__}")
            return True
        except ImportError:
            self.check("MuJoCo", False, "mujoco package not installed")
            return False

    def check_simulation_standalone(self):
        """Check that the simulation can load without ROS."""
        try:
            from forge_core.simulation import ForgeSimulation
            sim = ForgeSimulation("basic_workspace")
            state = sim.get_robot_state()
            objects = sim.get_object_states()
            sim.shutdown()
            self.check("Simulation", True,
                        f"{len(state.joint_names)} joints, {len(objects)} objects")
            return True
        except Exception as e:
            self.check("Simulation", False, str(e))
            return False

    def report(self):
        """Print health check results."""
        print()
        print("Forge Health")
        print("=" * 50)

        all_pass = True
        for name, passed, detail in self.results:
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            detail_str = f"  ({detail})" if detail else ""
            print(f"  {name:<20} {status}{detail_str}")

        print("=" * 50)
        if all_pass:
            print("System healthy.")
        else:
            print("Some checks failed.")
            print("Run 'python3 scripts/run_forge.py' to start the system.")
        print()
        return all_pass


def main():
    hc = HealthCheck()

    # Check MuJoCo independently (no ROS needed)
    hc.check_mujoco_importable()
    hc.check_simulation_standalone()

    # Check ROS topics (requires running system)
    topics = hc.check_ros_available()

    if topics:
        hc.check_topic_active("Joint State", "/forge/joint_states", topics)
        hc.check_topic_active("Object State", "/forge/object_states", topics)
        hc.check_topic_active("EE Pose", "/forge/ee_pose", topics)
        hc.check_topic_active("Controller", "/forge/joint_commands", topics)

        # Check nodes
        hc.check_node_running("forge_simulator")
        hc.check_node_running("forge_controller")
        hc.check_node_running("forge_state")
    else:
        print("\n  [info] ROS not available or system not running.")
        print("  Skipping ROS-dependent checks.\n")

    success = hc.report()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
