#!/usr/bin/env python3
"""Run the complete Forge system in a single process.

This script composes all three ROS 2 nodes (forge_simulator, forge_controller,
forge_state) in a single process, sharing the same MuJoCo simulation instance.
This is the recommended way to run Forge during Phase 1 development.

Usage:
    python3 scripts/run_forge.py
    python3 scripts/run_forge.py --scenario basic_workspace
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure forge_core is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import rclpy
from rclpy.executors import MultiThreadedExecutor

from forge_core.simulation import ForgeSimulation

# Add ROS package paths
_ros2_src = Path(__file__).resolve().parent.parent / "ros2_ws" / "src"
for pkg_dir in ["forge_sim", "forge_control", "forge_state"]:
    pkg_path = str(_ros2_src / pkg_dir)
    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)

from forge_sim.simulator_node import SimulatorNode
from forge_control.controller_node import ControllerNode
from forge_state.state_node import StateNode


def main():
    parser = argparse.ArgumentParser(description="Run the Forge simulation system")
    parser.add_argument("--scenario", default="basic_workspace",
                        help="Scenario name (default: basic_workspace)")
    args = parser.parse_args()

    rclpy.init()

    # Create nodes
    sim_node = SimulatorNode()

    # Share simulation instance with controller and state nodes
    sim = sim_node.get_sim()
    lock = sim_node.get_sim_lock()

    ctrl_node = ControllerNode()
    ctrl_node.set_sim(sim, lock)

    state_node = StateNode()
    state_node.set_sim(sim, lock)

    # Run all nodes in a multi-threaded executor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(sim_node)
    executor.add_node(ctrl_node)
    executor.add_node(state_node)

    sim_node.get_logger().info("Forge system started — all nodes running")

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        sim.shutdown()
        sim_node.destroy_node()
        ctrl_node.destroy_node()
        state_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
