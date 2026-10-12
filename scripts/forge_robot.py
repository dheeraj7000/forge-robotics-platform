#!/usr/bin/env python3
"""Forge robot CLI — connect, run policies, safety checks, calibration, and sim comparison.

Usage:
    python3 scripts/forge_robot.py connect --type simulated --scenario basic_workspace
    python3 scripts/forge_robot.py run --policy null --task pick_and_place_basic
    python3 scripts/forge_robot.py safety-check
    python3 scripts/forge_robot.py calibrate
    python3 scripts/forge_robot.py compare-sim
    python3 scripts/forge_robot.py status
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np

from forge_core.hardware import (
    RobotCommand,
    SimulatedRobot,
    run_policy_on_robot,
)
from forge_core.policy import NullPolicy, create_policy
from forge_core.recorder import TrajectoryBuffer
from forge_core.safety import SafetyConfig, SafetyMonitor, load_safety_config
from forge_core.sim2real import (
    DomainRandomizationConfig,
    apply_domain_randomization,
    calibrate_joint_offsets,
    compare_trajectories,
    load_sim2real_config,
)

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Subcommand handlers ────────────────────────────────────


def cmd_connect(args: argparse.Namespace) -> None:
    """Connect to a robot, print status, disconnect."""
    robot = SimulatedRobot(
        scenario_name=args.scenario,
    )
    robot.connect()
    status = robot.get_status()
    print(f"Connected: {status.connected}")
    print(f"Status:    {status.status}")
    print(f"Uptime:    {status.uptime:.2f}s")
    state = robot.get_state()
    print(f"Joints:    {state.joint_positions}")
    print(f"EE pos:    {state.ee_position}")
    print(f"Gripper:   {state.gripper_state:.2f}")
    robot.disconnect()
    print("Disconnected.")


def cmd_run(args: argparse.Namespace) -> None:
    """Run a policy on a simulated robot."""
    robot = SimulatedRobot(scenario_name="basic_workspace")
    robot.connect()

    policy = create_policy(args.policy)

    monitor = None
    if not args.no_safety:
        monitor = SafetyMonitor(robot)

    recorder = TrajectoryBuffer() if args.record else None

    summary = run_policy_on_robot(
        robot=robot,
        policy=policy,
        safety_monitor=monitor,
        recorder=recorder,
        verbose=True,
    )

    print("\nRun summary:")
    print(f"  Steps:             {summary['steps']}")
    print(f"  Completed:         {summary['completed']}")
    print(f"  Safety violations: {summary['safety_violations']}")
    print(f"  E-stopped:         {summary['e_stopped']}")

    if recorder is not None:
        print(f"  Recorded steps:    {recorder.length}")

    robot.disconnect()


def cmd_safety_check(args: argparse.Namespace) -> None:
    """Load and display safety configuration."""
    config = load_safety_config()
    print("Safety configuration:")
    print(f"  Joint limits lower: {config.joint_limits_lower}")
    print(f"  Joint limits upper: {config.joint_limits_upper}")
    print(f"  Velocity limits:    {config.velocity_limits}")
    print(f"  Workspace min:      {config.workspace_min}")
    print(f"  Workspace max:      {config.workspace_max}")
    print(f"  Max gripper force:  {config.max_gripper_force}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    """Run mock calibration using two slightly different trajectories."""
    robot1 = SimulatedRobot(scenario_name="basic_workspace")
    robot1.connect()

    # Collect trajectory from first robot
    sim_positions = []
    for _ in range(20):
        state = robot1.get_state()
        sim_positions.append(state.joint_positions.copy())
        # Send a small command to get some variation
        cmd = RobotCommand(
            joint_targets=state.joint_positions + np.random.uniform(-0.01, 0.01, size=7),
            gripper_command=0.0,
            timestamp=0.0,
        )
        robot1.send_command(cmd)
    robot1.disconnect()

    sim_arr = np.array(sim_positions)
    # Create "real" data with a known offset
    offset = np.array([0.01, -0.02, 0.015, -0.01, 0.005, -0.005, 0.01])
    real_arr = sim_arr + offset

    result = calibrate_joint_offsets(sim_arr, real_arr)
    print("Calibration result:")
    print(f"  Joint offsets:    {result.joint_offsets}")
    print(f"  Residual error:   {result.residual_error:.6f}")
    print(f"  Samples:          {result.n_samples}")


def cmd_compare_sim(args: argparse.Namespace) -> None:
    """Compare trajectories with and without domain randomization."""
    # Run 1: baseline
    robot1 = SimulatedRobot(scenario_name="basic_workspace")
    robot1.connect()
    policy1 = NullPolicy()
    policy1._max_steps = 20
    policy1.reset(robot1._sim)

    positions1, velocities1, ee_positions1 = [], [], []
    while not policy1.done:
        policy1.act(robot1._sim)
        state = robot1.get_state()
        positions1.append(state.joint_positions.copy())
        velocities1.append(state.joint_velocities.copy())
        ee_positions1.append(state.ee_position.copy())
    robot1.disconnect()

    # Run 2: with domain randomization
    robot2 = SimulatedRobot(scenario_name="basic_workspace")
    robot2.connect()
    dr_config = load_sim2real_config()
    apply_domain_randomization(robot2._sim, dr_config)
    policy2 = NullPolicy()
    policy2._max_steps = 20
    policy2.reset(robot2._sim)

    positions2, velocities2, ee_positions2 = [], [], []
    while not policy2.done:
        policy2.act(robot2._sim)
        state = robot2.get_state()
        positions2.append(state.joint_positions.copy())
        velocities2.append(state.joint_velocities.copy())
        ee_positions2.append(state.ee_position.copy())
    robot2.disconnect()

    traj1 = {
        "joint_positions": np.array(positions1),
        "joint_velocities": np.array(velocities1),
        "ee_positions": np.array(ee_positions1),
    }
    traj2 = {
        "joint_positions": np.array(positions2),
        "joint_velocities": np.array(velocities2),
        "ee_positions": np.array(ee_positions2),
    }

    comparison = compare_trajectories(traj1, traj2)
    print("Sim comparison (baseline vs domain randomized):")
    print(f"  Position RMSE:     {comparison.position_rmse:.6f}")
    print(f"  Velocity RMSE:     {comparison.velocity_rmse:.6f}")
    print(f"  EE Position RMSE:  {comparison.ee_position_rmse:.6f}")
    print(f"  Max position err:  {comparison.max_position_error:.6f}")
    print(f"  Correlation:       {comparison.correlation:.6f}")
    print(f"  Summary:           {comparison.summary}")


def cmd_status(args: argparse.Namespace) -> None:
    """Connect, print RobotStatus, disconnect."""
    robot = SimulatedRobot(scenario_name="basic_workspace")
    robot.connect()
    status = robot.get_status()
    print(f"Connected:      {status.connected}")
    print(f"Status:         {status.status}")
    print(f"Error message:  {status.error_message}")
    print(f"Uptime:         {status.uptime:.2f}s")
    print(f"Last heartbeat: {status.last_heartbeat:.2f}")
    robot.disconnect()
    print("Disconnected.")


# ── CLI ─────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="forge robot",
        description="Forge physical robot management — connect, run, calibrate, compare",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # connect
    connect_parser = subparsers.add_parser(
        "connect", help="Connect to a robot and print status",
    )
    connect_parser.add_argument(
        "--type", type=str, default="simulated",
        help="Robot type (default: simulated)",
    )
    connect_parser.add_argument(
        "--scenario", type=str, default="basic_workspace",
        help="Scenario name (default: basic_workspace)",
    )
    connect_parser.set_defaults(func=cmd_connect)

    # run
    run_parser = subparsers.add_parser(
        "run", help="Run a policy on the robot",
    )
    run_parser.add_argument(
        "--policy", type=str, default="null",
        help="Policy name (default: null)",
    )
    run_parser.add_argument(
        "--task", type=str, default=None,
        help="Task name (loads scenario from task)",
    )
    run_parser.add_argument(
        "--record", action="store_true",
        help="Record trajectory",
    )
    run_parser.add_argument(
        "--no-safety", action="store_true",
        help="Disable safety monitoring",
    )
    run_parser.set_defaults(func=cmd_run)

    # safety-check
    safety_parser = subparsers.add_parser(
        "safety-check", help="Display safety configuration",
    )
    safety_parser.set_defaults(func=cmd_safety_check)

    # calibrate
    calibrate_parser = subparsers.add_parser(
        "calibrate", help="Run mock joint calibration",
    )
    calibrate_parser.set_defaults(func=cmd_calibrate)

    # compare-sim
    compare_parser = subparsers.add_parser(
        "compare-sim", help="Compare sim trajectories with/without domain randomization",
    )
    compare_parser.set_defaults(func=cmd_compare_sim)

    # status
    status_parser = subparsers.add_parser(
        "status", help="Print robot status",
    )
    status_parser.set_defaults(func=cmd_status)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(getattr(args, "verbose", False))

    if not args.command:
        parser.print_help()
        return

    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
