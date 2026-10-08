#!/usr/bin/env python3
"""Forge chaos testing CLI — run tasks with fault injection.

Usage:
    # List fault profiles
    python3 scripts/forge_chaos.py --list

    # Run a task with a fault profile
    python3 scripts/forge_chaos.py --task pick_red_cube --fault object_moved

    # Run with recording
    python3 scripts/forge_chaos.py --task pick_red_cube --fault cascading --record

    # Quick inline fault (no YAML needed)
    python3 scripts/forge_chaos.py --task pick_red_cube --inject object_moved --at 5.0

    # Compare: run with and without faults
    python3 scripts/forge_chaos.py --task pick_red_cube --fault object_moved --compare
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.faults import (
    FaultConfig,
    FaultInjector,
    list_fault_profiles,
    load_fault_profile,
    FAULT_REGISTRY,
)
from forge_core.policy import create_policy
from forge_core.runner import run_task_by_name, format_suite_summary


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        prog="forge chaos",
        description="Forge chaos testing — run tasks with fault injection",
    )
    parser.add_argument("--task", type=str, help="Task to run")
    parser.add_argument("--fault", type=str, help="Fault profile name (from faults/)")
    parser.add_argument("--inject", type=str, help="Quick inline fault type")
    parser.add_argument("--at", type=float, default=5.0,
                        help="Trigger time for inline fault (default: 5.0)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="Duration for inline fault (default: 0 = instant)")
    parser.add_argument("--policy", type=str, default="scripted_pick_and_place")
    parser.add_argument("--compare", action="store_true",
                        help="Run with AND without faults, show comparison")
    parser.add_argument("--record", action="store_true", help="Record trajectories")
    parser.add_argument("--list", action="store_true", help="List fault profiles")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.list:
        print("\nAvailable fault profiles:")
        for name in list_fault_profiles():
            print(f"  {name}")
        print("\nAvailable fault types:")
        for name in sorted(FAULT_REGISTRY.keys()):
            print(f"  {name}")
        print()
        return

    if not args.task:
        parser.print_help()
        return

    # Build fault injector
    injector = None
    if args.fault:
        injector = FaultInjector.from_profile(args.fault)
    elif args.inject:
        config = FaultConfig(
            type=args.inject,
            trigger_time=args.at,
            duration=args.duration,
            params=_default_params(args.inject),
        )
        injector = FaultInjector.from_configs([config])

    policy = create_policy(args.policy)

    if args.compare:
        # Run without faults
        print("\n--- Baseline (no faults) ---")
        baseline = run_task_by_name(args.task, policy, verbose=args.verbose,
                                     record=args.record)
        print(baseline.summary())

        # Reset policy for second run
        print("\n--- With faults ---")
        policy2 = create_policy(args.policy)
        result = run_task_by_name(args.task, policy2, verbose=args.verbose,
                                   record=args.record, fault_injector=injector)
        print(result.summary())

        # Comparison
        print("\n--- Comparison ---")
        b_status = "PASS" if baseline.success else "FAIL"
        f_status = "PASS" if result.success else "FAIL"
        print(f"  Baseline: {b_status}  Faulted: {f_status}")
        if "faults" in result.metrics:
            print(f"  Faults: {result.metrics['faults']}")
        print()
    else:
        result = run_task_by_name(args.task, policy, verbose=args.verbose,
                                   record=args.record, fault_injector=injector)
        print()
        print(result.summary())

    sys.exit(0 if result.success else 1)


def _default_params(fault_type: str) -> dict:
    """Provide sensible default params for quick inline faults."""
    defaults = {
        "object_moved": {"object_id": "red_cube", "new_position": [0.6, -0.1, 0.245]},
        "sensor_noise": {"std": 0.05},
        "sensor_delay": {"delay_steps": 50},
        "actuator_stuck": {"joints": [1]},
        "controller_dropout": {},
        "gravity_shift": {"gravity": [0, 0, -4.905]},
    }
    return defaults.get(fault_type, {})


if __name__ == "__main__":
    main()
