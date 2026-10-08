#!/usr/bin/env python3
"""Forge evaluation CLI — run tasks and suites, report results.

Usage:
    # Run a single task
    python3 scripts/forge_eval.py --task pick_red_cube

    # Run a single task with verbose output
    python3 scripts/forge_eval.py --task pick_red_cube --verbose

    # Run a test suite
    python3 scripts/forge_eval.py --suite manipulation-v1

    # Run all available tasks
    python3 scripts/forge_eval.py --all

    # Save results to JSON
    python3 scripts/forge_eval.py --task pick_red_cube --save

    # List available tasks and suites
    python3 scripts/forge_eval.py --list

    # Specify a policy
    python3 scripts/forge_eval.py --task pick_red_cube --policy scripted_pick_and_place
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.policy import create_policy
from forge_core.runner import (
    run_task_by_name,
    run_suite,
    save_result,
    format_suite_summary,
)
from forge_core.task import list_tasks, load_suite


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_list():
    """List available tasks and suites."""
    print("\nAvailable tasks:")
    for name in list_tasks():
        print(f"  {name}")

    suites_dir = Path(_project_root) / "tasks" / "suites"
    if suites_dir.exists():
        print("\nAvailable suites:")
        for p in sorted(suites_dir.glob("*.yaml")):
            print(f"  {p.stem}")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="forge eval",
        description="Forge task evaluation CLI",
    )
    parser.add_argument("--task", type=str, help="Run a single task by name")
    parser.add_argument("--suite", type=str, help="Run a test suite by name")
    parser.add_argument("--all", action="store_true", help="Run all available tasks")
    parser.add_argument("--list", action="store_true", help="List tasks and suites")
    parser.add_argument("--policy", type=str, default="scripted_pick_and_place",
                        help="Policy to use (default: scripted_pick_and_place)")
    parser.add_argument("--target-object", type=str, default=None,
                        help="Override target object for policy")
    parser.add_argument("--save", action="store_true", help="Save results to runs/")
    parser.add_argument("--record", action="store_true",
                        help="Record trajectories to runs/run_XXXXX/")
    parser.add_argument("--fault", type=str, default=None,
                        help="Fault profile name for chaos testing")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.list:
        cmd_list()
        return

    if not (args.task or args.suite or args.all):
        parser.print_help()
        return

    # Create policy
    policy_kwargs = {}
    if args.target_object:
        policy_kwargs["target_object"] = args.target_object
    policy = create_policy(args.policy, **policy_kwargs)

    # Create fault injector if requested
    fault_injector = None
    if args.fault:
        from forge_core.faults import FaultInjector
        fault_injector = FaultInjector.from_profile(args.fault)

    results = []

    if args.task:
        result = run_task_by_name(args.task, policy, verbose=args.verbose,
                                   record=args.record, fault_injector=fault_injector)
        results.append(result)
        print()
        print(result.summary())

    elif args.suite:
        task_names = load_suite(args.suite)
        print(f"\nRunning suite '{args.suite}' ({len(task_names)} tasks)...\n")
        results = run_suite(task_names, policy, verbose=args.verbose,
                            record=args.record, fault_injector=fault_injector)
        print(format_suite_summary(results))

        for r in results:
            print(r.summary())
            print()

    elif args.all:
        task_names = list_tasks()
        print(f"\nRunning all {len(task_names)} tasks...\n")
        results = run_suite(task_names, policy, verbose=args.verbose,
                            record=args.record, fault_injector=fault_injector)
        print(format_suite_summary(results))

    # Save results if requested
    if args.save:
        for r in results:
            path = save_result(r)
            print(f"Saved: {path}")

    # Exit code: 0 if all passed, 1 if any failed
    all_passed = all(r.success for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
