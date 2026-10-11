#!/usr/bin/env python3
"""Forge regression evaluation CLI — manage baselines and detect regressions.

Usage:
    # Check for regressions against a baseline
    python3 scripts/forge_regression.py --check --suite manipulation-v1

    # Create a new baseline from current results
    python3 scripts/forge_regression.py --create-baseline --suite manipulation-v1

    # Update (overwrite) an existing baseline
    python3 scripts/forge_regression.py --update-baseline --suite manipulation-v1

    # Show contents of a saved baseline
    python3 scripts/forge_regression.py --show-baseline --suite manipulation-v1

    # Compare two baseline files
    python3 scripts/forge_regression.py --compare baselines/a.json baselines/b.json

    # Override policy
    python3 scripts/forge_regression.py --check --suite manipulation-v1 --policy scripted_pick_and_place

    # Override regression config
    python3 scripts/forge_regression.py --check --suite manipulation-v1 --threshold-config path/to/config.yaml
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

from forge_core.regression import (
    check_regression,
    load_baseline,
    load_regression_config,
    save_baseline,
)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _run_suite(suite_name: str, policy_name: str):
    """Run a suite and return evaluation results."""
    from forge_core.policy import create_policy
    from forge_core.runner import run_suite
    from forge_core.task import load_suite

    task_names = load_suite(suite_name)
    policy = create_policy(policy_name)
    print(f"\nRunning suite '{suite_name}' ({len(task_names)} tasks)...\n")
    return run_suite(task_names, policy)


def cmd_check(args):
    """Check for regressions against a saved baseline."""
    config = load_regression_config(args.threshold_config)
    suite = args.suite or config.get("suite", "manipulation-v1")
    tolerance = config.get("metric_tolerance", 0.1)
    allow_new = config.get("allow_new_failures", False)

    results = _run_suite(suite, args.policy)
    baseline = load_baseline(suite)
    report = check_regression(
        results, baseline,
        metric_tolerance=tolerance,
        allow_new_failures=allow_new,
    )
    print(report.summary())
    sys.exit(0 if report.overall_passed else 1)


def cmd_create_baseline(args):
    """Create a new baseline from current evaluation results."""
    config = load_regression_config(args.threshold_config)
    suite = args.suite or config.get("suite", "manipulation-v1")

    results = _run_suite(suite, args.policy)
    path = save_baseline(results, suite)
    print(f"\nBaseline saved: {path}")


def cmd_show_baseline(args):
    """Pretty-print a saved baseline."""
    config = load_regression_config(args.threshold_config)
    suite = args.suite or config.get("suite", "manipulation-v1")

    baseline = load_baseline(suite)
    print(f"\nBaseline: {baseline.suite_name}")
    print(f"Timestamp: {baseline.timestamp}")
    print(f"Git commit: {baseline.git_commit}")
    print(f"Tasks ({len(baseline.tasks)}):")
    for name, data in baseline.tasks.items():
        status = "PASS" if data["success"] else "FAIL"
        sim_time = data.get("sim_time", 0.0)
        print(f"  {name:<30} {status:<6} sim_time={sim_time:.3f}s")
    print()


def cmd_compare(args):
    """Compare two baseline JSON files side by side."""
    path_a, path_b = Path(args.files[0]), Path(args.files[1])

    with open(path_a, "r") as f:
        data_a = json.load(f)
    with open(path_b, "r") as f:
        data_b = json.load(f)

    tasks_a = data_a.get("tasks", {})
    tasks_b = data_b.get("tasks", {})
    all_tasks = sorted(set(tasks_a.keys()) | set(tasks_b.keys()))

    print(f"\nComparing baselines:")
    print(f"  A: {path_a} ({data_a.get('timestamp', '?')})")
    print(f"  B: {path_b} ({data_b.get('timestamp', '?')})")
    print(f"\n{'Task':<30} {'A':<10} {'B':<10} {'Delta sim_time'}")
    print("-" * 70)
    for task in all_tasks:
        a = tasks_a.get(task, {})
        b = tasks_b.get(task, {})
        a_status = "PASS" if a.get("success") else "FAIL" if a else "N/A"
        b_status = "PASS" if b.get("success") else "FAIL" if b else "N/A"
        a_time = a.get("sim_time", 0.0)
        b_time = b.get("sim_time", 0.0)
        delta = b_time - a_time if a and b else float("nan")
        delta_str = f"{delta:+.3f}s" if a and b else "N/A"
        print(f"  {task:<28} {a_status:<10} {b_status:<10} {delta_str}")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="forge regression",
        description="Forge regression evaluation CLI — manage baselines and detect regressions",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="Run suite and check for regressions against baseline")
    mode.add_argument("--create-baseline", action="store_true",
                      help="Run suite and save results as a new baseline")
    mode.add_argument("--update-baseline", action="store_true",
                      help="Run suite and overwrite existing baseline")
    mode.add_argument("--show-baseline", action="store_true",
                      help="Display contents of a saved baseline")
    mode.add_argument("--compare", nargs=2, metavar="FILE", dest="files",
                      help="Compare two baseline JSON files")

    parser.add_argument("--suite", type=str, default=None,
                        help="Suite name (default: from config or manipulation-v1)")
    parser.add_argument("--policy", type=str, default="scripted_pick_and_place",
                        help="Policy to use (default: scripted_pick_and_place)")
    parser.add_argument("--threshold-config", type=str, default=None,
                        help="Path to regression config YAML (overrides config/regression.yaml)")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.check:
        cmd_check(args)
    elif args.create_baseline or args.update_baseline:
        cmd_create_baseline(args)
    elif args.show_baseline:
        cmd_show_baseline(args)
    elif args.files:
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
