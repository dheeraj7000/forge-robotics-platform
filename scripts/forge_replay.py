#!/usr/bin/env python3
"""Forge replay CLI — inspect recorded runs.

Usage:
    python3 scripts/forge_replay.py --list
    python3 scripts/forge_replay.py runs/run_00001/
    python3 scripts/forge_replay.py runs/run_00001/ --summary
    python3 scripts/forge_replay.py runs/run_00001/ --step 50
    python3 scripts/forge_replay.py runs/run_00001/ --time 5.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.replay import load_run, list_runs


def main():
    parser = argparse.ArgumentParser(
        prog="forge replay",
        description="Inspect recorded Forge runs",
    )
    parser.add_argument("run_dir", nargs="?", help="Path to a run directory")
    parser.add_argument("--list", action="store_true", help="List all recorded runs")
    parser.add_argument("--summary", action="store_true", help="Print run summary")
    parser.add_argument("--step", type=int, help="Show state at step N")
    parser.add_argument("--time", type=float, help="Show state at time T seconds")

    args = parser.parse_args()

    if args.list:
        runs = list_runs()
        if not runs:
            print("No recorded runs found.")
            return
        print(f"\nRecorded runs ({len(runs)}):")
        for run_dir in runs:
            meta_path = run_dir / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                status = "PASS" if meta.get("success") else "FAIL"
                task = meta.get("task_name", "?")
                print(f"  {run_dir.name}  {task:<25} {status}")
            else:
                print(f"  {run_dir.name}  (no metadata)")
        print()
        return

    if not args.run_dir:
        parser.print_help()
        return

    record = load_run(args.run_dir)

    if args.step is not None:
        state = record.state_at_step(args.step)
        print(json.dumps(state, indent=2))
    elif args.time is not None:
        state = record.state_at_time(args.time)
        print(json.dumps(state, indent=2))
    else:
        # Default: print summary
        print()
        print(record.summary())
        print()


if __name__ == "__main__":
    main()
