#!/usr/bin/env python3
"""Forge training CLI — train learned policies from demonstrations.

Usage:
    # Collect demos from scripted policy, then train
    python3 scripts/forge_train.py --task pick_red_cube --episodes 5

    # Train from a recorded run directory
    python3 scripts/forge_train.py --from-run runs/run_00001/

    # Train with custom params
    python3 scripts/forge_train.py --task pick_red_cube --episodes 10 --epochs 200 --lr 0.0005

    # Evaluate the trained policy
    python3 scripts/forge_eval.py --task pick_red_cube --policy learned --model-path models/bc_policy.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.learned_policy import (
    collect_demos,
    train_bc,
    DemoDataset,
)
from forge_core.policy import create_policy


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        prog="forge train",
        description="Train learned policies via behavioral cloning",
    )
    parser.add_argument("--task", type=str, default="pick_red_cube",
                        help="Task to collect demonstrations from")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of demo episodes to collect")
    parser.add_argument("--from-run", type=str, default=None,
                        help="Train from an existing recorded run directory")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Where to save the trained model")
    parser.add_argument("--demo-policy", type=str, default="scripted_pick_and_place",
                        help="Policy to use for demo collection")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.from_run:
        # Train from recorded run
        print(f"\nLoading demos from {args.from_run}...")
        dataset = DemoDataset.from_run_dir(args.from_run)
        print(f"  {len(dataset)} observation-action pairs")
    else:
        # Collect fresh demos
        print(f"\nCollecting {args.episodes} demo episodes for '{args.task}'...")
        demo_policy = create_policy(args.demo_policy)
        dataset = collect_demos(
            task_name=args.task,
            policy=demo_policy,
            n_episodes=args.episodes,
        )
        print(f"  {len(dataset)} observation-action pairs")

    # Train
    print(f"\nTraining for {args.epochs} epochs (lr={args.lr}, batch={args.batch_size})...")
    result = train_bc(
        dataset=dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
    )

    print(f"\n{result.summary()}")
    print(f"\nTo evaluate:")
    print(f"  python3 scripts/forge_eval.py --task {args.task} "
          f"--policy learned --model-path {result.model_path}")


if __name__ == "__main__":
    main()
