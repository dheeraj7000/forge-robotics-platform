#!/usr/bin/env python3
"""Forge training CLI — train learned policies from demonstrations.

Usage:
    # Collect demos from scripted policy, then train
    python3 scripts/forge_train.py --task pick_red_cube --episodes 5

    # Train from a recorded run directory
    python3 scripts/forge_train.py --from-run runs/run_00001/

    # Train from multiple recorded runs
    python3 scripts/forge_train.py --from-run runs/run_00001/ runs/run_00002/

    # Train with custom params
    python3 scripts/forge_train.py --task pick_red_cube --episodes 10 --epochs 200 --lr 0.0005

    # Train with a YAML config
    python3 scripts/forge_train.py --config config/training.yaml

    # List saved models
    python3 scripts/forge_train.py --list-models

    # Evaluate the trained policy
    python3 scripts/forge_eval.py --task pick_red_cube --policy learned --model-path models/bc_policy.pt
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Any

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.config import PROJECT_ROOT
from forge_core.learned_policy import (
    collect_demos,
    load_training_config,
    train_bc,
    DemoDataset,
    PolicyNetwork,
)
from forge_core.policy import create_policy


# ── Config merge helpers ────────────────────────────────────

_YAML_TO_FLAT: dict[str, str] = {
    "training.epochs": "epochs",
    "training.batch_size": "batch_size",
    "training.learning_rate": "lr",
    "training.validation_split": "validation_split",
    "training.checkpoint_interval": "checkpoint_interval",
    "model.hidden_layers": "hidden_layers",
    "model.obs_dim": "obs_dim",
    "model.act_dim": "act_dim",
    "data.target_object": "target_object",
    "data.demo_episodes": "episodes",
    "data.demo_policy": "demo_policy",
}

CODE_DEFAULTS: dict[str, Any] = {
    "epochs": 100,
    "batch_size": 64,
    "lr": 1e-3,
    "validation_split": 0.0,
    "checkpoint_interval": None,
    "target_object": "red_cube",
    "episodes": 3,
    "demo_policy": "scripted_pick_and_place",
    "hidden_layers": [256, 256, 128],
    "obs_dim": 31,
    "act_dim": 8,
}


def _flatten_training_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested YAML training config to flat keys matching CODE_DEFAULTS."""
    flat: dict[str, Any] = {}
    for yaml_key, flat_key in _YAML_TO_FLAT.items():
        section, key = yaml_key.split(".", 1)
        value = raw.get(section, {}).get(key)
        if value is not None:
            flat[flat_key] = value
    return flat


def merge_training_config(
    code_defaults: dict[str, Any],
    yaml_config: dict[str, Any],
    cli_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge config: code defaults ← YAML config ← CLI overrides.

    cli_overrides should only contain keys the user explicitly passed
    (non-None argparse values).
    """
    result = {**code_defaults}
    for k, v in yaml_config.items():
        if v is not None:
            result[k] = v
    for k, v in cli_overrides.items():
        if v is not None:
            result[k] = v
    return result


# ── Model listing ───────────────────────────────────────────

def list_models() -> None:
    """List saved model files in the models/ directory."""
    models_dir = PROJECT_ROOT / "models"
    if not models_dir.exists():
        print("No trained models found in models/")
        return

    pt_files = sorted(models_dir.glob("*.pt"))
    if not pt_files:
        print("No trained models found in models/")
        return

    print(f"\n{'Filename':<35} {'Size':>10} {'Modified':<20}")
    print("-" * 67)
    for f in pt_files:
        size = f.stat().st_size
        if size >= 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        else:
            size_str = f"{size / 1024:.1f} KB"
        mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime)
        mtime_str = mtime.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{f.name:<35} {size_str:>10} {mtime_str:<20}")


# ── CLI ─────────────────────────────────────────────────────

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
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML training config")
    parser.add_argument("--task", type=str, default="pick_red_cube",
                        help="Task to collect demonstrations from")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of demo episodes to collect")
    parser.add_argument("--from-run", type=str, nargs="+", default=None,
                        help="Train from existing recorded run directory(ies)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Where to save the trained model")
    parser.add_argument("--demo-policy", type=str, default=None,
                        help="Policy to use for demo collection")
    parser.add_argument("--list-models", action="store_true",
                        help="List saved models and exit")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)

    # List models and exit
    if args.list_models:
        list_models()
        return

    # Load and merge config
    yaml_flat: dict[str, Any] = {}
    if args.config:
        raw_config = load_training_config(args.config)
        yaml_flat = _flatten_training_config(raw_config)

    cli_overrides: dict[str, Any] = {}
    if args.episodes is not None:
        cli_overrides["episodes"] = args.episodes
    if args.demo_policy is not None:
        cli_overrides["demo_policy"] = args.demo_policy
    if args.epochs is not None:
        cli_overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        cli_overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        cli_overrides["lr"] = args.lr

    cfg = merge_training_config(CODE_DEFAULTS, yaml_flat, cli_overrides)

    # Build network from config
    network = PolicyNetwork(
        obs_dim=cfg["obs_dim"],
        act_dim=cfg["act_dim"],
        hidden=cfg["hidden_layers"],
    )

    if args.from_run:
        # Train from recorded run(s)
        print(f"\nLoading demos from {len(args.from_run)} run(s)...")
        dataset = DemoDataset.from_run_dirs(args.from_run, target_object=cfg["target_object"])
        print(f"  {len(dataset)} observation-action pairs")
    else:
        # Collect fresh demos
        print(f"\nCollecting {cfg['episodes']} demo episodes for '{args.task}'...")
        demo_policy = create_policy(cfg["demo_policy"])
        dataset = collect_demos(
            task_name=args.task,
            policy=demo_policy,
            n_episodes=cfg["episodes"],
            target_object=cfg["target_object"],
        )
        print(f"  {len(dataset)} observation-action pairs")

    # Train
    print(f"\nTraining for {cfg['epochs']} epochs (lr={cfg['lr']}, batch={cfg['batch_size']})...")
    result = train_bc(
        dataset=dataset,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        save_path=args.save_path,
        network=network,
        validation_split=cfg["validation_split"],
        checkpoint_interval=cfg["checkpoint_interval"],
    )

    print(f"\n{result.summary()}")
    print(f"\nTo evaluate:")
    print(f"  python3 scripts/forge_eval.py --task {args.task} "
          f"--policy learned --model-path {result.model_path}")


if __name__ == "__main__":
    main()
