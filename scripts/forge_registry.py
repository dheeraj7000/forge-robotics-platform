#!/usr/bin/env python3
"""Forge registry CLI — manage experiments and model registry.

Usage:
    # List experiments
    python3 scripts/forge_registry.py experiments list
    python3 scripts/forge_registry.py experiments list --status completed
    python3 scripts/forge_registry.py experiments list --tag bc

    # Show experiment details
    python3 scripts/forge_registry.py experiments show exp_00001

    # Compare experiments
    python3 scripts/forge_registry.py experiments compare exp_00001 exp_00002

    # Delete an experiment
    python3 scripts/forge_registry.py experiments delete exp_00001

    # List registered models
    python3 scripts/forge_registry.py models list
    python3 scripts/forge_registry.py models list --name bc_pick_red_cube

    # Show model details
    python3 scripts/forge_registry.py models show bc_pick_red_cube

    # Promote a model
    python3 scripts/forge_registry.py models promote bc_pick_red_cube 1 validated

    # Delete a model entry
    python3 scripts/forge_registry.py models delete bc_pick_red_cube 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.experiment import ExperimentTracker
from forge_core.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Experiment subcommands ──────────────────────────────────

def cmd_experiments_list(args):
    """List experiments."""
    tracker = ExperimentTracker()
    experiments = tracker.list_experiments(
        status=args.status,
        tag=args.tag,
    )
    if not experiments:
        print("No experiments found.")
        return

    print(f"\n{'ID':<12} {'Name':<30} {'Status':<12} {'Loss':>10} {'Timestamp':<22}")
    print("-" * 88)
    for exp in experiments:
        exp_id = exp.get("experiment_id", "")
        name = exp.get("name") or ""
        if len(name) > 28:
            name = name[:25] + "..."
        status = exp.get("status", "")
        metrics = exp.get("metrics", {})
        final_loss = metrics.get("final_loss")
        loss_str = f"{final_loss:.6f}" if final_loss is not None else "-"
        timestamp = exp.get("timestamp", "")[:19]
        print(f"{exp_id:<12} {name:<30} {status:<12} {loss_str:>10} {timestamp:<22}")
    print()


def cmd_experiments_show(args):
    """Show full experiment details."""
    tracker = ExperimentTracker()
    try:
        exp = tracker.get_experiment(args.experiment_id)
    except FileNotFoundError:
        print(f"Error: experiment '{args.experiment_id}' not found")
        sys.exit(1)

    print(f"\nExperiment: {exp.get('experiment_id')}")
    print(f"  Name:       {exp.get('name') or '-'}")
    print(f"  Status:     {exp.get('status')}")
    print(f"  Timestamp:  {exp.get('timestamp')}")
    print(f"  Git commit: {exp.get('git_commit') or '-'}")
    print(f"  Model path: {exp.get('model_path') or '-'}")
    print(f"  Duration:   {exp.get('duration_seconds') or '-'}s")
    print(f"  Tags:       {', '.join(exp.get('tags', [])) or '-'}")

    if exp.get("error"):
        print(f"  Error:      {exp['error']}")

    metrics = exp.get("metrics", {})
    if metrics:
        print(f"\n  Metrics:")
        for key in ("final_loss", "final_val_loss", "best_epoch", "best_loss"):
            val = metrics.get(key)
            if val is not None:
                print(f"    {key}: {val}")

    config = exp.get("config", {})
    if config:
        print(f"\n  Config:")
        for key, val in config.items():
            print(f"    {key}: {val}")

    dataset_info = exp.get("dataset_info", {})
    if dataset_info:
        print(f"\n  Dataset:")
        for key, val in dataset_info.items():
            print(f"    {key}: {val}")
    print()


def cmd_experiments_compare(args):
    """Compare experiments side by side."""
    tracker = ExperimentTracker()
    try:
        comparison = tracker.compare_experiments(args.experiment_ids)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    experiments = comparison["experiments"]
    if not experiments:
        print("No experiments to compare.")
        return

    print(f"\n{'ID':<12} {'Name':<20} {'Status':<12} {'Loss':>10} "
          f"{'Val Loss':>10} {'Epochs':>8} {'Duration':>10}")
    print("-" * 84)
    for exp in experiments:
        exp_id = exp.get("experiment_id", "")
        name = exp.get("name") or ""
        if len(name) > 18:
            name = name[:15] + "..."
        status = exp.get("status", "")
        loss = exp.get("final_loss")
        loss_str = f"{loss:.6f}" if loss is not None else "-"
        val_loss = exp.get("final_val_loss")
        val_str = f"{val_loss:.6f}" if val_loss is not None else "-"
        epochs = exp.get("epochs")
        epochs_str = str(epochs) if epochs is not None else "-"
        dur = exp.get("duration_seconds")
        dur_str = f"{dur:.1f}s" if dur is not None else "-"
        print(f"{exp_id:<12} {name:<20} {status:<12} {loss_str:>10} "
              f"{val_str:>10} {epochs_str:>8} {dur_str:>10}")
    print()


def cmd_experiments_delete(args):
    """Delete an experiment."""
    tracker = ExperimentTracker()
    if not args.force:
        answer = input(f"Delete experiment '{args.experiment_id}'? [y/N] ")
        if answer.lower() not in ("y", "yes"):
            print("Cancelled.")
            return
    try:
        tracker.delete_experiment(args.experiment_id)
        print(f"Deleted experiment: {args.experiment_id}")
    except FileNotFoundError:
        print(f"Error: experiment '{args.experiment_id}' not found")
        sys.exit(1)


# ── Model subcommands ───────────────────────────────────────

def cmd_models_list(args):
    """List registered models."""
    registry = ModelRegistry()
    models = registry.list_models(name=args.name, stage=args.stage)
    if not models:
        print("No models registered.")
        return

    print(f"\n{'Name':<25} {'Version':>8} {'Stage':<12} {'Experiment':<12} {'Registered':<22}")
    print("-" * 81)
    for m in models:
        name = m.get("name", "")
        version = m.get("version", "")
        stage = m.get("stage", "")
        exp_id = m.get("experiment_id") or "-"
        registered = m.get("registered_at", "")[:19]
        print(f"{name:<25} {version:>8} {stage:<12} {exp_id:<12} {registered:<22}")
    print()


def cmd_models_show(args):
    """Show full model details."""
    registry = ModelRegistry()
    entry = registry.get_model(args.name, version=args.version)
    if entry is None:
        version_str = f" v{args.version}" if args.version else ""
        print(f"Error: model '{args.name}{version_str}' not found")
        sys.exit(1)

    print(f"\nModel: {entry['name']} v{entry['version']}")
    print(f"  Stage:        {entry.get('stage')}")
    print(f"  Checkpoint:   {entry.get('checkpoint_path')}")
    print(f"  Experiment:   {entry.get('experiment_id') or '-'}")
    print(f"  Registered:   {entry.get('registered_at')}")
    print(f"  Description:  {entry.get('description') or '-'}")

    metadata = entry.get("metadata", {})
    if metadata:
        print(f"\n  Metadata:")
        for key, val in metadata.items():
            print(f"    {key}: {val}")
    print()


def cmd_models_promote(args):
    """Promote a model to a new stage."""
    registry = ModelRegistry()
    try:
        entry = registry.promote(args.name, args.version, args.stage)
        print(f"Promoted {entry['name']} v{entry['version']} -> {entry['stage']}")
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def cmd_models_delete(args):
    """Delete a model entry."""
    registry = ModelRegistry()
    if not args.force:
        answer = input(f"Delete model '{args.name}' v{args.version}? [y/N] ")
        if answer.lower() not in ("y", "yes"):
            print("Cancelled.")
            return
    try:
        registry.delete_model(args.name, args.version)
        print(f"Deleted model: {args.name} v{args.version}")
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


# ── CLI ─────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="forge registry",
        description="Forge experiment and model registry management",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    subparsers = parser.add_subparsers(dest="category", help="Category")

    # ── experiments ──────────────────────────────────────────
    exp_parser = subparsers.add_parser("experiments", help="Manage experiments")
    exp_sub = exp_parser.add_subparsers(dest="command", help="Command")

    # experiments list
    exp_list = exp_sub.add_parser("list", help="List experiments")
    exp_list.add_argument("--status", type=str, default=None,
                          help="Filter by status (running, completed, failed)")
    exp_list.add_argument("--tag", type=str, default=None,
                          help="Filter by tag")
    exp_list.set_defaults(func=cmd_experiments_list)

    # experiments show
    exp_show = exp_sub.add_parser("show", help="Show experiment details")
    exp_show.add_argument("experiment_id", type=str, help="Experiment ID")
    exp_show.set_defaults(func=cmd_experiments_show)

    # experiments compare
    exp_compare = exp_sub.add_parser("compare", help="Compare experiments")
    exp_compare.add_argument("experiment_ids", type=str, nargs="+",
                             help="Experiment IDs to compare")
    exp_compare.set_defaults(func=cmd_experiments_compare)

    # experiments delete
    exp_delete = exp_sub.add_parser("delete", help="Delete an experiment")
    exp_delete.add_argument("experiment_id", type=str, help="Experiment ID")
    exp_delete.add_argument("--force", action="store_true",
                            help="Skip confirmation prompt")
    exp_delete.set_defaults(func=cmd_experiments_delete)

    # ── models ───────────────────────────────────────────────
    model_parser = subparsers.add_parser("models", help="Manage model registry")
    model_sub = model_parser.add_subparsers(dest="command", help="Command")

    # models list
    model_list = model_sub.add_parser("list", help="List registered models")
    model_list.add_argument("--name", type=str, default=None,
                            help="Filter by model name")
    model_list.add_argument("--stage", type=str, default=None,
                            help="Filter by stage")
    model_list.set_defaults(func=cmd_models_list)

    # models show
    model_show = model_sub.add_parser("show", help="Show model details")
    model_show.add_argument("name", type=str, help="Model name")
    model_show.add_argument("--version", type=int, default=None,
                            help="Model version (default: latest)")
    model_show.set_defaults(func=cmd_models_show)

    # models promote
    model_promote = model_sub.add_parser("promote", help="Promote model stage")
    model_promote.add_argument("name", type=str, help="Model name")
    model_promote.add_argument("version", type=int, help="Model version")
    model_promote.add_argument("stage", type=str, help="Target stage")
    model_promote.set_defaults(func=cmd_models_promote)

    # models delete
    model_delete = model_sub.add_parser("delete", help="Delete model entry")
    model_delete.add_argument("name", type=str, help="Model name")
    model_delete.add_argument("version", type=int, help="Model version")
    model_delete.add_argument("--force", action="store_true",
                              help="Skip confirmation prompt")
    model_delete.set_defaults(func=cmd_models_delete)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(getattr(args, "verbose", False))

    if not args.category:
        parser.print_help()
        return

    if not args.command:
        # Print help for the category subparser
        if args.category == "experiments":
            parser.parse_args(["experiments", "--help"])
        elif args.category == "models":
            parser.parse_args(["models", "--help"])
        return

    args.func(args)


if __name__ == "__main__":
    main()
