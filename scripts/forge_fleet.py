#!/usr/bin/env python3
"""Forge fleet & OTA CLI — manage devices, OTA updates, and fleet overview.

Usage:
    # Register a device
    python3 scripts/forge_fleet.py devices register --id arm-01 --name "Panda Arm 01" --type panda

    # List devices
    python3 scripts/forge_fleet.py devices list
    python3 scripts/forge_fleet.py devices list --status online --group production

    # Update device status
    python3 scripts/forge_fleet.py devices status --id arm-01 --set online

    # Tag management
    python3 scripts/forge_fleet.py devices tag --id arm-01 --add gpu
    python3 scripts/forge_fleet.py devices tag --id arm-01 --remove gpu

    # Group management
    python3 scripts/forge_fleet.py devices group --id arm-01 --add production

    # Deregister a device
    python3 scripts/forge_fleet.py devices deregister --id arm-01

    # Create an OTA package
    python3 scripts/forge_fleet.py ota create-package --source-dir deploy/bc_policy/ --version 1.0.0 --model-name bc_pick

    # Deploy an OTA package
    python3 scripts/forge_fleet.py ota deploy --package-id pkg_00001 --target-group production

    # Check rollout status
    python3 scripts/forge_fleet.py ota status --rollout-id rollout_00001

    # Rollback a rollout
    python3 scripts/forge_fleet.py ota rollback --rollout-id rollout_00001

    # View device update history
    python3 scripts/forge_fleet.py ota history --device-id arm-01

    # List packages and rollouts
    python3 scripts/forge_fleet.py ota list-packages
    python3 scripts/forge_fleet.py ota list-rollouts --status completed

    # Fleet overview
    python3 scripts/forge_fleet.py overview
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

from forge_core.fleet import FleetManager
from forge_core.ota import OTAManager

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Device subcommands ──────────────────────────────────────

def cmd_devices_register(args):
    """Register a new device."""
    fleet = FleetManager()
    capabilities = [c.strip() for c in args.capabilities.split(",")] if args.capabilities else []
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else []
    groups = [g.strip() for g in args.groups.split(",")] if args.groups else []
    metadata = json.loads(args.metadata) if args.metadata else {}
    try:
        device = fleet.register_device(
            device_id=args.id,
            name=args.name,
            device_type=args.type,
            capabilities=capabilities,
            tags=tags,
            groups=groups,
            metadata=metadata,
        )
        print(f"Registered device: {device['device_id']} ({device['name']})")
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def cmd_devices_deregister(args):
    """Deregister a device."""
    fleet = FleetManager()
    if not args.force:
        answer = input(f"Deregister device '{args.id}'? [y/N] ")
        if answer.lower() not in ("y", "yes"):
            print("Cancelled.")
            return
    try:
        fleet.deregister_device(args.id)
        print(f"Deregistered device: {args.id}")
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def cmd_devices_list(args):
    """List registered devices."""
    fleet = FleetManager()
    devices = fleet.list_devices(
        status=args.status,
        group=args.group,
        tag=args.tag,
    )
    if not devices:
        print("No devices found.")
        return

    print(f"\n{'ID':<15} {'Name':<25} {'Type':<12} {'Status':<10} {'Model':<20} {'Groups':<20}")
    print("-" * 104)
    for d in devices:
        did = d.get("device_id", "")
        name = d.get("name", "")
        if len(name) > 23:
            name = name[:20] + "..."
        dtype = d.get("device_type", "")
        status = d.get("status", "")
        model = d.get("current_model") or "-"
        if d.get("current_model_version") is not None:
            model = f"{model} v{d['current_model_version']}"
        if len(model) > 18:
            model = model[:15] + "..."
        groups = ", ".join(d.get("groups", [])) or "-"
        if len(groups) > 18:
            groups = groups[:15] + "..."
        print(f"{did:<15} {name:<25} {dtype:<12} {status:<10} {model:<20} {groups:<20}")
    print()


def cmd_devices_status(args):
    """Update device status."""
    fleet = FleetManager()
    try:
        device = fleet.update_status(args.id, args.set)
        print(f"Updated {device['device_id']} status -> {device['status']}")
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def cmd_devices_tag(args):
    """Add or remove a tag from a device."""
    fleet = FleetManager()
    try:
        if args.add:
            device = fleet.add_tag(args.id, args.add)
            print(f"Added tag '{args.add}' to {device['device_id']}")
        elif args.remove:
            device = fleet.remove_tag(args.id, args.remove)
            print(f"Removed tag '{args.remove}' from {device['device_id']}")
        else:
            print("Error: specify --add or --remove")
            sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def cmd_devices_group(args):
    """Add or remove a device from a group."""
    fleet = FleetManager()
    try:
        if args.add:
            device = fleet.add_to_group(args.id, args.add)
            print(f"Added {device['device_id']} to group '{args.add}'")
        elif args.remove:
            device = fleet.remove_from_group(args.id, args.remove)
            print(f"Removed {device['device_id']} from group '{args.remove}'")
        else:
            print("Error: specify --add or --remove")
            sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


# ── OTA subcommands ─────────────────────────────────────────

def cmd_ota_create_package(args):
    """Create an OTA update package."""
    ota = OTAManager()
    requirements = [r.strip() for r in args.requirements.split(",")] if args.requirements else []
    try:
        pkg = ota.create_update_package(
            source_package_dir=args.source_dir,
            version=args.version,
            model_name=args.model_name,
            model_version=args.model_version,
            requirements=requirements,
        )
        print(f"Created package: {pkg['package_id']} (v{pkg['version']}, {pkg['model_name']})")
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def cmd_ota_deploy(args):
    """Deploy an OTA package to devices."""
    ota = OTAManager()

    target_devices = None
    if args.target_devices:
        target_devices = [d.strip() for d in args.target_devices.split(",")]

    try:
        rollout = ota.create_rollout(
            package_id=args.package_id,
            target_devices=target_devices,
            target_group=args.target_group,
            target_tag=args.target_tag,
            strategy=args.strategy,
            canary_percentage=args.canary_percentage,
        )
        print(f"Created rollout: {rollout['rollout_id']} ({rollout['strategy']}, "
              f"{len(rollout['target_devices'])} devices)")

        # Execute the rollout
        result = ota.execute_rollout(rollout["rollout_id"])
        success = sum(1 for r in result.get("results", []) if r["status"] == "success")
        failed = sum(1 for r in result.get("results", []) if r["status"] == "failed")
        skipped = sum(1 for r in result.get("results", []) if r["status"] == "skipped")
        print(f"Rollout {result['rollout_id']} {result['status']}: "
              f"{success} success, {failed} failed, {skipped} skipped")
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def cmd_ota_status(args):
    """Show rollout status."""
    ota = OTAManager()
    rollout = ota.get_rollout(args.rollout_id)
    if rollout is None:
        print(f"Error: rollout '{args.rollout_id}' not found")
        sys.exit(1)

    print(f"\nRollout: {rollout['rollout_id']}")
    print(f"  Package:    {rollout['package_id']}")
    print(f"  Strategy:   {rollout['strategy']}")
    print(f"  Status:     {rollout['status']}")
    print(f"  Created:    {rollout.get('created_at', '')[:19]}")
    print(f"  Targets:    {len(rollout['target_devices'])} devices")

    results = rollout.get("results", [])
    if results:
        print(f"\n  {'Device':<15} {'Status':<10} {'Previous':<12} {'New':<12} {'Message':<30}")
        print("  " + "-" * 81)
        for r in results:
            did = r.get("device_id", "")
            status = r.get("status", "")
            prev = r.get("previous_version") or "-"
            new = r.get("new_version", "")
            msg = r.get("message", "")
            if len(msg) > 28:
                msg = msg[:25] + "..."
            print(f"  {did:<15} {status:<10} {prev:<12} {new:<12} {msg:<30}")
    print()


def cmd_ota_rollback(args):
    """Roll back a rollout."""
    ota = OTAManager()
    if not args.force:
        answer = input(f"Roll back rollout '{args.rollout_id}'? [y/N] ")
        if answer.lower() not in ("y", "yes"):
            print("Cancelled.")
            return
    try:
        rollout = ota.rollback_rollout(args.rollout_id)
        print(f"Rolled back: {rollout['rollout_id']} -> {rollout['status']}")
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def cmd_ota_history(args):
    """Show update history for a device."""
    ota = OTAManager()
    history = ota.get_device_history(args.device_id)
    if not history:
        print(f"No update history for device '{args.device_id}'.")
        return

    print(f"\nUpdate history for {args.device_id}:")
    print(f"\n{'Status':<12} {'Previous':<12} {'New':<12} {'Message':<30} {'Timestamp':<22}")
    print("-" * 90)
    for h in history:
        status = h.get("status", "")
        prev = h.get("previous_version") or "-"
        new = h.get("new_version", "")
        msg = h.get("message", "")
        if len(msg) > 28:
            msg = msg[:25] + "..."
        ts = h.get("timestamp", "")[:19]
        print(f"{status:<12} {prev:<12} {new:<12} {msg:<30} {ts:<22}")
    print()


def cmd_ota_list_packages(args):
    """List OTA update packages."""
    ota = OTAManager()
    packages = ota.list_update_packages()
    if not packages:
        print("No update packages found.")
        return

    print(f"\n{'Package ID':<15} {'Version':<10} {'Model':<20} {'Model Ver':>10} {'Created':<22}")
    print("-" * 79)
    for p in packages:
        pid = p.get("package_id", "")
        version = p.get("version", "")
        model = p.get("model_name", "")
        if len(model) > 18:
            model = model[:15] + "..."
        mver = p.get("model_version", "")
        created = p.get("created_at", "")[:19]
        print(f"{pid:<15} {version:<10} {model:<20} {mver:>10} {created:<22}")
    print()


def cmd_ota_list_rollouts(args):
    """List OTA rollouts."""
    ota = OTAManager()
    rollouts = ota.list_rollouts(status=args.status)
    if not rollouts:
        print("No rollouts found.")
        return

    print(f"\n{'Rollout ID':<18} {'Package':<15} {'Strategy':<12} {'Status':<14} {'Devices':>8} {'Created':<22}")
    print("-" * 91)
    for r in rollouts:
        rid = r.get("rollout_id", "")
        pkg = r.get("package_id", "")
        strategy = r.get("strategy", "")
        status = r.get("status", "")
        devices = len(r.get("target_devices", []))
        created = r.get("created_at", "")[:19]
        print(f"{rid:<18} {pkg:<15} {strategy:<12} {status:<14} {devices:>8} {created:<22}")
    print()


# ── Overview subcommand ─────────────────────────────────────

def cmd_overview(args):
    """Print fleet summary dashboard."""
    fleet = FleetManager()
    ota = OTAManager()

    summary = fleet.get_fleet_summary()
    active_rollouts = ota.list_rollouts(status="in-progress")
    pending_rollouts = ota.list_rollouts(status="pending")

    print("\n" + "=" * 50)
    print("  Forge Fleet Overview")
    print("=" * 50)

    print(f"\n  Total devices: {summary['total']}")
    if summary["by_status"]:
        print("  Status breakdown:")
        for status, count in sorted(summary["by_status"].items()):
            print(f"    {status:<12} {count}")
    else:
        print("  No devices registered.")

    if summary["groups"]:
        print(f"\n  Groups: {', '.join(summary['groups'])}")
    if summary["tags"]:
        print(f"  Tags:   {', '.join(summary['tags'])}")

    print(f"\n  Active rollouts:  {len(active_rollouts)}")
    print(f"  Pending rollouts: {len(pending_rollouts)}")

    if active_rollouts:
        print("\n  Active rollout details:")
        for r in active_rollouts:
            print(f"    {r['rollout_id']} — {r['package_id']} "
                  f"({r['strategy']}, {len(r['target_devices'])} devices)")

    packages = ota.list_update_packages()
    print(f"\n  Total OTA packages: {len(packages)}")
    print("=" * 50)
    print()


# ── CLI ─────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="forge fleet",
        description="Forge fleet & OTA management",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    subparsers = parser.add_subparsers(dest="category", help="Category")

    # ── devices ─────────────────────────────────────────────
    dev_parser = subparsers.add_parser("devices", help="Manage fleet devices")
    dev_sub = dev_parser.add_subparsers(dest="command", help="Command")

    # devices register
    dev_register = dev_sub.add_parser("register", help="Register a new device")
    dev_register.add_argument("--id", type=str, required=True,
                              help="Unique device ID")
    dev_register.add_argument("--name", type=str, required=True,
                              help="Human-readable device name")
    dev_register.add_argument("--type", type=str, required=True,
                              help="Device type (e.g. panda, ur5)")
    dev_register.add_argument("--capabilities", type=str, default=None,
                              help="Comma-separated capabilities (e.g. gpu,camera)")
    dev_register.add_argument("--tags", type=str, default=None,
                              help="Comma-separated tags")
    dev_register.add_argument("--groups", type=str, default=None,
                              help="Comma-separated groups")
    dev_register.add_argument("--metadata", type=str, default=None,
                              help="JSON string of metadata")
    dev_register.set_defaults(func=cmd_devices_register)

    # devices deregister
    dev_deregister = dev_sub.add_parser("deregister", help="Deregister a device")
    dev_deregister.add_argument("--id", type=str, required=True,
                                help="Device ID to deregister")
    dev_deregister.add_argument("--force", action="store_true",
                                help="Skip confirmation prompt")
    dev_deregister.set_defaults(func=cmd_devices_deregister)

    # devices list
    dev_list = dev_sub.add_parser("list", help="List registered devices")
    dev_list.add_argument("--status", type=str, default=None,
                          help="Filter by status (online, offline, updating, error)")
    dev_list.add_argument("--group", type=str, default=None,
                          help="Filter by group")
    dev_list.add_argument("--tag", type=str, default=None,
                          help="Filter by tag")
    dev_list.set_defaults(func=cmd_devices_list)

    # devices status
    dev_status = dev_sub.add_parser("status", help="Update device status")
    dev_status.add_argument("--id", type=str, required=True,
                            help="Device ID")
    dev_status.add_argument("--set", type=str, required=True,
                            help="New status (online, offline, updating, error)")
    dev_status.set_defaults(func=cmd_devices_status)

    # devices tag
    dev_tag = dev_sub.add_parser("tag", help="Add or remove a device tag")
    dev_tag.add_argument("--id", type=str, required=True,
                         help="Device ID")
    tag_group = dev_tag.add_mutually_exclusive_group(required=True)
    tag_group.add_argument("--add", type=str, default=None,
                           help="Tag to add")
    tag_group.add_argument("--remove", type=str, default=None,
                           help="Tag to remove")
    dev_tag.set_defaults(func=cmd_devices_tag)

    # devices group
    dev_group = dev_sub.add_parser("group", help="Add or remove device from group")
    dev_group.add_argument("--id", type=str, required=True,
                           help="Device ID")
    grp_group = dev_group.add_mutually_exclusive_group(required=True)
    grp_group.add_argument("--add", type=str, default=None,
                           help="Group to add")
    grp_group.add_argument("--remove", type=str, default=None,
                           help="Group to remove")
    dev_group.set_defaults(func=cmd_devices_group)

    # ── ota ─────────────────────────────────────────────────
    ota_parser = subparsers.add_parser("ota", help="OTA update management")
    ota_sub = ota_parser.add_subparsers(dest="command", help="Command")

    # ota create-package
    ota_create = ota_sub.add_parser("create-package", help="Create an OTA update package")
    ota_create.add_argument("--source-dir", type=str, required=True,
                            help="Path to deployment package directory")
    ota_create.add_argument("--version", type=str, required=True,
                            help="Semver version string (e.g. 1.0.0)")
    ota_create.add_argument("--model-name", type=str, required=True,
                            help="Model name")
    ota_create.add_argument("--model-version", type=int, default=None,
                            help="Model version (auto-detected from manifest if omitted)")
    ota_create.add_argument("--requirements", type=str, default=None,
                            help="Comma-separated device capability requirements")
    ota_create.set_defaults(func=cmd_ota_create_package)

    # ota deploy
    ota_deploy = ota_sub.add_parser("deploy", help="Deploy an OTA package")
    ota_deploy.add_argument("--package-id", type=str, required=True,
                            help="Package ID to deploy")
    target = ota_deploy.add_mutually_exclusive_group()
    target.add_argument("--target-devices", type=str, default=None,
                        help="Comma-separated device IDs")
    target.add_argument("--target-group", type=str, default=None,
                        help="Target group name")
    target.add_argument("--target-tag", type=str, default=None,
                        help="Target tag name")
    ota_deploy.add_argument("--strategy", type=str, default="immediate",
                            choices=["immediate", "canary", "staged"],
                            help="Rollout strategy (default: immediate)")
    ota_deploy.add_argument("--canary-percentage", type=int, default=10,
                            help="Canary rollout percentage (default: 10)")
    ota_deploy.set_defaults(func=cmd_ota_deploy)

    # ota status
    ota_status = ota_sub.add_parser("status", help="Show rollout status")
    ota_status.add_argument("--rollout-id", type=str, required=True,
                            help="Rollout ID")
    ota_status.set_defaults(func=cmd_ota_status)

    # ota rollback
    ota_rollback = ota_sub.add_parser("rollback", help="Roll back a rollout")
    ota_rollback.add_argument("--rollout-id", type=str, required=True,
                              help="Rollout ID to roll back")
    ota_rollback.add_argument("--force", action="store_true",
                              help="Skip confirmation prompt")
    ota_rollback.set_defaults(func=cmd_ota_rollback)

    # ota history
    ota_history = ota_sub.add_parser("history", help="Show device update history")
    ota_history.add_argument("--device-id", type=str, required=True,
                             help="Device ID")
    ota_history.set_defaults(func=cmd_ota_history)

    # ota list-packages
    ota_list_pkg = ota_sub.add_parser("list-packages", help="List OTA update packages")
    ota_list_pkg.set_defaults(func=cmd_ota_list_packages)

    # ota list-rollouts
    ota_list_roll = ota_sub.add_parser("list-rollouts", help="List OTA rollouts")
    ota_list_roll.add_argument("--status", type=str, default=None,
                               help="Filter by status (pending, in-progress, completed, failed, rolled-back)")
    ota_list_roll.set_defaults(func=cmd_ota_list_rollouts)

    # ── overview ────────────────────────────────────────────
    overview_parser = subparsers.add_parser("overview", help="Fleet summary dashboard")
    overview_parser.set_defaults(func=cmd_overview)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(getattr(args, "verbose", False))

    if not args.category:
        parser.print_help()
        return

    if not hasattr(args, "func"):
        # Print help for the category subparser
        if args.category == "devices":
            parser.parse_args(["devices", "--help"])
        elif args.category == "ota":
            parser.parse_args(["ota", "--help"])
        return

    args.func(args)


if __name__ == "__main__":
    main()
