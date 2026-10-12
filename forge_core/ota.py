"""OTA (Over-The-Air) update system for Forge.

Simulated OTA update management with update packages, rollout strategies
(immediate, canary, staged), rollback support, compatibility checking,
and per-device history.  All operations are local and file-based.

Persistence layout::

    ota/
    ├── packages.json          — update package registry
    ├── rollouts.json          — rollout records
    └── history/
        └── {device_id}.json   — per-device update history

Phase 10 — FEAT-001.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from forge_core.config import PROJECT_ROOT
from forge_core.fleet import FleetManager

logger = logging.getLogger(__name__)

VALID_ROLLOUT_STATUSES = ("pending", "in-progress", "completed", "failed", "rolled-back")
VALID_STRATEGIES = ("immediate", "canary", "staged")


# ── Version utilities ───────────────────────────────────────

def parse_version(version_str: str) -> tuple[int, int, int]:
    """Parse a semver string ``'X.Y.Z'`` into an integer tuple.

    Raises ValueError on malformed input.
    """
    parts = version_str.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid semver string: {version_str!r}")
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        raise ValueError(f"Invalid semver string: {version_str!r}")


def compare_versions(v1: str, v2: str) -> int:
    """Compare two semver strings.

    Returns -1 if v1 < v2, 0 if equal, 1 if v1 > v2.
    """
    t1 = parse_version(v1)
    t2 = parse_version(v2)
    if t1 < t2:
        return -1
    elif t1 > t2:
        return 1
    return 0


# ── Dataclasses ─────────────────────────────────────────────

@dataclass
class UpdatePackage:
    """An OTA update package created from a deployment artifact."""

    package_id: str
    version: str
    model_name: str
    model_version: int
    source_package: str
    requirements: list[str] = field(default_factory=list)
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "version": self.version,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "source_package": self.source_package,
            "requirements": self.requirements,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UpdatePackage:
        return cls(
            package_id=data["package_id"],
            version=data["version"],
            model_name=data["model_name"],
            model_version=data["model_version"],
            source_package=data["source_package"],
            requirements=data.get("requirements", []),
            created_at=data.get("created_at", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RolloutResult:
    """Per-device result from an OTA rollout execution."""

    device_id: str
    status: str  # 'success', 'failed', 'skipped'
    previous_version: str | None
    new_version: str
    message: str
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "status": self.status,
            "previous_version": self.previous_version,
            "new_version": self.new_version,
            "message": self.message,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RolloutResult:
        return cls(
            device_id=data["device_id"],
            status=data["status"],
            previous_version=data.get("previous_version"),
            new_version=data["new_version"],
            message=data.get("message", ""),
            timestamp=data.get("timestamp", ""),
        )


# ── OTA Manager ─────────────────────────────────────────────

class OTAManager:
    """Manage OTA update packages, rollouts, and device history.

    Stores state in ``ota_dir/packages.json``, ``ota_dir/rollouts.json``,
    and per-device history files under ``ota_dir/history/``.
    """

    def __init__(
        self,
        ota_dir: Path | None = None,
        fleet_manager: FleetManager | None = None,
    ) -> None:
        self.ota_dir = ota_dir or PROJECT_ROOT / "ota"
        self.ota_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir = self.ota_dir / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.fleet_manager = fleet_manager or FleetManager()

    # ── Persistence helpers ─────────────────────────────────

    def _load_packages(self) -> dict[str, Any]:
        path = self.ota_dir / "packages.json"
        if not path.exists():
            return {"packages": []}
        with open(path, "r") as f:
            return json.load(f)

    def _save_packages(self, data: dict[str, Any]) -> None:
        path = self.ota_dir / "packages.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_rollouts(self) -> dict[str, Any]:
        path = self.ota_dir / "rollouts.json"
        if not path.exists():
            return {"rollouts": []}
        with open(path, "r") as f:
            return json.load(f)

    def _save_rollouts(self, data: dict[str, Any]) -> None:
        path = self.ota_dir / "rollouts.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ── ID generation ───────────────────────────────────────

    def _next_package_id(self) -> str:
        data = self._load_packages()
        if not data["packages"]:
            return "pkg_00001"
        ids: list[int] = []
        for p in data["packages"]:
            try:
                ids.append(int(p["package_id"].split("_")[1]))
            except (IndexError, ValueError):
                pass
        return f"pkg_{max(ids, default=0) + 1:05d}"

    def _next_rollout_id(self) -> str:
        data = self._load_rollouts()
        if not data["rollouts"]:
            return "rollout_00001"
        ids: list[int] = []
        for r in data["rollouts"]:
            try:
                ids.append(int(r["rollout_id"].split("_")[1]))
            except (IndexError, ValueError):
                pass
        return f"rollout_{max(ids, default=0) + 1:05d}"

    # ── Package management ──────────────────────────────────

    def create_update_package(
        self,
        source_package_dir: str | Path,
        version: str,
        model_name: str,
        model_version: int | None = None,
        requirements: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an update package, optionally reading manifest.yaml."""
        source_package_dir = Path(source_package_dir)
        # Validate version format
        parse_version(version)

        # Try to read manifest.yaml from the source deployment package
        manifest_path = source_package_dir / "manifest.yaml"
        manifest_data: dict[str, Any] = {}
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                manifest_data = yaml.safe_load(f) or {}

        # Auto-fill from manifest if not provided
        if model_version is None:
            model_version = manifest_data.get("model_version", 1)
        if not model_name and manifest_data.get("model_name"):
            model_name = manifest_data["model_name"]

        package_id = self._next_package_id()
        pkg = UpdatePackage(
            package_id=package_id,
            version=version,
            model_name=model_name,
            model_version=model_version,
            source_package=str(source_package_dir),
            requirements=requirements or [],
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )

        data = self._load_packages()
        data["packages"].append(pkg.to_dict())
        self._save_packages(data)
        logger.info("update_package_created: %s (v%s)", package_id, version)
        return pkg.to_dict()

    def get_update_package(self, package_id: str) -> dict[str, Any] | None:
        """Return a package entry or None."""
        data = self._load_packages()
        for p in data["packages"]:
            if p["package_id"] == package_id:
                return p
        return None

    def list_update_packages(self) -> list[dict[str, Any]]:
        """List all update packages."""
        data = self._load_packages()
        return data["packages"]

    # ── Compatibility ───────────────────────────────────────

    def check_compatibility(
        self, package_id: str, device_id: str,
    ) -> dict[str, Any]:
        """Check if a package is compatible with a device.

        Returns ``{'compatible': bool, 'missing_capabilities': [...]}``.
        """
        pkg = self.get_update_package(package_id)
        if pkg is None:
            raise ValueError(f"Package not found: {package_id}")
        device = self.fleet_manager.get_device(device_id)
        if device is None:
            raise ValueError(f"Device not found: {device_id}")

        device_caps = set(device.get("capabilities", []))
        required = set(pkg.get("requirements", []))
        missing = sorted(required - device_caps)

        return {
            "compatible": len(missing) == 0,
            "missing_capabilities": missing,
        }

    # ── Rollout management ──────────────────────────────────

    def create_rollout(
        self,
        package_id: str,
        target_devices: list[str] | None = None,
        target_group: str | None = None,
        target_tag: str | None = None,
        strategy: str = "immediate",
        canary_percentage: int = 10,
    ) -> dict[str, Any]:
        """Create a rollout targeting specific devices, a group, or a tag."""
        # Validate package exists
        pkg = self.get_update_package(package_id)
        if pkg is None:
            raise ValueError(f"Package not found: {package_id}")
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy: {strategy!r}. Must be one of {VALID_STRATEGIES}"
            )

        # Resolve target devices
        if target_devices is not None:
            resolved = target_devices
        elif target_group is not None:
            devices = self.fleet_manager.list_devices(group=target_group)
            resolved = [d["device_id"] for d in devices]
        elif target_tag is not None:
            devices = self.fleet_manager.list_devices(tag=target_tag)
            resolved = [d["device_id"] for d in devices]
        else:
            devices = self.fleet_manager.list_devices()
            resolved = [d["device_id"] for d in devices]

        rollout_id = self._next_rollout_id()
        rollout: dict[str, Any] = {
            "rollout_id": rollout_id,
            "package_id": package_id,
            "target_devices": resolved,
            "strategy": strategy,
            "canary_percentage": canary_percentage,
            "status": "pending",
            "results": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        data = self._load_rollouts()
        data["rollouts"].append(rollout)
        self._save_rollouts(data)
        logger.info("rollout_created: %s (%s, %d devices)", rollout_id, strategy, len(resolved))
        return rollout

    def get_rollout(self, rollout_id: str) -> dict[str, Any] | None:
        """Return a rollout record or None."""
        data = self._load_rollouts()
        for r in data["rollouts"]:
            if r["rollout_id"] == rollout_id:
                return r
        return None

    def list_rollouts(self, status: str | None = None) -> list[dict[str, Any]]:
        """List rollouts, optionally filtered by status."""
        data = self._load_rollouts()
        rollouts = data["rollouts"]
        if status is not None:
            rollouts = [r for r in rollouts if r["status"] == status]
        return rollouts

    def execute_rollout(self, rollout_id: str) -> dict[str, Any]:
        """Simulate executing a rollout.

        For each target device: checks compatibility, sets status to
        'updating', simulates deployment, updates device model info, and
        records per-device results and history.

        Strategy behaviour:
        - ``immediate``: deploy to all compatible devices.
        - ``canary``: deploy only to ``canary_percentage`` of devices.
        - ``staged``: deploy to all but mark rollout as requiring
          confirmation (status stays 'in-progress').
        """
        data = self._load_rollouts()
        rollout = None
        for r in data["rollouts"]:
            if r["rollout_id"] == rollout_id:
                rollout = r
                break
        if rollout is None:
            raise ValueError(f"Rollout not found: {rollout_id}")

        pkg = self.get_update_package(rollout["package_id"])
        if pkg is None:
            raise ValueError(f"Package not found: {rollout['package_id']}")

        rollout["status"] = "in-progress"
        target_devices = list(rollout["target_devices"])

        # For canary strategy, only deploy to a subset
        if rollout["strategy"] == "canary":
            count = max(1, math.ceil(len(target_devices) * rollout["canary_percentage"] / 100))
            target_devices = target_devices[:count]

        results: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc).isoformat()

        for did in target_devices:
            device = self.fleet_manager.get_device(did)
            if device is None:
                result = RolloutResult(
                    device_id=did,
                    status="failed",
                    previous_version=None,
                    new_version=pkg["version"],
                    message="Device not found",
                    timestamp=now,
                )
                results.append(result.to_dict())
                self._record_device_history(did, result.to_dict())
                continue

            # Check compatibility
            compat = self.check_compatibility(rollout["package_id"], did)
            if not compat["compatible"]:
                result = RolloutResult(
                    device_id=did,
                    status="skipped",
                    previous_version=None,
                    new_version=pkg["version"],
                    message=f"Missing capabilities: {compat['missing_capabilities']}",
                    timestamp=now,
                )
                results.append(result.to_dict())
                self._record_device_history(did, result.to_dict())
                continue

            # Set device to updating
            self.fleet_manager.update_status(did, "updating")

            # Determine previous version
            prev_version = None
            if device.get("current_model_version") is not None:
                prev_version = f"{device.get('current_model_version', 0)}"

            # Simulate successful deployment
            self.fleet_manager.update_device_model(
                did,
                model_name=pkg["model_name"],
                model_version=pkg["model_version"],
            )
            self.fleet_manager.update_status(did, "online")

            result = RolloutResult(
                device_id=did,
                status="success",
                previous_version=prev_version,
                new_version=pkg["version"],
                message="Update applied successfully",
                timestamp=now,
            )
            results.append(result.to_dict())
            self._record_device_history(did, result.to_dict())

        rollout["results"] = results

        # Determine final rollout status
        if rollout["strategy"] == "staged":
            # Staged stays in-progress until confirmed
            rollout["status"] = "in-progress"
        elif rollout["strategy"] == "canary":
            # Canary stays in-progress (partial deployment)
            rollout["status"] = "in-progress"
        else:
            # Immediate: completed if any succeeded
            statuses = {r["status"] for r in results}
            if "success" in statuses:
                rollout["status"] = "completed"
            else:
                rollout["status"] = "failed"

        self._save_rollouts(data)
        logger.info("rollout_executed: %s (%d results)", rollout_id, len(results))
        return rollout

    def rollback_rollout(self, rollout_id: str) -> dict[str, Any]:
        """Roll back a rollout: revert devices to previous model version."""
        data = self._load_rollouts()
        rollout = None
        for r in data["rollouts"]:
            if r["rollout_id"] == rollout_id:
                rollout = r
                break
        if rollout is None:
            raise ValueError(f"Rollout not found: {rollout_id}")

        now = datetime.now(timezone.utc).isoformat()
        pkg = self.get_update_package(rollout["package_id"])

        for result in rollout.get("results", []):
            if result["status"] != "success":
                continue
            did = result["device_id"]
            prev = result.get("previous_version")
            if prev is not None:
                try:
                    prev_int = int(prev)
                except ValueError:
                    prev_int = 0
                model_name = pkg["model_name"] if pkg else "unknown"
                self.fleet_manager.update_device_model(did, model_name, prev_int)
            else:
                # No previous version — clear model info
                device_data = self.fleet_manager._load_registry()
                for d in device_data["devices"]:
                    if d["device_id"] == did:
                        d["current_model"] = None
                        d["current_model_version"] = None
                        break
                self.fleet_manager._save_registry(device_data)

            # Record rollback in history
            history_record = {
                "device_id": did,
                "status": "rolled-back",
                "previous_version": result.get("new_version"),
                "new_version": prev,
                "message": "Rolled back",
                "timestamp": now,
            }
            self._record_device_history(did, history_record)

        rollout["status"] = "rolled-back"
        self._save_rollouts(data)
        logger.info("rollout_rolled_back: %s", rollout_id)
        return rollout

    # ── Device history ──────────────────────────────────────

    def get_device_history(self, device_id: str) -> list[dict[str, Any]]:
        """Return the update history for a device."""
        path = self.history_dir / f"{device_id}.json"
        if not path.exists():
            return []
        with open(path, "r") as f:
            return json.load(f)

    def _record_device_history(
        self, device_id: str, record: dict[str, Any],
    ) -> None:
        """Append a record to a device's history file."""
        history = self.get_device_history(device_id)
        history.append(record)
        path = self.history_dir / f"{device_id}.json"
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
