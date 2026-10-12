"""Fleet management for Forge.

File-based device registry with group/tag management, status tracking,
and fleet-wide queries.  All device entries live in a single
``devices.json`` file under the *fleet/* directory.

Phase 10 — FEAT-001.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forge_core.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

VALID_STATUSES = ("online", "offline", "updating", "error")


@dataclass
class DeviceInfo:
    """Metadata for a registered fleet device."""

    device_id: str
    name: str
    device_type: str
    capabilities: list[str] = field(default_factory=list)
    status: str = "offline"
    tags: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    last_seen: str = ""
    registered_at: str = ""
    current_model: str | None = None
    current_model_version: int | None = None
    experiment_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "device_type": self.device_type,
            "capabilities": self.capabilities,
            "status": self.status,
            "tags": self.tags,
            "groups": self.groups,
            "last_seen": self.last_seen,
            "registered_at": self.registered_at,
            "current_model": self.current_model,
            "current_model_version": self.current_model_version,
            "experiment_id": self.experiment_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeviceInfo:
        return cls(
            device_id=data["device_id"],
            name=data["name"],
            device_type=data["device_type"],
            capabilities=data.get("capabilities", []),
            status=data.get("status", "offline"),
            tags=data.get("tags", []),
            groups=data.get("groups", []),
            last_seen=data.get("last_seen", ""),
            registered_at=data.get("registered_at", ""),
            current_model=data.get("current_model"),
            current_model_version=data.get("current_model_version"),
            experiment_id=data.get("experiment_id"),
            metadata=data.get("metadata", {}),
        )


class FleetManager:
    """Manage a fleet of devices with JSON persistence.

    Stores all entries in ``fleet_dir/devices.json``.
    """

    def __init__(self, fleet_dir: Path | None = None) -> None:
        self.fleet_dir = fleet_dir or PROJECT_ROOT / "fleet"
        self.fleet_dir.mkdir(parents=True, exist_ok=True)

    # ── Persistence helpers ─────────────────────────────────

    def _load_registry(self) -> dict[str, Any]:
        """Load the device registry from disk, returning empty if missing."""
        path = self.fleet_dir / "devices.json"
        if not path.exists():
            return {"devices": []}
        with open(path, "r") as f:
            return json.load(f)

    def _save_registry(self, data: dict[str, Any]) -> None:
        """Write the device registry to disk."""
        path = self.fleet_dir / "devices.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Public API ──────────────────────────────────────────

    def register_device(
        self,
        device_id: str,
        name: str,
        device_type: str,
        capabilities: list[str] | None = None,
        tags: list[str] | None = None,
        groups: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a new device. Raises ValueError if device_id exists."""
        data = self._load_registry()
        for d in data["devices"]:
            if d["device_id"] == device_id:
                raise ValueError(f"Device already registered: {device_id}")

        now = datetime.now(timezone.utc).isoformat()
        device = DeviceInfo(
            device_id=device_id,
            name=name,
            device_type=device_type,
            capabilities=capabilities or [],
            tags=tags or [],
            groups=groups or [],
            last_seen=now,
            registered_at=now,
            metadata=metadata or {},
        )
        data["devices"].append(device.to_dict())
        self._save_registry(data)
        logger.info("device_registered: %s", device_id)
        return device.to_dict()

    def deregister_device(self, device_id: str) -> None:
        """Remove a device from the registry. Raises ValueError if not found."""
        data = self._load_registry()
        original_len = len(data["devices"])
        data["devices"] = [
            d for d in data["devices"] if d["device_id"] != device_id
        ]
        if len(data["devices"]) == original_len:
            raise ValueError(f"Device not found: {device_id}")
        self._save_registry(data)
        logger.info("device_deregistered: %s", device_id)

    def get_device(self, device_id: str) -> dict[str, Any] | None:
        """Return a device entry or None if not found."""
        data = self._load_registry()
        for d in data["devices"]:
            if d["device_id"] == device_id:
                return d
        return None

    def update_status(self, device_id: str, status: str) -> dict[str, Any]:
        """Update device status. Validates against VALID_STATUSES."""
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status: {status!r}. Must be one of {VALID_STATUSES}"
            )
        data = self._load_registry()
        for d in data["devices"]:
            if d["device_id"] == device_id:
                d["status"] = status
                d["last_seen"] = datetime.now(timezone.utc).isoformat()
                self._save_registry(data)
                logger.info("device_status_updated: %s -> %s", device_id, status)
                return d
        raise ValueError(f"Device not found: {device_id}")

    def list_devices(
        self,
        status: str | None = None,
        group: str | None = None,
        tag: str | None = None,
    ) -> list[dict[str, Any]]:
        """List devices, optionally filtered. Sorted by device_id."""
        data = self._load_registry()
        devices = data["devices"]
        if status is not None:
            devices = [d for d in devices if d["status"] == status]
        if group is not None:
            devices = [d for d in devices if group in d.get("groups", [])]
        if tag is not None:
            devices = [d for d in devices if tag in d.get("tags", [])]
        devices.sort(key=lambda d: d["device_id"])
        return devices

    def add_tag(self, device_id: str, tag: str) -> dict[str, Any]:
        """Add a tag to a device. No-op if tag already present."""
        data = self._load_registry()
        for d in data["devices"]:
            if d["device_id"] == device_id:
                if tag not in d["tags"]:
                    d["tags"].append(tag)
                self._save_registry(data)
                return d
        raise ValueError(f"Device not found: {device_id}")

    def remove_tag(self, device_id: str, tag: str) -> dict[str, Any]:
        """Remove a tag from a device."""
        data = self._load_registry()
        for d in data["devices"]:
            if d["device_id"] == device_id:
                if tag in d["tags"]:
                    d["tags"].remove(tag)
                self._save_registry(data)
                return d
        raise ValueError(f"Device not found: {device_id}")

    def add_to_group(self, device_id: str, group: str) -> dict[str, Any]:
        """Add a device to a group. No-op if already in group."""
        data = self._load_registry()
        for d in data["devices"]:
            if d["device_id"] == device_id:
                if group not in d["groups"]:
                    d["groups"].append(group)
                self._save_registry(data)
                return d
        raise ValueError(f"Device not found: {device_id}")

    def remove_from_group(self, device_id: str, group: str) -> dict[str, Any]:
        """Remove a device from a group."""
        data = self._load_registry()
        for d in data["devices"]:
            if d["device_id"] == device_id:
                if group in d["groups"]:
                    d["groups"].remove(group)
                self._save_registry(data)
                return d
        raise ValueError(f"Device not found: {device_id}")

    def update_device_model(
        self,
        device_id: str,
        model_name: str,
        model_version: int,
        experiment_id: str | None = None,
    ) -> dict[str, Any]:
        """Update the model currently deployed on a device."""
        data = self._load_registry()
        for d in data["devices"]:
            if d["device_id"] == device_id:
                d["current_model"] = model_name
                d["current_model_version"] = model_version
                d["experiment_id"] = experiment_id
                self._save_registry(data)
                logger.info(
                    "device_model_updated: %s -> %s v%d",
                    device_id, model_name, model_version,
                )
                return d
        raise ValueError(f"Device not found: {device_id}")

    def get_fleet_summary(self) -> dict[str, Any]:
        """Return a summary of the fleet: counts by status, groups, tags."""
        data = self._load_registry()
        devices = data["devices"]
        status_counts: dict[str, int] = {}
        all_groups: set[str] = set()
        all_tags: set[str] = set()

        for d in devices:
            s = d.get("status", "offline")
            status_counts[s] = status_counts.get(s, 0) + 1
            all_groups.update(d.get("groups", []))
            all_tags.update(d.get("tags", []))

        return {
            "total": len(devices),
            "by_status": status_counts,
            "groups": sorted(all_groups),
            "tags": sorted(all_tags),
        }
