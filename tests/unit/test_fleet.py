"""Unit tests for the fleet management system."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.fleet import FleetManager, DeviceInfo, VALID_STATUSES


# ── TestDeviceRegistration ──────────────────────────────────

class TestDeviceRegistration:
    def test_register_device(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        device = fm.register_device("dev_001", "Robot Arm A", "arm")
        assert device["device_id"] == "dev_001"
        assert device["name"] == "Robot Arm A"
        assert device["device_type"] == "arm"
        assert device["status"] == "offline"

    def test_register_duplicate_raises(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        fm.register_device("dev_001", "Robot Arm A", "arm")
        with pytest.raises(ValueError, match="already registered"):
            fm.register_device("dev_001", "Robot Arm B", "arm")

    def test_register_with_metadata(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        device = fm.register_device(
            "dev_001", "Robot Arm A", "arm",
            metadata={"firmware": "2.1.0", "location": "lab_1"},
        )
        assert device["metadata"]["firmware"] == "2.1.0"
        assert device["metadata"]["location"] == "lab_1"

    def test_register_with_tags_and_groups(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        device = fm.register_device(
            "dev_001", "Robot Arm A", "arm",
            tags=["production", "v2"],
            groups=["floor_1", "assembly"],
        )
        assert "production" in device["tags"]
        assert "floor_1" in device["groups"]

    def test_initial_status_is_offline(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        device = fm.register_device("dev_001", "Robot Arm A", "arm")
        assert device["status"] == "offline"


# ── TestDeviceRetrieval ─────────────────────────────────────

class TestDeviceRetrieval:
    def test_get_device(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        fm.register_device("dev_001", "Robot Arm A", "arm")
        device = fm.get_device("dev_001")
        assert device is not None
        assert device["device_id"] == "dev_001"

    def test_get_nonexistent_returns_none(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        assert fm.get_device("nonexistent") is None

    def test_list_empty(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        assert fm.list_devices() == []

    def test_list_all(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        fm.register_device("dev_001", "Robot Arm A", "arm")
        fm.register_device("dev_002", "Robot Arm B", "arm")
        devices = fm.list_devices()
        assert len(devices) == 2

    def test_list_filter_by_status(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        fm.register_device("dev_001", "Robot Arm A", "arm")
        fm.register_device("dev_002", "Robot Arm B", "arm")
        fm.update_status("dev_001", "online")
        devices = fm.list_devices(status="online")
        assert len(devices) == 1
        assert devices[0]["device_id"] == "dev_001"

    def test_list_filter_by_group(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        fm.register_device("dev_001", "Robot Arm A", "arm", groups=["floor_1"])
        fm.register_device("dev_002", "Robot Arm B", "arm", groups=["floor_2"])
        devices = fm.list_devices(group="floor_1")
        assert len(devices) == 1
        assert devices[0]["device_id"] == "dev_001"

    def test_list_filter_by_tag(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        fm.register_device("dev_001", "Robot Arm A", "arm", tags=["production"])
        fm.register_device("dev_002", "Robot Arm B", "arm", tags=["staging"])
        devices = fm.list_devices(tag="production")
        assert len(devices) == 1
        assert devices[0]["device_id"] == "dev_001"

    def test_list_sorted(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        fm.register_device("dev_003", "Robot C", "arm")
        fm.register_device("dev_001", "Robot A", "arm")
        fm.register_device("dev_002", "Robot B", "arm")
        devices = fm.list_devices()
        ids = [d["device_id"] for d in devices]
        assert ids == ["dev_001", "dev_002", "dev_003"]


# ── TestDeviceStatusUpdate ──────────────────────────────────

class TestDeviceStatusUpdate:
    def test_update_status_online(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        fm.register_device("dev_001", "Robot Arm A", "arm")
        device = fm.update_status("dev_001", "online")
        assert device["status"] == "online"

    def test_update_status_updating(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        fm.register_device("dev_001", "Robot Arm A", "arm")
        device = fm.update_status("dev_001", "updating")
        assert device["status"] == "updating"

    def test_invalid_status_raises(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        fm.register_device("dev_001", "Robot Arm A", "arm")
        with pytest.raises(ValueError, match="Invalid status"):
            fm.update_status("dev_001", "unknown")

    def test_update_nonexistent_raises(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        with pytest.raises(ValueError, match="Device not found"):
            fm.update_status("nonexistent", "online")


# ── TestDeviceTagsAndGroups ─────────────────────────────────

class TestDeviceTagsAndGroups:
    def test_add_tag(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        fm.register_device("dev_001", "Robot Arm A", "arm")
        device = fm.add_tag("dev_001", "production")
        assert "production" in device["tags"]

    def test_remove_tag(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        fm.register_device("dev_001", "Robot Arm A", "arm", tags=["production"])
        device = fm.remove_tag("dev_001", "production")
        assert "production" not in device["tags"]

    def test_add_to_group(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        fm.register_device("dev_001", "Robot Arm A", "arm")
        device = fm.add_to_group("dev_001", "floor_1")
        assert "floor_1" in device["groups"]

    def test_remove_from_group(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        fm.register_device("dev_001", "Robot Arm A", "arm", groups=["floor_1"])
        device = fm.remove_from_group("dev_001", "floor_1")
        assert "floor_1" not in device["groups"]

    def test_add_duplicate_tag_is_noop(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        fm.register_device("dev_001", "Robot Arm A", "arm", tags=["production"])
        device = fm.add_tag("dev_001", "production")
        assert device["tags"].count("production") == 1


# ── TestDeviceModelTracking ─────────────────────────────────

class TestDeviceModelTracking:
    def test_update_device_model(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        fm.register_device("dev_001", "Robot Arm A", "arm")
        device = fm.update_device_model("dev_001", "bc_pick", 3)
        assert device["current_model"] == "bc_pick"
        assert device["current_model_version"] == 3

    def test_update_model_with_experiment_id(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        fm.register_device("dev_001", "Robot Arm A", "arm")
        device = fm.update_device_model(
            "dev_001", "bc_pick", 3, experiment_id="exp_00001",
        )
        assert device["experiment_id"] == "exp_00001"


# ── TestFleetSummary ────────────────────────────────────────

class TestFleetSummary:
    def test_summary_empty(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        summary = fm.get_fleet_summary()
        assert summary["total"] == 0
        assert summary["by_status"] == {}
        assert summary["groups"] == []
        assert summary["tags"] == []

    def test_summary_with_devices(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        fm.register_device("dev_001", "Robot A", "arm", tags=["v2"], groups=["floor_1"])
        fm.register_device("dev_002", "Robot B", "arm", tags=["v2"], groups=["floor_2"])
        fm.update_status("dev_001", "online")
        summary = fm.get_fleet_summary()
        assert summary["total"] == 2
        assert summary["by_status"]["online"] == 1
        assert summary["by_status"]["offline"] == 1
        assert "floor_1" in summary["groups"]
        assert "floor_2" in summary["groups"]
        assert "v2" in summary["tags"]


# ── TestDeviceDeregistration ────────────────────────────────

class TestDeviceDeregistration:
    def test_deregister_device(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        fm.register_device("dev_001", "Robot Arm A", "arm")
        fm.deregister_device("dev_001")
        assert fm.get_device("dev_001") is None

    def test_deregister_nonexistent_raises(self, tmp_path):
        fm = FleetManager(fleet_dir=tmp_path)
        with pytest.raises(ValueError, match="Device not found"):
            fm.deregister_device("nonexistent")
