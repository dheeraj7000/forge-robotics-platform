"""Unit tests for the OTA update system."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.fleet import FleetManager
from forge_core.ota import (
    OTAManager,
    UpdatePackage,
    RolloutResult,
    parse_version,
    compare_versions,
    VALID_STRATEGIES,
)


# ── Helpers ─────────────────────────────────────────────────

def _make_source_package(tmp_path: Path, name: str = "deploy_pkg") -> Path:
    """Create a minimal source deployment package directory."""
    pkg_dir = tmp_path / name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    return pkg_dir


def _make_source_package_with_manifest(
    tmp_path: Path,
    name: str = "deploy_pkg",
    model_name: str = "bc_pick",
    model_format: str = "onnx",
) -> Path:
    """Create a source package with a manifest.yaml."""
    pkg_dir = _make_source_package(tmp_path, name)
    manifest = {
        "model_name": model_name,
        "model_format": model_format,
        "opset_version": 17,
        "input_shape": [1, 31],
        "output_shape": [1, 8],
        "model_version": 2,
    }
    with open(pkg_dir / "manifest.yaml", "w") as f:
        yaml.dump(manifest, f)
    return pkg_dir


def _setup_fleet_and_ota(tmp_path: Path) -> tuple[FleetManager, OTAManager]:
    """Create FleetManager + OTAManager sharing the same tmp_path."""
    fleet_dir = tmp_path / "fleet"
    ota_dir = tmp_path / "ota"
    fm = FleetManager(fleet_dir=fleet_dir)
    ota = OTAManager(ota_dir=ota_dir, fleet_manager=fm)
    return fm, ota


def _register_devices(fm: FleetManager, count: int = 3) -> list[str]:
    """Register N devices with onnx capability and return their IDs."""
    ids = []
    for i in range(1, count + 1):
        did = f"dev_{i:03d}"
        fm.register_device(did, f"Robot {i}", "arm", capabilities=["onnx"])
        fm.update_status(did, "online")
        ids.append(did)
    return ids


# ── TestUpdatePackageCreation ───────────────────────────────

class TestUpdatePackageCreation:
    def test_create_package(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        src = _make_source_package(tmp_path)
        pkg = ota.create_update_package(src, "1.0.0", "bc_pick", model_version=1)
        assert pkg["package_id"] == "pkg_00001"
        assert pkg["version"] == "1.0.0"
        assert pkg["model_name"] == "bc_pick"

    def test_create_package_with_manifest(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        src = _make_source_package_with_manifest(tmp_path, model_name="bc_place")
        pkg = ota.create_update_package(src, "2.0.0", "bc_place")
        assert pkg["model_name"] == "bc_place"
        assert pkg["model_version"] == 2  # from manifest

    def test_sequential_package_ids(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        src = _make_source_package(tmp_path)
        p1 = ota.create_update_package(src, "1.0.0", "bc_pick", model_version=1)
        p2 = ota.create_update_package(src, "1.1.0", "bc_pick", model_version=2)
        assert p1["package_id"] == "pkg_00001"
        assert p2["package_id"] == "pkg_00002"

    def test_list_packages(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        src = _make_source_package(tmp_path)
        ota.create_update_package(src, "1.0.0", "bc_pick", model_version=1)
        ota.create_update_package(src, "1.1.0", "bc_pick", model_version=2)
        packages = ota.list_update_packages()
        assert len(packages) == 2


# ── TestCompatibilityChecking ───────────────────────────────

class TestCompatibilityChecking:
    def test_compatible_device(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        fm.register_device("dev_001", "Robot A", "arm", capabilities=["onnx", "gpu"])
        src = _make_source_package(tmp_path)
        pkg = ota.create_update_package(
            src, "1.0.0", "bc_pick", model_version=1, requirements=["onnx"],
        )
        result = ota.check_compatibility(pkg["package_id"], "dev_001")
        assert result["compatible"] is True
        assert result["missing_capabilities"] == []

    def test_incompatible_device_missing_capabilities(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        fm.register_device("dev_001", "Robot A", "arm", capabilities=["onnx"])
        src = _make_source_package(tmp_path)
        pkg = ota.create_update_package(
            src, "1.0.0", "bc_pick", model_version=1,
            requirements=["onnx", "tensorrt"],
        )
        result = ota.check_compatibility(pkg["package_id"], "dev_001")
        assert result["compatible"] is False
        assert "tensorrt" in result["missing_capabilities"]

    def test_version_comparison(self, tmp_path):
        """Version comparison utility works correctly."""
        assert compare_versions("1.0.0", "2.0.0") == -1
        assert compare_versions("2.0.0", "1.0.0") == 1
        assert compare_versions("1.0.0", "1.0.0") == 0
        assert compare_versions("1.2.3", "1.2.4") == -1
        assert compare_versions("1.3.0", "1.2.9") == 1


# ── TestRolloutCreation ─────────────────────────────────────

class TestRolloutCreation:
    def test_create_rollout_immediate(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        ids = _register_devices(fm)
        src = _make_source_package(tmp_path)
        pkg = ota.create_update_package(src, "1.0.0", "bc_pick", model_version=1)
        rollout = ota.create_rollout(
            pkg["package_id"], target_devices=ids, strategy="immediate",
        )
        assert rollout["rollout_id"] == "rollout_00001"
        assert rollout["status"] == "pending"
        assert rollout["strategy"] == "immediate"
        assert len(rollout["target_devices"]) == 3

    def test_create_rollout_by_group(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        fm.register_device("dev_001", "Robot A", "arm", capabilities=["onnx"], groups=["floor_1"])
        fm.register_device("dev_002", "Robot B", "arm", capabilities=["onnx"], groups=["floor_2"])
        src = _make_source_package(tmp_path)
        pkg = ota.create_update_package(src, "1.0.0", "bc_pick", model_version=1)
        rollout = ota.create_rollout(pkg["package_id"], target_group="floor_1")
        assert len(rollout["target_devices"]) == 1
        assert "dev_001" in rollout["target_devices"]

    def test_create_rollout_by_tag(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        fm.register_device("dev_001", "Robot A", "arm", capabilities=["onnx"], tags=["canary"])
        fm.register_device("dev_002", "Robot B", "arm", capabilities=["onnx"], tags=["stable"])
        src = _make_source_package(tmp_path)
        pkg = ota.create_update_package(src, "1.0.0", "bc_pick", model_version=1)
        rollout = ota.create_rollout(pkg["package_id"], target_tag="canary")
        assert len(rollout["target_devices"]) == 1
        assert "dev_001" in rollout["target_devices"]

    def test_create_rollout_canary(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        ids = _register_devices(fm, count=5)
        src = _make_source_package(tmp_path)
        pkg = ota.create_update_package(src, "1.0.0", "bc_pick", model_version=1)
        rollout = ota.create_rollout(
            pkg["package_id"], target_devices=ids,
            strategy="canary", canary_percentage=20,
        )
        assert rollout["strategy"] == "canary"
        assert rollout["canary_percentage"] == 20

    def test_create_rollout_invalid_package_raises(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        with pytest.raises(ValueError, match="Package not found"):
            ota.create_rollout("pkg_99999")


# ── TestRolloutExecution ────────────────────────────────────

class TestRolloutExecution:
    def test_execute_immediate_rollout(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        ids = _register_devices(fm)
        src = _make_source_package(tmp_path)
        pkg = ota.create_update_package(
            src, "1.0.0", "bc_pick", model_version=1, requirements=["onnx"],
        )
        rollout = ota.create_rollout(
            pkg["package_id"], target_devices=ids, strategy="immediate",
        )
        result = ota.execute_rollout(rollout["rollout_id"])
        assert result["status"] == "completed"
        assert len(result["results"]) == 3
        assert all(r["status"] == "success" for r in result["results"])

    def test_execute_canary_rollout_partial(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        ids = _register_devices(fm, count=10)
        src = _make_source_package(tmp_path)
        pkg = ota.create_update_package(
            src, "1.0.0", "bc_pick", model_version=1, requirements=["onnx"],
        )
        rollout = ota.create_rollout(
            pkg["package_id"], target_devices=ids,
            strategy="canary", canary_percentage=20,
        )
        result = ota.execute_rollout(rollout["rollout_id"])
        # 20% of 10 = 2 devices
        assert result["status"] == "in-progress"
        assert len(result["results"]) == 2

    def test_execute_rollout_updates_device_model(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        fm.register_device("dev_001", "Robot A", "arm", capabilities=["onnx"])
        fm.update_status("dev_001", "online")
        src = _make_source_package(tmp_path)
        pkg = ota.create_update_package(
            src, "1.0.0", "bc_pick", model_version=3, requirements=["onnx"],
        )
        rollout = ota.create_rollout(
            pkg["package_id"], target_devices=["dev_001"],
        )
        ota.execute_rollout(rollout["rollout_id"])
        device = fm.get_device("dev_001")
        assert device["current_model"] == "bc_pick"
        assert device["current_model_version"] == 3

    def test_execute_incompatible_device_skipped(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        fm.register_device("dev_001", "Robot A", "arm", capabilities=["onnx"])
        fm.update_status("dev_001", "online")
        src = _make_source_package(tmp_path)
        pkg = ota.create_update_package(
            src, "1.0.0", "bc_pick", model_version=1,
            requirements=["onnx", "tensorrt"],
        )
        rollout = ota.create_rollout(
            pkg["package_id"], target_devices=["dev_001"],
        )
        result = ota.execute_rollout(rollout["rollout_id"])
        assert result["results"][0]["status"] == "skipped"

    def test_execute_sets_device_status(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        fm.register_device("dev_001", "Robot A", "arm", capabilities=["onnx"])
        fm.update_status("dev_001", "online")
        src = _make_source_package(tmp_path)
        pkg = ota.create_update_package(
            src, "1.0.0", "bc_pick", model_version=1, requirements=["onnx"],
        )
        rollout = ota.create_rollout(
            pkg["package_id"], target_devices=["dev_001"],
        )
        ota.execute_rollout(rollout["rollout_id"])
        device = fm.get_device("dev_001")
        # After successful deployment, device should be back online
        assert device["status"] == "online"


# ── TestRolloutRollback ─────────────────────────────────────

class TestRolloutRollback:
    def test_rollback_rollout(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        ids = _register_devices(fm)
        src = _make_source_package(tmp_path)
        pkg = ota.create_update_package(
            src, "1.0.0", "bc_pick", model_version=1, requirements=["onnx"],
        )
        rollout = ota.create_rollout(
            pkg["package_id"], target_devices=ids, strategy="immediate",
        )
        ota.execute_rollout(rollout["rollout_id"])
        result = ota.rollback_rollout(rollout["rollout_id"])
        assert result["status"] == "rolled-back"

    def test_rollback_reverts_device_model(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        fm.register_device("dev_001", "Robot A", "arm", capabilities=["onnx"])
        fm.update_status("dev_001", "online")
        fm.update_device_model("dev_001", "bc_pick", 2)
        src = _make_source_package(tmp_path)
        pkg = ota.create_update_package(
            src, "1.0.0", "bc_pick", model_version=3, requirements=["onnx"],
        )
        rollout = ota.create_rollout(
            pkg["package_id"], target_devices=["dev_001"],
        )
        ota.execute_rollout(rollout["rollout_id"])
        device = fm.get_device("dev_001")
        assert device["current_model_version"] == 3
        ota.rollback_rollout(rollout["rollout_id"])
        device = fm.get_device("dev_001")
        assert device["current_model_version"] == 2

    def test_rollback_nonexistent_raises(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        with pytest.raises(ValueError, match="Rollout not found"):
            ota.rollback_rollout("rollout_99999")


# ── TestDeviceHistory ───────────────────────────────────────

class TestDeviceHistory:
    def test_history_recorded_on_execute(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        fm.register_device("dev_001", "Robot A", "arm", capabilities=["onnx"])
        fm.update_status("dev_001", "online")
        src = _make_source_package(tmp_path)
        pkg = ota.create_update_package(
            src, "1.0.0", "bc_pick", model_version=1, requirements=["onnx"],
        )
        rollout = ota.create_rollout(
            pkg["package_id"], target_devices=["dev_001"],
        )
        ota.execute_rollout(rollout["rollout_id"])
        history = ota.get_device_history("dev_001")
        assert len(history) == 1
        assert history[0]["status"] == "success"

    def test_history_recorded_on_rollback(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        fm.register_device("dev_001", "Robot A", "arm", capabilities=["onnx"])
        fm.update_status("dev_001", "online")
        src = _make_source_package(tmp_path)
        pkg = ota.create_update_package(
            src, "1.0.0", "bc_pick", model_version=1, requirements=["onnx"],
        )
        rollout = ota.create_rollout(
            pkg["package_id"], target_devices=["dev_001"],
        )
        ota.execute_rollout(rollout["rollout_id"])
        ota.rollback_rollout(rollout["rollout_id"])
        history = ota.get_device_history("dev_001")
        assert len(history) == 2
        assert history[1]["status"] == "rolled-back"

    def test_get_empty_history(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        history = ota.get_device_history("dev_nonexistent")
        assert history == []


# ── TestVersionParsing ──────────────────────────────────────

class TestVersionParsing:
    def test_parse_version(self):
        assert parse_version("1.2.3") == (1, 2, 3)
        assert parse_version("0.0.1") == (0, 0, 1)
        assert parse_version("10.20.30") == (10, 20, 30)

    def test_compare_versions(self):
        assert compare_versions("1.0.0", "2.0.0") == -1
        assert compare_versions("2.0.0", "1.0.0") == 1
        assert compare_versions("1.0.0", "1.0.0") == 0
        assert compare_versions("1.0.0", "1.0.1") == -1
        assert compare_versions("1.1.0", "1.0.9") == 1

    def test_invalid_version_raises(self):
        with pytest.raises(ValueError, match="Invalid semver"):
            parse_version("1.0")
        with pytest.raises(ValueError, match="Invalid semver"):
            parse_version("abc")
        with pytest.raises(ValueError, match="Invalid semver"):
            parse_version("1.0.0.0")


# ── TestRolloutListing ──────────────────────────────────────

class TestRolloutListing:
    def test_list_rollouts(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        ids = _register_devices(fm)
        src = _make_source_package(tmp_path)
        pkg = ota.create_update_package(src, "1.0.0", "bc_pick", model_version=1)
        ota.create_rollout(pkg["package_id"], target_devices=ids)
        ota.create_rollout(pkg["package_id"], target_devices=ids)
        rollouts = ota.list_rollouts()
        assert len(rollouts) == 2

    def test_list_rollouts_filter_by_status(self, tmp_path):
        fm, ota = _setup_fleet_and_ota(tmp_path)
        ids = _register_devices(fm)
        src = _make_source_package(tmp_path)
        pkg = ota.create_update_package(
            src, "1.0.0", "bc_pick", model_version=1, requirements=["onnx"],
        )
        r1 = ota.create_rollout(pkg["package_id"], target_devices=ids)
        ota.create_rollout(pkg["package_id"], target_devices=ids)
        ota.execute_rollout(r1["rollout_id"])
        completed = ota.list_rollouts(status="completed")
        pending = ota.list_rollouts(status="pending")
        assert len(completed) == 1
        assert len(pending) == 1
