"""Unit tests for the model registry system."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.model_registry import ModelRegistry, VALID_STAGES, STAGE_TRANSITIONS


# ── TestModelRegistration ───────────────────────────────────

class TestModelRegistration:
    def test_register_model(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        entry = reg.register_model("bc_pick", checkpoint_path="models/bc_pick.pt")
        assert entry["name"] == "bc_pick"
        assert entry["version"] == 1
        assert entry["checkpoint_path"] == "models/bc_pick.pt"

    def test_auto_version_increment(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        e1 = reg.register_model("bc_pick", checkpoint_path="models/v1.pt")
        e2 = reg.register_model("bc_pick", checkpoint_path="models/v2.pt")
        assert e1["version"] == 1
        assert e2["version"] == 2

    def test_register_with_experiment_id(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        entry = reg.register_model(
            "bc_pick", checkpoint_path="models/v1.pt", experiment_id="exp_00001",
        )
        assert entry["experiment_id"] == "exp_00001"

    def test_register_with_metadata(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        entry = reg.register_model(
            "bc_pick", checkpoint_path="models/v1.pt",
            metadata={"hidden_layers": [256, 128], "obs_dim": 31},
        )
        assert entry["metadata"]["hidden_layers"] == [256, 128]

    def test_initial_stage_is_training(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        entry = reg.register_model("bc_pick", checkpoint_path="models/v1.pt")
        assert entry["stage"] == "training"


# ── TestModelRetrieval ──────────────────────────────────────

class TestModelRetrieval:
    def test_get_by_name_version(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        reg.register_model("bc_pick", checkpoint_path="models/v1.pt")
        reg.register_model("bc_pick", checkpoint_path="models/v2.pt")
        entry = reg.get_model("bc_pick", version=1)
        assert entry is not None
        assert entry["version"] == 1
        assert entry["checkpoint_path"] == "models/v1.pt"

    def test_get_latest_version(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        reg.register_model("bc_pick", checkpoint_path="models/v1.pt")
        reg.register_model("bc_pick", checkpoint_path="models/v2.pt")
        entry = reg.get_model("bc_pick")
        assert entry is not None
        assert entry["version"] == 2

    def test_get_by_stage(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        reg.register_model("bc_pick", checkpoint_path="models/v1.pt")
        e2 = reg.register_model("bc_pick", checkpoint_path="models/v2.pt")
        reg.promote("bc_pick", 1, "validated")
        entry = reg.get_model("bc_pick", stage="validated")
        assert entry is not None
        assert entry["version"] == 1
        assert entry["stage"] == "validated"

    def test_get_nonexistent_returns_none(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        assert reg.get_model("nonexistent") is None
        assert reg.get_model("nonexistent", version=1) is None


# ── TestModelListing ────────────────────────────────────────

class TestModelListing:
    def test_list_empty(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        assert reg.list_models() == []

    def test_list_all(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        reg.register_model("bc_pick", checkpoint_path="models/v1.pt")
        reg.register_model("bc_place", checkpoint_path="models/place_v1.pt")
        models = reg.list_models()
        assert len(models) == 2

    def test_list_filter_by_name(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        reg.register_model("bc_pick", checkpoint_path="models/v1.pt")
        reg.register_model("bc_place", checkpoint_path="models/place_v1.pt")
        models = reg.list_models(name="bc_pick")
        assert len(models) == 1
        assert models[0]["name"] == "bc_pick"

    def test_list_filter_by_stage(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        reg.register_model("bc_pick", checkpoint_path="models/v1.pt")
        reg.register_model("bc_pick", checkpoint_path="models/v2.pt")
        reg.promote("bc_pick", 1, "validated")
        models = reg.list_models(stage="validated")
        assert len(models) == 1
        assert models[0]["version"] == 1

    def test_list_sorted(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        reg.register_model("bc_place", checkpoint_path="models/place_v1.pt")
        reg.register_model("bc_pick", checkpoint_path="models/v1.pt")
        reg.register_model("bc_pick", checkpoint_path="models/v2.pt")
        models = reg.list_models()
        names_versions = [(m["name"], m["version"]) for m in models]
        assert names_versions == [("bc_pick", 1), ("bc_pick", 2), ("bc_place", 1)]


# ── TestModelPromotion ──────────────────────────────────────

class TestModelPromotion:
    def test_promote_training_to_validated(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        reg.register_model("bc_pick", checkpoint_path="models/v1.pt")
        entry = reg.promote("bc_pick", 1, "validated")
        assert entry["stage"] == "validated"

    def test_promote_validated_to_production(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        reg.register_model("bc_pick", checkpoint_path="models/v1.pt")
        reg.promote("bc_pick", 1, "validated")
        entry = reg.promote("bc_pick", 1, "production")
        assert entry["stage"] == "production"

    def test_promote_to_archived(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        reg.register_model("bc_pick", checkpoint_path="models/v1.pt")
        entry = reg.promote("bc_pick", 1, "archived")
        assert entry["stage"] == "archived"

    def test_promote_archived_to_validated(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        reg.register_model("bc_pick", checkpoint_path="models/v1.pt")
        reg.promote("bc_pick", 1, "archived")
        entry = reg.promote("bc_pick", 1, "validated")
        assert entry["stage"] == "validated"

    def test_invalid_promotion_raises(self, tmp_path):
        """training -> production directly is not allowed."""
        reg = ModelRegistry(registry_dir=tmp_path)
        reg.register_model("bc_pick", checkpoint_path="models/v1.pt")
        with pytest.raises(ValueError, match="Invalid stage transition"):
            reg.promote("bc_pick", 1, "production")

    def test_promote_nonexistent_raises(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        with pytest.raises(ValueError, match="Model not found"):
            reg.promote("nonexistent", 1, "validated")


# ── TestModelDeletion ───────────────────────────────────────

class TestModelDeletion:
    def test_delete_model(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        reg.register_model("bc_pick", checkpoint_path="models/v1.pt")
        reg.delete_model("bc_pick", 1)
        assert reg.list_models() == []

    def test_delete_nonexistent_raises(self, tmp_path):
        reg = ModelRegistry(registry_dir=tmp_path)
        with pytest.raises(ValueError, match="Model not found"):
            reg.delete_model("nonexistent", 1)
