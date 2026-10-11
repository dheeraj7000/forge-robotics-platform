"""Model registry for Forge.

File-based model versioning with stage transitions and zero external
dependencies.  All model entries live in a single ``registry.json`` file
under the *registry/* directory.

Stage lifecycle::

    training ─> validated ─> production ─> archived
                    ^                          │
                    └──────────────────────────┘
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forge_core.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

VALID_STAGES = ("training", "validated", "production", "archived")

STAGE_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "training": ("validated", "archived"),
    "validated": ("production", "archived"),
    "production": ("archived",),
    "archived": ("validated",),
}


class ModelRegistry:
    """Version and stage-manage trained model checkpoints.

    Stores all entries in ``registry_dir/registry.json``.
    """

    def __init__(self, registry_dir: Path | None = None) -> None:
        self.registry_dir = registry_dir or PROJECT_ROOT / "registry"
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    # ── Persistence helpers ─────────────────────────────────

    def _load_registry(self) -> dict[str, Any]:
        """Load the registry from disk, returning empty if missing."""
        path = self.registry_dir / "registry.json"
        if not path.exists():
            return {"models": []}
        with open(path, "r") as f:
            return json.load(f)

    def _save_registry(self, data: dict[str, Any]) -> None:
        """Write the registry to disk."""
        path = self.registry_dir / "registry.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Version helpers ─────────────────────────────────────

    def _next_version(self, name: str) -> int:
        """Return the next version number for *name*."""
        data = self._load_registry()
        versions = [
            m["version"] for m in data["models"] if m["name"] == name
        ]
        return max(versions, default=0) + 1

    # ── Public API ──────────────────────────────────────────

    def register_model(
        self,
        name: str,
        checkpoint_path: str,
        experiment_id: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a new model version and return the entry."""
        data = self._load_registry()
        version = self._next_version(name)
        entry: dict[str, Any] = {
            "name": name,
            "version": version,
            "stage": "training",
            "checkpoint_path": checkpoint_path,
            "experiment_id": experiment_id,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "description": description,
            "metadata": metadata or {},
        }
        data["models"].append(entry)
        self._save_registry(data)
        logger.info("model_registered: %s v%d", name, version)
        return entry

    def get_model(
        self,
        name: str,
        version: int | None = None,
        stage: str | None = None,
    ) -> dict[str, Any] | None:
        """Retrieve a model entry.

        - If *version* is given, return the exact match.
        - If *stage* is given (without version), return the latest version
          in that stage.
        - If neither, return the latest version overall.
        """
        data = self._load_registry()
        candidates = [m for m in data["models"] if m["name"] == name]
        if not candidates:
            return None
        if version is not None:
            for m in candidates:
                if m["version"] == version:
                    return m
            return None
        if stage is not None:
            candidates = [m for m in candidates if m["stage"] == stage]
        if not candidates:
            return None
        return max(candidates, key=lambda m: m["version"])

    def list_models(
        self,
        name: str | None = None,
        stage: str | None = None,
    ) -> list[dict[str, Any]]:
        """List model entries, optionally filtered by name and/or stage.

        Returns entries sorted by (name, version).
        """
        data = self._load_registry()
        models = data["models"]
        if name is not None:
            models = [m for m in models if m["name"] == name]
        if stage is not None:
            models = [m for m in models if m["stage"] == stage]
        models.sort(key=lambda m: (m["name"], m["version"]))
        return models

    def promote(self, name: str, version: int, target_stage: str) -> dict[str, Any]:
        """Promote a model to *target_stage*.

        Validates the transition against :data:`STAGE_TRANSITIONS` and
        raises :class:`ValueError` for illegal moves.
        """
        data = self._load_registry()
        entry = None
        for m in data["models"]:
            if m["name"] == name and m["version"] == version:
                entry = m
                break
        if entry is None:
            raise ValueError(f"Model not found: {name} v{version}")
        current = entry["stage"]
        allowed = STAGE_TRANSITIONS.get(current, ())
        if target_stage not in allowed:
            raise ValueError(
                f"Invalid stage transition: {current} -> {target_stage}. "
                f"Allowed: {allowed}"
            )
        entry["stage"] = target_stage
        self._save_registry(data)
        logger.info("model_promoted: %s v%d -> %s", name, version, target_stage)
        return entry

    def delete_model(self, name: str, version: int) -> None:
        """Remove a model entry from the registry.

        Does **not** delete the checkpoint file on disk.
        """
        data = self._load_registry()
        original_len = len(data["models"])
        data["models"] = [
            m for m in data["models"]
            if not (m["name"] == name and m["version"] == version)
        ]
        if len(data["models"]) == original_len:
            raise ValueError(f"Model not found: {name} v{version}")
        self._save_registry(data)
        logger.info("model_deleted: %s v%d", name, version)

    def get_latest(
        self,
        name: str,
        stage: str | None = None,
    ) -> dict[str, Any] | None:
        """Convenience wrapper around :meth:`get_model`."""
        return self.get_model(name, stage=stage)
