"""Menagerie model locator for Forge.

Finds and caches the path to MuJoCo Menagerie robot models.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_MENAGERIE_CACHE = Path.home() / ".cache" / "mujoco_menagerie" / "models"


def get_menagerie_panda_dir() -> Path:
    """Return the directory containing the Menagerie Panda model files.

    Tries to download if not cached.
    """
    # Look for existing cached model
    pattern = str(_MENAGERIE_CACHE / "franka_emika_panda-*")
    matches = glob.glob(pattern)
    if matches:
        d = Path(matches[0])
        if (d / "panda.xml").exists():
            return d

    # Try downloading via mujoco_menagerie
    try:
        import mujoco_menagerie
        mujoco_menagerie.load("franka_emika_panda")
        # Re-check
        matches = glob.glob(pattern)
        if matches:
            return Path(matches[0])
    except Exception as e:
        logger.warning("Could not download Menagerie Panda: %s", e)

    raise FileNotFoundError(
        "Menagerie Panda model not found. Install: pip install mujoco_menagerie"
    )
