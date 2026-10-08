"""Perception system for Forge.

Replaces ground-truth object state with camera-based detection.
Uses MuJoCo offscreen rendering + OpenCV color segmentation to
estimate object positions from camera images.

Ground truth remains available exclusively for evaluation —
the policy sees only perception output.

Phase 5 uses color-based detection (HSV thresholding). Future phases
can swap in learned detectors without changing the interface.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import cv2
import mujoco
import numpy as np

from forge_core.simulation import ForgeSimulation, ObjectState

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single object detection from the perception system."""

    object_id: str
    position: np.ndarray  # estimated (x, y, z) in world frame
    confidence: float = 1.0
    pixel_center: tuple[int, int] = (0, 0)
    pixel_area: int = 0


@dataclass
class PerceptionResult:
    """Output of a perception cycle."""

    detections: list[Detection] = field(default_factory=list)
    image: np.ndarray | None = None  # raw camera image (H, W, 3) RGB
    timestamp: float = 0.0

    def get_detection(self, object_id: str) -> Detection | None:
        """Find a detection by object ID."""
        for d in self.detections:
            if d.object_id == object_id:
                return d
        return None

    def to_object_states(self) -> list[ObjectState]:
        """Convert detections to ObjectState format for policy compatibility."""
        return [
            ObjectState(
                id=d.object_id,
                position=d.position,
                orientation=np.array([1.0, 0, 0, 0]),  # unknown orientation
                timestamp=self.timestamp,
            )
            for d in self.detections
        ]


class Detector(ABC):
    """Base interface for object detectors."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect objects in an RGB image."""


class ColorDetector(Detector):
    """Detect objects by HSV color segmentation.

    Each known object is associated with an HSV range. The detector
    finds contours matching each color and returns their centroids.
    HSV ranges are calibrated for MuJoCo's rendering with the Menagerie
    Panda scene from a side-angle camera.
    """

    COLOR_PROFILES: dict[str, dict[str, Any]] = {
        "red_cube": {
            # H=0-7, S=168-227, V=113-229 (from segmentation analysis)
            "lower": np.array([0, 130, 80]),
            "upper": np.array([10, 255, 255]),
        },
        "blue_cube": {
            # H=120-170, S=60-227, V=139-255
            "lower": np.array([115, 50, 100]),
            "upper": np.array([175, 255, 255]),
        },
        "target_bin": {
            "lower": np.array([35, 40, 40]),
            "upper": np.array([85, 255, 255]),
        },
    }

    def __init__(self, min_area: int = 20):
        self.min_area = min_area

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect colored objects via HSV thresholding."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        detections = []

        for obj_id, profile in self.COLOR_PROFILES.items():
            mask = cv2.inRange(hsv, profile["lower"], profile["upper"])

            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area < self.min_area:
                continue

            M = cv2.moments(largest)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            confidence = min(1.0, area / 500.0)

            detections.append(Detection(
                object_id=obj_id,
                position=np.zeros(3),
                confidence=confidence,
                pixel_center=(cx, cy),
                pixel_area=int(area),
            ))

        return detections


class PerceptionPipeline:
    """Complete perception pipeline: render → detect → project to world.

    Uses a programmatic MjvCamera (free camera) positioned to see the
    workspace. This avoids the complexity of defining camera orientation
    in MJCF XML.
    """

    def __init__(
        self,
        sim: ForgeSimulation,
        width: int = 640,
        height: int = 480,
        detector: Detector | None = None,
        azimuth: float = 180.0,
        elevation: float = -35.0,
        distance: float = 1.2,
        lookat: tuple[float, float, float] = (0.5, 0.0, 0.22),
    ):
        self.sim = sim
        self.width = width
        self.height = height
        self.detector = detector or ColorDetector()

        self._renderer = mujoco.Renderer(sim.model, height=height, width=width)

        # Programmatic scene camera
        self._scene_option = mujoco.MjvOption()
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._camera.lookat[:] = lookat
        self._camera.distance = distance
        self._camera.azimuth = azimuth
        self._camera.elevation = elevation

        # Cache camera position for projection
        self._fovy = 45.0  # MuJoCo free camera default fovy
        self._update_cam_pos()

        logger.info("perception_initialized: free camera (%dx%d) "
                     "az=%.0f el=%.0f dist=%.1f",
                     width, height, azimuth, elevation, distance)

    def _update_cam_pos(self) -> None:
        """Compute camera world position from orbital parameters."""
        az = np.deg2rad(self._camera.azimuth)
        el = np.deg2rad(self._camera.elevation)
        d = self._camera.distance
        la = np.array(self._camera.lookat)

        self._cam_pos = np.array([
            la[0] + d * np.cos(el) * np.cos(az),
            la[1] + d * np.cos(el) * np.sin(az),
            la[2] + d * np.sin(-el),  # negative because elevation down is negative
        ])

        # Look direction
        self._look_dir = la - self._cam_pos
        self._look_dir = self._look_dir / np.linalg.norm(self._look_dir)

        # Camera axes
        up = np.array([0.0, 0.0, 1.0])
        self._right = np.cross(self._look_dir, up)
        norm = np.linalg.norm(self._right)
        if norm < 1e-6:
            self._right = np.array([1.0, 0.0, 0.0])
        else:
            self._right = self._right / norm
        self._up = np.cross(self._right, self._look_dir)
        self._up = self._up / np.linalg.norm(self._up)

    def render(self) -> np.ndarray:
        """Render the current scene. Returns RGB image."""
        self._renderer.update_scene(self.sim.data, self._camera, self._scene_option)
        return self._renderer.render().copy()

    def perceive(self) -> PerceptionResult:
        """Full perception cycle: render → detect → project."""
        image = self.render()
        detections = self.detector.detect(image)

        for det in detections:
            det.position = self._pixel_to_world(det.pixel_center, image.shape)

        return PerceptionResult(
            detections=detections,
            image=image,
            timestamp=self.sim.sim_time,
        )

    def _pixel_to_world(
        self, pixel: tuple[int, int], img_shape: tuple[int, ...]
    ) -> np.ndarray:
        """Project pixel coordinate to world position via ray-plane intersection."""
        h, w = img_shape[:2]
        cx, cy = pixel

        fovy_rad = np.deg2rad(self._fovy)
        f = h / (2.0 * np.tan(fovy_rad / 2.0))

        u = (cx - w / 2.0) / f
        v = -(cy - h / 2.0) / f

        ray = self._look_dir + u * self._right + v * self._up
        ray = ray / np.linalg.norm(ray)

        # Intersect with table plane z = 0.24
        table_z = 0.24
        if abs(ray[2]) < 1e-6:
            return self._cam_pos.copy()
        t = (table_z - self._cam_pos[2]) / ray[2]
        if t < 0:
            return self._cam_pos.copy()
        return self._cam_pos + t * ray

    def close(self):
        """Release rendering resources."""
        self._renderer.close()

    def __del__(self):
        try:
            self._renderer.close()
        except Exception:
            pass
