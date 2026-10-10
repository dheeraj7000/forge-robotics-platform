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
from dataclasses import dataclass, field

import cv2
import mujoco
import numpy as np

from forge_core.policy import Policy
from forge_core.simulation import ForgeSimulation, ObjectState

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single object detection from the perception system."""

    object_id: str
    pixel_center: tuple  # (row, col) in image coordinates
    pixel_area: int
    confidence: float
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))


class ColorDetector:
    """Detect colored objects by HSV filtering.

    Each known object is associated with one or more HSV ranges. The
    detector finds contours matching each color and returns their centroids.
    HSV ranges are calibrated for MuJoCo's rendered colors.
    """

    COLOR_RANGES: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "red_cube": [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([170, 100, 100]), np.array([180, 255, 255])),
        ],
        "blue_cube": [
            (np.array([100, 100, 100]), np.array([130, 255, 255])),
        ],
    }

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect colored objects in an RGB image."""
        if image is None or not isinstance(image, np.ndarray):
            logger.warning("detect called with invalid image")
            return []
        if image.dtype != np.uint8 or image.size == 0:
            logger.warning("detect called with wrong dtype or empty image")
            return []

        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        except cv2.error:
            logger.warning("failed to convert image to HSV")
            return []

        detections: list[Detection] = []
        for obj_id, ranges in self.COLOR_RANGES.items():
            # Build combined mask from all ranges for this color
            combined_mask = None
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)

            if combined_mask is None:
                continue

            # Morphological cleanup with 5x5 kernel
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.morphologyEx(
                combined_mask, cv2.MORPH_OPEN, kernel
            )

            # Find contours
            contours, _ = cv2.findContours(
                combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            # Filter by minimum area (50px) and take the largest
            valid = [
                (c, cv2.contourArea(c))
                for c in contours
                if cv2.contourArea(c) >= 50
            ]
            if not valid:
                continue

            largest, area = max(valid, key=lambda x: x[1])
            M = cv2.moments(largest)
            if M["m00"] == 0:
                continue

            # pixel_center = (cy, cx) for (row, col) order
            cy = int(M["m01"] / M["m00"])
            cx = int(M["m10"] / M["m00"])
            pixel_center = (cy, cx)

            confidence = min(1.0, area / 500.0)

            detections.append(Detection(
                object_id=obj_id,
                pixel_center=pixel_center,
                pixel_area=int(area),
                confidence=confidence,
                position=np.zeros(3),
            ))

        return detections


@dataclass
class PerceptionResult:
    """Output of a perception cycle."""

    image: np.ndarray
    detections: list[Detection]
    timestamp: float = 0.0

    def to_object_states(self) -> list[ObjectState]:
        """Convert detections to ObjectState format for compatibility."""
        return [
            ObjectState(
                id=d.object_id,
                position=d.position,
                orientation=np.array([1.0, 0, 0, 0]),
                timestamp=self.timestamp,
            )
            for d in self.detections
        ]

    def get_detection(self, object_id: str) -> Detection | None:
        """Look up a detection by object_id. Returns None if not found."""
        for d in self.detections:
            if d.object_id == object_id:
                return d
        return None


class PerceptionPipeline:
    """Full perception pipeline: render → detect → project to world.

    Uses a programmatic MjvCamera (free camera) positioned to see the
    workspace. Default scene options are sufficient — no MjvOption needed.
    """

    def __init__(
        self,
        sim: ForgeSimulation,
        *,
        width: int = 640,
        height: int = 480,
        azimuth: float = 180,
        elevation: float = -30,
        distance: float = 1.8,
        lookat: tuple = (0.5, 0.0, 0.22),
    ):
        """Initialize pipeline with simulation and camera params."""
        if distance <= 0:
            raise ValueError(f"distance must be > 0, got {distance}")
        if len(lookat) != 3:
            raise ValueError(
                f"lookat must have 3 elements, got {len(lookat)}"
            )
        if width <= 0 or height <= 0:
            raise ValueError(
                f"width and height must be > 0, got {width}x{height}"
            )

        self._sim = sim
        self._width = width
        self._height = height
        self._azimuth = azimuth
        self._elevation = elevation
        self._distance = distance
        self._lookat = np.array(lookat, dtype=np.float64)

        self._renderer = mujoco.Renderer(sim.model, height=height, width=width)

        # Programmatic free camera — default scene options are sufficient
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._camera.azimuth = azimuth
        self._camera.elevation = elevation
        self._camera.distance = distance
        self._camera.lookat[:] = lookat

        self._detector = ColorDetector()
        self._rng = np.random.default_rng(42)

        logger.info(
            "perception_initialized: free camera (%dx%d) "
            "az=%.0f el=%.0f dist=%.1f",
            width, height, azimuth, elevation, distance,
        )

    def render(self) -> np.ndarray:
        """Render the current scene from the camera viewpoint.

        Returns:
            (height, width, 3) uint8 RGB image.
        """
        self._renderer.update_scene(self._sim.data, self._camera)
        return self._renderer.render()

    def render_depth(self) -> np.ndarray:
        """Render depth buffer.

        Returns:
            (height, width) float32 array with depth in meters.
            0.0 indicates no geometry hit.
        """
        self._renderer.update_scene(self._sim.data, self._camera)
        self._renderer.enable_depth_rendering()
        raw = self._renderer.render()
        # raw.max() <= 1.0 heuristic assumes camera distance >= 1.0m —
        # may misfire for very close-up cameras where linearized depth
        # values also fall below 1.0
        if raw.max() <= 1.0:
            # Raw OpenGL depth buffer — linearize to meters
            extent = self._sim.model.stat.extent
            near = self._sim.model.vis.map.znear * extent
            far = self._sim.model.vis.map.zfar * extent
            depth = np.where(
                raw < 0.999,
                near * far / (far - raw * (far - near)),
                0.0,
            )
        else:
            # Already linearized in meters; 0.0 = no geometry
            depth = raw
        self._renderer.disable_depth_rendering()
        return depth.astype(np.float32)

    def perceive(self) -> PerceptionResult:
        """Full perception cycle: render + detect + estimate 3D positions.

        Returns:
            PerceptionResult with image, detections, and timestamp.
        """
        # Single update_scene at top
        self._renderer.update_scene(self._sim.data, self._camera)

        # .copy() prevents buffer reuse corruption when depth render follows
        image = self._renderer.render().copy()

        # Depth rendering
        self._renderer.enable_depth_rendering()
        raw = self._renderer.render()
        # raw.max() <= 1.0 heuristic assumes camera distance >= 1.0m
        if raw.max() <= 1.0:
            extent = self._sim.model.stat.extent
            near = self._sim.model.vis.map.znear * extent
            far = self._sim.model.vis.map.zfar * extent
            depth = np.where(
                raw < 0.999,
                near * far / (far - raw * (far - near)),
                0.0,
            )
        else:
            depth = raw
        self._renderer.disable_depth_rendering()
        depth = depth.astype(np.float32)

        # Detect objects
        detections = self._detector.detect(image)

        # Estimate 3D positions via depth unprojection
        fovy = self._sim.model.vis.global_.fovy
        f = 0.5 * self._height / np.tan(np.radians(fovy / 2))

        az_rad = np.radians(self._azimuth)
        el_rad = np.radians(self._elevation)
        cam_pos = self._lookat + self._distance * np.array([
            np.cos(el_rad) * np.cos(az_rad),
            np.cos(el_rad) * np.sin(az_rad),
            -np.sin(el_rad),
        ])

        forward = self._lookat - cam_pos
        forward = forward / np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / right_norm
        up = np.cross(right, forward)
        # Camera-to-world rotation: columns are right, up, -forward
        R = np.column_stack([right, up, -forward])

        for det in detections:
            # Destructure row, col from pixel_center (row, col) order
            row, col = det.pixel_center
            d = depth[row, col]
            if d == 0.0:
                # Fallback: lookat + small gaussian noise (seeded RNG)
                det.position = (
                    self._lookat.copy() + self._rng.normal(0, 0.05, 3)
                )
            else:
                x_cam = (col - self._width / 2) * d / f
                y_cam = (self._height / 2 - row) * d / f
                z_cam = -d
                cam_point = np.array([x_cam, y_cam, z_cam])
                det.position = R @ cam_point + cam_pos

        timestamp = self._sim.sim_time

        return PerceptionResult(
            image=image,
            detections=detections,
            timestamp=timestamp,
        )

    def close(self) -> None:
        """Release the MuJoCo renderer resources. Idempotent."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


class PerceptionPolicy(Policy):
    """Wrapper that adds perception observations to any policy."""

    def __init__(
        self,
        inner_policy: Policy,
        sim: ForgeSimulation,
        **camera_kwargs,
    ):
        self._inner = inner_policy
        self._pipeline = PerceptionPipeline(sim, **camera_kwargs)
        self._last_result: PerceptionResult | None = None

    def reset(self, sim: ForgeSimulation) -> None:
        """Reset inner policy and clear perception state."""
        self._inner.reset(sim)
        self._last_result = None

    def act(self, sim: ForgeSimulation) -> np.ndarray:
        """Perceive, then delegate to inner policy."""
        self._last_result = self._pipeline.perceive()
        return self._inner.act(sim)

    @property
    def done(self) -> bool:
        """Whether the inner policy has finished."""
        return self._inner.done

    @property
    def name(self) -> str:
        """Name reflecting the wrapper."""
        return f"Perception({self._inner.name})"

    @property
    def last_perception(self) -> PerceptionResult | None:
        """Most recent perception result, or None before first act()."""
        return self._last_result

    def close(self) -> None:
        """Release pipeline resources."""
        self._pipeline.close()
