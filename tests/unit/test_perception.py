"""Unit tests for the perception system."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.simulation import ForgeSimulation
from forge_core.perception import (
    PerceptionPipeline,
    PerceptionResult,
    Detection,
    ColorDetector,
)


@pytest.fixture
def sim():
    s = ForgeSimulation("basic_workspace")
    s.step(100)  # let physics settle
    yield s
    s.shutdown()


class TestColorDetector:
    def test_detector_returns_list(self, sim):
        pipeline = PerceptionPipeline(sim)
        img = pipeline.render()
        detector = ColorDetector()
        detections = detector.detect(img)
        assert isinstance(detections, list)
        pipeline.close()

    def test_detections_have_required_fields(self, sim):
        pipeline = PerceptionPipeline(sim)
        img = pipeline.render()
        detector = ColorDetector()
        detections = detector.detect(img)
        for d in detections:
            assert isinstance(d.object_id, str)
            assert isinstance(d.pixel_center, tuple)
            assert isinstance(d.pixel_area, int)
            assert 0 <= d.confidence <= 1.0
        pipeline.close()


class TestPerceptionPipeline:
    def test_render_returns_image(self, sim):
        pipeline = PerceptionPipeline(sim)
        img = pipeline.render()
        assert img.shape == (480, 640, 3)
        assert img.dtype == np.uint8
        pipeline.close()

    def test_perceive_returns_result(self, sim):
        pipeline = PerceptionPipeline(sim)
        result = pipeline.perceive()
        assert isinstance(result, PerceptionResult)
        assert result.image is not None
        assert result.image.shape[0] > 0
        pipeline.close()

    def test_detects_at_least_one_object(self, sim):
        pipeline = PerceptionPipeline(sim)
        result = pipeline.perceive()
        assert len(result.detections) >= 1
        pipeline.close()

    def test_detections_have_world_positions(self, sim):
        pipeline = PerceptionPipeline(sim)
        result = pipeline.perceive()
        for d in result.detections:
            assert d.position.shape == (3,)
            # Position should be finite
            assert np.all(np.isfinite(d.position))
        pipeline.close()

    def test_to_object_states(self, sim):
        pipeline = PerceptionPipeline(sim)
        result = pipeline.perceive()
        states = result.to_object_states()
        assert len(states) == len(result.detections)
        for s in states:
            assert s.position.shape == (3,)
        pipeline.close()

    def test_get_detection_by_id(self, sim):
        pipeline = PerceptionPipeline(sim)
        result = pipeline.perceive()
        # At least one of the cubes should be detected
        found = result.get_detection("red_cube") or result.get_detection("blue_cube")
        assert found is not None
        pipeline.close()

    def test_perception_differs_from_ground_truth(self, sim):
        """Perception estimates should not be identical to ground truth
        (they should have some error from the detection process)."""
        pipeline = PerceptionPipeline(sim)
        result = pipeline.perceive()
        gt = {o.id: o.position for o in sim.get_object_states()}

        for d in result.detections:
            if d.object_id in gt:
                # Some error expected from color detection + projection
                err = np.linalg.norm(d.position - gt[d.object_id])
                # Should be nonzero (perception is imperfect)
                # but not wildly wrong (< 1m for a reasonable detector)
                assert err > 0.001, "Suspiciously perfect — might be using ground truth"
                assert err < 2.0, f"Perception error too large: {err:.3f}m"
        pipeline.close()

    def test_custom_camera_params(self, sim):
        """Pipeline accepts custom camera parameters."""
        pipeline = PerceptionPipeline(
            sim, azimuth=160, elevation=-30, distance=1.5,
            lookat=(0.5, 0.0, 0.22),
        )
        result = pipeline.perceive()
        assert result.image is not None
        pipeline.close()
