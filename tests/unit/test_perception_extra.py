"""Extra unit tests for the perception system (Phase 5)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.simulation import ForgeSimulation
from forge_core.policy import NullPolicy
from forge_core.perception import (
    PerceptionPipeline,
    PerceptionResult,
    Detection,
    ColorDetector,
    PerceptionPolicy,
)


@pytest.fixture
def sim():
    s = ForgeSimulation("basic_workspace")
    s.step(100)  # let physics settle
    yield s
    s.shutdown()


class TestColorDetectorExtra:
    def test_color_detector_empty_image(self):
        """Blank image should produce no detections."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = ColorDetector().detect(img)
        assert isinstance(detections, list)
        assert len(detections) == 0


class TestPerceptionPipelineExtra:
    def test_perception_pipeline_close_idempotent(self, sim):
        """Calling close() twice should not raise."""
        pipeline = PerceptionPipeline(sim)
        pipeline.close()
        pipeline.close()  # second call should be safe

    def test_perception_pipeline_context_manager(self, sim):
        """Pipeline works as a context manager."""
        with PerceptionPipeline(sim) as p:
            img = p.render()
            assert img.shape == (480, 640, 3)
        # exiting the context should not raise

    def test_render_depth_shape(self, sim):
        """render_depth() returns a (480, 640) float array."""
        pipeline = PerceptionPipeline(sim)
        depth = pipeline.render_depth()
        assert depth.shape == (480, 640)
        assert depth.dtype == np.float32
        pipeline.close()

    def test_perception_result_timestamp(self, sim):
        """perceive() timestamp should be close to sim.sim_time."""
        pipeline = PerceptionPipeline(sim)
        result = pipeline.perceive()
        assert abs(result.timestamp - sim.sim_time) < 0.1
        pipeline.close()


class TestPerceptionPolicyExtra:
    def test_perception_policy_wrapper(self, sim):
        """PerceptionPolicy wraps a policy and exposes last_perception."""
        inner = NullPolicy()
        inner.reset(sim)
        pp = PerceptionPolicy(inner, sim)
        targets = pp.act(sim)
        assert isinstance(targets, np.ndarray)
        assert isinstance(pp.last_perception, PerceptionResult)
        pp.close()
