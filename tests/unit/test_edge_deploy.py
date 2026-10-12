"""Unit tests for edge deployment integration (Phase 9 — FEAT-002)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.edge_deploy import (
    has_onnxruntime,
    has_tensorrt,
    ExportConfig,
    TensorRTConfig,
    OnnxExporter,
    OnnxPolicy,
    TensorRTPolicy,
    TensorRTOptimizer,
    EdgePolicy,
    DeploymentManifest,
    DeploymentPackager,
    export_from_registry,
    benchmark_inference,
    OBS_DIM,
    ACT_DIM,
)
from forge_core.learned_policy import PolicyNetwork
from forge_core.policy import Policy, create_policy


# ── Helpers ─────────────────────────────────────────────────

def _make_mock_sim():
    """Create a mock ForgeSimulation that returns valid robot/object states."""
    sim = MagicMock()

    robot = MagicMock()
    robot.joint_positions = np.zeros(9, dtype=np.float32)
    robot.joint_velocities = np.zeros(9, dtype=np.float32)
    robot.ee_position = np.array([0.3, 0.0, 0.5], dtype=np.float32)
    robot.ee_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    sim.get_robot_state.return_value = robot

    obj = MagicMock()
    obj.position = np.array([0.4, 0.1, 0.02], dtype=np.float32)
    obj.orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obj.linear_velocity = np.zeros(3, dtype=np.float32)
    obj.angular_velocity = np.zeros(3, dtype=np.float32)
    sim.get_object_state.return_value = obj

    return sim


def _export_onnx(tmp_path: Path) -> Path:
    """Export a fresh PolicyNetwork to ONNX and return the path."""
    torch.manual_seed(42)
    network = PolicyNetwork()
    exporter = OnnxExporter()
    onnx_path = tmp_path / "model.onnx"
    exporter.export(network, onnx_path)
    return onnx_path


# ── TestRuntimeDetection ────────────────────────────────────

class TestRuntimeDetection:
    def test_has_onnxruntime_returns_bool(self):
        result = has_onnxruntime()
        assert isinstance(result, bool)

    def test_has_tensorrt_returns_false(self):
        # No TensorRT in the test environment
        assert has_tensorrt() is False


# ── TestExportConfig ────────────────────────────────────────

class TestExportConfig:
    def test_default_values(self):
        config = ExportConfig()
        assert config.opset_version == 17
        assert config.dynamic_batch is False
        assert config.input_names == ["observation"]
        assert config.output_names == ["action"]

    def test_custom_values(self):
        config = ExportConfig(opset_version=13, dynamic_batch=True,
                              input_names=["obs"], output_names=["act"])
        assert config.opset_version == 13
        assert config.dynamic_batch is True
        assert config.input_names == ["obs"]
        assert config.output_names == ["act"]


# ── TestTensorRTConfig ──────────────────────────────────────

class TestTensorRTConfig:
    def test_default_values(self):
        config = TensorRTConfig()
        assert config.precision == "fp32"
        assert config.max_batch_size == 1
        assert config.workspace_size_mb == 256

    def test_custom_precision(self):
        config = TensorRTConfig(precision="fp16")
        assert config.precision == "fp16"


# ── TestOnnxExporter ────────────────────────────────────────

@pytest.mark.skipif(not has_onnxruntime(), reason="onnxruntime not installed")
class TestOnnxExporter:
    def test_export_creates_onnx_file(self, tmp_path):
        torch.manual_seed(42)
        network = PolicyNetwork()
        exporter = OnnxExporter()
        onnx_path = tmp_path / "model.onnx"
        result = exporter.export(network, onnx_path)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_export_custom_opset(self, tmp_path):
        torch.manual_seed(42)
        network = PolicyNetwork()
        config = ExportConfig(opset_version=13)
        exporter = OnnxExporter(config=config)
        onnx_path = tmp_path / "model_opset13.onnx"
        result = exporter.export(network, onnx_path)
        assert result.exists()

    def test_export_dynamic_batch(self, tmp_path):
        torch.manual_seed(42)
        network = PolicyNetwork()
        config = ExportConfig(dynamic_batch=True)
        exporter = OnnxExporter(config=config)
        onnx_path = tmp_path / "model_dynamic.onnx"
        result = exporter.export(network, onnx_path)
        assert result.exists()

    def test_validate_passes(self, tmp_path):
        torch.manual_seed(42)
        network = PolicyNetwork()
        exporter = OnnxExporter()
        onnx_path = tmp_path / "model.onnx"
        exporter.export(network, onnx_path)
        assert exporter.validate(onnx_path, network) is True

    def test_validate_fails_on_bad_tolerance(self, tmp_path):
        torch.manual_seed(42)
        network = PolicyNetwork()
        exporter = OnnxExporter()
        onnx_path = tmp_path / "model.onnx"
        exporter.export(network, onnx_path)

        # Create a different network to force mismatched outputs
        torch.manual_seed(99)
        different_network = PolicyNetwork()

        with pytest.raises(ValueError, match="Numeric mismatch"):
            exporter.validate(onnx_path, different_network, tolerance=0)

    def test_export_from_checkpoint(self, tmp_path):
        torch.manual_seed(42)
        network = PolicyNetwork()
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(network.state_dict(), str(checkpoint_path))

        exporter = OnnxExporter()
        onnx_path = tmp_path / "from_ckpt.onnx"
        result = exporter.export_from_checkpoint(checkpoint_path, onnx_path)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_validate_shape_check(self, tmp_path):
        import onnx

        torch.manual_seed(42)
        network = PolicyNetwork()
        exporter = OnnxExporter()
        onnx_path = tmp_path / "model.onnx"
        exporter.export(network, onnx_path)

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)


# ── TestOnnxPolicy ──────────────────────────────────────────

@pytest.mark.skipif(not has_onnxruntime(), reason="onnxruntime not installed")
class TestOnnxPolicy:
    def test_implements_policy_abc(self):
        assert issubclass(OnnxPolicy, Policy)

    def test_act_returns_correct_shape(self, tmp_path):
        onnx_path = _export_onnx(tmp_path)
        policy = OnnxPolicy(onnx_path=onnx_path)
        sim = _make_mock_sim()
        policy.reset(sim)
        action = policy.act(sim)
        assert action.shape == (ACT_DIM,)

    def test_reset_clears_state(self, tmp_path):
        onnx_path = _export_onnx(tmp_path)
        policy = OnnxPolicy(onnx_path=onnx_path, max_steps=5)
        sim = _make_mock_sim()
        policy.reset(sim)
        for _ in range(10):
            policy.act(sim)
        assert policy.done
        policy.reset(sim)
        assert not policy.done

    def test_done_after_max_steps(self, tmp_path):
        onnx_path = _export_onnx(tmp_path)
        policy = OnnxPolicy(onnx_path=onnx_path, max_steps=3)
        sim = _make_mock_sim()
        policy.reset(sim)
        for _ in range(5):
            policy.act(sim)
        assert policy.done

    def test_name_property(self, tmp_path):
        onnx_path = _export_onnx(tmp_path)
        policy = OnnxPolicy(onnx_path=onnx_path)
        assert policy.name == "OnnxPolicy"

    def test_benchmark(self, tmp_path):
        onnx_path = _export_onnx(tmp_path)
        policy = OnnxPolicy(onnx_path=onnx_path)
        result = policy.benchmark(n_iterations=10)
        assert "mean_ms" in result
        assert "std_ms" in result
        assert "min_ms" in result
        assert "max_ms" in result


# ── TestTensorRTPolicy ──────────────────────────────────────

@pytest.mark.skipif(has_tensorrt(), reason="TensorRT is installed — cannot test unavailable path")
class TestTensorRTPolicy:
    def test_available_returns_false(self):
        assert TensorRTPolicy.available() is False

    def test_init_raises_without_tensorrt(self, tmp_path):
        dummy_engine = tmp_path / "model.trt"
        dummy_engine.write_bytes(b"dummy")
        with pytest.raises(RuntimeError, match="TensorRT not available"):
            TensorRTPolicy(engine_path=dummy_engine)


# ── TestTensorRTOptimizer ───────────────────────────────────

@pytest.mark.skipif(has_tensorrt(), reason="TensorRT is installed — cannot test unavailable path")
class TestTensorRTOptimizer:
    def test_available_returns_false(self):
        assert TensorRTOptimizer.available() is False

    def test_optimize_raises_without_tensorrt(self, tmp_path):
        optimizer = TensorRTOptimizer()
        dummy_onnx = tmp_path / "model.onnx"
        dummy_onnx.write_bytes(b"dummy")
        with pytest.raises(RuntimeError, match="TensorRT not available"):
            optimizer.optimize(dummy_onnx, tmp_path / "model.trt")


# ── TestEdgePolicy ──────────────────────────────────────────

@pytest.mark.skipif(not has_onnxruntime(), reason="onnxruntime not installed")
class TestEdgePolicy:
    def test_selects_onnx_when_no_tensorrt(self, tmp_path):
        onnx_path = _export_onnx(tmp_path)
        # EdgePolicy expects a directory with .onnx files
        policy = EdgePolicy(model_dir=tmp_path)
        assert policy.runtime == "onnx"

    def test_raises_on_empty_dir(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No .trt, .engine, or .onnx"):
            EdgePolicy(model_dir=empty_dir)

    def test_name_property(self, tmp_path):
        _export_onnx(tmp_path)
        policy = EdgePolicy(model_dir=tmp_path)
        assert "EdgePolicy" in policy.name
        assert "OnnxPolicy" in policy.name

    def test_delegates_act(self, tmp_path):
        _export_onnx(tmp_path)
        policy = EdgePolicy(model_dir=tmp_path)
        sim = _make_mock_sim()
        policy.reset(sim)
        action = policy.act(sim)
        assert action.shape == (ACT_DIM,)


# ── TestDeploymentManifest ──────────────────────────────────

class TestDeploymentManifest:
    def test_to_dict(self):
        manifest = DeploymentManifest(
            model_name="test_model",
            model_format="onnx",
            opset_version=17,
            input_shape=[1, OBS_DIM],
            output_shape=[1, ACT_DIM],
            created_at="2024-01-01T00:00:00Z",
        )
        d = manifest.to_dict()
        assert d["model_name"] == "test_model"
        assert d["model_format"] == "onnx"
        assert d["opset_version"] == 17
        assert d["input_shape"] == [1, OBS_DIM]
        assert d["output_shape"] == [1, ACT_DIM]

    def test_from_dict_roundtrip(self):
        manifest = DeploymentManifest(
            model_name="test_model",
            model_format="onnx",
            opset_version=17,
            input_shape=[1, OBS_DIM],
            output_shape=[1, ACT_DIM],
            source_experiment="exp_001",
            source_checkpoint="models/v1.pt",
            created_at="2024-01-01T00:00:00Z",
            metadata={"accuracy": 0.95},
        )
        d = manifest.to_dict()
        restored = DeploymentManifest.from_dict(d)
        assert restored.model_name == manifest.model_name
        assert restored.model_format == manifest.model_format
        assert restored.opset_version == manifest.opset_version
        assert restored.input_shape == manifest.input_shape
        assert restored.output_shape == manifest.output_shape
        assert restored.source_experiment == manifest.source_experiment
        assert restored.source_checkpoint == manifest.source_checkpoint
        assert restored.created_at == manifest.created_at
        assert restored.metadata == manifest.metadata


# ── TestDeploymentPackager ──────────────────────────────────

@pytest.mark.skipif(not has_onnxruntime(), reason="onnxruntime not installed")
class TestDeploymentPackager:
    def test_package_creates_directory(self, tmp_path):
        onnx_path = _export_onnx(tmp_path)
        output_dir = tmp_path / "deploy_pkg"
        packager = DeploymentPackager()
        result = packager.package(onnx_path, output_dir)
        assert result.exists()
        # Check .onnx file copied
        onnx_files = list(output_dir.glob("*.onnx"))
        assert len(onnx_files) >= 1
        # Check manifest
        assert (output_dir / "manifest.yaml").exists()

    def test_manifest_contents(self, tmp_path):
        import yaml

        onnx_path = _export_onnx(tmp_path)
        output_dir = tmp_path / "deploy_pkg"
        packager = DeploymentPackager()
        packager.package(onnx_path, output_dir, model_name="test_model")

        with open(output_dir / "manifest.yaml") as f:
            manifest_data = yaml.safe_load(f)
        assert manifest_data["model_name"] == "test_model"
        assert manifest_data["model_format"] == "onnx"
        assert "opset_version" in manifest_data
        assert "input_shape" in manifest_data
        assert "output_shape" in manifest_data
        assert "created_at" in manifest_data

    def test_package_without_tensorrt(self, tmp_path):
        onnx_path = _export_onnx(tmp_path)
        output_dir = tmp_path / "deploy_pkg"
        packager = DeploymentPackager()
        result = packager.package(onnx_path, output_dir, include_tensorrt=True)
        # TensorRT not available, package should still succeed (logs warning)
        assert result.exists()
        # No .trt file should be created
        trt_files = list(output_dir.glob("*.trt"))
        assert len(trt_files) == 0


# ── TestExportFromRegistry ──────────────────────────────────

@pytest.mark.skipif(not has_onnxruntime(), reason="onnxruntime not installed")
class TestExportFromRegistry:
    def test_export_from_registry(self, tmp_path):
        from forge_core.model_registry import ModelRegistry

        # Create a real checkpoint
        torch.manual_seed(42)
        network = PolicyNetwork()
        checkpoint_path = tmp_path / "model.pt"
        torch.save(network.state_dict(), str(checkpoint_path))

        # Register in a temporary registry
        registry = ModelRegistry(registry_dir=tmp_path / "registry")
        registry.register_model("test_model", checkpoint_path=str(checkpoint_path))

        # Export from registry
        onnx_path = tmp_path / "exported.onnx"
        result = export_from_registry(
            model_name="test_model",
            output_path=onnx_path,
            registry=registry,
        )
        assert result.exists()
        assert result.stat().st_size > 0


# ── TestBenchmarkInference ──────────────────────────────────

@pytest.mark.skipif(not has_onnxruntime(), reason="onnxruntime not installed")
class TestBenchmarkInference:
    def test_benchmark_onnx_policy(self, tmp_path):
        onnx_path = _export_onnx(tmp_path)
        policy = OnnxPolicy(onnx_path=onnx_path)
        result = benchmark_inference(policy, n_iterations=10)
        assert "mean_ms" in result
        assert "std_ms" in result
        assert "min_ms" in result
        assert "max_ms" in result
        assert result["n_iterations"] == 10


# ── TestCreatePolicy integration ────────────────────────────

@pytest.mark.skipif(not has_onnxruntime(), reason="onnxruntime not installed")
class TestCreatePolicyIntegration:
    def test_create_onnx_policy(self, tmp_path):
        onnx_path = _export_onnx(tmp_path)
        policy = create_policy("onnx", onnx_path=str(onnx_path))
        assert isinstance(policy, OnnxPolicy)

    def test_create_edge_policy(self, tmp_path):
        _export_onnx(tmp_path)
        policy = create_policy("edge", model_dir=str(tmp_path))
        assert isinstance(policy, EdgePolicy)

    def test_create_onnx_policy_missing_path_raises(self):
        with pytest.raises(ValueError, match="onnx_path is required"):
            create_policy("onnx")

    def test_create_edge_policy_missing_dir_raises(self):
        with pytest.raises(ValueError, match="model_dir is required"):
            create_policy("edge")
