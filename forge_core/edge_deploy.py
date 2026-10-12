"""Edge deployment module for Forge.

Provides ONNX export, optional TensorRT optimization, edge inference
runtime policies (OnnxPolicy / TensorRTPolicy / EdgePolicy), deployment
packaging, and benchmarking utilities.

All onnx, onnxruntime, and tensorrt imports are lazy so that importing
this module never fails even when those packages are absent.  TensorRT
is runtime-detected only — it is never a pip dependency.

Phase 9 — FEAT-001.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from forge_core.config import PROJECT_ROOT, load_yaml
from forge_core.policy import Policy

logger = logging.getLogger(__name__)


# ── Runtime detection helpers ───────────────────────────────

def has_onnxruntime() -> bool:
    """Return True if onnxruntime is importable."""
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def has_tensorrt() -> bool:
    """Return True if tensorrt is importable."""
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


# ── Configuration dataclasses ──────────────────────────────

@dataclass
class ExportConfig:
    """Configuration for ONNX export."""
    opset_version: int = 17
    dynamic_batch: bool = False
    input_names: list[str] = field(default_factory=lambda: ["observation"])
    output_names: list[str] = field(default_factory=lambda: ["action"])


@dataclass
class TensorRTConfig:
    """Configuration for TensorRT optimization."""
    precision: str = "fp32"
    max_batch_size: int = 1
    workspace_size_mb: int = 256

    def __post_init__(self) -> None:
        if self.precision not in ("fp16", "fp32"):
            raise ValueError(
                f"precision must be 'fp16' or 'fp32', got {self.precision!r}"
            )


# ── Constants (shared with learned_policy) ──────────────────

OBS_DIM = 31
ACT_DIM = 8


# ── ONNX Exporter ──────────────────────────────────────────

class OnnxExporter:
    """Export a PolicyNetwork to ONNX and validate numeric equivalence."""

    def __init__(self, config: ExportConfig | None = None) -> None:
        self.config = config or ExportConfig()

    def export(
        self,
        network: Any,
        output_path: str | Path,
        obs_dim: int = OBS_DIM,
        act_dim: int = ACT_DIM,
    ) -> Path:
        """Export a PolicyNetwork to ONNX format.

        Creates a dummy input, runs torch.onnx.export, and returns the
        output path.
        """
        import torch

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        network.eval()
        dummy_input = torch.randn(1, obs_dim)

        dynamic_axes = None
        if self.config.dynamic_batch:
            dynamic_axes = {
                name: {0: "batch"} for name in
                self.config.input_names + self.config.output_names
            }

        torch.onnx.export(
            network,
            dummy_input,
            str(output_path),
            opset_version=self.config.opset_version,
            input_names=self.config.input_names,
            output_names=self.config.output_names,
            dynamic_axes=dynamic_axes,
        )
        logger.info("onnx_exported: %s (opset=%d)", output_path, self.config.opset_version)
        return output_path

    def validate(
        self,
        onnx_path: str | Path,
        network: Any,
        obs_dim: int = OBS_DIM,
        tolerance: float = 1e-5,
        n_samples: int = 10,
    ) -> bool:
        """Validate ONNX model against the PyTorch network.

        Checks structural validity with ``onnx.checker`` then compares
        n_samples random inputs through both runtimes.  Raises ValueError
        on mismatch.
        """
        import onnx
        import onnxruntime as ort
        import torch

        onnx_path = Path(onnx_path)

        # Structural validation
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

        # Numeric validation
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name

        network.eval()
        for i in range(n_samples):
            obs = torch.randn(1, obs_dim)

            with torch.no_grad():
                pt_out = network(obs).numpy()

            ort_out = session.run(None, {input_name: obs.numpy().astype(np.float32)})[0]

            if not np.allclose(pt_out, ort_out, atol=tolerance):
                raise ValueError(
                    f"Numeric mismatch at sample {i}: "
                    f"max_diff={np.max(np.abs(pt_out - ort_out)):.8f}, "
                    f"tolerance={tolerance}"
                )

        logger.info("onnx_validated: %s (%d samples, tol=%.1e)", onnx_path, n_samples, tolerance)
        return True

    def export_from_checkpoint(
        self,
        checkpoint_path: str | Path,
        output_path: str | Path,
        obs_dim: int = OBS_DIM,
        act_dim: int = ACT_DIM,
        hidden: list[int] | None = None,
    ) -> Path:
        """Load a .pt checkpoint and export to ONNX.

        Instantiates a fresh PolicyNetwork, loads the state_dict,
        and calls :meth:`export`.
        """
        import torch
        from forge_core.learned_policy import PolicyNetwork

        hidden = hidden or [256, 256, 128]
        network = PolicyNetwork(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)
        state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        network.load_state_dict(state_dict)
        return self.export(network, output_path, obs_dim=obs_dim, act_dim=act_dim)


# ── TensorRT Optimizer ──────────────────────────────────────

class TensorRTOptimizer:
    """Optimize an ONNX model to a TensorRT engine."""

    def __init__(self, config: TensorRTConfig | None = None) -> None:
        self.config = config or TensorRTConfig()

    @staticmethod
    def available() -> bool:
        """Return True if TensorRT is available."""
        return has_tensorrt()

    def optimize(
        self,
        onnx_path: str | Path,
        output_path: str | Path,
        obs_dim: int = OBS_DIM,
    ) -> Path:
        """Build a TensorRT engine from an ONNX model.

        Raises RuntimeError if TensorRT is not available.
        """
        if not has_tensorrt():
            raise RuntimeError("TensorRT not available")

        import tensorrt as trt

        onnx_path = Path(onnx_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, trt_logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = [parser.get_error(i) for i in range(parser.num_errors)]
                raise RuntimeError(f"ONNX parse failed: {errors}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, self.config.workspace_size_mb * (1 << 20)
        )

        if self.config.precision == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                logger.warning("tensorrt_fp16_not_supported: falling back to fp32")

        profile = builder.create_optimization_profile()
        profile.set_shape(
            "observation",
            min=(1, obs_dim),
            opt=(1, obs_dim),
            max=(self.config.max_batch_size, obs_dim),
        )
        config.add_optimization_profile(profile)

        engine = builder.build_serialized_network(network, config)
        if engine is None:
            raise RuntimeError("TensorRT engine build failed")

        with open(output_path, "wb") as f:
            f.write(engine)

        logger.info("tensorrt_optimized: %s -> %s (precision=%s)",
                     onnx_path, output_path, self.config.precision)
        return output_path


# ── ONNX Policy ─────────────────────────────────────────────

class OnnxPolicy(Policy):
    """Policy that runs inference via ONNX Runtime.

    Implements the Policy ABC so it can be used as a drop-in replacement
    for LearnedPolicy.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        target_object: str = "red_cube",
        max_steps: int = 1500,
        **kwargs: Any,
    ) -> None:
        import onnxruntime as ort

        self._onnx_path = Path(onnx_path)
        self._target_object = target_object
        self._max_steps = max_steps
        self._steps = 0
        self._is_done = False

        self._session = ort.InferenceSession(str(self._onnx_path))
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        logger.info("onnx_policy_loaded: %s", self._onnx_path)

    def reset(self, sim: Any) -> None:
        """Reset policy state for a new episode."""
        self._steps = 0
        self._is_done = False

    def act(self, sim: Any) -> np.ndarray:
        """Run ONNX inference on the current observation."""
        from forge_core.learned_policy import extract_observation

        self._steps += 1
        if self._steps >= self._max_steps:
            self._is_done = True
            return np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0])

        obs = extract_observation(sim, self._target_object)
        obs_input = obs.reshape(1, -1).astype(np.float32)
        result = self._session.run(None, {self._input_name: obs_input})
        return result[0].squeeze(0)

    @property
    def done(self) -> bool:
        return self._is_done

    @property
    def name(self) -> str:
        return "OnnxPolicy"

    def benchmark(
        self,
        n_iterations: int = 100,
        obs_dim: int = OBS_DIM,
    ) -> dict[str, float]:
        """Benchmark inference latency with random inputs."""
        times: list[float] = []
        for _ in range(n_iterations):
            obs = np.random.randn(1, obs_dim).astype(np.float32)
            t0 = time.perf_counter()
            self._session.run(None, {self._input_name: obs})
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "n_iterations": n_iterations,
        }


# ── TensorRT Policy ─────────────────────────────────────────

class TensorRTPolicy(Policy):
    """Policy that runs inference via TensorRT.

    If TensorRT is not available the class still exists but ``__init__``
    raises ``RuntimeError``.
    """

    def __init__(
        self,
        engine_path: str | Path,
        target_object: str = "red_cube",
        max_steps: int = 1500,
        **kwargs: Any,
    ) -> None:
        if not has_tensorrt():
            raise RuntimeError("TensorRT not available")

        import tensorrt as trt

        self._engine_path = Path(engine_path)
        self._target_object = target_object
        self._max_steps = max_steps
        self._steps = 0
        self._is_done = False

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(self._engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()

        # Allocate CUDA buffers
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "pycuda is required for TensorRTPolicy CUDA buffer allocation"
            )

        self._inputs: list[Any] = []
        self._outputs: list[Any] = []
        self._bindings: list[int] = []
        self._stream = cuda.Stream()

        for i in range(self._engine.num_io_tensors):
            tensor_name = self._engine.get_tensor_name(i)
            shape = self._engine.get_tensor_shape(tensor_name)
            size = abs(int(np.prod(shape)))
            dtype = np.float32
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self._bindings.append(int(device_mem))
            mode = self._engine.get_tensor_mode(tensor_name)
            if mode == trt.TensorIOMode.INPUT:
                self._inputs.append({"host": host_mem, "device": device_mem, "name": tensor_name})
            else:
                self._outputs.append({"host": host_mem, "device": device_mem, "name": tensor_name})

        logger.info("tensorrt_policy_loaded: %s", self._engine_path)

    @staticmethod
    def available() -> bool:
        """Return True if TensorRT is available."""
        return has_tensorrt()

    def reset(self, sim: Any) -> None:
        """Reset policy state for a new episode."""
        self._steps = 0
        self._is_done = False

    def act(self, sim: Any) -> np.ndarray:
        """Run TensorRT inference on the current observation."""
        from forge_core.learned_policy import extract_observation

        self._steps += 1
        if self._steps >= self._max_steps:
            self._is_done = True
            return np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0])

        import pycuda.driver as cuda

        obs = extract_observation(sim, self._target_object)
        np.copyto(self._inputs[0]["host"], obs.ravel().astype(np.float32))
        cuda.memcpy_htod_async(self._inputs[0]["device"], self._inputs[0]["host"], self._stream)

        for inp in self._inputs:
            self._context.set_tensor_address(inp["name"], int(inp["device"]))
        for out in self._outputs:
            self._context.set_tensor_address(out["name"], int(out["device"]))
        self._context.execute_async_v3(stream_handle=self._stream.handle)

        cuda.memcpy_dtoh_async(self._outputs[0]["host"], self._outputs[0]["device"], self._stream)
        self._stream.synchronize()

        return self._outputs[0]["host"].copy()

    @property
    def done(self) -> bool:
        return self._is_done

    @property
    def name(self) -> str:
        return "TensorRTPolicy"

    def benchmark(
        self,
        n_iterations: int = 100,
        obs_dim: int = OBS_DIM,
    ) -> dict[str, float]:
        """Benchmark inference latency with random inputs."""
        import pycuda.driver as cuda

        times: list[float] = []
        for _ in range(n_iterations):
            obs = np.random.randn(obs_dim).astype(np.float32)
            np.copyto(self._inputs[0]["host"], obs)

            t0 = time.perf_counter()
            cuda.memcpy_htod_async(
                self._inputs[0]["device"], self._inputs[0]["host"], self._stream
            )
            for inp in self._inputs:
                self._context.set_tensor_address(inp["name"], int(inp["device"]))
            for out in self._outputs:
                self._context.set_tensor_address(out["name"], int(out["device"]))
            self._context.execute_async_v3(stream_handle=self._stream.handle)
            cuda.memcpy_dtoh_async(
                self._outputs[0]["host"], self._outputs[0]["device"], self._stream
            )
            self._stream.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "n_iterations": n_iterations,
        }


# ── Edge Policy (auto-select runtime) ──────────────────────

class EdgePolicy(Policy):
    """Auto-selecting edge inference policy.

    Looks in *model_dir* for a TensorRT engine or ONNX model and
    delegates to the appropriate runtime policy.
    """

    def __init__(
        self,
        model_dir: str | Path,
        target_object: str = "red_cube",
        max_steps: int = 1500,
        **kwargs: Any,
    ) -> None:
        model_dir = Path(model_dir)

        # Try TensorRT first
        trt_files = list(model_dir.glob("*.trt")) + list(model_dir.glob("*.engine"))
        if trt_files and has_tensorrt():
            self._inner = TensorRTPolicy(
                engine_path=trt_files[0],
                target_object=target_object,
                max_steps=max_steps,
                **kwargs,
            )
            self._runtime = "tensorrt"
            logger.info("edge_policy_selected: TensorRT (%s)", trt_files[0])
            return

        # Fall back to ONNX
        onnx_files = list(model_dir.glob("*.onnx"))
        if onnx_files:
            self._inner = OnnxPolicy(
                onnx_path=onnx_files[0],
                target_object=target_object,
                max_steps=max_steps,
                **kwargs,
            )
            self._runtime = "onnx"
            logger.info("edge_policy_selected: ONNX (%s)", onnx_files[0])
            return

        raise FileNotFoundError(
            f"No .trt, .engine, or .onnx model found in {model_dir}"
        )

    def reset(self, sim: Any) -> None:
        self._inner.reset(sim)

    def act(self, sim: Any) -> np.ndarray:
        return self._inner.act(sim)

    @property
    def done(self) -> bool:
        return self._inner.done

    @property
    def name(self) -> str:
        return f"EdgePolicy({self._inner.name})"

    @property
    def runtime(self) -> str:
        return self._runtime


# ── Deployment Manifest ─────────────────────────────────────

@dataclass
class DeploymentManifest:
    """Metadata for a packaged deployment artifact."""
    model_name: str
    model_format: str  # "onnx" or "tensorrt"
    opset_version: int
    input_shape: list[int]
    output_shape: list[int]
    source_experiment: str | None = None
    source_checkpoint: str | None = None
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_format": self.model_format,
            "opset_version": self.opset_version,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "source_experiment": self.source_experiment,
            "source_checkpoint": self.source_checkpoint,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeploymentManifest:
        return cls(
            model_name=data["model_name"],
            model_format=data["model_format"],
            opset_version=data["opset_version"],
            input_shape=data["input_shape"],
            output_shape=data["output_shape"],
            source_experiment=data.get("source_experiment"),
            source_checkpoint=data.get("source_checkpoint"),
            created_at=data.get("created_at", ""),
            metadata=data.get("metadata", {}),
        )


# ── Deployment Packager ─────────────────────────────────────

class DeploymentPackager:
    """Package ONNX models (and optional TensorRT engines) for deployment."""

    def package(
        self,
        onnx_path: str | Path,
        output_dir: str | Path,
        model_name: str | None = None,
        source_experiment: str | None = None,
        source_checkpoint: str | None = None,
        include_tensorrt: bool = False,
        tensorrt_config: TensorRTConfig | None = None,
    ) -> Path:
        """Create a deployment package directory.

        Copies the ONNX file, writes a manifest.yaml, and optionally
        builds a TensorRT engine.
        """
        import yaml
        from datetime import datetime, timezone

        onnx_path = Path(onnx_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy ONNX model
        dest_onnx = output_dir / onnx_path.name
        shutil.copy2(str(onnx_path), str(dest_onnx))

        if model_name is None:
            model_name = onnx_path.stem

        # Determine shapes from ONNX model
        try:
            import onnx
            model = onnx.load(str(onnx_path))
            inp = model.graph.input[0]
            out = model.graph.output[0]
            input_shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            output_shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            opset = model.opset_import[0].version
        except Exception:
            input_shape = [1, OBS_DIM]
            output_shape = [1, ACT_DIM]
            opset = 17

        model_format = "onnx"

        # Optional TensorRT optimization
        if include_tensorrt:
            if has_tensorrt():
                trt_config = tensorrt_config or TensorRTConfig()
                optimizer = TensorRTOptimizer(config=trt_config)
                trt_output = output_dir / f"{model_name}.trt"
                optimizer.optimize(dest_onnx, trt_output)
                model_format = "tensorrt"
            else:
                logger.warning(
                    "tensorrt_not_available: skipping TensorRT optimization"
                )

        manifest = DeploymentManifest(
            model_name=model_name,
            model_format=model_format,
            opset_version=opset,
            input_shape=input_shape,
            output_shape=output_shape,
            source_experiment=source_experiment,
            source_checkpoint=source_checkpoint,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        manifest_path = output_dir / "manifest.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(manifest.to_dict(), f, default_flow_style=False)

        logger.info("deployment_packaged: %s -> %s", model_name, output_dir)
        return output_dir


# ── Benchmark helper ────────────────────────────────────────

def benchmark_inference(
    policy: Policy,
    sim: Any | None = None,
    n_iterations: int = 100,
) -> dict[str, float]:
    """Benchmark a policy's inference latency.

    If the policy has a ``.benchmark()`` method (OnnxPolicy / TensorRTPolicy),
    delegates to it.  Otherwise times ``act()`` calls with a real sim.
    """
    if hasattr(policy, "benchmark") and callable(policy.benchmark):
        return policy.benchmark(n_iterations=n_iterations)

    if sim is None:
        raise ValueError(
            "sim is required for benchmarking policies without a benchmark() method"
        )

    times: list[float] = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        policy.act(sim)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "n_iterations": n_iterations,
    }


# ── Registry integration ───────────────────────────────────

def export_from_registry(
    model_name: str,
    version: int | None = None,
    output_path: str | Path | None = None,
    registry: Any | None = None,
    **export_kwargs: Any,
) -> Path:
    """Export a registered model to ONNX.

    Loads the model entry from the registry, exports from its checkpoint,
    and returns the output path.
    """
    from forge_core.model_registry import ModelRegistry

    if registry is None:
        registry = ModelRegistry()

    entry = registry.get_model(model_name, version=version)
    if entry is None:
        raise ValueError(f"Model not found in registry: {model_name} v{version}")

    checkpoint_path = entry["checkpoint_path"]
    v = entry["version"]

    if output_path is None:
        models_dir = PROJECT_ROOT / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        output_path = models_dir / f"{model_name}_v{v}.onnx"

    exporter = OnnxExporter(**export_kwargs)
    return exporter.export_from_checkpoint(checkpoint_path, output_path)


# ── Deploy config loader ───────────────────────────────────

def load_deploy_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load deployment configuration from a YAML file.

    When *path* is given, loads that file (raises FileNotFoundError if
    missing).  When *path* is None, tries the default
    ``config/deploy.yaml``; returns ``{}`` if the default file is missing.
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Deploy config not found: {p}")
        return load_yaml(p)

    default_path = PROJECT_ROOT / "config" / "deploy.yaml"
    if not default_path.exists():
        logger.debug("no default deploy config found")
        return {}
    return load_yaml(default_path)
