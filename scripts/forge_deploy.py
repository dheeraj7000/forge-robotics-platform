#!/usr/bin/env python3
"""Forge edge deployment CLI — export, validate, benchmark, and package models.

Usage:
    # Export a PyTorch model to ONNX
    python3 scripts/forge_deploy.py --export --model models/bc_policy.pt --output models/bc_policy.onnx

    # Validate ONNX vs PyTorch numeric equivalence
    python3 scripts/forge_deploy.py --validate --model models/bc_policy.onnx --checkpoint models/bc_policy.pt

    # Benchmark ONNX inference latency
    python3 scripts/forge_deploy.py --benchmark --model models/bc_policy.onnx

    # Package model for deployment
    python3 scripts/forge_deploy.py --package --model models/bc_policy.onnx --output deploy/bc_policy/

    # Optimize with TensorRT (requires TensorRT)
    python3 scripts/forge_deploy.py --optimize --model models/bc_policy.onnx --output models/bc_policy.trt

    # Export a model from the registry
    python3 scripts/forge_deploy.py --export-registry --name bc_pick --output models/bc_pick.onnx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from forge_core.edge_deploy import (
    OnnxExporter,
    OnnxPolicy,
    TensorRTOptimizer,
    DeploymentPackager,
    ExportConfig,
    TensorRTConfig,
    export_from_registry,
    has_tensorrt,
)


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_export(args):
    """Export a PyTorch model to ONNX."""
    config = ExportConfig(
        opset_version=args.opset,
        dynamic_batch=args.dynamic_batch,
    )
    exporter = OnnxExporter(config=config)
    output = exporter.export_from_checkpoint(args.model, args.output)
    print(f"Exported: {output}")


def cmd_validate(args):
    """Validate ONNX model against a PyTorch checkpoint."""
    import torch
    from forge_core.learned_policy import PolicyNetwork

    exporter = OnnxExporter()
    network = PolicyNetwork()
    state_dict = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
    network.load_state_dict(state_dict)

    try:
        result = exporter.validate(
            args.model,
            network,
            tolerance=args.tolerance,
            n_samples=args.n_samples,
        )
        if result:
            print(f"Validation PASSED ({args.n_samples} samples, tol={args.tolerance})")
    except ValueError as exc:
        print(f"Validation FAILED: {exc}")
        sys.exit(1)


def cmd_benchmark(args):
    """Benchmark ONNX or TensorRT inference latency."""
    model_path = Path(args.model)
    if model_path.suffix in (".trt", ".engine") and has_tensorrt():
        from forge_core.edge_deploy import TensorRTPolicy
        policy = TensorRTPolicy(engine_path=model_path)
    else:
        policy = OnnxPolicy(onnx_path=model_path)

    result = policy.benchmark(n_iterations=args.n_iterations)
    print(f"Benchmark results ({result['n_iterations']} iterations):")
    print(f"  Mean:  {result['mean_ms']:.3f} ms")
    print(f"  Std:   {result['std_ms']:.3f} ms")
    print(f"  Min:   {result['min_ms']:.3f} ms")
    print(f"  Max:   {result['max_ms']:.3f} ms")


def cmd_package(args):
    """Package a model for deployment."""
    packager = DeploymentPackager()
    output_dir = packager.package(
        onnx_path=args.model,
        output_dir=args.output,
        model_name=args.name,
        include_tensorrt=args.include_tensorrt,
    )
    print(f"Packaged: {output_dir}")


def cmd_optimize(args):
    """Optimize an ONNX model with TensorRT."""
    if not has_tensorrt():
        print("Warning: TensorRT is not available. Cannot optimize.")
        sys.exit(1)

    config = TensorRTConfig(precision=args.precision)
    optimizer = TensorRTOptimizer(config=config)
    output = optimizer.optimize(args.model, args.output)
    print(f"Optimized: {output}")


def cmd_export_registry(args):
    """Export a model from the registry to ONNX."""
    output_path = export_from_registry(
        model_name=args.name,
        version=args.version,
        output_path=args.output,
    )
    print(f"Exported from registry: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="forge deploy",
        description="Forge edge deployment CLI — export, validate, benchmark, and package models",
    )

    # Action flags (mutually exclusive)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--export", action="store_true",
                        help="Export a PyTorch .pt model to ONNX")
    action.add_argument("--validate", action="store_true",
                        help="Validate ONNX model against PyTorch checkpoint")
    action.add_argument("--benchmark", action="store_true",
                        help="Benchmark inference latency")
    action.add_argument("--package", action="store_true",
                        help="Package model for deployment")
    action.add_argument("--optimize", action="store_true",
                        help="Optimize ONNX model with TensorRT")
    action.add_argument("--export-registry", action="store_true",
                        help="Export a model from the registry to ONNX")

    # Common arguments
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model file (.pt, .onnx, .trt)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (file or directory)")
    parser.add_argument("--verbose", "-v", action="store_true")

    # Export-specific
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    parser.add_argument("--dynamic-batch", action="store_true",
                        help="Enable dynamic batch size in ONNX export")

    # Validate-specific
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to PyTorch checkpoint for validation")
    parser.add_argument("--tolerance", type=float, default=1e-5,
                        help="Numeric tolerance for validation (default: 1e-5)")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of random samples for validation (default: 10)")

    # Benchmark-specific
    parser.add_argument("--n-iterations", type=int, default=100,
                        help="Number of benchmark iterations (default: 100)")

    # Package-specific
    parser.add_argument("--name", type=str, default=None,
                        help="Model name for packaging or registry lookup")
    parser.add_argument("--include-tensorrt", action="store_true",
                        help="Include TensorRT engine in deployment package")

    # Optimize-specific
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp16", "fp32"],
                        help="TensorRT precision (default: fp32)")

    # Registry-specific
    parser.add_argument("--version", type=int, default=None,
                        help="Model version for registry operations")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.export:
        if not args.model or not args.output:
            parser.error("--export requires --model and --output")
        cmd_export(args)
    elif args.validate:
        if not args.model or not args.checkpoint:
            parser.error("--validate requires --model and --checkpoint")
        cmd_validate(args)
    elif args.benchmark:
        if not args.model:
            parser.error("--benchmark requires --model")
        cmd_benchmark(args)
    elif args.package:
        if not args.model or not args.output:
            parser.error("--package requires --model and --output")
        cmd_package(args)
    elif args.optimize:
        if not args.model or not args.output:
            parser.error("--optimize requires --model and --output")
        cmd_optimize(args)
    elif args.export_registry:
        if not args.name:
            parser.error("--export-registry requires --name")
        cmd_export_registry(args)


if __name__ == "__main__":
    main()
