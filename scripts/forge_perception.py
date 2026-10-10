#!/usr/bin/env python3
"""Forge perception CLI - render scenes, detect objects, annotate images.

Usage:
    # Render a scene
    python3 scripts/forge_perception.py --scenario basic_workspace --render -o output.png

    # Detect objects
    python3 scripts/forge_perception.py --scenario basic_workspace --detect

    # Detect and annotate
    python3 scripts/forge_perception.py --scenario basic_workspace --detect --annotate -o annotated.png
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np

from forge_core.perception import PerceptionPipeline
from forge_core.simulation import ForgeSimulation


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        prog="forge perception",
        description="Forge perception CLI - render, detect, and annotate",
    )
    parser.add_argument("--scenario", type=str, required=True,
                        help="Simulation scenario name")
    parser.add_argument("--render", action="store_true",
                        help="Render the scene to an image")
    parser.add_argument("--detect", action="store_true",
                        help="Run object detection")
    parser.add_argument("--annotate", action="store_true",
                        help="Draw bounding boxes on detected objects (requires --detect)")
    parser.add_argument("-o", "--output", type=str, default="perception_output.png",
                        help="Output image path (default: perception_output.png)")
    parser.add_argument("--azimuth", type=float, default=None,
                        help="Camera azimuth angle")
    parser.add_argument("--elevation", type=float, default=None,
                        help="Camera elevation angle")
    parser.add_argument("--distance", type=float, default=None,
                        help="Camera distance from lookat point")
    parser.add_argument("--lookat", nargs=3, type=float, default=None,
                        help="Camera lookat point (x y z)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Simulation steps to settle physics (default: 100)")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Build camera kwargs from provided args (only include non-None)
    camera_kwargs: dict = {}
    if args.azimuth is not None:
        camera_kwargs["azimuth"] = args.azimuth
    if args.elevation is not None:
        camera_kwargs["elevation"] = args.elevation
    if args.distance is not None:
        camera_kwargs["distance"] = args.distance
    if args.lookat is not None:
        camera_kwargs["lookat"] = tuple(args.lookat)

    # Create simulation
    try:
        sim = ForgeSimulation(args.scenario)
    except Exception as e:
        print(f"Error: failed to create simulation for scenario '{args.scenario}': {e}",
              file=sys.stderr)
        sys.exit(1)

    # Settle physics
    sim.step(args.steps)

    # Create perception pipeline
    pipeline = PerceptionPipeline(sim, **camera_kwargs)

    output_path = args.output

    try:
        if args.render:
            img = pipeline.render()
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, bgr_img)
            print(f"Rendered scene saved to {output_path}")

        if args.detect:
            result = pipeline.perceive()
            detections = result.detections

            if not detections:
                logger.warning("No objects detected in scene")
            else:
                for det in detections:
                    print(f"Detected: {det.object_id} | "
                          f"pixel_center={det.pixel_center} | "
                          f"pixel_area={det.pixel_area} | "
                          f"confidence={det.confidence:.2%} | "
                          f"position={det.position}")

            if args.annotate:
                # Color map: red for red_cube (BGR), blue for blue_cube (BGR), gray for others
                color_map = {
                    "red_cube": (0, 0, 255),
                    "blue_cube": (255, 0, 0),
                }
                default_color = (128, 128, 128)

                # Convert rendered image to BGR for annotation
                img = result.image
                bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                for det in detections:
                    row, col = det.pixel_center
                    box_half = 30
                    top_left = (col - box_half, row - box_half)
                    bottom_right = (col + box_half, row + box_half)
                    color = color_map.get(det.object_id, default_color)
                    cv2.rectangle(bgr_img, top_left, bottom_right, color, 2)

                    label = f"{det.object_id} ({det.confidence:.0%})"
                    cv2.putText(bgr_img, label, (col - box_half, row - box_half - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                cv2.imwrite(output_path, bgr_img)
                print(f"Annotated image saved to {output_path}")
    finally:
        pipeline.close()
        sim.shutdown()


if __name__ == "__main__":
    main()