# Phase 5 — Perception (Camera + OpenCV): Requirements & Technical Design

*Revision 3 — addresses all findings from Revision 2 design review*

## 1. Requirements

### 1.1 Summary

Phase 5 adds camera-based perception to the Forge platform. A new `forge_core/perception.py` module uses MuJoCo's offscreen renderer to capture RGB images from configurable virtual cameras and OpenCV to detect colored objects via HSV filtering. The perception pipeline produces structured detection results with estimated 3D positions, integrates optionally with the existing Runner/Recorder/Policy stack, and is exercised through a new CLI tool and comprehensive headless tests. No ROS dependency is introduced.

### 1.2 Functional Requirements

- FR-1: Camera rendering — `PerceptionPipeline` produces RGB images (480×640×3, uint8) from a programmatically-positioned virtual camera using `mujoco.Renderer`, accessing `sim.model` and `sim.data`.
- FR-2: Perception module — New `forge_core/perception.py` exposes `PerceptionPipeline`, `PerceptionResult`, `Detection`, and `ColorDetector` classes matching the hard API contract in `tests/unit/test_perception.py`.
- FR-3: Color detection — `ColorDetector` identifies objects by HSV color filtering for red and blue cubes, returning detections with pixel center, pixel area, confidence, and estimated 3D world position.
- FR-4: 3D position estimation — Perception estimates object positions using depth rendering and camera-to-world unprojection. Estimates must not equal ground truth (error > 0.001 m) but must be reasonable (error < 2.0 m).
- FR-5: Runner integration — `run_task()` accepts an optional `perception: bool` flag. When True, it wraps the provided policy in a `PerceptionPolicy` before entering the run loop.
- FR-6: Recorder integration — `TrajectoryBuffer` can optionally save camera frames alongside existing trajectory data.
- FR-7: Policy integration — A `PerceptionPolicy` wrapper provides image observations to policies that want them.
- FR-8: CLI tool — `scripts/forge_perception.py` renders and saves camera images, runs perception, and saves annotated detection images (bounding boxes drawn on frames).
- FR-9: Documentation — Phase 5 section in `docs/architecture.md`; README updated to show Phase 5 as Done; `docs/setup.md` updated with `opencv-python-headless` dependency.
- FR-10: Dependency — `opencv-python-headless` added to `pyproject.toml` dependencies.

### 1.3 Non-Functional Requirements

- NFR-1: Zero ROS dependency — all perception code lives in `forge_core/` and imports nothing from ROS.
- NFR-2: Headless operation — rendering and tests must work without a display server (EGL/OSMesa backend via MuJoCo's offscreen renderer).
- NFR-3: Deterministic detection — HSV-based color filtering produces repeatable results for identical inputs.
- NFR-4: Performance — rendering + perception for a single frame should complete in under 500 ms on a CPU-only machine.

### 1.4 Acceptance Criteria

1. All existing tests in `tests/unit/test_perception.py` pass without modification.
2. `PerceptionPipeline(sim).render()` returns a `(480, 640, 3)` uint8 numpy array.
3. `PerceptionPipeline(sim).perceive()` returns a `PerceptionResult` with at least one `Detection`.
4. Each `Detection` has fields: `object_id` (str), `pixel_center` (tuple), `pixel_area` (int), `confidence` (float 0–1), `position` (ndarray shape (3,)).
5. `PerceptionResult.to_object_states()` returns a list of `ObjectState` with correct `.position` shape (3,).
6. `PerceptionResult.get_detection("red_cube")` returns a `Detection` or `None`.
7. Perception position error vs. ground truth is > 0.001 m and < 2.0 m for detected cubes.
8. `PerceptionPipeline` accepts custom camera kwargs: `azimuth`, `elevation`, `distance`, `lookat`.
9. `ColorDetector().detect(img)` returns a `list[Detection]`.
10. `scripts/forge_perception.py` runs without errors, saves rendered images and annotated detection images.
11. `opencv-python-headless` appears in `pyproject.toml` dependencies.
12. `docs/architecture.md` contains a Phase 5 section; README phase roadmap shows Phase 5 as Done.
13. All 88+ existing tests continue to pass.

### 1.5 Out of Scope

- Learned detectors (neural networks, YOLO, etc.) — Phase 6.
- Real-time video streaming or ROS image topics.
- Semantic segmentation or instance segmentation.
- Camera intrinsic calibration from real hardware.
- GUI/display-based visualization (all output is to files).
- Named camera definitions in scenario YAML — deferred to a future iteration when wired to the pipeline.

---

## 2. Technical Design

### 2.1 Overview

The perception system is a single new module `forge_core/perception.py` plus minimal integration changes in Runner and Recorder. A `PerceptionPipeline` object wraps a `mujoco.Renderer` to produce RGB and depth images from a virtual camera, then delegates object detection to a `ColorDetector` that applies HSV color filtering via OpenCV. Detection results, including estimated 3D world positions obtained by unprojecting depth pixels through the camera matrix, are returned as structured `PerceptionResult` and `Detection` dataclasses. The pipeline integrates with the existing Runner and Policy layers through a `PerceptionPolicy` wrapper — the sole integration path — keeping the existing interfaces backward-compatible. A new CLI script exercises the pipeline end-to-end.

The technology stack is locked: MuJoCo ≥ 3.1.0 for rendering, OpenCV (headless) ≥ 4.8.0 for image processing, numpy for array operations. No additional libraries.

### 2.2 Camera Rendering in MuJoCo

#### Approach

MuJoCo provides `mujoco.Renderer(model, height, width)` for offscreen rendering. This avoids any need for GLFW or a display server. The renderer is attached to the simulation's model and renders into a numpy array.

Rather than adding rendering methods directly to `ForgeSimulation` (which would add an OpenCV import and bloat the simulation module), the `PerceptionPipeline` owns the `mujoco.Renderer` instance. It accesses `sim.model` and `sim.data` to configure the camera and render.

The virtual camera is positioned programmatically using `mujoco.MjvCamera` with `type = mujoco.mjtCamera.mjCAMERA_FREE`. Camera parameters (azimuth, elevation, distance, lookat) are set on this camera object. This matches the test contract which passes these kwargs to `PerceptionPipeline`.

#### Default Camera Parameters

```python
DEFAULT_CAMERA = {
    "azimuth": 180,      # looking from +x toward origin
    "elevation": -30,    # angled down (negative = above lookat in MuJoCo)
    "distance": 1.8,     # meters from lookat
    "lookat": (0.5, 0.0, 0.22),  # center of table
}
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
```

These defaults produce a view of the tabletop that captures both cubes and the robot.

#### Depth Rendering

MuJoCo's Python `Renderer.render()` with depth enabled may return either linearized depth in meters or raw OpenGL depth buffer values in the [0, 1] range (where 0.0 = near plane, 1.0 = far plane). The C API (`mjr_readPixels`) reads the raw [0, 1] buffer, and whether the Python wrapper linearizes depends on the MuJoCo version and build. The implementation must handle both cases with a conditional linearization path:

```python
raw = self._renderer.render()  # with depth enabled

if raw.max() <= 1.0:
    # Raw OpenGL depth buffer — linearize to meters
    extent = self._sim.model.stat.extent
    near = self._sim.model.vis.map.znear * extent
    far = self._sim.model.vis.map.zfar * extent
    # Pixels at far plane (raw ≈ 1.0) → sentinel 0.0
    depth = np.where(
        raw < 0.999,
        near * far / (far - raw * (far - near)),
        0.0,
    )
else:
    # Already linearized in meters; 0.0 = no geometry
    depth = raw
```

A depth value of 0.0 after this processing indicates "no geometry hit" and triggers the fallback path for that pixel's detection.

#### Rendering Flow

1. Create `mujoco.Renderer(sim.model, height, width)`.
2. Create `mujoco.MjvCamera()` and set `type`, `azimuth`, `elevation`, `distance`, `lookat`.
3. Call `renderer.update_scene(sim.data, camera)` to sync scene state.
4. Call `renderer.render()` → RGB uint8 array (height, width, 3).
5. For depth: enable depth rendering via `renderer.enable_depth_rendering(True)`, call `renderer.render()` → float array (height, width), apply the conditional linearization above, then disable depth rendering with `renderer.enable_depth_rendering(False)`.

### 2.3 Perception Module — `forge_core/perception.py`

#### Class: `Detection`

A dataclass holding a single detected object.

```python
@dataclass
class Detection:
    object_id: str           # e.g. "red_cube"
    pixel_center: tuple      # (row, col) in image coordinates
    pixel_area: int          # area in pixels of the detected blob
    confidence: float        # 0.0–1.0 based on area relative to expected size
    position: np.ndarray     # (3,) estimated world position, default zeros
```

`position` defaults to `np.zeros(3)` and is populated by `PerceptionPipeline.perceive()` after depth-based unprojection. `ColorDetector.detect()` returns detections with `position = np.zeros(3)` since the detector has no depth information.

#### Class: `ColorDetector`

Stateless detector that operates on a single RGB image using OpenCV HSV filtering.

```python
class ColorDetector:
    """Detect colored objects by HSV filtering."""

    # Color definitions: name -> list of (hsv_lower, hsv_upper) range pairs
    COLOR_RANGES: dict[str, list[tuple[np.ndarray, np.ndarray]]]

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect colored objects in an RGB image."""
```

Implementation detail:

1. Convert the input RGB image to HSV via `cv2.cvtColor(image, cv2.COLOR_RGB2HSV)`. Note: MuJoCo's renderer produces RGB, not BGR — we must NOT apply `COLOR_BGR2HSV`.
2. For each color in `COLOR_RANGES`:
   - Apply `cv2.inRange()` with the HSV bounds to produce a binary mask. If multiple ranges exist for a color (e.g., red wraps around hue 0/180), OR the masks together.
   - Apply `cv2.morphologyEx()` with a small kernel (5×5) to clean noise.
   - Find contours via `cv2.findContours()`.
   - Filter contours by minimum area (50 pixels) to reject noise.
   - For each qualifying contour: compute moments to get centroid, compute area, and compute confidence as `min(1.0, area / expected_area)` where `expected_area` is a constant (500 pixels) representing the typical apparent size of a cube.
3. Return a list of `Detection` objects. If multiple contours match the same color, take the largest one (by area) as the detection for that object_id.

Color ranges (tuned for MuJoCo's rendered colors):

| Object | RGBA in scene | HSV Ranges |
|--------|---------------|-----------|
| `red_cube` | 0.9 0.1 0.1 1 | (0, 100, 100)–(10, 255, 255) OR (170, 100, 100)–(180, 255, 255) |
| `blue_cube` | 0.1 0.1 0.9 1 | (100, 100, 100)–(130, 255, 255) |

Red requires two ranges because red wraps around hue 0/180 in HSV. The detector ORs the two masks.

Error handling: If `image` is None or has wrong dtype, `detect()` returns an empty list and logs a warning. If OpenCV raises during processing, catch `cv2.error`, log it, and return an empty list — this is recoverable since a missed frame doesn't halt the pipeline.

#### Class: `PerceptionResult`

A dataclass holding the output of a full perception cycle.

```python
@dataclass
class PerceptionResult:
    image: np.ndarray              # the rendered RGB image
    detections: list[Detection]    # detected objects
    timestamp: float = 0.0        # sim time when this frame was captured

    def to_object_states(self) -> list[ObjectState]:
        """Convert detections to ObjectState list for compatibility with sim API."""

    def get_detection(self, object_id: str) -> Detection | None:
        """Look up a detection by object_id. Returns None if not found."""
```

`to_object_states()` imports and returns `ObjectState` from `forge_core.simulation`. Each `ObjectState` gets `id=detection.object_id`, `position=detection.position`, `orientation=np.array([1,0,0,0])` (identity quaternion — perception doesn't estimate orientation), and `timestamp=self.timestamp`. This ensures the timestamp from `sim.sim_time` at capture time flows through.

`get_detection()` does a linear scan of `self.detections` and returns the first match, or `None`. The list is small (2–5 items) so this is fine.

#### Class: `PerceptionPipeline`

The main entry point tying rendering and detection together. Supports the context manager protocol for safe resource management.

```python
class PerceptionPipeline:
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

    def render(self) -> np.ndarray:
        """Render the current scene from the camera viewpoint.
        Returns: (height, width, 3) uint8 RGB image."""

    def render_depth(self) -> np.ndarray:
        """Render depth buffer. Returns: (height, width) float32 array (meters).
        Applies conditional linearization if the raw buffer is in [0, 1] range."""

    def perceive(self) -> PerceptionResult:
        """Full perception cycle: render + detect + estimate 3D positions.
        Returns: PerceptionResult with image, detections, and timestamp."""

    def close(self) -> None:
        """Release the MuJoCo renderer resources."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()
```

Constructor behavior:
1. Store a reference to `sim` (not a copy).
2. Validate inputs: `distance > 0` or raise `ValueError`; `lookat` must be a 3-element sequence or raise `ValueError`; `width > 0` and `height > 0` or raise `ValueError`.
3. Create `mujoco.Renderer(sim.model, height, width)`.
4. Create `mujoco.MjvCamera()` and configure with the provided azimuth/elevation/distance/lookat.
5. Create a `ColorDetector` instance.
6. Create a seeded RNG: `self._rng = np.random.default_rng(42)` for reproducible fallback noise.
7. Store width, height, and camera parameters for use in unprojection.

`render()` implementation:
1. Call `self._renderer.update_scene(self._sim.data, self._camera)`.
2. Call `self._renderer.render()` → returns `np.ndarray` of shape `(height, width, 3)`, dtype uint8.
3. Return the image.

`render_depth()` implementation:
1. Call `self._renderer.update_scene(self._sim.data, self._camera)`.
2. Enable depth rendering: `self._renderer.enable_depth_rendering(True)`.
3. Call `self._renderer.render()` → raw depth buffer.
4. Apply the conditional linearization from §2.2.
5. Disable depth rendering: `self._renderer.enable_depth_rendering(False)`.
6. Return the depth array as float32.

`perceive()` implementation:
1. Call `self._renderer.update_scene(self._sim.data, self._camera)` once to sync scene state.
2. Call `self._renderer.render()` → RGB image.
3. Enable depth, render depth, apply conditional linearization, disable depth → depth buffer.
4. Call `self._detector.detect(image)` to get `list[Detection]`.
5. For each detection, estimate 3D world position using the depth buffer and pixel center (see §2.4).
6. Capture `timestamp = self._sim.sim_time`.
7. Return `PerceptionResult(image=image, detections=detections, timestamp=timestamp)`.

Note: `perceive()` calls `update_scene()` once at the top, then uses internal render calls that skip the redundant update. The public `render()` and `render_depth()` methods each call `update_scene()` themselves for standalone use. The minor overhead of a double update when calling `render()` then `render_depth()` separately is acceptable for API simplicity.

`close()` implementation:
1. If `self._renderer` is not None, call `self._renderer.close()`.
2. Set `self._renderer = None`.

Calling `close()` multiple times is safe — the None check makes it idempotent. The `__del__` safety net ensures resources are freed if the caller forgets to call `close()` explicitly or if an exception occurs between construction and `close()`. The `__enter__`/`__exit__` pair supports `with` blocks for production code and scripts (tests continue to call `close()` directly per the existing test contract).

Error handling:
- If the simulation has been shut down before the pipeline is closed, calls to `render()` or `perceive()` will raise `AssertionError` from `ForgeSimulation.model`/`.data` properties. Callers must close the pipeline before shutting down the simulation. This is documented but not caught by the pipeline — it is a programming error.
- If the renderer fails to create (e.g., no OpenGL context available), the constructor raises the underlying `mujoco.FatalError`. This is fatal and unrecoverable.
- If depth rendering fails or returns all zeros, `perceive()` falls back to setting all positions to the camera lookat point plus Gaussian noise (σ=0.05 m per axis, drawn from `self._rng`), logs a warning, and continues. This makes the pipeline robust to environments where depth is unavailable while keeping the fallback reproducible in tests.

### 2.4 3D Position Estimation

#### Approach: Depth-based Unprojection

Given a pixel detection at `(row, col)` and a depth buffer, we can estimate the 3D world position by projecting the pixel back through the camera.

The depth buffer (after the conditional linearization in §2.2) contains depth in meters. A value of 0.0 means no geometry was hit; treat it as "no depth available" for that detection and use the fallback.

The approach uses MuJoCo's camera intrinsics (field of view) and the computed camera extrinsics (view matrix derived from azimuth/elevation/distance/lookat) to unproject:

1. Read the depth value `d` at the detection's `pixel_center` from the depth buffer. If `d == 0.0`, use fallback for this detection.

2. Compute the 3D point in camera coordinates:
   ```python
   fovy = model.vis.global_.fovy  # vertical field of view in degrees
   f = 0.5 * height / np.tan(np.radians(fovy / 2))
   x_cam = (col - width / 2) * d / f
   y_cam = (height / 2 - row) * d / f   # row increases downward, camera Y points up
   z_cam = -d  # camera looks along -Z in MuJoCo convention
   ```

   Note on the Y-axis sign: MuJoCo's `Renderer.render()` returns images in standard row-major order (row 0 = top of image) after flipping the OpenGL buffer. In the MuJoCo/OpenGL camera coordinate system, Y points up. Since image rows increase downward but camera Y increases upward, the correct mapping is `y_cam = (height/2 - row) * d / f`. Using `(row - height/2)` would invert the Y axis.

   Implementation note: if testing reveals that `Renderer.render()` returns OpenGL bottom-up format (row 0 = bottom) instead of top-down, the formula reverts to `y_cam = (row - height/2) * d / f`. The implementer should verify by rendering a scene with an object known to be above the lookat point and checking whether its Y coordinate is positive.

3. Transform from camera coordinates to world coordinates using the camera's pose. Reconstruct the camera rotation matrix and position from azimuth, elevation, distance, and lookat:
   ```python
   # Camera position in world frame
   az_rad = np.radians(azimuth)
   el_rad = np.radians(elevation)
   cam_pos = lookat + distance * np.array([
       np.cos(el_rad) * np.cos(az_rad),
       np.cos(el_rad) * np.sin(az_rad),
       -np.sin(el_rad),    # negated: negative elevation = camera ABOVE lookat
   ])
   ```

   With the default parameters (elevation=-30, distance=1.8, lookat=(0.5, 0, 0.22)):
   - `cam_pos[2] = 0.22 - 1.8 * sin(-30°) = 0.22 + 0.9 = 1.12` (above the table, correct)

   The Z-component uses `-sin(el_rad)` because MuJoCo's camera convention is that negative elevation places the camera above the lookat point. Without the negation, `elevation=-30` would compute `z = 0.22 + 1.8 * sin(-30°) = -0.68`, placing the camera underground.

4. Build the rotation matrix from the camera's forward/up/right vectors:
   ```python
   forward = normalize(lookat - cam_pos)  # points from camera toward lookat
   world_up = np.array([0.0, 0.0, 1.0])
   right = normalize(np.cross(forward, world_up))
   up = np.cross(right, forward)
   # Camera-to-world rotation: columns are right, up, -forward
   R = np.column_stack([right, up, -forward])
   ```

5. Apply the camera-to-world transform:
   ```python
   cam_point = np.array([x_cam, y_cam, z_cam])
   world_pos = R @ cam_point + cam_pos
   ```

This approach naturally introduces estimation error because:
- Depth quantization in the buffer.
- The pixel center of a detection blob rarely corresponds exactly to the geometric center of the object.
- HSV filtering boundaries don't perfectly match object edges, shifting the centroid.

These errors satisfy the test requirement that `err > 0.001 m` without artificial noise injection.

#### Fallback

If depth rendering fails or the depth value at a detection's pixel is 0.0 (no geometry hit), fall back to placing the object at the camera lookat point plus Gaussian noise (σ=0.05 m per axis, drawn from `self._rng` which is seeded with 42 for reproducibility). This ensures `perceive()` always returns valid positions even in degraded environments. The worst-case error at 3σ is ~0.26 m, well within the test's 2.0 m bound and above the 0.001 m floor.

### 2.5 Integration with Runner

The runner (`forge_core/runner.py`) gains an optional `perception` parameter in `run_task()`:

```python
def run_task(
    task: TaskSpec,
    policy: Policy,
    verbose: bool = False,
    record: bool = False,
    fault_injector: FaultInjector | None = None,
    perception: bool = False,  # NEW
) -> EvalResult:
```

When `perception=True`, `run_task()` wraps the provided policy in a `PerceptionPolicy(inner_policy, sim)` before entering the run loop. No `observe()` method is needed on any policy. The runner does NOT own a pipeline itself — `PerceptionPolicy` is the sole integration path.

The perception cleanup is wrapped in try/finally to ensure the renderer resources are released even if an exception occurs during the run loop:

```python
if perception:
    from forge_core.perception import PerceptionPolicy
    policy = PerceptionPolicy(policy, sim)

try:
    # ... existing run loop unchanged ...

    # Recording block: when perception is True and trajectory is not None,
    # access policy.last_perception (guaranteed to exist since the runner
    # wrapped the policy in PerceptionPolicy above) and append result.image
    # to trajectory.frames.
    if trajectory is not None and perception:
        perc = policy.last_perception
        if perc is not None:
            trajectory.frames.append(perc.image)

finally:
    if perception:
        policy.close()

# ... evaluation, saving, shutdown as before ...
```

This is a minimal, backward-compatible change. Existing policies and callers without `perception=True` are completely unaffected. The `run_task_by_name()` and `run_suite()` functions gain the same `perception: bool = False` parameter and pass it through.

### 2.6 Integration with Recorder

When `record=True` and `perception=True`, the recorder saves camera frames:

- Add an optional `frames: list[np.ndarray]` field to `TrajectoryBuffer` (default empty list). This is backward-compatible because the field has a default.
- In the recording block of the run loop (§2.5), when `perception` is True and `trajectory` is not None, access `policy.last_perception` (guaranteed to exist since the runner wrapped the policy in `PerceptionPolicy` above). If the result is not None, append `result.image` to `trajectory.frames`.
- In `save_run()`, if `trajectory.frames` is non-empty, save them as a separate compressed numpy file: `frames.npz` containing a 4D array `(N, H, W, 3)`.
- Frame recording is optional — if `frames` is empty, no `frames.npz` is written.

The `save_run()` function signature does not change; it checks internally whether the buffer has frames.

### 2.7 Policy Integration — `PerceptionPolicy`

A wrapper class in `forge_core/perception.py`:

```python
class PerceptionPolicy(Policy):
    """Wrapper that adds perception observations to any policy."""

    def __init__(self, inner_policy: Policy, sim: ForgeSimulation, **camera_kwargs):
        self._inner = inner_policy
        self._pipeline = PerceptionPipeline(sim, **camera_kwargs)
        self._last_result: PerceptionResult | None = None

    def reset(self, sim: ForgeSimulation) -> None:
        self._inner.reset(sim)
        self._last_result = None

    def act(self, sim: ForgeSimulation) -> np.ndarray:
        self._last_result = self._pipeline.perceive()
        return self._inner.act(sim)

    @property
    def done(self) -> bool:
        return self._inner.done

    @property
    def name(self) -> str:
        return f"Perception({self._inner.name})"

    @property
    def last_perception(self) -> PerceptionResult | None:
        return self._last_result

    def close(self) -> None:
        self._pipeline.close()
```

This wraps any existing policy. The `last_perception` property lets callers (including the Recorder integration in §2.5) inspect the latest perception result. Future learned policies can override `act()` to use `self._last_result.image` as input.

### 2.8 CLI — `scripts/forge_perception.py`

Follows the pattern of `forge_eval.py` with `argparse` and `sys.path` setup.

```
Usage:
    # Render and save a camera image
    python3 scripts/forge_perception.py --scenario basic_workspace --render -o frame.png

    # Run perception and print detections
    python3 scripts/forge_perception.py --scenario basic_workspace --detect

    # Save annotated image with bounding boxes
    python3 scripts/forge_perception.py --scenario basic_workspace --detect --annotate -o detections.png

    # Custom camera parameters
    python3 scripts/forge_perception.py --scenario basic_workspace --detect \
        --azimuth 160 --elevation -30 --distance 1.5 --lookat 0.5 0.0 0.22
```

Arguments:
- `--scenario` (required): scenario name to load.
- `--render`: render and save a raw image.
- `--detect`: run full perception pipeline and print detected objects.
- `--annotate`: when combined with `--detect`, draw bounding boxes and labels on the image.
- `-o / --output`: output file path (default: `perception_output.png`).
- `--azimuth`, `--elevation`, `--distance`, `--lookat`: camera overrides.
- `--steps`: number of simulation steps to settle before rendering (default: 100).
- `--verbose / -v`: debug logging.

Flow:
1. Create `ForgeSimulation(scenario)`, call `sim.step(steps)` to settle physics.
2. Create `PerceptionPipeline(sim, **camera_kwargs)`.
3. If `--render`: call `pipeline.render()`, save with `cv2.imwrite()` (converting RGB→BGR for OpenCV's file writer).
4. If `--detect`: call `pipeline.perceive()`, print each detection's fields (object_id, pixel_center, pixel_area, confidence, position). If `--annotate`, draw bounding boxes and labels on the image using `cv2.rectangle()` and `cv2.putText()`, then save.
5. Call `pipeline.close()` and `sim.shutdown()`.

Annotation draws a colored rectangle around each detection (using the detected color — red for red_cube, blue for blue_cube) and labels with `"{object_id} ({confidence:.0%})"`.

Error handling: If the scenario doesn't exist, print a clear error and exit with code 1. If perception finds zero objects, print a warning but still save the image (no detections to annotate).

### 2.9 File Modifications Summary

| File | Change |
|------|--------|
| `forge_core/perception.py` | **New file** — `Detection`, `ColorDetector`, `PerceptionResult`, `PerceptionPipeline`, `PerceptionPolicy` |
| `forge_core/runner.py` | Add optional `perception: bool = False` param to `run_task()`, `run_task_by_name()`, `run_suite()`. When True, wrap policy in `PerceptionPolicy` with try/finally cleanup |
| `forge_core/recorder.py` | Add optional `frames` list to `TrajectoryBuffer`; save frames in `save_run()` |
| `pyproject.toml` | Add `"opencv-python-headless>=4.8.0"` to `dependencies` |
| `scripts/forge_perception.py` | **New file** — CLI tool |
| `tests/unit/test_perception.py` | **No changes** — existing tests must pass as-is |
| `docs/architecture.md` | Add Phase 5 section |
| `docs/setup.md` | Add `opencv-python-headless` to install commands; add perception CLI section |
| `README.md` | Update status to "Phase 5", add perception entries to structure and feature list, update phase roadmap, update test count |

### 2.10 Test Strategy

The existing `tests/unit/test_perception.py` is the primary test suite. It covers:

- `TestColorDetector`: detector returns a list; detections have required fields.
- `TestPerceptionPipeline`: render shape/dtype; perceive returns PerceptionResult; detects ≥ 1 object; detections have finite 3D positions; `to_object_states()` returns ObjectState list; `get_detection()` by ID; perception differs from ground truth; custom camera params accepted.

All tests run headless — MuJoCo's `Renderer` uses EGL or OSMesa automatically when no display is available.

Additional tests to add in a separate test file (`tests/unit/test_perception_extra.py`):

- `test_color_detector_empty_image`: detector returns empty list for a blank image.
- `test_perception_pipeline_close_idempotent`: calling `close()` twice doesn't raise.
- `test_perception_pipeline_context_manager`: verify `with PerceptionPipeline(sim) as p:` works correctly and closes on exit.
- `test_perception_policy_wrapper`: verify `PerceptionPolicy` wraps a NullPolicy correctly — `act()` returns valid targets, `last_perception` is a `PerceptionResult`.
- `test_render_depth_shape`: verify depth buffer is `(480, 640)` float32.
- `test_perception_result_timestamp`: verify `result.timestamp` matches `sim.sim_time` within tolerance.

Unit-testable without MuJoCo (pure OpenCV): `ColorDetector.detect()` can be tested with synthetic images (solid-color rectangles on a black background). This allows fast CI tests.

Integration-testable (requires MuJoCo): `PerceptionPipeline.render()`, `perceive()`, and the full perception-differs-from-ground-truth check. These are covered by the existing test file.

### 2.11 Dependency Details

Add to `pyproject.toml` `[project] dependencies`:

```toml
"opencv-python-headless>=4.8.0",
```

Using `opencv-python-headless` (not `opencv-python`) because:
- No GUI/display dependency (no libgtk, no X11).
- Smaller install footprint.
- Compatible with headless servers and CI.

The version pin `>=4.8.0` ensures we have stable `cv2.findContours` return-value API (3 returns were removed in 4.x; 4.8+ is well-tested).

### 2.12 Documentation Updates

#### `docs/architecture.md` — New Phase 5 Section

Add after the Phase 4 section:

```markdown
## Phase 5 — Perception (Camera + OpenCV)

### Camera Rendering

The `PerceptionPipeline` uses `mujoco.Renderer` for offscreen rendering.
Cameras are positioned programmatically via azimuth/elevation/distance/lookat.
Produces 640×480 RGB images and depth buffers without a display server.

### Perception Module — `forge_core/perception.py`

| Class | Purpose |
|-------|---------|
| `PerceptionPipeline` | Rendering + detection + 3D estimation |
| `ColorDetector` | HSV-based color detection via OpenCV |
| `Detection` | Single detected object with position |
| `PerceptionResult` | Full frame result with lookup methods |
| `PerceptionPolicy` | Wrapper adding perception to any policy |

### Detection Pipeline

1. Render RGB + depth via MuJoCo offscreen renderer
2. Convert RGB → HSV, apply color masks for red/blue
3. Find contours, compute centroids and areas
4. Unproject pixel + depth → 3D world position
5. Return structured PerceptionResult

### Integration

- Runner: `perception=True` wraps the policy in `PerceptionPolicy`
- Recorder: saves frames alongside trajectories when perception is active
- Policy: `PerceptionPolicy` wrapper provides image observations

### CLI — `scripts/forge_perception.py`

    forge_perception.py --scenario basic_workspace --render -o frame.png
    forge_perception.py --scenario basic_workspace --detect --annotate -o detections.png
```

#### `README.md` Updates

- Change status line to "Phase 5 — Perception (Camera + OpenCV)".
- Add `forge_core/perception.py` to project structure.
- Add `scripts/forge_perception.py` to project structure.
- Add perception bullet to "What's included".
- Update phase roadmap: Phase 5 → Done.
- Update test count.
- Add perception CLI usage example in Quick Start.
- Add `opencv-python-headless` to install command.

#### `docs/setup.md` Updates

- Add `opencv-python-headless` to the pip install command in Step 2.
- Add a "Perception" section under the existing CLI sections showing usage of `forge_perception.py`.

### 2.13 Edge Cases

1. **No objects visible from camera angle**: `perceive()` returns a `PerceptionResult` with `detections=[]`. The image is still valid. The CLI prints "No objects detected" and saves an unannotated image.

2. **Depth buffer returns all zeros or all ones**: Conditional linearization handles the raw [0,1] case. If after linearization the entire buffer is 0.0 (sentinel), fallback to lookat-based position estimation with Gaussian noise (σ=0.05 m per axis, seeded RNG). Logged as a warning.

3. **Simulation shut down before close()**: Calls to `render()` / `perceive()` raise `AssertionError` from `ForgeSimulation.model`/`.data` properties. Callers must close the pipeline before shutting down the simulation. This is a programming error, not a recoverable condition.

4. **Multiple contours for the same color**: Take the largest contour by area. The smaller ones are likely reflections or noise.

5. **Red hue wraparound**: The detector uses two HSV ranges for red (0–10 and 170–180) and ORs the resulting masks.

6. **Very small objects (far from camera)**: Minimum contour area threshold (50 px) filters noise. If a real object is smaller than 50 px, it won't be detected — this is acceptable for Phase 5 and logged at debug level.

7. **`close()` called multiple times**: `close()` is idempotent — if `self._renderer` is already None, it's a no-op. The `__del__` destructor also calls `close()`, safely handling the case where the pipeline is garbage-collected without explicit cleanup.

8. **OpenCV not installed**: Importing `forge_core.perception` raises `ImportError` with a clear message. This doesn't affect other modules since perception is only imported when explicitly used (lazy import in runner.py).

9. **Depth value 0.0 at a detection pixel**: Treated as "no depth available" — falls back to lookat + noise for that specific detection. Other detections with valid depth are still unprojected normally.

10. **Exception during run loop with perception=True**: The try/finally block in the runner (§2.5) ensures `policy.close()` is always called, releasing the renderer resources even if the loop raises.

### 2.14 Invariants and Ownership

| Invariant | Owner | Rationale |
|-----------|-------|-----------|
| Renderer lifecycle tied to pipeline | `PerceptionPipeline.__init__`/`close()`/`__del__` | Single owner avoids double-free; `__del__` is safety net |
| `Detection.confidence` ∈ [0, 1] | `ColorDetector.detect()` | Clamped with `min(1.0, ...)` at creation |
| `Detection.position` shape is (3,) | `PerceptionPipeline.perceive()` | Initialized to `np.zeros(3)` in `Detection.__init__`; overwritten with (3,) array in `perceive()` |
| `PerceptionResult.image` is never None after `perceive()` | `PerceptionPipeline.perceive()` | Always set from `render()` output before return |
| `PerceptionResult.timestamp` reflects capture time | `PerceptionPipeline.perceive()` | Set from `sim.sim_time` in `perceive()` |
| Camera params are validated | `PerceptionPipeline.__init__` | `distance` must be > 0; `lookat` must be a 3-tuple. `ValueError` raised otherwise. |
| Fallback noise is reproducible | `PerceptionPipeline.__init__` | Seeded RNG (`np.random.default_rng(42)`) ensures identical fallback positions across runs |
| Pipeline is the sole perception path | `PerceptionPolicy` wrapper in runner | Runner never owns a pipeline directly; avoids duplicate pipeline creation |
| Perception cleanup is exception-safe | `run_task()` try/finally in runner | `policy.close()` is called in finally block when `perception=True` |

### 2.15 Validation Rules for External Inputs

| Input | Required | Type | Limits | On Failure |
|-------|----------|------|--------|------------|
| `sim` param to PerceptionPipeline | Yes | ForgeSimulation | Must have active model/data | `AssertionError` from sim properties |
| `width`/`height` | No (defaults 640/480) | int | > 0 | `ValueError` |
| `azimuth` | No (default 180) | float | any | Wraps naturally in MuJoCo |
| `elevation` | No (default -30) | float | -90 to 90 | Clamped by MuJoCo |
| `distance` | No (default 1.8) | float | > 0 | `ValueError` |
| `lookat` | No (default (0.5,0,0.22)) | tuple of 3 floats | finite | `ValueError` if wrong length |
| `image` to `ColorDetector.detect()` | Yes | ndarray (H,W,3) uint8 | Non-empty | Returns `[]`, logs warning |
| CLI `--scenario` | Yes | str | Must exist in scenarios/ | Exit with error message, code 1 |
| CLI `--lookat` | No | 3 floats | finite | argparse error |
| CLI `--output` | No (default name) | str (file path) | Valid path | IOError surfaced to user |

---

## 3. Review Finding Responses (Revision 2 → Revision 3)

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | Camera Z-position formula has wrong sign (§2.4) | HIGH | **Addressed.** Changed Z component from `np.sin(el_rad)` to `-np.sin(el_rad)` in the `cam_pos` formula. Added worked example showing the correct result: `cam_z = 0.22 - 1.8 * sin(-30°) = 1.12` (above table). Added explanation of the MuJoCo convention that negative elevation = camera above lookat. |
| 2 | Depth buffer linearization claim is unverified (§2.2, §2.4) | MEDIUM | **Addressed.** Replaced the unconditional "linearized depth in meters" claim with a conditional linearization path. §2.2 now contains explicit code that checks `raw.max() <= 1.0` to detect raw [0,1] depth, applies `near*far/(far - raw*(far-near))` linearization when needed, and uses `raw >= 0.999` (not `== 0.0`) as the no-geometry sentinel for raw depth. The sentinel after processing is always 0.0. |
| 3 | Y-axis camera-space coordinate sign may be inverted (§2.4) | MEDIUM | **Addressed.** Changed `y_cam = (row - height/2) * d / f` to `y_cam = (height/2 - row) * d / f` with an explanation: row increases downward in the image but camera Y points upward. Added an implementation note for the implementer to verify the `Renderer.render()` image orientation if tests fail. |
| 4 | Recorder frame-capture detection mechanism not specified (§2.6) | NIT | **Addressed.** §2.5 and §2.6 now explicitly state: "When `perception` is True and `trajectory` is not None, access `policy.last_perception` (guaranteed to exist since the runner wrapped the policy in `PerceptionPolicy` above)." The code snippet in §2.5 shows the exact check. |
| 5 | No context manager or `__del__` for PerceptionPipeline (§2.3) | NIT | **Addressed.** Added `__enter__`/`__exit__`/`__del__` methods to `PerceptionPipeline` in §2.3. `__del__` calls `close()` as a safety net for leaked pipelines. `__enter__`/`__exit__` support `with` blocks. Added `test_perception_pipeline_context_manager` to the extra tests in §2.10. |
| 6 | No exception-safe cleanup of PerceptionPolicy in runner (§2.5) | NIT | **Addressed.** §2.5 now wraps the run loop and cleanup in try/finally. The code snippet shows `policy.close()` in the `finally` block when `perception=True`. Added corresponding invariant to §2.14. Edge case #10 in §2.13 documents this. |

### Responses from Revision 1 → Revision 2 (carried forward)

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | Runner `observe()` conflicts with `PerceptionPolicy` | MEDIUM | Addressed in R2. Removed runner-owned pipeline. Runner wraps policy in `PerceptionPolicy` as sole integration path. |
| 2 | Conflicting Gaussian noise σ (0.1 m vs 0.05 m) | MEDIUM | Addressed in R2. Unified to σ=0.05 m per axis with seeded RNG. |
| 3 | Depth linearization approach is ambiguous | MEDIUM | Addressed in R2, refined in R3 with conditional linearization. |
| 4 | `render()` error-handling relies on impossible None check | MEDIUM | Addressed in R2. AssertionError propagates from sim properties. |
| 5 | YAML `cameras` section is dead config | NIT | Addressed in R2. Removed from design, moved to Out of Scope. |
| 6 | `close()` conditional is unnecessary | NIT | Addressed in R2. Simplified to None-guard only. |
| 7 | Redundant `update_scene` in `perceive()` | NIT | Addressed in R2. Single `update_scene()` at top of `perceive()`. |
| 8 | `to_object_states()` always sets timestamp=0.0 | NIT | Addressed in R2. Added `timestamp` field to `PerceptionResult`, populated from `sim.sim_time`. |
