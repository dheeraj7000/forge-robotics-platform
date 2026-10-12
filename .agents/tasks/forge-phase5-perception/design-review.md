# Phase 5 Perception — Design Review (Revision 3)

**Reviewer:** Design Review Agent
**Date:** Auto-generated
**Verdict:** CHANGES_REQUESTED (2 MEDIUM findings)

---

## Findings

### 1. MEDIUM — `enable_depth_rendering` API calling convention is wrong (§2.2, §2.3)

**Problem:** The design uses `self._renderer.enable_depth_rendering(True)` and `self._renderer.enable_depth_rendering(False)` in the rendering flow (§2.2 step 5), `render_depth()` (§2.3), and inside `perceive()`. The MuJoCo Python API (`mujoco >= 3.1.0`) provides two separate parameterless methods: `enable_depth_rendering()` and `disable_depth_rendering()`. Calling `enable_depth_rendering(True)` would raise `TypeError: enable_depth_rendering() takes 1 positional argument but 2 were given` because the method signature is `def enable_depth_rendering(self)` with no additional parameters.

This affects every code path that renders depth — `render_depth()` and `perceive()` — making the core perception pipeline non-functional as specified.

**Locations:**
- §2.2 Rendering Flow, step 5
- §2.3 `render_depth()` steps 2 and 5
- §2.3 `perceive()` step 3

**Fix:** Replace all occurrences:
```python
# Before (wrong):
self._renderer.enable_depth_rendering(True)
self._renderer.enable_depth_rendering(False)

# After (correct):
self._renderer.enable_depth_rendering()
self._renderer.disable_depth_rendering()
```

Update the four call sites: §2.2 step 5 (two calls), §2.3 `render_depth()` steps 2 and 5, and the `perceive()` depth rendering step.

---

### 2. MEDIUM — `pixel_center` tuple ordering is ambiguous and likely inconsistent with unprojection (§2.3, §2.4)

**Problem:** The `Detection` dataclass defines `pixel_center: tuple  # (row, col) in image coordinates`. However, the `ColorDetector.detect()` description says to "compute moments to get centroid" — OpenCV moments yield `cx = M["m10"]/M["m00"]` (column/x) and `cy = M["m01"]/M["m00"]` (row/y). The design never shows the explicit `Detection(pixel_center=...)` construction in the detector, so the implementer must choose between:

- **(a)** `pixel_center = (cx, cy)` — natural OpenCV order = (col, row), contradicting the docstring
- **(b)** `pixel_center = (cy, cx)` — matches the docstring (row, col) but requires an unobvious swap

The unprojection in §2.4 uses `row` and `col` separately:
```python
x_cam = (col - width / 2) * d / f
y_cam = (height / 2 - row) * d / f
```

If the implementer picks option (a) and the unprojection destructures as `row, col = pixel_center`, the X and Y camera coordinates get swapped, producing wrong world positions. For cubes at y=±0.15m from center, the resulting position error could approach or exceed the test's 2.0m bound.

**Fix:** Add explicit construction code in the ColorDetector section to lock the convention:

```python
# In ColorDetector.detect(), after computing moments:
cx = int(M["m10"] / M["m00"])   # column (x in image)
cy = int(M["m01"] / M["m00"])   # row (y in image)
pixel_center = (cy, cx)          # (row, col) per Detection docstring
```

And in §2.4, show the destructuring:
```python
row, col = detection.pixel_center  # (row, col) convention
```

Alternatively, change the convention to (col, row) everywhere and update the unprojection accordingly. Either is fine — just pick one and be explicit in both the detector and the unprojection.

---

### 3. NIT — Depth linearization heuristic `raw.max() <= 1.0` is fragile for close-up cameras (§2.2)

**Problem:** The conditional linearization uses `raw.max() <= 1.0` to distinguish raw OpenGL depth (always in [0, 1]) from already-linearized depth (in meters). This fails when MuJoCo returns linearized depth and all visible geometry is within 1.0m of the camera — `raw.max()` would be ≤ 1.0, triggering the linearization formula on already-linear data and producing distorted depth values.

With the default camera (distance=1.8m), all visible geometry is beyond 1.0m, so the heuristic works. But custom cameras with `distance < 1.0` or close-up views could trigger the false positive.

**Fix:** Add a comment documenting the limitation:
```python
# NOTE: This heuristic assumes at least one visible surface is > 1.0m from
# the camera. For close-up cameras (distance < 1.0m), verify depth output
# empirically or use a version-specific check.
```

---

### 4. NIT — `render()` buffer reuse risk not addressed (§2.3)

**Problem:** MuJoCo's `Renderer.render()` may return a view of an internal buffer that is overwritten by the next `render()` call. In `perceive()`, the design calls `render()` for RGB, then enables depth and calls `render()` again for depth. If the RGB buffer is a view, the second render may overwrite it. The existing implementation already handles this with `.copy()`, but the design does not mention it.

**Fix:** In the `perceive()` implementation description, add: "Copy the RGB image (`image = self._renderer.render().copy()`) before the depth render call to prevent buffer reuse."

---

### 5. NIT — Runner frame recording placement within the while loop is ambiguous (§2.5)

**Problem:** The code snippet in §2.5 shows the frame recording block inside the `try` but formatted as if it's at the same level as the run loop comment, not clearly inside the `while not policy.done` loop:

```python
try:
    # ... existing run loop unchanged ...

    # Recording block:
    if trajectory is not None and perception:
        perc = policy.last_perception
        ...
```

The intent is for this to execute each iteration (after `policy.act()` triggers perception), but the layout could be read as "after the loop." Per-step recording must be inside the while loop, adjacent to the existing `trajectory.record_step()` call.

**Fix:** Add a clarifying note: "The frame recording block goes inside the `while not policy.done` loop, immediately after the existing `trajectory.record_step(...)` call." Or show it inline with surrounding loop code.

---

### 6. NIT — `PerceptionPolicy` constructor failure leaves `finally` calling `close()` on unwrapped policy (§2.5)

**Problem:** If `PerceptionPolicy(policy, sim)` raises (e.g., `mujoco.FatalError` from renderer creation), the variable `policy` still points to the original unwrapped `Policy` object. The `finally` block checks `if perception:` (still True) and calls `policy.close()`. The base `Policy` ABC does not define `close()`, so this raises `AttributeError` — masking the original error.

```python
if perception:
    policy = PerceptionPolicy(policy, sim)  # if this raises...
try:
    ...
finally:
    if perception:
        policy.close()  # ...calls close() on the original policy → AttributeError
```

**Fix:** Guard with `hasattr` or set a flag:
```python
perception_active = False
if perception:
    from forge_core.perception import PerceptionPolicy
    policy = PerceptionPolicy(policy, sim)
    perception_active = True

try:
    ...
finally:
    if perception_active:
        policy.close()
```

---

### 7. NIT — `MjvOption` argument omitted from `update_scene()` calls (§2.2, §2.3)

**Problem:** The design shows `renderer.update_scene(sim.data, camera)` with two arguments. The existing codebase (in `forge_core/perception.py`) calls `self._renderer.update_scene(self.sim.data, self._camera, self._scene_option)` with three arguments, passing an `MjvOption()` for scene options. While `scene_option` is optional in the MuJoCo API (defaults to None), the existing code creates and passes one. The design doesn't mention `MjvOption` at all.

**Fix:** Either add `self._scene_option = mujoco.MjvOption()` to the constructor and pass it as the third argument to `update_scene()`, or add a note that the default scene options are sufficient and `MjvOption()` is intentionally omitted. Both approaches work.

---

## Verified Assumptions

| # | Claim in Design | Verification | Status |
|---|----------------|--------------|--------|
| 1 | `sim.model` and `sim.data` are assert-guarded properties | Read `forge_core/simulation.py` lines 259-268: `assert self._model is not None` / `assert self._data is not None` | ✅ Verified |
| 2 | `sim.sim_time` returns simulation clock | Read `forge_core/simulation.py` line 267: `return self._data.time` | ✅ Verified |
| 3 | `ObjectState` has `id`, `position`, `orientation`, `timestamp` | Read `forge_core/simulation.py` dataclass definition (line ~30) | ✅ Verified |
| 4 | `Policy` ABC has `reset()`, `act()`, `done`, `name` — no `close()` | Read `forge_core/policy.py`: abstract methods match; no `close()` defined | ✅ Verified |
| 5 | `TrajectoryBuffer` has no `frames` field currently | Read `forge_core/recorder.py`: only `timestamps`, `joint_*`, `ee_*`, `actions`, `object_*` fields | ✅ Verified |
| 6 | `run_task()` has no `perception` param currently | Read `forge_core/runner.py`: signature is `(task, policy, verbose, record, fault_injector)` | ✅ Verified |
| 7 | `pyproject.toml` has no `opencv-python-headless` dependency | Read `pyproject.toml`: deps are mujoco, numpy, PyYAML only | ✅ Verified |
| 8 | Red cube RGBA = `0.9 0.1 0.1 1`, Blue cube RGBA = `0.1 0.1 0.9 1` | Read `simulation.py` `_material_to_rgba()` and `basic_workspace.yaml` | ✅ Verified |
| 9 | `save_run()` signature is stable — internal frame check won't break callers | Read `forge_core/recorder.py`: `save_run(run_id, trajectory, metadata, scenario_config, result, output_dir)` | ✅ Verified |
| 10 | Scenario has `red_cube`, `blue_cube`, and `target_bin` objects | Read `simulation/scenarios/basic_workspace.yaml` | ✅ Verified |
| 11 | MuJoCo box geom `size` is half-extents (cubes are 4cm per side) | MuJoCo documentation convention; `size=[0.02, 0.02, 0.02]` → 4cm cube | ✅ Verified (by convention) |
| 12 | `mujoco.Renderer` has `close()` method | Existing `forge_core/perception.py` calls `self._renderer.close()` successfully | ✅ Verified (by existing usage) |
| 13 | Existing CLI scripts use `argparse` with `sys.path` setup | Checked `scripts/forge_eval.py`, `forge_replay.py`, `run_forge.py` — all follow this pattern | ✅ Verified |

## Unverified / Wrong Assumptions

| # | Claim in Design | Issue | Risk |
|---|----------------|-------|------|
| 1 | `Renderer` has `enable_depth_rendering(bool)` method | **WRONG** — the API uses parameterless `enable_depth_rendering()` and `disable_depth_rendering()` as separate methods. Passing `True` raises `TypeError`. | HIGH — runtime failure in perceive() |
| 2 | `Renderer.render()` returns top-down images (row 0 = top of frame) | ⚠️ **Unverified** — likely true (Python bindings typically flip the OpenGL bottom-up buffer), but if wrong, the Y-axis sign in §2.4 unprojection is inverted. Design includes an implementation note to verify. | MEDIUM — Y position inverted if wrong |
| 3 | `model.vis.global_.fovy` is the FOV used by the free camera | ⚠️ **Unverified** — MuJoCo documentation states the free camera uses the global fovy, and the default is 45°. Plausible but not tested against the actual binary. | LOW — the default 45° is well-known |
| 4 | Depth buffer from `Renderer.render()` in depth mode is raw [0,1] or linearized meters | ⚠️ **Unverified** — behavior depends on MuJoCo version and build. The conditional linearization in §2.2 handles both cases, which is the correct defensive approach. | LOW — design already handles both |
| 5 | `Renderer.update_scene(data, camera)` accepts `MjvCamera` as the camera arg | ⚠️ **Unverified** — likely correct (MuJoCo Python API accepts MjvCamera or camera_id int). Existing code in `forge_core/perception.py` uses this pattern successfully. | LOW — confirmed by existing usage |

---

## Summary

The design is thorough and well-structured. Revision 3 has addressed all prior HIGH and MEDIUM findings. Two new MEDIUM issues were found:

1. The depth rendering API call uses `enable_depth_rendering(True/False)` instead of the separate parameterless methods — a straight TypeError at runtime.
2. The `pixel_center` tuple ordering is ambiguous between the (row, col) docstring and the natural (col, row) output from OpenCV moments, which would silently produce wrong 3D positions.

Both fixes are small and surgical. All other design elements — the test contract conformance, camera rendering approach, integration points, error handling, edge cases, and documentation plan — are solid and ready for implementation after these two corrections.
