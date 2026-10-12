# Implementation Plan — Phase 5: Perception (Camera + OpenCV)

## Design-Review Fixes Reference

| Fix ID | Severity | Summary |
|--------|----------|---------|
| (a) | MEDIUM | Use parameterless `enable_depth_rendering()` / `disable_depth_rendering()` — no args |
| (b) | MEDIUM | Construct `pixel_center = (cy, cx)` for (row, col); destructure as `row, col = detection.pixel_center` |
| (c) | NIT | Add comment that `raw.max() <= 1.0` heuristic assumes camera distance >= 1.0 m |
| (d) | NIT | Add `.copy()` to RGB render call in `perceive()`: `image = self._renderer.render().copy()` |
| (e) | NIT | Frame recording goes inside the while loop after `trajectory.record_step()` |
| (f) | NIT | Use `perception_active` flag; check it in finally block before calling `policy.close()` |
| (g) | NIT | Do NOT pass `MjvOption()` — default scene options are sufficient |

---

- [ ] 1. Add `opencv-python-headless>=4.8.0` to `pyproject.toml` dependencies.
      Add the line `"opencv-python-headless>=4.8.0",` to the `[project] dependencies` array after `"PyYAML>=6.0"`.
      Files: `pyproject.toml`
      Verify: `pip install opencv-python-headless>=4.8.0` succeeds; `python3 -c "import cv2; print(cv2.__version__)"` prints a version.

- [ ] 2. Create `forge_core/perception.py` — the core perception module.
      Implement all five classes: `Detection`, `ColorDetector`, `PerceptionResult`, `PerceptionPipeline`, `PerceptionPolicy`.
      
      **Detection** (dataclass): `object_id: str`, `pixel_center: tuple`, `pixel_area: int`, `confidence: float`, `position: np.ndarray` (default `np.zeros(3)`).
      
      **ColorDetector**: `COLOR_RANGES` dict for `red_cube` (two HSV ranges: `(0,100,100)-(10,255,255)` and `(170,100,100)-(180,255,255)`) and `blue_cube` (`(100,100,100)-(130,255,255)`). `detect(image)` converts RGB→HSV, applies `inRange`, morphology (5×5 kernel), `findContours`, filters by min area 50px, takes largest contour per color, computes moments. **Fix (b)**: construct `pixel_center = (cy, cx)` where `cy = int(M['m01']/M['m00'])`, `cx = int(M['m10']/M['m00'])`. Confidence = `min(1.0, area / 500)`. Returns `list[Detection]` with `position=np.zeros(3)`.
      
      **PerceptionResult** (dataclass): `image`, `detections`, `timestamp=0.0`. `to_object_states()` returns `ObjectState` list with identity quaternion and `timestamp=self.timestamp`. `get_detection(id)` linear scan.
      
      **PerceptionPipeline**: Constructor validates `distance > 0`, `len(lookat) == 3`, `width > 0`, `height > 0`. Creates `mujoco.Renderer(sim.model, height, width)`, `MjvCamera` with `mjCAMERA_FREE`. **Fix (g)**: do NOT create/pass `MjvOption`. Creates `ColorDetector`, `self._rng = np.random.default_rng(42)`.
      - `render()`: `update_scene(sim.data, camera)`, return `renderer.render()`.
      - `render_depth()`: `update_scene`, **Fix (a)**: `self._renderer.enable_depth_rendering()` (no args), render, conditional linearization (**Fix (c)**: comment about heuristic), `self._renderer.disable_depth_rendering()` (no args), return float32.
      - `perceive()`: `update_scene` once. **Fix (d)**: `image = self._renderer.render().copy()`. Enable depth, render, linearize, disable depth. Detect. For each detection: **Fix (b)**: `row, col = detection.pixel_center`. Read depth at `[row, col]`. If `depth == 0`, fallback (lookat + rng gaussian σ=0.05). Else unproject via fovy, cam_pos (Z uses `-sin(el_rad)`), rotation matrix. Set `timestamp = sim.sim_time`. Return `PerceptionResult`.
      - `close()`: idempotent renderer close. `__enter__`/`__exit__`/`__del__`.
      
      **PerceptionPolicy** (extends `Policy`): wraps inner policy, creates `PerceptionPipeline`. `act()` calls `perceive()` then `inner.act()`. Exposes `last_perception` property. `close()` calls `pipeline.close()`.
      
      Files: `forge_core/perception.py` (NEW)
      Verify: `python3 -m pytest tests/unit/test_perception.py -v` — all 10 tests pass.

- [ ] 3. Modify `forge_core/runner.py` — add `perception: bool = False` parameter.
      Add to `run_task()`, `run_task_by_name()`, `run_suite()`. In `run_task()`:
      - After `policy.reset(sim)`, set `perception_active = False`. If `perception`: lazy import `PerceptionPolicy`, wrap policy, set `perception_active = True`. **Fix (f)**: flag guards the finally block.
      - Wrap the while loop + settle + evaluate + save in `try/finally`. In `finally`: `if perception_active: policy.close()`.
      - **Fix (e)**: Inside the while loop, after the existing `trajectory.record_step()` block, add: `if trajectory is not None and perception_active: perc = policy.last_perception; if perc is not None: trajectory.frames.append(perc.image)`.
      - Move `sim.shutdown()` after the finally block (it must stay outside).
      - Pass `perception=perception` through in `run_task_by_name()` and `run_suite()`.
      Files: `forge_core/runner.py`
      Verify: `python3 -m pytest tests/ -v` — all existing tests pass, no regressions.

- [ ] 4. Modify `forge_core/recorder.py` — add optional frames field and save logic.
      Add `frames: list[np.ndarray] = field(default_factory=list)` to `TrajectoryBuffer` (after existing fields).
      In `save_run()`, after saving `trajectory.npz`, add: `if trajectory.frames: frames_array = np.stack(trajectory.frames); np.savez_compressed(run_dir / "frames.npz", frames=frames_array)`.
      Files: `forge_core/recorder.py`
      Verify: `python3 -m pytest tests/ -v` — all existing tests pass.

- [ ] 5. Create `tests/unit/test_perception_extra.py` — additional perception tests.
      Follow existing test patterns (imports, sys.path setup, sim fixture).
      Tests:
      - `test_color_detector_empty_image`: blank 100×100×3 uint8 zeros → `detect()` returns `[]`.
      - `test_perception_pipeline_close_idempotent`: `close()` twice, no exception.
      - `test_perception_pipeline_context_manager`: `with PerceptionPipeline(sim) as p:` works, `p.render()` returns correct shape.
      - `test_perception_policy_wrapper`: wrap `NullPolicy` in `PerceptionPolicy`, call `act(sim)`, verify `last_perception` is `PerceptionResult`, call `close()`.
      - `test_render_depth_shape`: `pipeline.render_depth()` returns `(480, 640)` float array.
      - `test_perception_result_timestamp`: `perceive().timestamp` ≈ `sim.sim_time` within 0.1s.
      Files: `tests/unit/test_perception_extra.py` (NEW)
      Verify: `python3 -m pytest tests/unit/test_perception_extra.py -v` — all 6 tests pass.

- [ ] 6. Create `scripts/forge_perception.py` — perception CLI tool.
      Follow `scripts/forge_eval.py` pattern (sys.path setup, argparse). Arguments: `--scenario` (required), `--render`, `--detect`, `--annotate`, `-o/--output` (default `perception_output.png`), `--azimuth`, `--elevation`, `--distance`, `--lookat` (nargs=3), `--steps` (default 100), `--verbose`.
      Flow: create sim, step to settle, create pipeline with camera kwargs, render/detect/annotate as requested, save images (RGB→BGR for cv2.imwrite), print detections, cleanup.
      Draw bounding boxes with `cv2.rectangle` (±30px around pixel center) and `cv2.putText` labels.
      Handle nonexistent scenario with print + `sys.exit(1)`.
      Files: `scripts/forge_perception.py` (NEW)
      Verify: `python3 scripts/forge_perception.py --scenario basic_workspace --render -o /tmp/test_frame.png` exits 0; `python3 scripts/forge_perception.py --scenario basic_workspace --detect --annotate -o /tmp/test_det.png` exits 0.

- [ ] 7. Update `docs/architecture.md` — add Phase 5 section.
      Add `## Phase 5 — Perception (Camera + OpenCV)` section after Phase 4 (before Design Principles). Include: Camera Rendering, Perception Module table, Detection Pipeline steps, Integration, CLI subsections per design §2.12. Update overview to mention Phase 5.
      Files: `docs/architecture.md`
      Verify: `grep 'Phase 5' docs/architecture.md` returns matches.

- [ ] 8. Update `docs/setup.md` — add opencv dep and perception CLI docs.
      Add `opencv-python-headless` to Step 2 pip install command. Add `## Perception` section with CLI usage examples after the existing Replay section.
      Files: `docs/setup.md`
      Verify: `grep 'opencv-python-headless' docs/setup.md` returns a match.

- [ ] 9. Update `README.md` — status, structure, features, roadmap, test count.
      (a) Status → "Phase 5 — Perception (Camera + OpenCV)". (b) Description → "Phases 1–5 are complete". (c) Add perception bullet to What's included. (d) Add `perception.py` and `forge_perception.py` to structure. (e) Add `opencv-python-headless` to install command. (f) Add perception Quick Start section. (g) Update test count. (h) Phase 5 → Done in roadmap.
      Files: `README.md`
      Verify: `grep 'Phase 5' README.md` shows updated status; `python3 -m pytest tests/ -v` — full suite passes.
