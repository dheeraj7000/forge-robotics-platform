# Implementation Plan — Phase 9: Edge Deployment (ONNX, TensorRT)

## Codebase Facts (from exploration)

**Policy ABC** (`forge_core/policy.py`):
- `reset(self, sim: ForgeSimulation) -> None`
- `act(self, sim: ForgeSimulation) -> np.ndarray`
- `done` property → `bool`
- `name` property → `str` (default: `self.__class__.__name__`)
- `create_policy(name, **kwargs)` uses lazy import for "learned"; POLICY_REGISTRY dict for others
- Available string shows all registered + "learned"

**PolicyNetwork** (`forge_core/learned_policy.py`):
- `PolicyNetwork(obs_dim=31, act_dim=8, hidden=[256, 256, 128])`
- Architecture: obs → 256 → 256 → 128 → action (Tanh activations)
- `forward(obs: Tensor) -> Tensor`
- `OBS_DIM = 31`, `ACT_DIM = 8`

**extract_observation** (`forge_core/learned_policy.py`):
- `extract_observation(sim, target_object='red_cube') -> np.ndarray` (31-dim float32)
- Used by `LearnedPolicy.act()` — same function for ONNX inference

**ModelRegistry** (`forge_core/model_registry.py`):
- `register_model(name, checkpoint_path, experiment_id, description, metadata) -> dict`
- `get_model(name, version=None, stage=None) -> dict | None`
- Entry has: name, version, stage, checkpoint_path, experiment_id, metadata
- Stages: training → validated → production → archived

**Runner** (`forge_core/runner.py`):
- `load_policy_from_registry(name, version, stage, **policy_kwargs)` → uses `create_policy("learned", model_path=...)`
- `run_task(task, policy, ...)` — policy is any `Policy`

**Eval CLI** (`scripts/forge_eval.py`):
- argparse, `--policy`, `--model-path`, `--registry-model`, `--registry-version`
- Calls `create_policy(policy_name, **policy_kwargs)`

**pyproject.toml**: `[ml]` extra for torch, `[dev]` for pytest. No edge extra yet.

**Test pattern**: sys.path insert, `TestXxx` classes, `tmp_path` fixtures, no ROS deps.

---

## Design Decisions

1. **Single module**: All edge deployment code goes in `forge_core/edge_deploy.py`, following the pattern of `learned_policy.py` being self-contained.
2. **Lazy imports**: `onnx`, `onnxruntime`, and `tensorrt` are imported inside functions/methods, never at module level. This ensures `from forge_core.edge_deploy import ...` always works regardless of installed extras.
3. **Policy naming**: Register `"onnx"` and `"edge"` in `create_policy()` via lazy import, matching the `"learned"` pattern.
4. **[edge] extra**: Named `edge` (not `deploy`) to match the module name and be concise. Contains `onnx>=1.14.0` and `onnxruntime>=1.15.0`.
5. **TensorRT handling**: Never a pip dep. Runtime-detected via `has_tensorrt()`. `TensorRTPolicy.__init__` raises RuntimeError when unavailable. All TensorRT tests use `pytest.mark.skipif`.
6. **extract_observation reuse**: OnnxPolicy/TensorRTPolicy reuse `extract_observation` from `learned_policy.py` for consistency with LearnedPolicy.
7. **Config file**: `config/deploy.yaml` for deployment settings, loaded by `load_deploy_config()` following the `load_training_config()` pattern.
8. **Manifest format**: YAML (`manifest.yaml`) — consistent with the project's YAML-driven philosophy.

---

## Plan

- [ ] 1. **Update pyproject.toml with [edge] optional extra.**
      Add `edge = ["onnx>=1.14.0", "onnxruntime>=1.15.0"]` under `[project.optional-dependencies]`. Add pytest marker `edge_deploy` in `[tool.pytest.ini_options]`.
      Files: `pyproject.toml`
      Verify: `pip install -e '.[edge]'` succeeds; `python3 -c "import onnx; import onnxruntime"` succeeds.

- [ ] 2. **Create config/deploy.yaml with default deployment settings.**
      Contains: opset_version, dynamic_batch, tensorrt precision/batch/workspace, package settings.
      Files: `config/deploy.yaml`
      Verify: `python3 -c "from forge_core.config import load_yaml; print(load_yaml('config/deploy.yaml'))"` prints the config dict.

- [ ] 3. **Create forge_core/edge_deploy.py — the core edge deployment module.**
      Contains all classes and functions described in FEAT-001 steps 3a–3l: `has_onnxruntime()`, `has_tensorrt()`, `ExportConfig`, `TensorRTConfig`, `OnnxExporter`, `TensorRTOptimizer`, `OnnxPolicy(Policy)`, `TensorRTPolicy(Policy)`, `EdgePolicy(Policy)`, `DeploymentManifest`, `DeploymentPackager`, `benchmark_inference()`, `export_from_registry()`, `load_deploy_config()`. All onnx/onnxruntime/tensorrt imports are lazy (inside methods). Zero ROS dependency.
      Files: `forge_core/edge_deploy.py`
      Verify: `python3 -c "from forge_core.edge_deploy import OnnxExporter, OnnxPolicy, TensorRTPolicy, EdgePolicy, DeploymentPackager, export_from_registry, has_onnxruntime, has_tensorrt, TensorRTOptimizer; print('OK')"` prints OK.

- [ ] 4. **Update forge_core/policy.py to register onnx and edge policies.**
      In `create_policy()`, add lazy-import branches for `"onnx"` (returns `OnnxPolicy`) and `"edge"` (returns `EdgePolicy`), following the existing `"learned"` pattern. Update the available-policies string.
      Files: `forge_core/policy.py`
      Verify: `python3 -c "from forge_core.policy import create_policy"` succeeds; existing tests pass.

- [ ] 5. **Update forge_core/runner.py with edge policy loading.**
      Add `load_edge_policy_from_registry(name, version, stage, output_dir, **policy_kwargs) -> Policy` function that exports a registered model to ONNX and returns an OnnxPolicy.
      Files: `forge_core/runner.py`
      Verify: `python3 -c "from forge_core.runner import load_edge_policy_from_registry"` succeeds.

- [ ] 6. **Update scripts/forge_eval.py to support --onnx-model flag.**
      Add `--onnx-model` argument. When provided, set `policy_name = "onnx"` and `policy_kwargs["onnx_path"] = args.onnx_model`.
      Files: `scripts/forge_eval.py`
      Verify: `python3 scripts/forge_eval.py --help` shows `--onnx-model` in output.

- [ ] 7. **Create scripts/forge_deploy.py CLI.**
      argparse-based script with 6 action flags: `--export`, `--validate`, `--benchmark`, `--package`, `--optimize`, `--export-registry`. Each flag triggers its corresponding operation from `forge_core/edge_deploy`. Follows the pattern of `forge_eval.py`.
      Files: `scripts/forge_deploy.py`
      Verify: `python3 scripts/forge_deploy.py --help` prints usage with all flags.

- [ ] 8. **Create tests/unit/test_edge_deploy.py with comprehensive tests.**
      Test classes: TestRuntimeDetection, TestExportConfig, TestTensorRTConfig, TestOnnxExporter, TestOnnxPolicy, TestTensorRTPolicy, TestTensorRTOptimizer, TestEdgePolicy, TestDeploymentManifest, TestDeploymentPackager, TestExportFromRegistry, TestBenchmarkInference. All onnxruntime-dependent tests guarded with `@pytest.mark.skipif(not has_onnxruntime(), ...)`. TensorRT unavailable-path tests guarded with `@pytest.mark.skipif(has_tensorrt(), ...)`.
      Files: `tests/unit/test_edge_deploy.py`
      Verify: `python3 -m pytest tests/unit/test_edge_deploy.py -v` — all tests pass (TensorRT tests skipped).

- [ ] 9. **Run full test suite to verify no regressions.**
      Files: (none modified)
      Verify: `python3 -m pytest tests/ -v` — all tests pass (233 existing + new edge deploy tests).

- [ ] 10. **Update docs/architecture.md with Phase 9 section.**
       Add `## Phase 9 — Edge Deployment (ONNX / TensorRT)` section after Phase 8, covering: overview, module table, ONNX export, TensorRT optimization, edge inference runtime, deployment packaging, integration points, CLI reference. Update overview paragraph to say Phases 1–9. Add design principle 7 about optional edge deps.
       Files: `docs/architecture.md`
       Verify: `grep "Phase 9" docs/architecture.md` finds the section heading.

- [ ] 11. **Update docs/setup.md with edge deployment info.**
       Add Step 2c for edge deps installation (`pip install -e '.[edge]'`). Add Edge Deployment usage section with `forge_deploy.py` examples.
       Files: `docs/setup.md`
       Verify: `grep "edge" docs/setup.md` finds the new sections.

- [ ] 12. **Update README.md with Phase 9 status and features.**
       Update status to Phase 9, add edge deployment features to "What's included", add CLI examples, update project structure tree (edge_deploy.py, forge_deploy.py, deploy.yaml), update test count, set Phase 9 to "Done" in roadmap.
       Files: `README.md`
       Verify: `grep "Phase 9" README.md` finds updated status; `grep "edge_deploy" README.md` finds structure entry.
