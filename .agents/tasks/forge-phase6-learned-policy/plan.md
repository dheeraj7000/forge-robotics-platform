# Implementation Plan — Phase 6: Learned Policy (PyTorch)

Source: approved design at `.agents/tasks/forge-phase6-learned-policy/design.md`
Review fixes from: `.agents/tasks/forge-phase6-learned-policy/design-review.json`

---

## FEAT-001: Core Implementation (Code + Tests)

- [ ] 1. Add `[ml]` optional dependency to `pyproject.toml`
      Add `ml = ["torch>=2.0"]` under `[project.optional-dependencies]`, below the existing `dev` group.
      Files: `/root/forge/pyproject.toml`
      Verify: `python3 -c "import torch; print(torch.__version__)"` — already installed, just formalizing.

- [ ] 2. Create `config/training.yaml` with default training configuration
      Three sections matching design §1: `training` (epochs: 100, batch_size: 64, learning_rate: 0.001, validation_split: 0.1, checkpoint_interval: 20), `model` (hidden_layers: [256, 256, 128], obs_dim: 31, act_dim: 8), `data` (target_object: "red_cube", demo_episodes: 3, demo_policy: "scripted_pick_and_place"). Include a comment header: `# Forge training configuration — behavioral cloning`.
      Files: `/root/forge/config/training.yaml`
      Verify: `python3 -c "from forge_core.config import load_yaml; print(load_yaml('config/training.yaml'))"` — prints the dict.

- [ ] 3. Extend `TrainResult` dataclass in `learned_policy.py`
      Add `from dataclasses import dataclass, field` (add `field` to existing import). Add `val_history: list[float] = field(default_factory=list)` to `TrainResult`. Update `summary()`: if `self.val_history`, append `, val_loss={self.val_history[-1]:.6f}` before the `, saved=` part.
      Files: `/root/forge/forge_core/learned_policy.py`
      Verify: `python3 -m pytest tests/unit/test_learned_policy.py::TestBehavioralCloning -v` — existing tests still pass (val_history defaults to []).

- [ ] 4. Add `obs_mode` parameter to `LearnedPolicy`
      Add class constant `OBS_MODES = ("state", "vision")`. Add `obs_mode: str = "state"` parameter before `**kwargs`. Validate: if `obs_mode not in self.OBS_MODES` raise `ValueError`. If `obs_mode == "vision"` raise `NotImplementedError("Vision-based policy input is planned for a future phase. Use obs_mode='state' for MLP policies.")`. Store as `self._obs_mode = obs_mode`.
      Files: `/root/forge/forge_core/learned_policy.py`
      Verify: `python3 -m pytest tests/unit/test_learned_policy.py::TestLearnedPolicy -v` — existing tests pass (default obs_mode="state" is transparent).

- [ ] 5. Add warning logs to `DemoDataset.from_run_dir`
      In the existing `from_run_dir` method, when `f"obj_{target_object}_pos" not in data`: add `logger.warning("missing_object_data: %s not found in %s, using zeros", target_object, run_dir)`. Same for `"obj_target_bin_pos" not in data`: add `logger.warning("missing_object_data: target_bin not found in %s, using zeros", run_dir)`.
      Files: `/root/forge/forge_core/learned_policy.py`
      Verify: `python3 -m pytest tests/unit/test_learned_policy.py -v` — all existing tests pass.

- [ ] 6. Add `DemoDataset.from_run_dirs` classmethod
      New classmethod: `from_run_dirs(cls, run_dirs: list[str | Path], target_object: str = "red_cube") -> DemoDataset`. Guard: `if not run_dirs: raise ValueError("No run directories provided")`. Call `cls.from_run_dir(d, target_object)` for each, concatenate via `np.concatenate([ds.observations.numpy() for ds in datasets])` and same for actions. Log: `logger.info("loaded_run_dirs: %d dirs, %d total samples", len(run_dirs), len(all_obs))`. Return `cls(all_obs, all_act)`. This resolves the naming discrepancy: `from_run_dir` (singular, existing) loads one dir; `from_run_dirs` (plural, new) loads multiple and always calls the singular version. The CLI will always call the plural form.
      Files: `/root/forge/forge_core/learned_policy.py`
      Verify: `python3 -m pytest tests/unit/test_learned_policy.py -v` — existing tests pass.

- [ ] 7. Add `load_training_config` and `_validate_training_config` functions
      Both in `forge_core/learned_policy.py`. `load_training_config(path: str | Path | None = None) -> dict[str, Any]`: when `path` is given, call `load_yaml(path)` (FileNotFoundError propagates). When `path` is None, try `PROJECT_ROOT / "config" / "training.yaml"`; if missing, `logger.debug("no default training config found")` and return `{}`. Call `_validate_training_config(config)` before returning. `_validate_training_config(config: dict) -> None`: for each nested key, use `config.get(section, {}).get(key)` to safely access. Validate: `training.epochs` int ≥ 1, `training.batch_size` int ≥ 1, `training.learning_rate` float > 0, `training.validation_split` float in [0.0, 0.5], `training.checkpoint_interval` int ≥ 1, `model.hidden_layers` list of ints each ≥ 1, `model.obs_dim` int ≥ 1, `model.act_dim` int ≥ 1. Raise `ValueError` with descriptive message on invalid. Unrecognized keys silently ignored.
      Files: `/root/forge/forge_core/learned_policy.py`
      Verify: `python3 -c "from forge_core.learned_policy import load_training_config; print(load_training_config())"` — loads config/training.yaml successfully.

- [ ] 8. Extend `train_bc` with validation, checkpointing, and improved logging
      Add parameters: `validation_split: float = 0.0`, `checkpoint_interval: int | None = None`, `checkpoint_dir: str | None = None`. Add `import time` at module top. Add `from torch.utils.data import random_split` to existing torch imports.
      
      **Input guards** (top of function, before any training):
      - `if len(dataset) == 0: raise ValueError("Dataset is empty")` (review fix #3)
      - `if not (0.0 <= validation_split <= 0.5): raise ValueError(...)`
      - `if checkpoint_interval is not None and checkpoint_interval < 1: raise ValueError(...)`
      
      **Validation split**: when `validation_split > 0`: `val_size = max(1, int(len(dataset) * validation_split))`, `train_size = len(dataset) - val_size`, guard `train_size < 1`. Use `random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))`. Create `val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)`. Train loader uses `train_subset`. When `validation_split == 0.0`: use entire dataset, `val_history = []`.
      
      **Logging**: record `start_time = time.monotonic()`. Log training start: `bc_training_start: %d samples (train=%d, val=%d), epochs=%d, lr=%s, batch=%d`. Every epoch (remove the `% 20` guard): log `bc_epoch %d/%d: loss=%.6f val_loss=%.6f` (or without val_loss if no val). Log training end: `bc_training_complete: final_loss=%.6f, val_loss=%.6f, model=%s, time=%.1fs`.
      
      **Per-epoch validation**: after each training epoch, if `val_loader`: set `network.eval()`, run full pass with `torch.no_grad()`, compute avg MSE, append to `val_history`, set `network.train()`.
      
      **Checkpointing**: if `checkpoint_interval is not None` and `((epoch + 1) % checkpoint_interval == 0 or epoch == epochs - 1)`: save to `checkpoint_dir / f"bc_policy_epoch_{epoch+1:03d}.pt"`. Default `checkpoint_dir = PROJECT_ROOT / "models"`, create with `mkdir(parents=True, exist_ok=True)`.
      
      **Return**: `TrainResult(..., val_history=val_history)`.
      
      Files: `/root/forge/forge_core/learned_policy.py`
      Verify: `python3 -m pytest tests/unit/test_learned_policy.py::TestBehavioralCloning -v` — existing test_train_bc passes (default validation_split=0.0, no checkpointing).

- [ ] 9. Update `scripts/forge_train.py` with config merge, list-models, and multi-run CLI
      
      **New imports**: add `load_training_config`, `PolicyNetwork` to imports from `forge_core.learned_policy`.
      
      **Review fix #1 — YAML key mapping**: define `_YAML_TO_FLAT` dict mapping nested YAML paths to flat code keys: `{"training.epochs": "epochs", "training.batch_size": "batch_size", "training.learning_rate": "lr", "training.validation_split": "validation_split", "training.checkpoint_interval": "checkpoint_interval", "model.hidden_layers": "hidden_layers", "model.obs_dim": "obs_dim", "model.act_dim": "act_dim", "data.target_object": "target_object", "data.demo_episodes": "episodes", "data.demo_policy": "demo_policy"}`. Define `_flatten_training_config(raw: dict) -> dict` that iterates the mapping, splits on `.` to get section/key, does `raw.get(section, {}).get(key)`, and includes non-None values in the result.
      
      **Review fix #4 — complete CODE_DEFAULTS**: `CODE_DEFAULTS = {"epochs": 100, "batch_size": 64, "lr": 1e-3, "validation_split": 0.0, "checkpoint_interval": None, "target_object": "red_cube", "episodes": 3, "demo_policy": "scripted_pick_and_place", "hidden_layers": [256, 256, 128], "obs_dim": 31, "act_dim": 8}`.
      
      **merge_training_config**: `merge_training_config(code_defaults: dict, yaml_config: dict, cli_overrides: dict) -> dict`: `result = {**code_defaults}`, overlay yaml (skip None), overlay cli (skip None). Return result.
      
      **list_models**: `list_models() -> None`: scan `PROJECT_ROOT / "models"` for `*.pt` files. Print table (filename, size in KB/MB, mtime as YYYY-MM-DD HH:MM:SS). If none found, print `"No trained models found in models/"`.
      
      **Argument parser updates**: add `--config` (str, default=None), `--list-models` (store_true). Change `--from-run` to `nargs="+"`. Change `--epochs`, `--batch-size`, `--lr` defaults to `None`.
      
      **Main flow update** (review fix #2):
      1. Parse args, setup logging.
      2. If `--list-models`: call `list_models()`, return.
      3. Load YAML: if `args.config`, call `load_training_config(args.config)` and flatten with `_flatten_training_config`. Else try `load_training_config()` (default path) and flatten.
      4. Build CLI overrides dict from non-None argparse values (`args.epochs`, `args.batch_size`, `args.lr`, etc.).
      5. Merge: `cfg = merge_training_config(CODE_DEFAULTS, yaml_flat, cli_overrides)`.
      6. Construct `PolicyNetwork(obs_dim=cfg["obs_dim"], act_dim=cfg["act_dim"], hidden=cfg["hidden_layers"])`.
      7. If `--from-run`: `DemoDataset.from_run_dirs(args.from_run, target_object=cfg["target_object"])`.
         Else: `collect_demos(task_name=args.task, policy=create_policy(cfg["demo_policy"]), n_episodes=cfg["episodes"], target_object=cfg["target_object"])`.
      8. Call `train_bc(dataset=dataset, epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"], save_path=args.save_path, network=network, validation_split=cfg["validation_split"], checkpoint_interval=cfg["checkpoint_interval"])`.
      9. Print `result.summary()` and eval hint.
      
      Files: `/root/forge/scripts/forge_train.py`
      Verify: `python3 scripts/forge_train.py --list-models` — prints "No trained models found" or table. `python3 scripts/forge_train.py --help` — shows all new flags.

- [ ] 10. Add all new test classes to `tests/unit/test_learned_policy.py`
      
      Do NOT modify any existing test class. Add new imports at the top: `load_training_config` from `forge_core.learned_policy`. For `TestConfigMerge`, add `scripts/` parent to sys.path and import `merge_training_config`, `_flatten_training_config`, `CODE_DEFAULTS` from `forge_train` (the module name without `.py`).
      
      **Shared fixture** — `synthetic_run_dir(tmp_path)`: creates `trajectory.npz` with T=50 synthetic data (joint_positions, joint_velocities, ee_positions, ee_orientations, actions, object_names, obj_red_cube_pos, obj_target_bin_pos). Uses `np.random.seed(42)`.
      
      **TestTrainingConfig** (7 tests):
      - `test_load_default_config`: `load_training_config()` returns dict with training/model/data sections
      - `test_load_custom_config`: write a custom YAML to tmp_path, load it, verify values
      - `test_load_missing_explicit_raises`: `load_training_config("/nonexistent.yaml")` raises FileNotFoundError
      - `test_load_missing_default_returns_empty`: mock/monkeypatch PROJECT_ROOT to a temp dir without training.yaml, returns {}
      - `test_invalid_epochs_raises`: config with `training: {epochs: -1}` raises ValueError
      - `test_invalid_validation_split_raises`: config with `training: {validation_split: 0.8}` raises ValueError
      - `test_invalid_lr_raises`: config with `training: {learning_rate: -0.01}` raises ValueError
      
      **TestConfigMerge** (5 tests):
      - `test_code_defaults_used_when_no_yaml_no_cli`: `merge_training_config(defaults, {}, {})` returns defaults
      - `test_yaml_overrides_defaults`: yaml sets epochs=50, result has epochs=50
      - `test_cli_overrides_yaml`: yaml sets epochs=50, cli sets epochs=25, result has epochs=25
      - `test_cli_overrides_defaults_without_yaml`: cli sets lr=0.0005, result has that lr
      - `test_partial_yaml_partial_cli`: yaml sets epochs+batch_size, cli sets lr, all three reflected
      
      **TestValidationMetrics** (7 tests):
      - `test_train_bc_with_validation`: train_bc(ds, epochs=5, validation_split=0.2), result.val_history length 5, all > 0
      - `test_train_bc_no_validation`: train_bc(ds, epochs=5), result.val_history == []
      - `test_summary_includes_val_loss`: result with val_history, summary() contains "val_loss="
      - `test_summary_without_val_loss`: result without val_history, summary() does not contain "val_loss="
      - `test_validation_split_too_large_raises`: 3-sample dataset with validation_split=0.2 raises ValueError
      - `test_validation_split_out_of_range_raises`: validation_split=0.8 raises ValueError
      - `test_small_dataset_gets_at_least_one_val_sample`: 10-sample dataset, validation_split=0.05, succeeds
      
      **TestCheckpointing** (6 tests):
      - `test_checkpoints_created`: epochs=10, checkpoint_interval=5, creates epoch_005.pt and epoch_010.pt
      - `test_final_always_checkpointed`: epochs=7, checkpoint_interval=5, creates epoch_005.pt and epoch_007.pt
      - `test_checkpoint_loadable`: load a checkpoint with LearnedPolicy(model_path=...), act produces correct shape
      - `test_no_checkpoints_by_default`: epochs=5, no checkpoint_interval, no epoch-numbered files
      - `test_checkpoint_interval_zero_raises`: checkpoint_interval=0 raises ValueError
      - `test_checkpoint_interval_negative_raises`: checkpoint_interval=-1 raises ValueError
      
      **TestMultiRunDataset** (4 tests):
      - `test_from_run_dirs_merges`: two synthetic_run_dirs, from_run_dirs returns len = sum of individual lens
      - `test_from_run_dirs_empty_raises`: from_run_dirs([]) raises ValueError
      - `test_from_run_dir_missing_object_warns`: run dir without obj_red_cube_pos, caplog has "missing_object_data"
      - `test_from_run_dirs_logs_aggregate_stats`: after loading 2 dirs, caplog contains "loaded_run_dirs: 2 dirs"
      
      **TestObsMode** (3 tests):
      - `test_default_obs_mode`: LearnedPolicy()._obs_mode == "state"
      - `test_invalid_obs_mode_raises`: LearnedPolicy(obs_mode="lidar") raises ValueError
      - `test_vision_obs_mode_not_implemented`: LearnedPolicy(obs_mode="vision") raises NotImplementedError
      
      Files: `/root/forge/tests/unit/test_learned_policy.py`
      Verify: `python3 -m pytest tests/unit/test_learned_policy.py -v` — all tests pass. Then `python3 -m pytest tests/ -v` — 117 existing + ~32 new tests pass.

---

## FEAT-002: Documentation Updates

- [ ] 11. Add Phase 6 section to `docs/architecture.md`
      Insert after the Phase 5 section, before "Design Principles". Cover: module overview, PolicyNetwork architecture (31→[256,256,128]→8, Tanh, configurable), training pipeline (collect_demos / from_run_dir → train_bc → .pt), YAML config (config/training.yaml), integration (create_policy("learned")), extension point (obs_mode), CLI (forge_train.py).
      Files: `/root/forge/docs/architecture.md`
      Verify: `grep -q "Phase 6" docs/architecture.md` confirms the section exists.

- [ ] 12. Add ML dependencies and training CLI section to `docs/setup.md`
      Add "Machine Learning Dependencies" section after Step 2 with `pip install -e ".[ml]"`. Add "Training a Learned Policy" section with CLI examples: basic training, from recorded runs, with YAML config, list models, evaluating trained model. Note shell glob behavior on Windows.
      Files: `/root/forge/docs/setup.md`
      Verify: `grep -q "ml" docs/setup.md` confirms ML section exists.

- [ ] 13. Update `README.md` for Phase 6
      Change heading to "Phase 6 — Learned Policy (PyTorch)". Update description paragraph. Add `learned_policy.py`, `forge_train.py`, `config/training.yaml` to project structure. Phase 6 row: "Done". Add training CLI subsection. Add "Learned policy" to "What's included" list. Update test count.
      Files: `/root/forge/README.md`
      Verify: `grep "Phase 6" README.md && grep "Done.*Learned" README.md` — both match.

---

## Review Fix Traceability

| Review Finding | ID | Where Applied |
|---|---|---|
| MEDIUM: YAML-to-flat key mapping | #1 | Step 9 — `_YAML_TO_FLAT` + `_flatten_training_config` in forge_train.py |
| MEDIUM: model/data config plumbing | #2 | Step 9 — Main flow constructs PolicyNetwork from config, passes target_object |
| NIT: empty dataset guard | #3 | Step 8 — `if len(dataset) == 0` at top of train_bc |
| NIT: complete CODE_DEFAULTS | #4 | Step 9 — full CODE_DEFAULTS dict with all 11 keys |

## Naming Discrepancy Resolution

`from_run_dir` (singular, existing) — loads one directory, unchanged.
`from_run_dirs` (plural, new, Step 6) — loads multiple directories by calling `from_run_dir` for each.
CLI (Step 9) always calls `from_run_dirs` regardless of argument count — no branching needed.
