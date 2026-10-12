# Phase 6 — Learned Policy (PyTorch): Technical Design

## Overview

Phase 6 completes and hardens the existing PyTorch-based learned policy subsystem in Forge. The core module (`forge_core/learned_policy.py`), unit tests (`tests/unit/test_learned_policy.py`), and training CLI (`scripts/forge_train.py`) already exist with working implementations of `LearnedPolicy`, `PolicyNetwork`, `DemoDataset`, `train_bc`, and `collect_demos`. This design adds five capabilities on top: YAML-driven training configuration, validation metrics with train/val split, model checkpointing, multi-run dataset loading, and a model listing subcommand. It also adds the `[ml]` optional dependency to `pyproject.toml`, extends the test suite, and updates documentation.

Every existing API signature is preserved exactly. The `TrainResult` dataclass gains one new field (`val_history`). The `train_bc` function gains new keyword arguments with defaults that reproduce the current behavior. No existing test is modified.

---

## 1. YAML Training Configuration

### Config File: `config/training.yaml`

A new YAML file at `config/training.yaml` provides default training hyperparameters. The structure uses nested keys consistent with `config/default.yaml` and `tasks/*.yaml`:

```yaml
# Forge training configuration — behavioral cloning
training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001
  validation_split: 0.1
  checkpoint_interval: 20

model:
  hidden_layers: [256, 256, 128]
  obs_dim: 31
  act_dim: 8

data:
  target_object: "red_cube"
  demo_episodes: 3
  demo_policy: "scripted_pick_and_place"
```

### Loading Function: `load_training_config`

A new function in `forge_core/learned_policy.py`:

```python
def load_training_config(path: str | Path | None = None) -> dict[str, Any]:
```

Uses `forge_core.config.load_yaml` to read the file. When `path` is `None`, loads the default at `PROJECT_ROOT / "config" / "training.yaml"`. Returns the raw dict; the caller is responsible for extracting keys. If the file does not exist and `path` was explicitly provided, raises `FileNotFoundError`. If `path` is `None` and the default file is missing, returns an empty dict (graceful fallback).

### Validation Rules

| Key | Type | Required | Range | On Invalid |
|-----|------|----------|-------|------------|
| `training.epochs` | int | no | ≥ 1 | `ValueError` |
| `training.batch_size` | int | no | ≥ 1 | `ValueError` |
| `training.learning_rate` | float | no | > 0 | `ValueError` |
| `training.validation_split` | float | no | [0.0, 0.5] | `ValueError` |
| `training.checkpoint_interval` | int | no | ≥ 1 | `ValueError` |
| `model.hidden_layers` | list[int] | no | each ≥ 1 | `ValueError` |
| `model.obs_dim` | int | no | ≥ 1 | `ValueError` |
| `model.act_dim` | int | no | ≥ 1 | `ValueError` |
| `data.*` | str/int | no | — | `ValueError` |

Validation happens in a helper `_validate_training_config(config: dict) -> None` called by `load_training_config`. This keeps validation co-located with loading rather than scattered across the CLI. Unrecognized keys are silently ignored (forward compatibility).

Each top-level section (`training`, `model`, `data`) is optional. The validator accesses nested keys via `config.get(section, {}).get(key)` so that a missing section is treated as all-defaults rather than raising a `KeyError`. For example, a YAML file containing only a `model:` block is valid — all `training.*` and `data.*` keys take their code defaults.

### CLI Integration

`forge_train.py` gains a `--config <path>` flag. The merge order is: defaults from code → YAML config → CLI arguments. Concretely:

1. Start with hardcoded defaults (current values: `epochs=100`, `batch_size=64`, `lr=1e-3`).
2. If `--config` is provided, call `load_training_config(path)` and overlay its values.
3. CLI arguments (`--epochs`, `--batch-size`, `--lr`) override when explicitly passed. argparse's `default=None` sentinel distinguishes "user passed a value" from "use the fallback".

The merge logic is extracted into a standalone, testable helper function:

```python
def merge_training_config(
    code_defaults: dict[str, Any],
    yaml_config: dict[str, Any],
    cli_overrides: dict[str, Any],
) -> dict[str, Any]:
```

`code_defaults` contains the hardcoded fallbacks (`{"epochs": 100, "batch_size": 64, "lr": 1e-3, ...}`). `yaml_config` is the flattened dict from `load_training_config`. `cli_overrides` contains only the keys the user explicitly passed (non-`None` argparse values). The function returns the merged flat dict. The merge is a simple left-to-right dict update: `result = {**code_defaults}`, then overlay YAML values for any keys present, then overlay CLI overrides for any keys present. This function lives in `forge_train.py` (not in `forge_core/`) since it is CLI-specific.

---

## 2. Validation Metrics

### Train/Validation Split

`train_bc` gains two new keyword arguments:

```python
def train_bc(
    dataset: DemoDataset,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    save_path: str | None = None,
    network: PolicyNetwork | None = None,
    validation_split: float = 0.0,       # NEW
    checkpoint_interval: int | None = None,  # NEW (§3)
    checkpoint_dir: str | None = None,       # NEW (§3)
) -> TrainResult:
```

Input validation for `validation_split` runs at the top of `train_bc`, before any data splitting or training:

```python
if not (0.0 <= validation_split <= 0.5):
    raise ValueError(f"validation_split must be in [0.0, 0.5], got {validation_split}")
```

When `validation_split > 0`, the dataset is split into training and validation subsets using `torch.utils.data.random_split`. Split sizes are computed as:

```python
val_size = max(1, int(len(dataset) * validation_split))  # at least 1 when split > 0
train_size = len(dataset) - val_size
if train_size < 1:
    raise ValueError(
        f"Dataset too small for requested validation split: "
        f"{len(dataset)} samples with validation_split={validation_split} "
        f"leaves {train_size} training samples"
    )
```

The `max(1, ...)` floor guarantees at least 1 validation sample whenever validation is requested, preventing a zero-division edge case in the validation loss averaging loop. The subsequent `train_size < 1` guard covers the other direction. The split uses a fixed generator seed (`torch.Generator().manual_seed(42)`) for reproducibility. The training DataLoader draws from the train subset; a separate validation DataLoader (no shuffle, same batch size) draws from the val subset.

When `validation_split == 0.0` (the default), no split occurs and the entire dataset is used for training. `val_history` will be an empty list. This preserves backward compatibility: existing callers that do not pass `validation_split` get identical behavior to today.

### Per-Epoch Validation

At the end of each epoch, after the training pass, the network is set to `eval()` mode and a full pass over the validation DataLoader computes the average MSE loss (no gradient). The network is then set back to `train()` mode. Both training loss and validation loss are appended to their respective history lists.

### TrainResult Extension

```python
@dataclass
class TrainResult:
    epochs: int
    final_loss: float
    model_path: str
    history: list[float]
    val_history: list[float] = field(default_factory=list)  # NEW

    def summary(self) -> str:
        s = (f"Training complete: {self.epochs} epochs, "
             f"final_loss={self.final_loss:.6f}")
        if self.val_history:
            s += f", val_loss={self.val_history[-1]:.6f}"
        s += f", saved={self.model_path}"
        return s
```

The new `val_history` field has a default of `[]`, so existing code that constructs `TrainResult` with positional/keyword args for the original four fields continues to work. The `summary()` method conditionally includes validation loss only when `val_history` is non-empty.

### Logging

Using the existing `logger = logging.getLogger(__name__)` pattern:

- **Training start**: `logger.info("bc_training_start: %d samples (train=%d, val=%d), epochs=%d, lr=%s, batch=%d", ...)` — dataset size, split sizes, config summary.
- **Per-epoch**: the epoch log is emitted every epoch (the current every-20th-epoch modulo guard is removed). Format: `logger.info("bc_epoch %d/%d: loss=%.6f val_loss=%.6f", ...)`. When no validation split is active, the val_loss portion is omitted from the format string: `logger.info("bc_epoch %d/%d: loss=%.6f", ...)`. This satisfies FR-4's "Per-epoch" requirement. At 100 epochs the output is 100 INFO lines, which is manageable and makes validation loss tracking continuously visible.
- **Training end**: `logger.info("bc_training_complete: final_loss=%.6f, val_loss=%.6f, model=%s, time=%.1fs", ...)` — final losses, model path, wall-clock training time (measured with `time.monotonic()`).

---

## 3. Model Checkpointing

### Checkpoint Saving in `train_bc`

When `checkpoint_interval` is not `None`, `train_bc` saves intermediate checkpoints during training. Checkpoints are saved at the end of every `checkpoint_interval`-th epoch (i.e., when `(epoch + 1) % checkpoint_interval == 0`). The final epoch always saves a checkpoint regardless of interval alignment.

Input validation: at the top of `train_bc`, before the training loop, the following guard runs:

```python
if checkpoint_interval is not None and checkpoint_interval < 1:
    raise ValueError(f"checkpoint_interval must be >= 1, got {checkpoint_interval}")
```

This prevents `ZeroDivisionError` from `% 0` and ensures negative values are rejected. The guard runs even when the YAML config validator has already checked the value, because `train_bc` is a public function callable directly without going through YAML/CLI.

The checkpoint directory defaults to `PROJECT_ROOT / "models"` but can be overridden via `checkpoint_dir`. The directory is created with `mkdir(parents=True, exist_ok=True)`.

Checkpoint filename format: `bc_policy_epoch_{epoch+1:03d}.pt`. For example, with `checkpoint_interval=20` and `epochs=100`:
- `models/bc_policy_epoch_020.pt`
- `models/bc_policy_epoch_040.pt`
- `models/bc_policy_epoch_060.pt`
- `models/bc_policy_epoch_080.pt`
- `models/bc_policy_epoch_100.pt`

The final model is saved twice: once as the checkpoint and once at `save_path` (which defaults to `models/bc_policy.pt`).

Each checkpoint is a bare `state_dict` saved via `torch.save(network.state_dict(), path)` — same format as the existing final model save. This keeps `LearnedPolicy._load_model` compatible with any checkpoint file.

### Error Handling

If `torch.save` fails mid-write (e.g., disk full, permission error), the target file may be corrupt. The exception propagates as-is. A failure to checkpoint is treated as fatal — silently continuing training and losing the model is worse than crashing. The stack trace from `OSError` or `RuntimeError` is sufficient for diagnosis.

---

## 4. Multi-Run Directory Loading

### New Class Method: `DemoDataset.from_run_dirs`

```python
@classmethod
def from_run_dirs(cls, run_dirs: list[str | Path],
                  target_object: str = "red_cube") -> DemoDataset:
```

Iterates over each path, calls `from_run_dir` for each, and concatenates the resulting observations and actions. Since `DemoDataset.__init__` converts inputs to `torch.Tensor` immediately, the sub-datasets store `.observations` and `.actions` as tensors, not numpy arrays. The concatenation must account for this by converting back to numpy before reconstructing:

```python
@classmethod
def from_run_dirs(cls, run_dirs: list[str | Path],
                  target_object: str = "red_cube") -> DemoDataset:
    if not run_dirs:
        raise ValueError("No run directories provided")
    datasets = [cls.from_run_dir(d, target_object) for d in run_dirs]
    all_obs = np.concatenate([ds.observations.numpy() for ds in datasets])
    all_act = np.concatenate([ds.actions.numpy() for ds in datasets])
    logger.info("loaded_run_dirs: %d dirs, %d total samples", len(run_dirs), len(all_obs))
    return cls(all_obs, all_act)
```

This numpy→tensor→numpy→tensor round-trip is negligible for demo-sized data (typically thousands to tens of thousands of rows). It is simpler and safer than using `torch.cat` with `__new__` to bypass the constructor.

If `run_dirs` is empty, raises `ValueError("No run directories provided")`. If a specific run directory is missing `trajectory.npz`, the `np.load` call inside `from_run_dir` raises `FileNotFoundError` — this propagates as-is, since a user-specified run that doesn't exist is a caller error.

When a run directory's data is missing optional arrays (target object positions, goal positions), the existing fallback-to-zeros behavior in `from_run_dir` handles it. A `logger.warning` is added to `from_run_dir` when a fallback occurs:

```python
if f"obj_{target_object}_pos" not in data:
    logger.warning("missing_object_data: %s not found in %s, using zeros",
                   target_object, run_dir)
```

Same for `obj_target_bin_pos`.

### CLI Changes

The `--from-run` argument in `forge_train.py` changes from `type=str` to `nargs="+"`:

```python
parser.add_argument("--from-run", type=str, nargs="+", default=None,
                    help="Train from recorded run directory(ies)")
```

This accepts one or more paths. The CLI handler always calls `DemoDataset.from_run_dirs(args.from_run, ...)` regardless of how many paths are in the list — `from_run_dirs` handles single-element lists identically to multi-element lists, so no branching is needed.

Glob patterns in `--from-run` are expanded by the shell. This is standard Unix/bash/zsh behavior and requires no special code. On Windows, use explicit paths or a Unix-like shell (WSL). This limitation is noted in `docs/setup.md`.

---

## 5. CLI Design: `scripts/forge_train.py`

### Full Argument Specification

The existing `forge_train.py` is extended (not rewritten). New arguments are added to the existing `argparse.ArgumentParser`:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--config` | str | None | Path to YAML training config |
| `--task` | str | "pick_red_cube" | Task for demo collection (existing) |
| `--episodes` | int | 3 | Demo episodes to collect (existing) |
| `--from-run` | str (nargs=+) | None | Run dir(s) to train from (extended) |
| `--epochs` | int | None* | Training epochs |
| `--batch-size` | int | None* | Batch size |
| `--lr` | float | None* | Learning rate |
| `--save-path` | str | None | Model output path (existing) |
| `--demo-policy` | str | "scripted_pick_and_place" | Policy for demo collection (existing) |
| `--list-models` | action=store_true | False | List saved models |
| `--verbose` / `-v` | action=store_true | False | Debug logging (existing) |

*Note: `--epochs`, `--batch-size`, `--lr` change from having hardcoded argparse defaults to `default=None`. This lets the merge logic distinguish "user specified" from "use config/code default". The code defaults (100, 64, 1e-3) and YAML config values fill in when these are None.

### `--list-models` Subcommand

When `--list-models` is passed, the CLI:

1. Scans `PROJECT_ROOT / "models"` for `*.pt` files.
2. For each file, collects: filename, file size (human-readable: KB/MB), and `mtime` formatted as `YYYY-MM-DD HH:MM:SS`.
3. Prints a table to stdout.
4. If no `.pt` files are found, prints `"No trained models found in models/"`.
5. Exits immediately (no training).

Implementation: a standalone function `list_models()` in `forge_train.py`, called before the training logic when `args.list_models` is True.

### Main Flow (Updated)

```
1. Parse args
2. Setup logging
3. If --list-models: list_models() then return
4. Load config: merge code defaults ← YAML config ← CLI args (via merge_training_config)
5. If --from-run: load DemoDataset via from_run_dirs(args.from_run, ...)
   Else: collect demos using --demo-policy
6. Call train_bc(...) with merged params
7. Print result.summary()
8. Print eval command hint
```

---

## 6. Dependency Plan: PyTorch under `[ml]` Extra

### `pyproject.toml` Change

Add a new optional dependency group:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-timeout>=2.0",
]
ml = [
    "torch>=2.0",
]
```

This allows `pip install -e ".[ml]"` to pull in PyTorch. The rest of Forge remains functional without it — the lazy import in `create_policy("learned", ...)` already handles the `ImportError` case.

No `torch` import appears at the top level of any module outside `forge_core/learned_policy.py`. The existing `create_policy` function uses a conditional import inside the `if name == "learned":` branch, which is the correct pattern and requires no change.

### Test Environment

The CI/test environment already has PyTorch installed (all existing learned policy tests import torch at module level). The `[ml]` extra formalizes this but does not change what's installed in the current dev environment.

---

## 7. LearnedPolicy Class — Preserving the Contract

The `LearnedPolicy` class is already fully implemented and tested. No changes to its public API. The only internal changes are:

1. **`_load_model` enhancement**: Currently loads a bare `state_dict`. This is the only supported format. No change needed — the existing implementation matches the checkpoint format from §3.

2. **Perception extension point (future hook)**: The class gains a class-level constant and constructor parameter for future use:

```python
class LearnedPolicy(Policy):
    # Observation mode: "state" (MLP, Phase 6) or "vision" (CNN, future)
    OBS_MODES = ("state", "vision")

    def __init__(self, model_path: str | None = None,
                 target_object: str = "red_cube",
                 max_steps: int = 1500,
                 obs_mode: str = "state",   # NEW
                 **kwargs):
        if obs_mode not in self.OBS_MODES:
            raise ValueError(f"Unknown obs_mode '{obs_mode}'. Must be one of {self.OBS_MODES}")
        if obs_mode == "vision":
            raise NotImplementedError(
                "Vision-based policy input is planned for a future phase. "
                "Use obs_mode='state' for MLP policies."
            )
        self._obs_mode = obs_mode
        # ... rest unchanged
```

This is a non-breaking addition: `obs_mode` defaults to `"state"` and the `**kwargs` pattern already absorbs unknown arguments. The `"vision"` path raises `NotImplementedError` immediately, making the extension point explicit without dead code.

### Network Architecture Summary

The `PolicyNetwork` MLP is unchanged:
- Input: 31-dim state vector (9 joint pos + 9 joint vel + 3 EE pos + 4 EE quat + 3 target pos + 3 goal pos)
- Hidden layers: [256, 256, 128] with Tanh activations (configurable via `hidden` param)
- Output: 8-dim action vector (7 arm joints + 1 finger actuator)
- No output activation (simulation handles actuator limits)

### Save/Load Format

Models are saved as bare PyTorch `state_dict` dicts via `torch.save`. Loaded via `torch.load(path, map_location="cpu", weights_only=True)` then `load_state_dict`. This is the standard PyTorch pattern and is already implemented. The `weights_only=True` flag is a security measure that prevents arbitrary code execution from untrusted model files — this is already in place.

---

## 8. Integration Points

### Runner Integration (No Changes Required)

`run_task_by_name()` already supports learned policies through the `policy` argument. The caller creates a policy with `create_policy("learned", model_path=...)` and passes it to `run_task_by_name`. The `perception=True` flag wraps any policy (including learned) in `PerceptionPolicy`. No changes to `runner.py`.

### Evaluation CLI Integration (No Changes Required)

`forge_eval.py` already passes `--model-path` through to `create_policy`:

```python
if args.model_path:
    policy_kwargs["model_path"] = args.model_path
policy = create_policy(args.policy, **policy_kwargs)
```

This works for `--policy learned --model-path models/bc_policy.pt`. No changes to `forge_eval.py`.

### Perception Integration (Extension Seam Only)

The future vision-input path would work as follows (not implemented in Phase 6):
1. `LearnedPolicy(obs_mode="vision")` would instantiate a CNN-based `VisionPolicyNetwork` instead of `PolicyNetwork`.
2. `act()` would call the `PerceptionPipeline` (or receive an image from `PerceptionPolicy`) and concatenate a feature vector with the state vector.
3. The `PerceptionPolicy` wrapper in `runner.py` already runs perception before `inner.act(sim)`, so the learned policy could access `sim` plus the latest image.

The seam is the `obs_mode` parameter and the `NotImplementedError` guard. The architecture doc will describe this extension point.

---

## 9. File Modification Plan

| File | Action | What Changes |
|------|--------|-------------|
| `forge_core/learned_policy.py` | Modify | Add `load_training_config`, `_validate_training_config`, `DemoDataset.from_run_dirs`, `val_history` to `TrainResult`, `validation_split`/`checkpoint_interval`/`checkpoint_dir` params to `train_bc`, `obs_mode` param to `LearnedPolicy`, warning logs in `from_run_dir` |
| `scripts/forge_train.py` | Modify | Add `--config`, `--list-models`, `nargs="+"` for `--from-run`, config merge logic (`merge_training_config` helper), `list_models()` function, `default=None` for `--epochs`/`--batch-size`/`--lr` |
| `config/training.yaml` | Create | Default training config YAML |
| `pyproject.toml` | Modify | Add `ml = ["torch>=2.0"]` under `[project.optional-dependencies]` |
| `tests/unit/test_learned_policy.py` | Modify | Add new test classes/methods for config loading, validation metrics, checkpointing, multi-run loading, config merge |
| `docs/architecture.md` | Modify | Add Phase 6 section |
| `docs/setup.md` | Modify | Add PyTorch installation and training CLI usage |
| `README.md` | Modify | Update phase roadmap, project structure, add training CLI docs |

Files NOT modified: `forge_core/policy.py`, `forge_core/runner.py`, `forge_core/perception.py`, `forge_core/recorder.py`, `forge_core/replay.py`, `forge_core/config.py`, `scripts/forge_eval.py`, `config/default.yaml`.

---

## 10. Error Handling

### `load_training_config`

| Condition | Recoverable | Behavior | Logged |
|-----------|-------------|----------|--------|
| Explicit path not found | No (fatal) | Raises `FileNotFoundError` | No (caller catches) |
| Default path not found | Yes | Returns `{}` | `logger.debug("no default training config found")` |
| YAML parse error | No (fatal) | Raises `yaml.YAMLError` (from `load_yaml`) | No (caller catches) |
| Validation failure | No (fatal) | Raises `ValueError` with descriptive message | No (caller catches) |

### `train_bc` (extended)

| Condition | Recoverable | Behavior | Logged |
|-----------|-------------|----------|--------|
| Empty dataset | No (fatal) | Raises `ValueError("Dataset is empty")` | No |
| `validation_split` out of [0.0, 0.5] | No (fatal) | Raises `ValueError` | No |
| `validation_split > 0` but dataset too small for ≥1 train + ≥1 val | No (fatal) | Raises `ValueError("Dataset too small for requested validation split")` | No |
| `checkpoint_interval` < 1 | No (fatal) | Raises `ValueError(f"checkpoint_interval must be >= 1, got {checkpoint_interval}")` | No |
| Val split leaves 0 training samples | No (fatal) | Raises `ValueError("Validation split too large")` | No |
| Checkpoint save fails (disk full) | No (fatal) | `torch.save` raises `OSError`, propagates. Target file may be corrupt. | No (stack trace is sufficient) |
| Final model save fails | No (fatal) | Same as above | No |

### `DemoDataset.from_run_dirs`

| Condition | Recoverable | Behavior | Logged |
|-----------|-------------|----------|--------|
| Empty list | No (fatal) | Raises `ValueError` | No |
| Run dir missing | No (fatal) | `FileNotFoundError` from `np.load` | No |
| Missing target object data | Yes | Falls back to zeros | `logger.warning` |

### `DemoDataset.from_run_dirs` (aggregate)

| Condition | Recoverable | Behavior | Logged |
|-----------|-------------|----------|--------|
| All dirs loaded successfully | N/A | Returns merged dataset | `logger.info("loaded_run_dirs: %d dirs, %d total samples", ...)` |

### `LearnedPolicy.__init__` (obs_mode)

| Condition | Recoverable | Behavior | Logged |
|-----------|-------------|----------|--------|
| Unknown `obs_mode` | No (fatal) | Raises `ValueError` | No |
| `obs_mode="vision"` | No (not implemented) | Raises `NotImplementedError` | No |

---

## 11. Test Plan

All new tests are added to `tests/unit/test_learned_policy.py`. Existing test classes (`TestObservation`, `TestPolicyNetwork`, `TestLearnedPolicy`, `TestDemoDataset`, `TestBehavioralCloning`) are not modified. New test classes are appended.

### TestTrainingConfig (new class)

| Test | What It Verifies |
|------|-----------------|
| `test_load_default_config` | `load_training_config()` returns a dict with expected keys when `config/training.yaml` exists |
| `test_load_custom_config` | `load_training_config(tmp_path / "custom.yaml")` loads values from a custom file |
| `test_load_missing_explicit_raises` | `load_training_config("/nonexistent.yaml")` raises `FileNotFoundError` |
| `test_load_missing_default_returns_empty` | When default config is absent, returns `{}` (requires temp directory manipulation or mocking `PROJECT_ROOT`) |
| `test_invalid_epochs_raises` | Config with `epochs: -1` raises `ValueError` |
| `test_invalid_validation_split_raises` | Config with `validation_split: 0.8` raises `ValueError` |
| `test_invalid_lr_raises` | Config with `learning_rate: -0.01` raises `ValueError` |

### TestConfigMerge (new class)

| Test | What It Verifies |
|------|-----------------|
| `test_code_defaults_used_when_no_yaml_no_cli` | `merge_training_config(defaults, {}, {})` returns the code defaults verbatim |
| `test_yaml_overrides_defaults` | `merge_training_config(defaults, {"epochs": 50}, {})` returns `epochs=50` with all other defaults |
| `test_cli_overrides_yaml` | `merge_training_config(defaults, {"epochs": 50}, {"epochs": 25})` returns `epochs=25` |
| `test_cli_overrides_defaults_without_yaml` | `merge_training_config(defaults, {}, {"lr": 0.0005})` returns `lr=0.0005` |
| `test_partial_yaml_partial_cli` | YAML sets `epochs` and `batch_size`, CLI sets only `lr` — all three are reflected, remaining keys are code defaults |

### TestValidationMetrics (new class)

| Test | What It Verifies |
|------|-----------------|
| `test_train_bc_with_validation` | `train_bc(ds, epochs=5, validation_split=0.2)` returns `result.val_history` of length 5, all > 0 |
| `test_train_bc_no_validation` | `train_bc(ds, epochs=5)` returns `result.val_history == []` |
| `test_summary_includes_val_loss` | `result.summary()` contains `"val_loss="` when val_history is non-empty |
| `test_summary_without_val_loss` | `result.summary()` does not contain `"val_loss="` when val_history is empty |
| `test_validation_split_too_large_raises` | `train_bc(ds_small, validation_split=0.2)` on a 3-sample dataset raises `ValueError` |
| `test_validation_split_out_of_range_raises` | `train_bc(ds, validation_split=0.8)` raises `ValueError` |
| `test_small_dataset_gets_at_least_one_val_sample` | `train_bc(ds_10, epochs=2, validation_split=0.05)` succeeds with `val_size=1` (10 * 0.05 = 0.5 → max(1, 0) = 1) |

### TestCheckpointing (new class)

| Test | What It Verifies |
|------|-----------------|
| `test_checkpoints_created` | `train_bc(ds, epochs=10, checkpoint_interval=5, checkpoint_dir=tmp)` creates `bc_policy_epoch_005.pt` and `bc_policy_epoch_010.pt` |
| `test_final_always_checkpointed` | `train_bc(ds, epochs=7, checkpoint_interval=5, checkpoint_dir=tmp)` creates `bc_policy_epoch_005.pt` and `bc_policy_epoch_007.pt` |
| `test_checkpoint_loadable` | A checkpoint file can be loaded by `LearnedPolicy(model_path=...)` and produces correct action shape |
| `test_no_checkpoints_by_default` | `train_bc(ds, epochs=5)` with no `checkpoint_interval` does not create epoch-numbered files |
| `test_checkpoint_interval_zero_raises` | `train_bc(ds, checkpoint_interval=0)` raises `ValueError` |
| `test_checkpoint_interval_negative_raises` | `train_bc(ds, checkpoint_interval=-1)` raises `ValueError` |

### TestMultiRunDataset (new class)

| Test | What It Verifies |
|------|-----------------|
| `test_from_run_dirs_merges` | Two synthetic run dirs → `from_run_dirs` returns a dataset whose length equals sum of individual datasets |
| `test_from_run_dirs_empty_raises` | `from_run_dirs([])` raises `ValueError` |
| `test_from_run_dir_missing_object_warns` | A run dir without target object data logs a warning and falls back to zeros (use `caplog` fixture) |
| `test_from_run_dirs_logs_aggregate_stats` | After loading 2 dirs, `caplog` contains `"loaded_run_dirs: 2 dirs"` log message |

### TestObsMode (new class)

| Test | What It Verifies |
|------|-----------------|
| `test_default_obs_mode` | `LearnedPolicy()._obs_mode == "state"` |
| `test_invalid_obs_mode_raises` | `LearnedPolicy(obs_mode="lidar")` raises `ValueError` |
| `test_vision_obs_mode_not_implemented` | `LearnedPolicy(obs_mode="vision")` raises `NotImplementedError` |

### Determinism

All new tests that involve randomness use fixed seeds:
- `torch.manual_seed(42)` and `np.random.seed(42)` in a fixture or at the start of each test that creates synthetic data.
- The `random_split` generator in `train_bc` uses a fixed seed (42) so the same dataset always produces the same split.

### Synthetic Run Directory Fixture

A shared `pytest` fixture creates a temporary run directory with a valid `trajectory.npz` containing synthetic data matching the recorder format:

```python
@pytest.fixture
def synthetic_run_dir(tmp_path):
    T = 50
    np.random.seed(42)
    data = {
        "joint_positions": np.random.randn(T, 9).astype(np.float32),
        "joint_velocities": np.random.randn(T, 9).astype(np.float32),
        "ee_positions": np.random.randn(T, 3).astype(np.float32),
        "ee_orientations": np.random.randn(T, 4).astype(np.float32),
        "actions": np.random.randn(T, 8).astype(np.float32),
        "object_names": np.array(["red_cube", "target_bin"]),
        "obj_red_cube_pos": np.random.randn(T, 3).astype(np.float32),
        "obj_target_bin_pos": np.random.randn(T, 3).astype(np.float32),
    }
    np.savez_compressed(tmp_path / "trajectory.npz", **data)
    return tmp_path
```

### What Can Be Unit Tested vs. Integration Tested

**Unit testable** (all in `tests/unit/test_learned_policy.py`, no simulation required):
- Config loading and validation (file I/O only)
- Config merge logic (pure function, `TestConfigMerge`)
- `TrainResult` field additions and `summary()` output
- `DemoDataset.from_run_dirs` with synthetic npz files
- `train_bc` with synthetic data (small epochs, no simulation)
- Checkpoint file creation and `checkpoint_interval` validation
- Validation loss computation and split size edge cases
- `obs_mode` validation
- `list_models()` output formatting

**Integration testable** (requires MuJoCo simulation, already covered by existing `sim` fixture):
- `extract_observation` shape and content (existing tests)
- `LearnedPolicy.act()` producing correct action shape (existing tests)
- `collect_demos` running a policy through simulation (existing test)
- End-to-end training CLI (manual testing / AC-4 through AC-6)

---

## 12. Documentation Updates

### `docs/architecture.md`

Add a "Phase 6 — Learned Policy (PyTorch)" section after Phase 5. Content:

- Module overview: `forge_core/learned_policy.py` provides behavior-cloning training and a drop-in `LearnedPolicy` that implements the `Policy` ABC.
- Network architecture: MLP with configurable hidden layers, Tanh activations, state-only input (31-dim → 8-dim).
- Training pipeline: `collect_demos` or `DemoDataset.from_run_dir` → `train_bc` → `PolicyNetwork` state dict `.pt` file.
- YAML config: `config/training.yaml` with training/model/data sections.
- Integration: `create_policy("learned", model_path=...)` for runner and eval CLI.
- Extension point: `obs_mode` parameter for future vision-based input.
- CLI: `forge_train.py` with collect/train/list-models workflows.

### `docs/setup.md`

Add a section "Machine Learning Dependencies" that covers:
- `pip install -e ".[ml]"` to install PyTorch
- Basic training CLI usage examples
- How to evaluate a trained model with `forge_eval.py`
- Note: glob patterns in `--from-run` are expanded by the shell; on Windows, use explicit paths or a Unix-like shell (WSL).

### `README.md`

- Update the Phase Roadmap to show Phase 6 as "Done".
- Update the "Current Status" line.
- Add `forge_core/learned_policy.py` and `scripts/forge_train.py` to the project structure section.
- Add a training CLI usage subsection.

---

## Technology Stack (Locked)

| Component | Technology | Version |
|-----------|------------|---------|
| Neural network | PyTorch | ≥ 2.0 |
| Network architecture | `nn.Sequential` MLP | — |
| Loss function | `nn.MSELoss` | — |
| Optimizer | `optim.Adam` | — |
| Data loading | `torch.utils.data.DataLoader` | — |
| Model format | PyTorch `state_dict` (`.pt`) | — |
| Config format | YAML via `PyYAML` | ≥ 6.0 (existing) |
| Config loader | `forge_core.config.load_yaml` | — |
| Test framework | pytest | ≥ 7.0 (existing) |

---

## Appendix: Design Review Responses (Round 1)

These responses address findings from the first design review.

### Finding 1 (MEDIUM): `DemoDataset.from_run_dirs` specifies `np.concatenate` but data is stored as `torch.Tensor`

**Addressed.** §4 specifies calling `.numpy()` on each sub-dataset's tensors before `np.concatenate`, then passing the numpy arrays to the `DemoDataset` constructor (which converts back to tensors). The code snippet in §4 shows the exact implementation. The numpy→tensor round-trip is negligible for demo-sized data.

### Finding 2 (MEDIUM): Epoch logging frequency is ambiguous

**Addressed.** §2 Logging subsection explicitly states: "the epoch log is emitted every epoch (the current every-20th-epoch modulo guard is removed)." This satisfies FR-4's "Per-epoch" requirement.

### Finding 3 (MEDIUM): `train_bc` has no input validation for `checkpoint_interval`

**Addressed.** §3 includes an explicit guard at the top of `train_bc`:
```python
if checkpoint_interval is not None and checkpoint_interval < 1:
    raise ValueError(...)
```
A row is in the §10 `train_bc` error table for this condition. Two matching tests (`test_checkpoint_interval_zero_raises`, `test_checkpoint_interval_negative_raises`) are in the §11 TestCheckpointing class.

### Finding 4 (NIT): `torch.save` is not atomic on POSIX

**Addressed.** §3 error handling prose reads: "If `torch.save` fails mid-write, the target file may be corrupt. The exception propagates as-is." The incorrect atomicity claim has been removed.

### Finding 5 (NIT): Unnecessary single-vs-multiple branching in `--from-run` CLI handler

**Addressed.** §4 CLI Changes states: "The CLI handler always calls `DemoDataset.from_run_dirs(args.from_run, ...)` regardless of how many paths are in the list." No single-path special case.

### Finding 6 (NIT): `_validate_training_config` doesn't specify safe access pattern for nested keys

**Addressed.** §1 Validation Rules includes: "Each top-level section is optional. The validator accesses nested keys via `config.get(section, {}).get(key)` so that a missing section is treated as all-defaults."

### Finding 7 (NIT): Design §8 refers to `run_task()` but actual function is `run_task_by_name()`

**Addressed.** §8 Runner Integration correctly references `run_task_by_name()`.

---

## Appendix: Design Review Responses (Round 2)

These responses address findings from the second design review (`.agents/tasks/forge-phase6-learned-policy/design-review.md`).

### Finding 1 (MEDIUM): Validation split size calculation is unspecified; zero-val-size edge case can cause division by zero

**Addressed.** §2 now includes an explicit split-size calculation formula:
```python
val_size = max(1, int(len(dataset) * validation_split))  # at least 1 when split > 0
train_size = len(dataset) - val_size
if train_size < 1:
    raise ValueError("Dataset too small for requested validation split: ...")
```
The `max(1, ...)` floor guarantees at least 1 validation sample whenever validation is requested, preventing a zero-division edge case. The `train_size < 1` guard covers the other direction. A corresponding row was added to the §10 error table for `train_bc`. A new test `test_small_dataset_gets_at_least_one_val_sample` was added to `TestValidationMetrics` in §11, and `test_validation_split_too_large_raises` covers the case where the dataset is too small.

### Finding 2 (NIT): `validation_split` guard code is not shown inline, unlike `checkpoint_interval`

**Addressed.** §2 now shows the inline guard snippet immediately after the new keyword arguments:
```python
if not (0.0 <= validation_split <= 0.5):
    raise ValueError(f"validation_split must be in [0.0, 0.5], got {validation_split}")
```
This matches the style of the `checkpoint_interval` guard in §3.

### Finding 3 (NIT): No unit test for the config merge logic (code defaults ← YAML ← CLI)

**Addressed.** §1 now specifies a standalone `merge_training_config(code_defaults, yaml_config, cli_overrides) -> dict` helper function in `forge_train.py`. §11 adds a `TestConfigMerge` class with five tests covering: code defaults only, YAML overrides defaults, CLI overrides YAML, CLI overrides defaults without YAML, and partial YAML + partial CLI.

### Finding 4 (NIT): `from_run_dirs` does not log aggregate stats

**Addressed.** The `from_run_dirs` code snippet in §4 now includes:
```python
logger.info("loaded_run_dirs: %d dirs, %d total samples", len(run_dirs), len(all_obs))
```
A corresponding test `test_from_run_dirs_logs_aggregate_stats` was added to `TestMultiRunDataset` in §11.

### Finding 5 (NIT): Shell glob expansion for `--from-run` is platform-dependent

**Addressed.** §4 CLI Changes now states: "Glob patterns in `--from-run` are expanded by the shell. This is standard Unix/bash/zsh behavior. On Windows, use explicit paths or a Unix-like shell (WSL)." §12 notes that `docs/setup.md` will include this caveat.
