# Phase 6 — Learned Policy: Design Review (Round 3)

**Reviewer**: Design Review Sub-agent  
**Date**: 2025-01-20  
**Verdict**: CHANGES_REQUESTED (2 MEDIUM, 2 NIT)

---

## Findings

### 1. MEDIUM — YAML-to-flat-dict key mapping is unspecified

**Where**: §1 (YAML Training Configuration), §5 (CLI Design), `merge_training_config`

**Problem**: The YAML schema uses nested keys with names that differ from the code/CLI namespace. For example:

| YAML key | Code / CLI key |
|---|---|
| `training.learning_rate` | `lr` (argparse flag `--lr`) |
| `training.batch_size` | `batch_size` (argparse flag `--batch-size`) |
| `training.validation_split` | `validation_split` |
| `training.checkpoint_interval` | `checkpoint_interval` |
| `data.demo_episodes` | `episodes` (argparse flag `--episodes`) |
| `data.demo_policy` | `demo_policy` (argparse flag `--demo-policy`) |

The design states that `load_training_config` "returns the raw dict" and `merge_training_config` accepts "the flattened dict from `load_training_config`". But no flattening or key-renaming function is specified. The implementer must guess:
- Does `training.learning_rate` become `"learning_rate"` or `"lr"` in the flat dict?
- Does `data.demo_episodes` become `"demo_episodes"` or `"episodes"`?

Since `merge_training_config` does a simple dict update (`{**code_defaults}` then overlay), the key names must match exactly between all three dicts. Without an explicit mapping, the merge silently drops mismatched keys.

**Fix**: Add a key-mapping table or a `_flatten_training_config(raw: dict) -> dict` helper specification that maps from the nested YAML namespace to the flat code namespace. For example:

```python
_YAML_TO_FLAT = {
    ("training", "epochs"): "epochs",
    ("training", "batch_size"): "batch_size",
    ("training", "learning_rate"): "lr",
    ("training", "validation_split"): "validation_split",
    ("training", "checkpoint_interval"): "checkpoint_interval",
    ("data", "target_object"): "target_object",
    ("data", "demo_episodes"): "episodes",
    ("data", "demo_policy"): "demo_policy",
    ("model", "hidden_layers"): "hidden_layers",
    ("model", "obs_dim"): "obs_dim",
    ("model", "act_dim"): "act_dim",
}

def _flatten_training_config(raw: dict) -> dict:
    flat = {}
    for (section, key), flat_key in _YAML_TO_FLAT.items():
        val = raw.get(section, {}).get(key)
        if val is not None:
            flat[flat_key] = val
    return flat
```

This function would live in `forge_train.py` alongside `merge_training_config`, and be called between `load_training_config` and `merge_training_config`.

---

### 2. MEDIUM — `model.*` and `data.*` YAML config keys have no specified code path to consumers

**Where**: §1 (YAML config schema), §5 (CLI Main Flow step 6)

**Problem**: The design defines these YAML config keys and validates them, but never specifies how they reach the functions that consume them:

**`model.hidden_layers`, `model.obs_dim`, `model.act_dim`**: `train_bc` creates a `PolicyNetwork()` with default arguments when `network is None`. The `train_bc` signature does not accept `hidden_layers`, `obs_dim`, or `act_dim`. The Main Flow step 6 says "Call `train_bc(...)` with merged params" but the merged params can't include model architecture values because `train_bc` has no parameters for them. Result: YAML model config values are validated but silently discarded at runtime.

**`data.target_object`**: Used by `collect_demos(target_object=...)` and `DemoDataset.from_run_dirs(target_object=...)`, but the Main Flow doesn't show extraction of this value from the merged config. The training CLI has no `--target-object` flag.

**`data.demo_episodes` and `data.demo_policy`**: The CLI has `--episodes` and `--demo-policy` flags, so these can override the YAML values — but only if Finding 1's key-mapping is resolved. The Main Flow step 5 references `--demo-policy` but doesn't reference the merged config for these values.

**Fix** (choose one):

*Option A — Wire the config values through:*
1. Add `target_object` to the flat-key namespace and pass it to `collect_demos` / `from_run_dirs` in the Main Flow.
2. In the CLI, construct the `PolicyNetwork` from model config when values are present, then pass it to `train_bc(network=net)`:
   ```python
   net = PolicyNetwork(
       obs_dim=merged.get("obs_dim", OBS_DIM),
       act_dim=merged.get("act_dim", ACT_DIM),
       hidden=merged.get("hidden_layers"),
   )
   result = train_bc(dataset, network=net, ...)
   ```
3. Update the Main Flow (§5 step 5-6) to show these data flows explicitly.

*Option B — Remove the unused config keys:*
If `model.*` and `data.target_object` are intended as documentation only (the MLP architecture is fixed for Phase 6), remove them from the validation rules and mark them as comments in `config/training.yaml`. This avoids dead validation code.

Option A is recommended since FR-1 lists these as "required config keys."

---

### 3. NIT — Empty dataset guard in `train_bc` is in the error table but not in the narrative

**Where**: §2 (Validation Metrics) vs §10 (Error Handling)

**Problem**: The §10 error table for `train_bc (extended)` lists "Empty dataset → raises `ValueError('Dataset is empty')`" but §2, where the `train_bc` modifications are described in detail, does not mention adding this guard. The existing code has no empty-dataset check (it would return `final_loss=0.0` with an empty history). An implementer focused on §2 could miss this.

**Fix**: Add a one-liner to §2 under the input validation block, before the validation_split check:

```python
if len(dataset) == 0:
    raise ValueError("Dataset is empty")
```

---

### 4. NIT — `code_defaults` dict in `merge_training_config` is incompletely specified

**Where**: §1 (CLI Integration), §5 (CLI Design)

**Problem**: The `code_defaults` parameter is shown as `{"epochs": 100, "batch_size": 64, "lr": 1e-3, ...}` with a `...` placeholder. The complete set of keys that participate in the merge is ambiguous. Should it include `validation_split`, `checkpoint_interval`, `target_object`, `hidden_layers`, etc.? The answer depends on Finding 1's key mapping and Finding 2's decision about which config keys are wired through.

**Fix**: After resolving Findings 1 and 2, list the complete `code_defaults` dict. For example:

```python
CODE_DEFAULTS = {
    "epochs": 100,
    "batch_size": 64,
    "lr": 1e-3,
    "validation_split": 0.0,
    "checkpoint_interval": None,
    "target_object": "red_cube",
    "episodes": 3,
    "demo_policy": "scripted_pick_and_place",
    # model keys if Option A from Finding 2:
    "hidden_layers": [256, 256, 128],
    "obs_dim": 31,
    "act_dim": 8,
}
```

---

## Verified Assumptions

These claims from the design were checked against the actual source code and confirmed correct.

| # | Design Claim | Verified Against | Status |
|---|---|---|---|
| 1 | `LearnedPolicy` implements the `Policy` ABC with `reset`, `act`, `done`, `name` | `forge_core/learned_policy.py` lines 99-130 | ✅ Correct |
| 2 | `train_bc` current signature: `(dataset, epochs=100, batch_size=64, lr=1e-3, save_path=None, network=None) → TrainResult` | `forge_core/learned_policy.py` lines 216-221 | ✅ Correct |
| 3 | `TrainResult` has 4 fields: `epochs`, `final_loss`, `model_path`, `history` | `forge_core/learned_policy.py` lines 202-212 | ✅ Correct |
| 4 | `DemoDataset.from_run_dir` falls back to zeros when target object data is missing | `forge_core/learned_policy.py` lines 161-168 | ✅ Correct |
| 5 | `DemoDataset.__init__` converts numpy arrays to `torch.Tensor` | `forge_core/learned_policy.py` lines 139-141 | ✅ Correct — `torch.from_numpy` is called |
| 6 | `create_policy("learned", ...)` uses lazy import inside `if name == "learned":` branch | `forge_core/policy.py` lines 134-136 | ✅ Correct |
| 7 | `forge_eval.py` already supports `--model-path` passed to `create_policy` | `scripts/forge_eval.py` lines 85-111 | ✅ Correct |
| 8 | `run_task_by_name()` exists and accepts a `policy` argument | `forge_core/runner.py` line 176 | ✅ Correct |
| 9 | `load_yaml` uses `yaml.safe_load` and returns `{}` for empty YAML | `forge_core/config.py` lines 21-28 | ✅ Correct |
| 10 | `load_yaml` raises `FileNotFoundError` when file doesn't exist (bare `open()`) | `forge_core/config.py` line 26 | ✅ Correct — `load_training_config` must catch this for the default-missing case |
| 11 | `pyproject.toml` already has `[project.optional-dependencies]` with a `dev` group | `pyproject.toml` lines 18-22 | ✅ Correct |
| 12 | Existing `--from-run` is `type=str, default=None` (single path) | `scripts/forge_train.py` line 61 | ✅ Correct |
| 13 | Existing `--epochs`, `--batch-size`, `--lr` have hardcoded defaults | `scripts/forge_train.py` lines 63-66 | ✅ Correct — defaults are 100, 64, 1e-3 |
| 14 | `torch.load` uses `weights_only=True` in `_load_model` | `forge_core/learned_policy.py` line 107 | ✅ Correct |
| 15 | No ROS imports in `forge_core/learned_policy.py` | grep search | ✅ Confirmed zero matches |
| 16 | Current epoch logging is every 20th epoch via modulo guard | `forge_core/learned_policy.py` line 258 | ✅ Correct — `if (epoch + 1) % 20 == 0 or epoch == 0` |
| 17 | Existing tests don't construct `TrainResult` directly; they receive it from `train_bc` | `tests/unit/test_learned_policy.py` | ✅ Correct — backward-compatible `val_history` addition is safe |
| 18 | Policy ABC defines `reset`, `act`, `done` (abstract), `name` (property with default) | `forge_core/policy.py` lines 14-28 | ✅ Correct |

## Unverified / Wrong Assumptions

None found. All factual claims about existing code were verified and found to be accurate.

---

## Summary

The design is thorough, well-structured, and has already incorporated feedback from two previous review rounds. The existing code claims are all verified correct. The two MEDIUM findings are related: the YAML config schema defines keys in a nested namespace that differs from the flat code/CLI namespace, and the path from config values to the consuming functions is not fully specified. These are straightforward to fix by adding a key-mapping table and wiring `model.*` / `data.*` values through the Main Flow.

The two NIT findings are minor documentation gaps (empty-dataset guard placement and incomplete code_defaults dict) that would not block a careful implementer but are easy to resolve.
