# Phase 6 — Learned Policy (PyTorch): Requirements

## Summary

Phase 6 adds a PyTorch-based learned policy subsystem to Forge. The core module (`forge_core/learned_policy.py`), unit tests (`tests/unit/test_learned_policy.py`), and training CLI (`scripts/forge_train.py`) already exist with substantial implementations. This phase completes and hardens those implementations, adds YAML-driven training configuration, validation metrics, model checkpointing, perception-aware policy input, documentation, and dependency management so the learned policy is production-ready and fully integrated with the existing Forge platform.

---

## Existing API Contract (Verbatim from Codebase)

The following signatures and contracts are already defined and **must be preserved exactly**.

### Policy ABC (`forge_core/policy.py`)

```python
class Policy(ABC):
    @abstractmethod
    def reset(self, sim: ForgeSimulation) -> None: ...
    @abstractmethod
    def act(self, sim: ForgeSimulation) -> np.ndarray: ...
    @property
    @abstractmethod
    def done(self) -> bool: ...
    @property
    def name(self) -> str:
        return self.__class__.__name__
```

### LearnedPolicy (`forge_core/learned_policy.py`)

```python
class LearnedPolicy(Policy):
    def __init__(self, model_path: str | None = None,
                 target_object: str = "red_cube",
                 max_steps: int = 1500, **kwargs): ...
    def _load_model(self, path: str) -> None: ...
    def reset(self, sim: ForgeSimulation) -> None: ...
    def act(self, sim: ForgeSimulation) -> np.ndarray: ...
    @property
    def done(self) -> bool: ...
    @property
    def name(self) -> str:
        return "LearnedPolicy"
```

### PolicyNetwork

```python
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden: list[int] | None = None): ...
    def forward(self, obs: torch.Tensor) -> torch.Tensor: ...
```

Constants: `OBS_DIM = 31`, `ACT_DIM = 8`.

### DemoDataset

```python
class DemoDataset(Dataset):
    def __init__(self, observations: np.ndarray, actions: np.ndarray): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...
    @classmethod
    def from_run_dir(cls, run_dir: str | Path,
                     target_object: str = "red_cube") -> DemoDataset: ...
```

### train_bc

```python
def train_bc(
    dataset: DemoDataset,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    save_path: str | None = None,
    network: PolicyNetwork | None = None,
) -> TrainResult: ...
```

### TrainResult

```python
@dataclass
class TrainResult:
    epochs: int
    final_loss: float
    model_path: str
    history: list[float]
    def summary(self) -> str: ...
```

### collect_demos

```python
def collect_demos(
    task_name: str,
    policy: Policy,
    n_episodes: int = 1,
    target_object: str = "red_cube",
) -> DemoDataset: ...
```

### create_policy registration (`forge_core/policy.py`)

```python
def create_policy(name: str, **kwargs) -> Policy:
    if name == "learned":
        from forge_core.learned_policy import LearnedPolicy
        return LearnedPolicy(**kwargs)
    ...
```

### Test expectations (`tests/unit/test_learned_policy.py`)

The test file imports and exercises:
- `extract_observation(sim)` → shape `(OBS_DIM,)`, dtype `float32`, all finite, changes after sim step
- `PolicyNetwork()` forward → shape `(1, ACT_DIM)` for single input, `(16, ACT_DIM)` for batch, custom hidden `[64, 64]` works
- `LearnedPolicy(max_steps=10)` → `policy.act(sim)` returns len `ACT_DIM`, terminates after `max_steps`
- `create_policy("learned", max_steps=10)` → `policy.name == "LearnedPolicy"`
- Save/load roundtrip: `torch.save(net.state_dict(), path)` then `LearnedPolicy(model_path=path)` acts correctly
- `DemoDataset(obs, act)` from arrays: len, indexing, shapes
- `train_bc(ds, epochs=5, ...)` → `result.epochs == 5`, `result.final_loss > 0`, model file exists, `len(result.history) == 5`
- `collect_demos("pick_red_cube", NullPolicy(), n_episodes=1)` → `len(ds) > 0`, correct observation shape

---

## Functional Requirements

### FR-1: YAML Training Configuration

The system must support loading training hyperparameters from a YAML config file, consistent with Forge's YAML-driven convention. A default training config file must be provided at `config/training.yaml`.

Required config keys:
- `training.epochs` (int, default 100)
- `training.batch_size` (int, default 64)
- `training.learning_rate` (float, default 1e-3)
- `training.validation_split` (float, default 0.1)
- `training.checkpoint_interval` (int, epochs between checkpoints, default 20)
- `model.hidden_layers` (list of int, default [256, 256, 128])
- `model.obs_dim` (int, default 31)
- `model.act_dim` (int, default 8)
- `data.target_object` (str, default "red_cube")
- `data.demo_episodes` (int, default 3)
- `data.demo_policy` (str, default "scripted_pick_and_place")

The CLI (`forge_train.py`) must accept a `--config` flag to load a YAML config, with CLI arguments overriding YAML values.

### FR-2: Validation Metrics

`train_bc` must split the dataset into training and validation sets (configurable ratio). Each epoch must compute and log both training loss and validation loss. `TrainResult` must include `val_history: list[float]` alongside the existing `history` (training loss). The summary must report both final training and validation loss.

### FR-3: Model Checkpointing

During training, the system must save model checkpoints at a configurable epoch interval (default: every 20 epochs) and always at the final epoch. Checkpoints must be saved to a `models/` directory (created automatically). Checkpoint filenames must include the epoch number (e.g., `bc_policy_epoch_020.pt`). The final/best model must also be saved at the configured `save_path`.

### FR-4: Training Metrics Logging

Training must log structured metrics using Python `logging` at INFO level:
- Per-epoch: epoch number, training loss, validation loss
- Training start: dataset size, config summary
- Training end: final losses, model path, total time

This uses the existing `logger = logging.getLogger(__name__)` pattern.

### FR-5: Multiple Run Directory Loading

`DemoDataset` must support loading from multiple run directories and merging them into a single dataset. The `forge_train.py` CLI must accept `--from-run` with multiple paths or a glob pattern.

### FR-6: Integration with Evaluation CLI

`forge_eval.py` already supports `--policy learned --model-path <path>`. This integration is complete and must continue to work. No changes needed to `forge_eval.py` unless testing reveals a bug.

### FR-7: Integration with Runner

`run_task()` already supports learned policies via the `policy` argument and `create_policy("learned", model_path=...)`. The `perception=True` flag wraps any policy including learned policies in `PerceptionPolicy`. This integration path must be preserved.

### FR-8: CLI Completeness (`scripts/forge_train.py`)

The training CLI must support:
- `--config <path>` — load YAML training config
- `--task <name>` — task for demo collection (existing)
- `--episodes <n>` — number of demo episodes (existing)
- `--from-run <path> [<path> ...]` — train from recorded run directories (extend to support multiple)
- `--epochs`, `--batch-size`, `--lr` — override training params (existing)
- `--save-path` — model output path (existing)
- `--demo-policy` — policy for demo collection (existing)
- `--list-models` — list saved models/checkpoints in `models/`
- `--verbose` — debug logging (existing)

### FR-9: Model Listing

The CLI must support listing available trained models and checkpoints from the `models/` directory, showing filename, file size, and modification timestamp.

### FR-10: Perception-Aware Policy Input (Future Hook)

The `LearnedPolicy` architecture must be extensible to accept image observations from the perception system in addition to state observations. Phase 6 implements state-only (MLP) input. The design must include a clear extension point (e.g., an `obs_mode` parameter: `"state"` vs `"vision"`) without requiring implementation of the vision path now. Document the extension point.

---

## Non-Functional Requirements

### NFR-1: No ROS Dependency

All code in `forge_core/learned_policy.py` and related training infrastructure must have zero dependency on ROS 2. Tests must run with `pytest` alone, no ROS environment needed.

### NFR-2: PyTorch as Optional Dependency

PyTorch must be declared as an optional dependency in `pyproject.toml` under a `[ml]` extra group. The rest of Forge must remain functional without PyTorch installed — the lazy import in `create_policy("learned", ...)` already handles this.

### NFR-3: Backward Compatibility

All 117 existing tests must continue to pass without modification. Existing API signatures must not change in a breaking way. The `TrainResult` dataclass may gain new fields but existing fields (`epochs`, `final_loss`, `model_path`, `history`) must retain their types and semantics.

### NFR-4: Deterministic Test Behavior

Tests for the learned policy must be deterministic. Use fixed random seeds for PyTorch and NumPy in tests. Synthetic datasets in tests must produce consistent results.

### NFR-5: Code Style Consistency

Follow existing patterns: `logging.getLogger(__name__)`, dataclass usage, `from __future__ import annotations`, type hints, docstrings on public functions, imports from `forge_core.config.PROJECT_ROOT`.

---

## Acceptance Criteria

1. **AC-1**: `pip install torch` (or `pip install -e ".[ml]"`) succeeds. `pyproject.toml` declares `torch>=2.0` under `[project.optional-dependencies.ml]`.

2. **AC-2**: All 117 pre-existing tests pass unchanged: `python3 -m pytest tests/ -v` reports 117+ passed, 0 failed.

3. **AC-3**: All new and existing learned-policy tests pass: `python3 -m pytest tests/unit/test_learned_policy.py -v` reports all passed. Minimum test coverage includes the test classes already defined (`TestObservation`, `TestPolicyNetwork`, `TestLearnedPolicy`, `TestDemoDataset`, `TestBehavioralCloning`) plus new tests for YAML config loading, validation metrics, checkpointing, and multi-run dataset loading.

4. **AC-4**: Training CLI works end-to-end: `python3 scripts/forge_train.py --task pick_red_cube --episodes 1 --epochs 5` completes without error, produces a model file in `models/`, and prints a training summary with loss values.

5. **AC-5**: Training from a recorded run works: after recording a run with `forge_eval.py --task pick_red_cube --record`, running `python3 scripts/forge_train.py --from-run runs/run_XXXXX/` loads the trajectory and trains successfully.

6. **AC-6**: YAML config works: `python3 scripts/forge_train.py --config config/training.yaml --task pick_red_cube --episodes 1` reads training params from the YAML file.

7. **AC-7**: Model save/load roundtrip: a model saved by `train_bc` can be loaded by `LearnedPolicy(model_path=...)` and produces actions of shape `(ACT_DIM,)` from simulation observations.

8. **AC-8**: Evaluation CLI works with learned policy: `python3 scripts/forge_eval.py --task pick_red_cube --policy learned --model-path models/bc_policy.pt` completes and reports an `EvalResult`.

9. **AC-9**: Model checkpoints exist: after training with `epochs=100` and `checkpoint_interval=20`, files `bc_policy_epoch_020.pt`, `bc_policy_epoch_040.pt`, `bc_policy_epoch_060.pt`, `bc_policy_epoch_080.pt`, and `bc_policy_epoch_100.pt` (or equivalent final model) exist in the output directory.

10. **AC-10**: Validation loss is tracked: `TrainResult.val_history` is a list of floats with length equal to `epochs`, and training summary includes validation loss.

11. **AC-11**: `--list-models` flag prints available models from `models/` directory or a "no models found" message.

12. **AC-12**: `config/training.yaml` exists with documented default values for all training parameters.

13. **AC-13**: `docs/architecture.md` contains a Phase 6 section describing the learned policy module, training pipeline, network architecture, integration points, and the future vision-input extension point.

14. **AC-14**: `README.md` Phase Roadmap shows Phase 6 as "Done" and the "Current Status" line is updated. The project structure section lists `learned_policy.py` and `forge_train.py`. Training CLI usage is documented.

15. **AC-15**: `docs/setup.md` includes PyTorch installation instructions (under the `[ml]` extra) and training CLI usage examples.

16. **AC-16**: `forge_core/learned_policy.py` contains no ROS imports. Verified by: `grep -r "import rclpy\|from rclpy\|import rospy\|from rospy" forge_core/learned_policy.py` returns nothing.

---

## Out of Scope

- **Reinforcement learning training loop** — Phase 6 uses behavior cloning only. The architecture supports future RL but the RL training loop is not implemented.
- **CNN / vision-based policy network** — Phase 6 is MLP (state input) only. The extension point for vision input is documented but not implemented.
- **Experiment tracking** (MLflow, W&B) — deferred to Phase 7 (ML infrastructure).
- **Model registry / versioning** — deferred to Phase 7.
- **ONNX export / TensorRT optimization** — deferred to Phase 9 (edge deployment).
- **Hyperparameter search / tuning** — users set params manually or via YAML.
- **Distributed or GPU-specific training** — training runs on CPU by default. GPU is used if available via standard PyTorch device selection, but no multi-GPU or distributed training support.
- **Changes to ROS 2 nodes** — the learned policy integrates at the `forge_core` level only.
- **New task definitions** — existing tasks are sufficient for training and evaluation.

---

## Open Questions

1. **Validation split strategy**: Should validation be a random split of the dataset, or should entire episodes be held out? Random split is simpler; episode-level split is more realistic since consecutive frames are correlated. **Assumption**: random split is acceptable for Phase 6 behavior cloning — episode-level holdout can be added later.

2. **Model naming convention**: When training from multiple runs or with different configs, should models be auto-named with timestamps or config hashes? **Assumption**: default to `bc_policy.pt` (overwritten) with checkpoints named by epoch. Users can specify `--save-path` for custom naming.

3. **Training config in `config/training.yaml` vs inline in `config/default.yaml`**: Should training config be a separate file or a new section in `default.yaml`? **Assumption**: separate file `config/training.yaml` — keeps concerns separated and is consistent with how scenarios and faults have their own directories.

4. **`from_run_dir` with missing object data**: The current `DemoDataset.from_run_dir` falls back to zeros if target object or goal position arrays are missing. Should this log a warning? **Assumption**: yes, log a warning but do not fail — allows training on runs that may not have all objects.
