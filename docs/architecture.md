# Forge — Architecture

## Overview

Forge is a robotics reliability platform built on ROS 2 and MuJoCo.
It supports a lifecycle of develop → simulate → evaluate → record →
detect failures → replay → improve.

The current implementation covers Phases 1–11:
simulation, task evaluation, telemetry/replay, failure injection,
camera-based perception, learned policy via behavioral cloning,
ML infrastructure (experiment tracking + model registry),
CI/CD with regression evaluation,
edge deployment (ONNX export, TensorRT optimization),
fleet management with OTA updates, and
physical robot support (hardware abstraction, safety system, sim-to-real transfer).

## System Diagram

```
  Task YAML + Fault Profile
        │
        ▼
  ┌───────────┐     ┌──────────┐
  │  Runner    │────▶│ Recorder │──▶ runs/run_XXXXX/
  │            │     └──────────┘      metadata.json
  │ load       │                       trajectory.npz
  │ reset      │     ┌──────────┐      config.yaml
  │ inject     │────▶│ Faults   │      result.json
  │ loop:      │     └──────────┘
  │  policy    │
  │  sim.step  │     ┌──────────┐
  │ evaluate   │────▶│Evaluator │──▶ EvalResult
  └─────┬──────┘     └──────────┘
        │
        ▼
  ForgeSimulation (MuJoCo)
  ┌─────────────────────────────┐
  │ Menagerie Panda + Table     │
  │ + Objects (freejoint)       │
  │                             │
  │ step / reset / query state  │
  └─────────────────────────────┘
        │
        ▼
  ROS 2 Nodes (optional)
  ┌──────────┐ ┌────────────┐ ┌──────────┐
  │forge_sim │ │forge_control│ │forge_state│
  └──────────┘ └────────────┘ └──────────┘
  /forge/object_states  /forge/joint_commands  /forge/joint_states
  /forge/ee_pose        /forge/reset (srv)
```

---

## Phase 1 — Simulation Foundation

### Robot Model

The simulation uses the **MuJoCo Menagerie Franka Emika Panda** — a
mesh-based model with accurate collision geometry and kinematics.
The model is downloaded automatically via the `mujoco_menagerie` package.

- 7-DOF arm: `joint1`–`joint7`
- 2 coupled finger joints via tendon: `finger_joint1`, `finger_joint2`
- 8 actuators: 7 arm (general/affine bias) + 1 finger (tendon-coupled, 0–255)
- EE tracking via the `hand` body

### Simulation Engine — `forge_core/simulation.py`

- Loads Menagerie Panda + scene objects from scenario YAML
- Writes a temp XML with `<include file="panda.xml"/>` in the Menagerie cache dir
- Owns step, reset, and state queries
- No ROS dependency — testable standalone with pytest

### ROS 2 Nodes

| Node | Responsibility |
|------|---------------|
| `forge_simulator` | Simulation lifecycle, object state, reset service |
| `forge_controller` | Subscribes `/forge/joint_commands`, applies actuator targets |
| `forge_state` | Publishes `/forge/joint_states` and `/forge/ee_pose` |

All topics use standard ROS message types — no custom messages.

### Configuration

Scenarios are YAML files in `simulation/scenarios/`. They define the robot
home position, object types/positions, and simulation parameters.

---

## Phase 2 — Task Execution & Evaluation

### Task Specification — `forge_core/task.py`

Tasks are YAML files in `tasks/` that define:
- Scenario to load
- Target object
- Time limit
- Success criteria (measurable conditions)

Criterion types: `object_in_zone`, `object_above_height`,
`object_distance_to`, `robot_at_home`.

### Policy Interface — `forge_core/policy.py`

```python
class Policy(ABC):
    def reset(self, sim: ForgeSimulation) -> None: ...
    def act(self, sim: ForgeSimulation) -> np.ndarray: ...
    def done(self) -> bool: ...
```

Built-in policies:
- `ScriptedPickAndPlace` — deterministic waypoint sequence
- `NullPolicy` — do-nothing baseline

The interface is designed so learned policies (Phase 6) can be
drop-in replacements.

### Evaluator — `forge_core/evaluator.py`

Checks each criterion against simulation state and returns structured
`EvalResult` with per-criterion pass/fail, distance metrics, and
aggregate run metrics.

### Runner — `forge_core/runner.py`

Orchestrates: load scenario → reset → run policy loop → evaluate.
Supports recording and fault injection.

### CLI — `scripts/forge_eval.py`

```
forge_eval.py --task pick_red_cube
forge_eval.py --suite manipulation-v1
forge_eval.py --all --save --record
forge_eval.py --task pick_red_cube --fault object_moved
```

---

## Phase 3 — Telemetry & Replay

### Recorder — `forge_core/recorder.py`

Captures per-cycle snapshots during task execution:
- Joint positions, velocities
- End-effector pose
- Object positions and orientations
- Actions (joint targets sent)
- Timestamps

### Run Format

```
runs/run_00001/
  metadata.json     — run ID, task, policy, timestamp, success
  trajectory.npz    — numpy arrays of all recorded signals
  config.yaml       — scenario config snapshot
  result.json       — evaluation result
```

Trajectory storage uses numpy `.npz` (compressed). MCAP support is
planned for a future iteration.

### Replay — `forge_core/replay.py`

Loads a run directory and provides:
- Run summary (task, result, object displacement)
- State query at any step index or simulation time
- Object trajectory analysis

### CLI — `scripts/forge_replay.py`

```
forge_replay.py --list
forge_replay.py runs/run_00001/
forge_replay.py runs/run_00001/ --step 50
forge_replay.py runs/run_00001/ --time 5.0
```

---

## Phase 4 — Failure Injection / Chaos Testing

### Fault System — `forge_core/faults.py`

Faults are composable perturbations injected during task execution.
Each fault has a trigger time, optional duration, and type-specific params.

| Fault Type | Effect |
|------------|--------|
| `object_moved` | Teleport an object mid-run |
| `sensor_noise` | Gaussian noise on joint position readings |
| `sensor_delay` | Return stale robot state for N cycles |
| `actuator_stuck` | Freeze specific joints at current value |
| `controller_dropout` | Zero all commands — robot goes limp |
| `gravity_shift` | Change gravity vector mid-run |

### Fault Profiles

YAML files in `faults/` define sets of faults:

```yaml
faults:
  - type: object_moved
    trigger_time: 5.0
    params:
      object_id: red_cube
      new_position: [0.6, -0.1, 0.245]
```

Built-in profiles: `object_moved`, `sensor_noise`, `controller_dropout`,
`gravity_shift`, `cascading` (multi-fault stack).

### Integration

The `FaultInjector` wraps a list of faults and integrates with the runner:
- `pre_step()` — applies faults and filters joint targets
- `filter_state()` — modifies robot state before the policy sees it
- Faults compose: multiple faults stack in sequence

### CLI — `scripts/forge_chaos.py`

```
forge_chaos.py --list
forge_chaos.py --task pick_red_cube --fault object_moved
forge_chaos.py --task pick_red_cube --inject controller_dropout --at 3.0 --duration 2.0
forge_chaos.py --task pick_red_cube --fault cascading --compare
```

The `--compare` flag runs baseline (no faults) and faulted side by side.

---

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

```
forge_perception.py --scenario basic_workspace --render -o frame.png
forge_perception.py --scenario basic_workspace --detect --annotate -o detections.png
```

---

## Phase 6 — Learned Policy (PyTorch)

### Module Overview

`forge_core/learned_policy.py` provides behavioral cloning (BC) training
and a `LearnedPolicy` class that implements the Policy ABC. A trained
neural network replaces the scripted waypoint sequence — same interface,
learned behavior.

### Network Architecture

The default policy network is an MLP:

```
Input (31-dim state)
  → Linear(31, 256) → Tanh
  → Linear(256, 256) → Tanh
  → Linear(256, 128) → Tanh
  → Linear(128, 8)
Output (8-dim action: 7 joint targets + 1 gripper)
```

Hidden layer sizes are configurable via `config/training.yaml` or
CLI arguments.

### Observation Space (31-dim)

| Indices | Content |
|---------|---------|
| 0–8 | Joint positions (9) |
| 9–17 | Joint velocities (9) |
| 18–20 | End-effector position (xyz) |
| 21–24 | End-effector orientation (quaternion wxyz) |
| 25–27 | Target object position (xyz) |
| 28–30 | Goal position — target bin (xyz) |

### Training Pipeline

1. **Collect demonstrations** — `collect_demos` runs a scripted policy
   and records state-action pairs, or use `DemoDataset.from_run_dir` /
   `DemoDataset.from_run_dirs` to load from previously recorded runs.
2. **Train** — `train_bc` optimizes the MLP via MSE loss on the
   demonstration actions. Supports validation split, epoch checkpointing,
   and training history tracking.
3. **Save** — the trained `state_dict` is saved as a `.pt` file in
   `models/` (e.g. `models/bc_policy.pt`).

### YAML Configuration

`config/training.yaml` controls training, model, and data parameters:

```yaml
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

Config is loaded and validated by `load_training_config`. CLI arguments
override YAML values, which override code defaults.

### Integration

- **Runner / Eval CLI**: `create_policy('learned', model_path=...)` returns
  a `LearnedPolicy` instance — drop-in replacement for scripted policies.
- **Perception**: `PerceptionPolicy` wraps the learned policy when
  `perception=True` is set, adding camera observations.
- **Recorder**: training demonstrations can be collected from any recorded
  run via `DemoDataset.from_run_dir`.

### Extension Point

The `obs_mode` parameter controls how observations are built:
- `'state'` — flat 31-dim MLP input (Phase 6, current)
- `'vision'` — reserved for a future CNN-based policy

Passing `obs_mode='vision'` raises `NotImplementedError` today.

### CLI — `scripts/forge_train.py`

```
# Collect demos and train
forge_train.py --task pick_red_cube --episodes 5

# Train from recorded runs
forge_train.py --from-run runs/run_00001/ runs/run_00002/

# Train with a YAML config
forge_train.py --config config/training.yaml --task pick_red_cube --episodes 1

# List saved models
forge_train.py --list-models

# Evaluate the trained policy
forge_eval.py --task pick_red_cube --policy learned --model-path models/bc_policy.pt
```

---
## Phase 7 — ML Infrastructure (Experiment Tracking + Model Registry)

### Overview

Phase 7 adds a self-contained, file-based ML infrastructure layer with
zero external tracking dependencies (no MLflow/W&B). Two new modules
provide experiment tracking and model versioning, both stored as JSON
on disk and integrated into the existing training and evaluation pipeline.

### Experiment Tracker — `forge_core/experiment.py`

Tracks every training run with full metadata: configuration, dataset
info, metrics history, timing, git commit hash, and tags.

**On-disk format:** one JSON file per experiment under `experiments/`.

```
experiments/
  exp_00001.json
  exp_00002.json
  ...
```

Each file contains:

| Field | Content |
|-------|---------|
| `experiment_id` | Sequential ID (`exp_XXXXX`) |
| `name` | Human-readable name (auto-generated or user-supplied) |
| `timestamp` | ISO 8601 creation time |
| `status` | `running`, `completed`, or `failed` |
| `git_commit` | HEAD hash at creation time |
| `config` | Full training config snapshot (epochs, lr, layers, etc.) |
| `dataset_info` | Sample count, source type, source runs, task |
| `metrics` | Final loss, validation loss, full loss/val history, best epoch |
| `model_path` | Path to saved checkpoint |
| `duration_seconds` | Wall-clock training time |
| `tags` | User-defined labels |
| `error` | Error message if status is `failed` |

Sequential IDs use the same scanning pattern as `recorder.py`.

**API:**

| Method | Purpose |
|--------|---------|
| `create_experiment()` | Start a new experiment record |
| `complete_experiment()` | Mark as completed with metrics |
| `fail_experiment()` | Mark as failed with error message |
| `get_experiment()` | Load one experiment by ID |
| `list_experiments()` | List/filter experiments by status or tag |
| `compare_experiments()` | Side-by-side metric comparison |
| `delete_experiment()` | Remove experiment file from disk |

### Model Registry — `forge_core/model_registry.py`

Versioned model registry with enforced stage lifecycle. All entries
live in a single `registry/registry.json` file.

**Stage transitions:**

```
training ─> validated ─> production ─> archived
                ^                          │
                └──────────────────────────┘
```

Only adjacent transitions are allowed. Direct jumps (e.g., training →
production) are rejected. The only back-transition is archived →
validated (re-activation).

**Registry entry fields:**

| Field | Content |
|-------|---------|
| `name` | Model name (e.g. `bc_pick_red_cube`) |
| `version` | Auto-incremented integer per model name |
| `stage` | Current lifecycle stage |
| `checkpoint_path` | Path to `.pt` file |
| `experiment_id` | Link to the experiment that produced it |
| `registered_at` | ISO 8601 registration time |
| `description` | Free-text description |
| `metadata` | Arbitrary key-value metadata (loss, epochs, etc.) |

**API:**

| Method | Purpose |
|--------|---------|
| `register_model()` | Add a new model version |
| `get_model()` | Look up by name and optional version |
| `list_models()` | List/filter by name or stage |
| `promote()` | Advance a model to the next stage |
| `delete_model()` | Remove a model entry |
| `get_latest()` | Get the highest version of a given model name |

### Integration Points

- **`train_bc()`** in `forge_core/learned_policy.py` accepts optional
  `experiment_tracker` and `experiment_id` parameters. When provided,
  metrics are auto-logged on completion and the experiment is marked
  failed on error. Callers that omit these parameters get identical
  behavior to Phase 6.

- **`scripts/forge_train.py`** orchestrates the full pipeline: creates
  an experiment, passes the tracker to `train_bc()`, and registers the
  model in the registry on success. Flags: `--no-track`,
  `--experiment-name`, `--model-name`. Tracking failures are caught so
  they never crash training.

- **`scripts/forge_eval.py`** adds `--registry-model` and
  `--registry-version` arguments to load a model directly from the
  registry instead of specifying `--model-path`.

- **`forge_core/runner.py`** provides a `load_policy_from_registry()`
  convenience function that resolves a model name/version from the
  registry and returns a `LearnedPolicy` instance.

### CLI — `scripts/forge_registry.py`

Separate management CLI with two command groups:

```
forge_registry.py experiments list [--status STATUS] [--tag TAG]
forge_registry.py experiments show EXP_ID
forge_registry.py experiments compare EXP_ID1 EXP_ID2
forge_registry.py experiments delete EXP_ID

forge_registry.py models list [--name NAME] [--stage STAGE]
forge_registry.py models show NAME [--version VERSION]
forge_registry.py models promote NAME --version VERSION --stage STAGE
forge_registry.py models delete NAME --version VERSION
```

---


---

## Phase 8 — CI/CD (Regression Evaluation)

### Overview

Phase 8 adds automated regression evaluation to detect performance
degradations between code changes. A new regression module compares
current evaluation results against saved baselines, integrated into
GitHub Actions CI/CD pipelines for continuous quality gates.

### Regression Module — `forge_core/regression.py`

The module follows the same file-based JSON storage pattern used by
ExperimentTracker and ModelRegistry. Zero ROS dependency.

**Dataclasses:**

| Dataclass | Purpose |
|-----------|---------|
| `Baseline` | Stored reference results: suite name, timestamp, git commit, tasks dict |
| `RegressionResult` | Per-task comparison: passed, newly_failing, metric_deltas |
| `RegressionReport` | Aggregate result with overall_passed, results list, summary(), to_dict() |

**Key functions:**

| Function | Purpose |
|----------|---------|
| `save_baseline()` | Convert EvalResult list to baseline JSON, write to `baselines/<suite>.json` |
| `load_baseline()` | Load a saved baseline by suite name |
| `check_regression()` | Compare current results against baseline, flag regressions |
| `load_regression_config()` | Load thresholds from `config/regression.yaml` or return defaults |

`check_regression()` iterates tasks, compares success status and key
metrics (sim_time), flags newly failing tasks and metric degradations
beyond the configured tolerance.

### Baseline Management

Baselines are JSON files in `baselines/` — one per evaluation suite.
Structure:

```json
{
  "suite_name": "manipulation-v1",
  "timestamp": "2025-01-01T00:00:00",
  "git_commit": "abc123",
  "tasks": {
    "pick_red_cube": {"success": true, "sim_time": 12.5, "metrics": {}},
    "pick_blue_cube": {"success": false, "sim_time": 30.0, "metrics": {}}
  }
}
```

Baselines are created via CLI and committed to the repository so
regressions are detected relative to a known-good state.

### CI/CD Workflows

Two GitHub Actions workflows under `.github/workflows/`:

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| Tests | `ci.yml` | push/PR to main | Run full test suite with Python 3.10/3.11 matrix |
| Regression Evaluation | `regression.yml` | PR to main, manual | Run regression check against saved baselines |

**ci.yml** features:
- Python version matrix (3.10, 3.11)
- pip dependency caching via `actions/cache@v4`
- Headless MuJoCo rendering (`MUJOCO_GL=egl`)
- CPU-only PyTorch install

**regression.yml** features:
- Runs `forge_regression.py --check --suite manipulation-v1`
- Uploads `baselines/` as artifact on failure for debugging
- Manual trigger via `workflow_dispatch`

### CLI — `scripts/forge_regression.py`

```
forge_regression.py --check --suite manipulation-v1
forge_regression.py --create-baseline --suite manipulation-v1
forge_regression.py --update-baseline --suite manipulation-v1
forge_regression.py --show-baseline --suite manipulation-v1
forge_regression.py --compare baselines/v1.json baselines/v2.json
forge_regression.py --check --suite manipulation-v1 --threshold-config config/regression.yaml
```

### Integration

The regression module connects to existing components:

- **EvalResult** from `forge_core/evaluator.py` — baseline tasks store
  the same fields produced by evaluation
- **run_suite()** from `forge_core/runner.py` — used to generate current
  results for comparison
- **`_get_git_commit`** from `forge_core/experiment.py` — tags baselines
  with the commit hash at creation time
- **`config/regression.yaml`** — YAML config loaded via the same pattern
  as `config/training.yaml`

---


## Phase 9 — Edge Deployment (ONNX / TensorRT)

### Overview

Phase 9 adds edge deployment capabilities: exporting trained PyTorch
policies to ONNX format, optional TensorRT optimization for NVIDIA
hardware, an edge inference runtime implementing the Policy ABC, and
deployment packaging with metadata manifests. All edge dependencies
are optional — the platform works without them installed.

### Module — forge_core/edge_deploy.py

| Class / Function | Purpose |
|------------------|---------|
| OnnxExporter | Export PolicyNetwork to ONNX with validation |
| TensorRTOptimizer | Optional TensorRT engine builder (runtime-detected) |
| OnnxPolicy | Policy ABC implementation using ONNX Runtime |
| TensorRTPolicy | Policy ABC implementation using TensorRT (guarded) |
| EdgePolicy | Auto-selecting runtime: TensorRT if available, else ONNX |
| DeploymentPackager | Package model + manifest for deployment |

### ONNX Export

The OnnxExporter converts a trained PolicyNetwork (31-dim input,
8-dim output MLP) to ONNX format. Export is configurable: opset version
(default 17), static or dynamic batch size, and custom input/output
names. After export, validate() runs N random samples through both
PyTorch and ONNX Runtime, asserting numeric equivalence within a
configurable tolerance (default 1e-5). export_from_checkpoint() loads
a .pt state dict and exports in one step.

### TensorRT Optimization

TensorRTOptimizer converts an ONNX model to a TensorRT engine for
GPU-accelerated inference. TensorRT is **never a pip dependency** — it
is detected at runtime via has_tensorrt(). When unavailable, the
optimizer reports available() == False and raises RuntimeError on
use. Supports FP16 and FP32 precision modes, configurable max batch
size and workspace.

### Edge Inference Runtime

Three policy classes implement the Policy ABC for edge inference:

- **OnnxPolicy** — loads an ONNX model into an onnxruntime.InferenceSession,
  runs extract_observation() then session inference to produce an action ndarray.
- **TensorRTPolicy** — loads a TensorRT engine with CUDA buffers.
  Raises RuntimeError at init if TensorRT is not installed.
- **EdgePolicy** — auto-selects the best available runtime: checks for
  .trt/.engine files + TensorRT availability, falls back to .onnx.

All three support benchmark() for latency measurement (mean, std, min,
max over N iterations).

### Deployment Packaging

DeploymentPackager.package() creates a self-contained deployment
directory:

`
deploy_output/
  model.onnx          # Exported ONNX model
  manifest.yaml       # Deployment metadata
  model.trt           # Optional TensorRT engine
`

The manifest.yaml contains: model name, format, opset version,
input/output shapes, source experiment, source checkpoint, creation
timestamp, and arbitrary metadata.

### Integration

- **Model registry**: export_from_registry() looks up a registered
  model by name/version and exports its checkpoint to ONNX.
- **Runner**: load_edge_policy_from_registry() exports and wraps a
  registry model as an OnnxPolicy.
- **Eval CLI**: forge_eval.py --onnx-model <path> evaluates a task
  using an ONNX policy.
- **Policy factory**: create_policy('onnx', onnx_path=...) and
  create_policy('edge', model_dir=...) integrate with the existing
  policy creation pattern.

### CLI — scripts/forge_deploy.py

`
python3 scripts/forge_deploy.py --export --model models/bc_policy.pt --output models/bc_policy.onnx
python3 scripts/forge_deploy.py --validate --model models/bc_policy.onnx --checkpoint models/bc_policy.pt
python3 scripts/forge_deploy.py --benchmark --model models/bc_policy.onnx --n-iterations 200
python3 scripts/forge_deploy.py --package --model models/bc_policy.onnx --output deploy/ --name bc_pick_red_cube
python3 scripts/forge_deploy.py --optimize --model models/bc_policy.onnx --output models/bc_policy.trt --precision fp16
python3 scripts/forge_deploy.py --export-registry --name bc_pick_red_cube --version 1 --output models/exported.onnx
`

---

## Phase 10 — Fleet & OTA

### Overview

Phase 10 adds fleet device management and over-the-air (OTA) update
capabilities. A fleet manager tracks registered devices with status,
tags, and groups, while an OTA manager handles update package creation,
rollout strategies (immediate, canary, staged), rollback, and
per-device update history. All operations are local and simulated —
no network calls or real device communication. Data is persisted as
JSON on disk following the same patterns used by the model registry
and experiment tracker.

### Fleet Manager — `forge_core/fleet.py`

Manages a registry of fleet devices with JSON persistence in
`fleet/devices.json`.

**DeviceInfo dataclass fields:**

| Field | Content |
|-------|---------|
| `device_id` | Unique device identifier |
| `name` | Human-readable name |
| `device_type` | Device type (e.g. `panda`, `ur5`) |
| `capabilities` | List of device capabilities (e.g. `gpu`, `camera`) |
| `status` | Current status — one of `online`, `offline`, `updating`, `error` |
| `tags` | User-defined labels |
| `groups` | Group memberships |
| `last_seen` | ISO 8601 timestamp of last status update |
| `registered_at` | ISO 8601 registration timestamp |
| `current_model` | Currently deployed model name |
| `current_model_version` | Currently deployed model version |
| `experiment_id` | Link to the experiment that produced the deployed model |
| `metadata` | Arbitrary key-value metadata |

**API:**

| Method | Purpose |
|--------|---------|
| `register_device()` | Add a new device to the fleet |
| `deregister_device()` | Remove a device from the fleet |
| `get_device()` | Look up a device by ID |
| `list_devices()` | List/filter devices by status, group, or tag |
| `update_status()` | Update device status (validates against VALID_STATUSES) |
| `add_tag()` / `remove_tag()` | Manage device tags |
| `add_to_group()` / `remove_from_group()` | Manage device group memberships |
| `update_device_model()` | Update the model currently deployed on a device |
| `get_fleet_summary()` | Fleet-wide summary: counts by status, groups, tags |

### OTA Update System — `forge_core/ota.py`

Manages OTA update packages, rollout execution, rollback, and
per-device update history. Persistence layout:

```
ota/
├── packages.json          — update package registry
├── rollouts.json          — rollout records
└── history/
    └── {device_id}.json   — per-device update history
```

**UpdatePackage dataclass fields:**

| Field | Content |
|-------|---------|
| `package_id` | Sequential ID (`pkg_XXXXX`) |
| `version` | Semver string (e.g. `1.0.0`) |
| `model_name` | Model name |
| `model_version` | Model version integer |
| `source_package` | Path to source deployment package |
| `requirements` | Required device capabilities |
| `created_at` | ISO 8601 creation timestamp |
| `metadata` | Arbitrary key-value metadata |

**Rollout strategies:**

| Strategy | Behaviour |
|----------|-----------|
| `immediate` | Deploy to all compatible target devices at once |
| `canary` | Deploy to a percentage of target devices first |
| `staged` | Deploy to all but require confirmation (stays in-progress) |

**API:**

| Method | Purpose |
|--------|---------|
| `create_update_package()` | Create a package from a deployment directory (reads manifest.yaml) |
| `get_update_package()` | Look up a package by ID |
| `list_update_packages()` | List all packages |
| `check_compatibility()` | Check if a package is compatible with a device |
| `create_rollout()` | Create a rollout targeting devices, a group, or a tag |
| `execute_rollout()` | Simulate rollout execution with status updates |
| `get_rollout()` / `list_rollouts()` | Query rollout records |
| `rollback_rollout()` | Revert devices to their previous model version |
| `get_device_history()` | Per-device update history |

**Version utilities:**

- `parse_version()` — splits semver `X.Y.Z` into an integer tuple
- `compare_versions()` — compares two semver strings (-1, 0, 1)

### Integration Points

- **Phase 9 deployment packages**: `create_update_package()` reads
  `manifest.yaml` from deployment packages created by
  `DeploymentPackager` to auto-fill model metadata.
- **Model registry**: OTA packages reference model names and versions
  from the registry.
- **Experiment tracking**: Devices track which experiment produced
  their currently deployed model.

### CLI — `scripts/forge_fleet.py`

Management CLI with three command groups:

```
# Device management
forge_fleet.py devices register --id arm-01 --name "Panda Arm 01" --type panda
forge_fleet.py devices list [--status STATUS] [--group GROUP] [--tag TAG]
forge_fleet.py devices status --id arm-01 --set online
forge_fleet.py devices tag --id arm-01 --add gpu
forge_fleet.py devices group --id arm-01 --add production
forge_fleet.py devices deregister --id arm-01

# OTA updates
forge_fleet.py ota create-package --source-dir deploy/bc_policy/ --version 1.0.0 --model-name bc_pick
forge_fleet.py ota deploy --package-id pkg_00001 --target-group production
forge_fleet.py ota status --rollout-id rollout_00001
forge_fleet.py ota rollback --rollout-id rollout_00001
forge_fleet.py ota history --device-id arm-01
forge_fleet.py ota list-packages
forge_fleet.py ota list-rollouts [--status STATUS]

# Fleet overview
forge_fleet.py overview
```

---
## Phase 11 — Physical Robot Support

### Overview

Phase 11 bridges the gap between simulation and physical hardware.
A hardware abstraction layer provides a uniform `RobotInterface` ABC so
policies can run on either a simulated or a real robot. A safety system
wraps the interface to enforce joint, velocity, and workspace limits
before any command reaches the actuators. Sim-to-real transfer utilities
handle domain randomization, trajectory comparison, and calibration to
close the reality gap.

### Hardware Abstraction Layer — `forge_core/hardware.py`

Defines the robot communication interface and a simulation-backed
implementation.

**Dataclasses:**

| Dataclass | Fields |
|-----------|--------|
| `RobotState` | `joint_positions`, `joint_velocities`, `ee_position`, `ee_orientation`, `gripper_state`, `timestamp` |
| `RobotCommand` | `joint_targets`, `gripper_command` (0=open, 255=closed), `timestamp` |
| `RobotStatus` | `connected`, `status` (idle/running/error/e_stopped), `error_message`, `uptime`, `last_heartbeat` |

**RobotInterface ABC methods:**

| Method | Purpose |
|--------|---------|
| `connect()` | Establish connection to the robot |
| `disconnect()` | Cleanly shut down the connection |
| `get_state()` | Return current `RobotState` |
| `send_command(cmd)` | Send a `RobotCommand` to the robot |
| `emergency_stop()` | Trigger an immediate emergency stop |
| `get_status()` | Return current `RobotStatus` |

**SimulatedRobot** implements `RobotInterface` backed by
`ForgeSimulation`. It translates `RobotCommand` into simulation
joint targets and maps simulation state back to `RobotState`.

**`run_policy_on_robot(robot, policy, ...)`** executes a `Policy` on
any `RobotInterface` implementation with optional recording via
`TrajectoryBuffer` and safety monitoring.

### Safety System — `forge_core/safety.py`

Wraps a `RobotInterface` to enforce configurable safety limits.

**SafetyConfig** is loaded from `config/safety.yaml` via
`load_safety_config()`. It defines joint position limits, velocity
limits, workspace boundaries, and maximum gripper force.

**SafetyViolation** records a single violation with type
(`joint_limit`, `velocity_limit`, `workspace_boundary`), message,
joint index, value, and limit.

**SafetyMonitor methods:**

| Method | Purpose |
|--------|---------|
| `check_command(cmd, state)` | Return list of `SafetyViolation` for a command |
| `is_safe(cmd, state)` | Boolean shorthand — no violations |
| `safe_send_command(cmd)` | Check then send; blocks unsafe commands |
| `emergency_stop()` | Delegate to wrapped robot's emergency stop |
| `get_violations_log()` | Return all recorded violations as dicts |

### Sim-to-Real Transfer — `forge_core/sim2real.py`

Utilities to narrow the sim-to-real gap.

- **Domain randomization** — `DomainRandomizationConfig` +
  `apply_domain_randomization(sim, config)` randomizes gravity,
  friction, mass, and sensor/actuator noise on a `ForgeSimulation`.
- **Trajectory comparison** — `compare_trajectories(sim, real)`
  computes position/velocity/EE RMSE, max errors, and correlation
  between simulated and real trajectories.
- **Calibration** — `calibrate_joint_offsets()` computes per-joint
  offsets; `calibrate_workspace()` computes a workspace translation
  transform via centroid alignment.

Configuration is loaded from `config/robot.yaml` via
`load_sim2real_config()`.

### Configuration

| File | Contents |
|------|----------|
| `config/robot.yaml` | Robot connection type, scenario, sim-to-real domain randomization parameters |
| `config/safety.yaml` | Joint limits, velocity limits, workspace boundaries, max gripper force |

### CLI — `scripts/forge_robot.py`

| Subcommand | Purpose |
|------------|---------|
| `connect` | Connect to a robot, print status, disconnect |
| `run` | Run a policy on the robot with optional recording and safety monitoring |
| `safety-check` | Run a safety check against current configuration |
| `calibrate` | Compute joint offsets between simulated and real positions |
| `compare-sim` | Compare simulated and real trajectories |
| `status` | Print current robot connection status |

### Integration Points

- **Policy** — `run_policy_on_robot` accepts any `Policy` implementation
  (scripted or learned) via the standard `reset/act/done` interface.
- **ForgeSimulation** — `SimulatedRobot` wraps `ForgeSimulation` for
  testing without hardware.
- **Recorder** — `run_policy_on_robot` accepts a `TrajectoryBuffer` to
  record runs in the same format used by the existing runner.

---
## Design Principles

1. **forge_core has no ROS dependency** — everything can be tested with pytest alone
2. **YAML-driven configuration** — scenarios, tasks, faults are data, not code
3. **Standard ROS message types only** — no custom messages
4. **Policy interface is ML-ready** — same `reset/act/done` interface for scripted and learned policies
5. **Faults compose** — stack any combination in a profile
6. **Runs are self-contained** — each run directory has everything needed to understand and reproduce it
7. **Edge deployment is optional** — onnx/onnxruntime under [edge] extra, TensorRT runtime-detected, never a pip dependency
