# Forge — Architecture

## Overview

Forge is a robotics reliability platform built on ROS 2 and MuJoCo.
It supports a lifecycle of develop → simulate → evaluate → record →
detect failures → replay → improve.

The current implementation covers Phases 1–6:
simulation, task evaluation, telemetry/replay, failure injection,
camera-based perception, and learned policy via behavioral cloning.

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
## Design Principles

1. **forge_core has no ROS dependency** — everything can be tested with pytest alone
2. **YAML-driven configuration** — scenarios, tasks, faults are data, not code
3. **Standard ROS message types only** — no custom messages
4. **Policy interface is ML-ready** — same `reset/act/done` interface for scripted and learned policies
5. **Faults compose** — stack any combination in a profile
6. **Runs are self-contained** — each run directory has everything needed to understand and reproduce it
