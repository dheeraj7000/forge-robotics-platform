# Forge — Architecture

## Overview

Forge is a robotics reliability platform built on ROS 2 and MuJoCo.
It supports a lifecycle of develop → simulate → evaluate → record →
detect failures → replay → improve.

The current implementation covers Phases 1–4:
simulation, task evaluation, telemetry/replay, and failure injection.

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

## Design Principles

1. **forge_core has no ROS dependency** — everything can be tested with pytest alone
2. **YAML-driven configuration** — scenarios, tasks, faults are data, not code
3. **Standard ROS message types only** — no custom messages
4. **Policy interface is ML-ready** — same `reset/act/done` interface for scripted and learned policies
5. **Faults compose** — stack any combination in a profile
6. **Runs are self-contained** — each run directory has everything needed to understand and reproduce it
