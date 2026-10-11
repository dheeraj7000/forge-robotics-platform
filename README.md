# Forge

**Robotics Reliability & Deployment Platform**

Forge is a robotics reliability platform for developing, evaluating, debugging, and
deploying ML-powered robotic systems. It answers the question:

> *What happens after a model or robot policy leaves the notebook?*

## Current Status: Phase 7 — ML Infrastructure

Phases 1–7 are complete: simulation foundation, task evaluation,
telemetry/replay, chaos testing, camera-based perception, learned
policy via behavioral cloning, and ML infrastructure (experiment
tracking + model registry).

### What's included

- **Simulation** — MuJoCo Menagerie Franka Panda (mesh collision, accurate kinematics)
- **ROS 2 integration** — three nodes publishing joint/object state via standard messages
- **Task evaluation** — YAML task definitions with measurable success criteria
- **Scripted policy** — deterministic pick-and-place waypoint sequence
- **Benchmark CLI** — run tasks, suites, save results
- **Telemetry** — per-cycle trajectory recording (joint state, EE pose, objects, actions)
- **Replay** — load and inspect any recorded run at any timestep
- **Chaos testing** — 6 fault types, composable YAML profiles, compare mode
- **Perception** — MuJoCo offscreen camera rendering, OpenCV HSV color detection, depth-based 3D position estimation
- **Learned policy** — PyTorch MLP trained via behavioral cloning from scripted demonstrations, configurable training pipeline with YAML config
- **Experiment tracking** — file-based experiment tracker logging config, metrics, git commit, and tags per training run (JSON, zero external deps)
- **Model registry** — versioned model registry with stage lifecycle (training → validated → production → archived), single-file JSON store
- **215 automated tests**, all passing

### What's NOT included (intentionally)

No LLM, VLM, cloud infrastructure, or physical robot.
Those belong to later phases.

## Quick Start

### Prerequisites

- Ubuntu 22.04/24.04
- Python 3.10+
- ROS 2 Humble or Jazzy (optional — evaluation works without ROS)

### Install

```bash
pip install mujoco mujoco_menagerie numpy PyYAML opencv-python-headless pytest
pip install -e '.[ml]'  # PyTorch for learned policies (optional)
```

### Verify

```bash
python3 -m pytest tests/ -v     # 215 tests, no ROS needed
python3 scripts/forge_doctor.py  # health check
```

### Evaluate a task

```bash
python3 scripts/forge_eval.py --task pick_red_cube
python3 scripts/forge_eval.py --suite manipulation-v1
```

### Record a run

```bash
python3 scripts/forge_eval.py --task pick_red_cube --record
python3 scripts/forge_replay.py --list
python3 scripts/forge_replay.py runs/run_00001/
```

### Chaos testing

```bash
python3 scripts/forge_chaos.py --list
python3 scripts/forge_chaos.py --task pick_red_cube --fault object_moved
python3 scripts/forge_chaos.py --task pick_red_cube --fault cascading --compare
python3 scripts/forge_chaos.py --task pick_red_cube --inject controller_dropout --at 3 --duration 2
```


### Perception

```bash
python3 scripts/forge_perception.py --scenario basic_workspace --render -o frame.png
python3 scripts/forge_perception.py --scenario basic_workspace --detect --annotate -o detections.png
```

### Training a Learned Policy

```bash
python3 scripts/forge_train.py --task pick_red_cube --episodes 5
python3 scripts/forge_train.py --from-run runs/run_00001/ runs/run_00002/
python3 scripts/forge_train.py --config config/training.yaml --task pick_red_cube --episodes 1
python3 scripts/forge_train.py --list-models
python3 scripts/forge_eval.py --task pick_red_cube --policy learned --model-path models/bc_policy.pt
```

### Experiment Tracking & Model Registry

```bash
# List all experiments
python3 scripts/forge_registry.py experiments list

# Show a specific experiment
python3 scripts/forge_registry.py experiments show exp_00001

# Compare two experiments side by side
python3 scripts/forge_registry.py experiments compare exp_00001 exp_00002

# List registered models
python3 scripts/forge_registry.py models list

# Show model details
python3 scripts/forge_registry.py models show bc_pick_red_cube

# Promote a model through stages
python3 scripts/forge_registry.py models promote bc_pick_red_cube --version 1 --stage validated

# Evaluate a model from the registry
python3 scripts/forge_eval.py --task pick_red_cube --policy learned --registry-model bc_pick_red_cube
```

### ROS 2 (optional)

```bash
python3 scripts/run_forge.py                              # start system
ros2 topic echo /forge/joint_states --once                # read state
ros2 service call /forge/reset std_srvs/srv/Trigger       # reset
```

## Project Structure

```
forge/
├── forge_core/              # Core library (no ROS dependency)
│   ├── simulation.py        # MuJoCo simulation engine
│   ├── menagerie.py         # Menagerie model locator
│   ├── task.py              # Task specification loading
│   ├── policy.py            # Policy interface + scripted policies
│   ├── evaluator.py         # Success criteria evaluation
│   ├── runner.py            # Task execution orchestrator
│   ├── faults.py            # Fault injection system
│   ├── recorder.py          # Trajectory recording
│   ├── replay.py            # Run replay and inspection
│   ├── perception.py        # Camera rendering + object detection
│   ├── learned_policy.py    # PyTorch learned policy + BC training
│   ├── experiment.py        # Experiment tracking (JSON-based)
│   ├── model_registry.py    # Model registry with stage lifecycle
│   └── config.py            # YAML configuration loading
├── ros2_ws/src/
│   ├── forge_sim/           # Simulator node + object state + reset
│   ├── forge_control/       # Joint command controller
│   └── forge_state/         # Robot state publisher
├── simulation/
│   ├── models/              # Robot MJCF models (Menagerie)
│   ├── worlds/              # Scene definitions
│   └── scenarios/           # YAML scenario configurations
├── tasks/                   # Task YAML definitions
│   ├── pick_red_cube.yaml
│   ├── pick_blue_cube.yaml
│   ├── pick_both_cubes.yaml
│   ├── pick_red_cube_spread.yaml
│   └── suites/
│       └── manipulation-v1.yaml
├── faults/                  # Fault profile YAML definitions
│   ├── object_moved.yaml
│   ├── sensor_noise.yaml
│   ├── controller_dropout.yaml
│   ├── gravity_shift.yaml
│   └── cascading.yaml
├── experiments/             # Experiment tracking data (JSON per experiment)
├── registry/                # Model registry (registry.json)
├── runs/                    # Recorded run outputs
├── models/                  # Saved model .pt files
├── tests/
│   ├── unit/                # 214 unit tests
│   └── integration/         # Acceptance tests
├── scripts/
│   ├── run_forge.py         # Single-process ROS 2 launcher
│   ├── forge_doctor.py      # Health check
│   ├── forge_eval.py        # Task evaluation CLI
│   ├── forge_chaos.py       # Chaos testing CLI
│   ├── forge_replay.py      # Run replay CLI
│   ├── forge_perception.py  # Perception CLI
│   ├── forge_train.py       # Training CLI
│   └── forge_registry.py   # Experiment & model registry CLI
├── config/
│   ├── default.yaml
│   └── training.yaml        # ML training configuration
└── docs/
    ├── architecture.md      # System design (all phases)
    └── setup.md             # Installation and usage guide
```

## Fault Types

| Type | Effect |
|------|--------|
| `object_moved` | Teleport an object to a new position mid-run |
| `sensor_noise` | Gaussian noise on joint position readings |
| `sensor_delay` | Return stale robot state for N cycles |
| `actuator_stuck` | Freeze specific joints at current value |
| `controller_dropout` | Zero all commands — robot goes limp |
| `gravity_shift` | Change gravity vector mid-run |

## Phase Roadmap

| Phase | Status | Capability |
|-------|--------|-----------|
| 1 | Done | Simulation foundation |
| 2 | Done | Task execution & evaluation |
| 3 | Done | Telemetry & replay |
| 4 | Done | Failure injection / chaos testing |
| 5 | Done | Perception (camera + OpenCV) |
| 6 | Done | Learned policy (PyTorch) |
| 7 | Done | ML infrastructure (experiment tracking, model registry) |
| 8 | — | CI/CD (regression evaluation) |
| 9 | — | Edge deployment (ONNX, TensorRT) |
| 10 | — | Fleet & OTA |
| 11 | — | Physical robot |

## Documentation

- [Architecture](docs/architecture.md) — system design across all phases
- [Setup Guide](docs/setup.md) — installation, usage, and all CLI commands

## License

MIT
