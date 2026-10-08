# Forge

**Robotics Reliability & Deployment Platform**

Forge is a robotics reliability platform for developing, evaluating, debugging, and
deploying ML-powered robotic systems. It answers the question:

> *What happens after a model or robot policy leaves the notebook?*

## Current Status: Phase 2 — Task Execution & Evaluation

Phase 2 adds task execution, scripted policies, and an evaluation framework
on top of the Phase 1 simulation foundation.

### What's included

- MuJoCo simulation with a 7-DOF Panda arm, tabletop, and manipulable objects
- Three ROS 2 nodes: `forge_simulator`, `forge_controller`, `forge_state`
- Joint state and object state publishing via standard ROS message types
- Joint position command interface
- Deterministic scenario configuration (YAML)
- Reset functionality
- Health check (`forge doctor`)
- **Task specification system** (YAML task definitions with success criteria)
- **Scripted push policy** (deterministic, no ML)
- **Evaluation engine** with structured metrics and per-criterion results
- **Benchmark CLI** (`forge_eval.py`) for running tasks, suites, and saving results
- **Multiple scenarios** (basic_workspace, spread_workspace)
- **Test suite** (`manipulation-v1`) with 3 tasks
- Automated tests (60 passing)

### What's NOT included (intentionally)

No LLM, VLM, learned policy, cloud infrastructure, or physical robot.
Those belong to later phases.

## Quick Start

### Prerequisites

- Ubuntu 22.04/24.04
- Python 3.10+
- ROS 2 Humble (22.04) or Jazzy (24.04)

### Install

```bash
# Install Python deps
cd /root/forge
pip install mujoco numpy PyYAML pytest

# Build ROS workspace
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### Run

```bash
# Option A: Single process (recommended)
python3 scripts/run_forge.py

# Option B: ROS 2 launch
ros2 launch forge_sim forge_launch.py
```

### Verify

```bash
# Unit tests (no ROS needed)
python3 -m pytest tests/unit/ -v

# Health check
python3 scripts/forge_doctor.py

# Echo joint state (requires running system)
ros2 topic echo /forge/joint_states --once
```

### Evaluate tasks

```bash
# List available tasks and suites
python3 scripts/forge_eval.py --list

# Run a single task
python3 scripts/forge_eval.py --task pick_red_cube

# Run the manipulation suite
python3 scripts/forge_eval.py --suite manipulation-v1

# Save results to runs/ directory
python3 scripts/forge_eval.py --suite manipulation-v1 --save
```

### Send a command

```bash
ros2 topic pub /forge/joint_commands sensor_msgs/msg/JointState \
  "{position: [0.5, -0.3, 0.2, -1.5, 0.1, 1.2, 0.3]}" --once
```

### Reset

```bash
ros2 service call /forge/reset std_srvs/srv/Trigger
```

## Project Structure

```
forge/
├── README.md
├── pyproject.toml
├── forge_core/              # Shared Python library (no ROS dependency)
│   ├── config.py            # YAML configuration loading
│   ├── simulation.py        # MuJoCo simulation engine
│   ├── task.py              # Task specification loading
│   ├── policy.py            # Policy interface + scripted policies
│   ├── evaluator.py         # Success criteria evaluation
│   └── runner.py            # Task execution orchestrator
├── ros2_ws/src/
│   ├── forge_sim/           # Simulator node + object state + reset
│   ├── forge_control/       # Joint command controller
│   └── forge_state/         # Robot state publisher
├── simulation/
│   ├── models/              # Robot MJCF models
│   ├── worlds/              # Scene definitions
│   └── scenarios/           # YAML scenario configurations
├── tests/
│   ├── unit/                # MuJoCo-only tests (no ROS)
│   └── integration/         # Acceptance tests
├── scripts/
│   ├── run_forge.py         # Single-process launcher
│   ├── forge_doctor.py      # Health check
│   └── forge_eval.py        # Benchmark CLI
├── config/
│   └── default.yaml         # Default configuration
├── tasks/                   # Task YAML definitions
│   ├── pick_red_cube.yaml
│   ├── pick_blue_cube.yaml
│   ├── pick_both_cubes.yaml
│   ├── pick_red_cube_spread.yaml
│   └── suites/
│       └── manipulation-v1.yaml
├── runs/                    # Evaluation result outputs
└── docs/
    ├── architecture.md
    └── setup.md
```

## Documentation

- [Architecture](docs/architecture.md) — system design and component responsibilities
- [Setup Guide](docs/setup.md) — full installation and verification instructions

## License

MIT
