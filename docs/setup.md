# Forge — Setup Guide

## Prerequisites

- **OS**: Ubuntu 22.04 or 24.04
- **Python**: 3.10+
- **RAM**: 16 GB recommended
- **GPU**: Not required (Phases 1–4 are CPU-only)

## Step 1: Install ROS 2

### Ubuntu 22.04 (Humble)

```bash
sudo apt update && sudo apt install -y software-properties-common curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-humble-desktop
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source /opt/ros/humble/setup.bash
```

### Ubuntu 24.04 (Jazzy)

```bash
sudo apt update && sudo apt install -y software-properties-common curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-jazzy-desktop
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source /opt/ros/jazzy/setup.bash
```

## Step 2: Install Python Dependencies

```bash
cd /root/forge
pip install mujoco mujoco_menagerie numpy PyYAML opencv-python-headless
pip install pytest  # for testing
```

The first run will download the Menagerie Panda model (~3 MB) automatically.

## Step 2b: Machine Learning Dependencies

To use the learned policy and training features (Phase 6), install the
ML extras:

```bash
pip install -e '.[ml]'
```

This adds PyTorch (>= 2.0). Alternatively, install it directly:

```bash
pip install torch>=2.0
```

PyTorch is only required for training and running learned policies.
All other Forge features work without it.


## Step 2c: Edge Deployment Dependencies

To use edge deployment features (Phase 9), install the edge extras:

```bash
pip install -e '.[edge]'
```
This adds `onnx` and `onnxruntime`. TensorRT must be installed
separately from [NVIDIA's documentation](https://developer.nvidia.com/tensorrt)
if GPU-accelerated inference is needed.

Edge deployment features work without TensorRT installed (ONNX-only
mode). TensorRT is detected at runtime and used when available.

## Step 3: Verify Installation

```bash
cd /root/forge

# Run all tests (no ROS needed)
python3 -m pytest tests/ -v

# Health check (checks MuJoCo + simulation without ROS)
python3 scripts/forge_doctor.py
```

## Step 4: Build ROS 2 Workspace (optional)

Only needed if you want to use the ROS 2 topic/service interface.

```bash
cd /root/forge/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Step 5: Run the System

### Option A: Single-process (recommended for development)

```bash
python3 scripts/run_forge.py
```

### Option B: ROS 2 launch

```bash
source /root/forge/ros2_ws/install/setup.bash
ros2 launch forge_sim forge_launch.py
```

## Step 6: Verify Running System

In a second terminal:

```bash
ros2 topic list
ros2 topic echo /forge/joint_states --once
ros2 service call /forge/reset std_srvs/srv/Trigger
python3 scripts/forge_doctor.py
```

## Task Evaluation

```bash
# List tasks and suites
python3 scripts/forge_eval.py --list

# Run a task
python3 scripts/forge_eval.py --task pick_red_cube

# Run with recording
python3 scripts/forge_eval.py --task pick_red_cube --record

# Run the manipulation suite
python3 scripts/forge_eval.py --suite manipulation-v1

# Run with fault injection
python3 scripts/forge_eval.py --task pick_red_cube --fault object_moved
```

## Chaos Testing

```bash
# List fault profiles and types
python3 scripts/forge_chaos.py --list

# Run with a fault profile
python3 scripts/forge_chaos.py --task pick_red_cube --fault object_moved

# Quick inline fault
python3 scripts/forge_chaos.py --task pick_red_cube --inject controller_dropout --at 3.0 --duration 2.0

# Compare baseline vs faulted
python3 scripts/forge_chaos.py --task pick_red_cube --fault cascading --compare
```

## Replay Recorded Runs

```bash
# List recorded runs
python3 scripts/forge_replay.py --list

# Show run summary
python3 scripts/forge_replay.py runs/run_00001/

# Inspect state at step 50
python3 scripts/forge_replay.py runs/run_00001/ --step 50

# Inspect state at time 5.0s
python3 scripts/forge_replay.py runs/run_00001/ --time 5.0
```

## Perception

```bash
# Render a camera frame from the simulation
python3 scripts/forge_perception.py --scenario basic_workspace --render -o frame.png

# Detect colored objects
python3 scripts/forge_perception.py --scenario basic_workspace --detect

# Detect and annotate the image
python3 scripts/forge_perception.py --scenario basic_workspace --detect --annotate -o detections.png

# Custom camera parameters
python3 scripts/forge_perception.py --scenario basic_workspace --detect \
  --azimuth 135 --elevation -25 --distance 1.8
```

## Training a Learned Policy

```bash
# Collect demonstrations from the scripted policy and train
python3 scripts/forge_train.py --task pick_red_cube --episodes 5

# Train from previously recorded runs
python3 scripts/forge_train.py --from-run runs/run_00001/ runs/run_00002/

# Train with a YAML config file
python3 scripts/forge_train.py --config config/training.yaml --task pick_red_cube --episodes 1

# List saved models
python3 scripts/forge_train.py --list-models

# Evaluate the trained policy
python3 scripts/forge_eval.py --task pick_red_cube --policy learned --model-path models/bc_policy.pt
```

> **Note**: Glob patterns in `--from-run` are expanded by the shell. On
> Windows, use explicit paths or run inside WSL.

## Experiment Tracking & Model Registry

Forge automatically tracks experiments and registers models during training.
No external dependencies (MLflow/W&B) are required — everything is stored
as JSON on disk.

### Experiment Management

```bash
# List all experiments (filter by status or tag)
python3 scripts/forge_registry.py experiments list
python3 scripts/forge_registry.py experiments list --status completed
python3 scripts/forge_registry.py experiments list --tag pick_red_cube

# Show full details for an experiment
python3 scripts/forge_registry.py experiments show exp_00001

# Compare two experiments (metrics side by side)
python3 scripts/forge_registry.py experiments compare exp_00001 exp_00002

# Delete an experiment
python3 scripts/forge_registry.py experiments delete exp_00003
```

### Model Registry

```bash
# List all registered models (filter by name or stage)
python3 scripts/forge_registry.py models list
python3 scripts/forge_registry.py models list --stage validated

# Show model details
python3 scripts/forge_registry.py models show bc_pick_red_cube
python3 scripts/forge_registry.py models show bc_pick_red_cube --version 1

# Promote a model through the stage lifecycle
python3 scripts/forge_registry.py models promote bc_pick_red_cube --version 1 --stage validated
python3 scripts/forge_registry.py models promote bc_pick_red_cube --version 1 --stage production

# Delete a model entry
python3 scripts/forge_registry.py models delete bc_pick_red_cube --version 1
```

### Training with Experiment Tracking

Experiments are tracked automatically by default when using `forge_train.py`:

```bash
# Training auto-creates an experiment and registers the model
python3 scripts/forge_train.py --task pick_red_cube --episodes 5

# Disable tracking
python3 scripts/forge_train.py --task pick_red_cube --episodes 5 --no-track

# Custom experiment and model names
python3 scripts/forge_train.py --task pick_red_cube --episodes 5 \
  --experiment-name my_experiment --model-name my_model
```

### Evaluating from Registry

```bash
# Load a model from the registry by name (uses latest version)
python3 scripts/forge_eval.py --task pick_red_cube --policy learned \
  --registry-model bc_pick_red_cube

# Load a specific version
python3 scripts/forge_eval.py --task pick_red_cube --policy learned \
  --registry-model bc_pick_red_cube --registry-version 1
```

Stage lifecycle: `training → validated → production → archived`.
Only adjacent transitions are allowed. The only back-transition is
`archived → validated` (re-activation).


## Regression Testing

Forge includes a regression evaluation system to detect performance
degradations between code changes. Baselines capture known-good
evaluation results; the regression checker compares current results
against these baselines.

### Configuration

Regression thresholds are defined in `config/regression.yaml`:

```yaml
metric_tolerance: 0.1    # 10% tolerance for metric degradation
allow_new_failures: false # fail if a previously passing task now fails
suite: manipulation-v1   # default suite name
```

### Baseline Management

```bash
# Create a new baseline from current evaluation results
python3 scripts/forge_regression.py --create-baseline --suite manipulation-v1

# View the saved baseline
python3 scripts/forge_regression.py --show-baseline --suite manipulation-v1

# Update baseline after intentional changes
python3 scripts/forge_regression.py --update-baseline --suite manipulation-v1

# Compare two baselines side by side
python3 scripts/forge_regression.py --compare baselines/old.json baselines/new.json
```

Baselines are stored as JSON in the `baselines/` directory and should
be committed to the repository.

### Regression Checking

```bash
# Run regression check against the saved baseline
python3 scripts/forge_regression.py --check --suite manipulation-v1

# Use a custom regression config
python3 scripts/forge_regression.py --check --suite manipulation-v1 \
  --threshold-config config/regression.yaml

# Override the policy used for evaluation
python3 scripts/forge_regression.py --check --suite manipulation-v1 \
  --policy scripted_pick_and_place
```

The check command exits with code 0 if no regressions are detected,
and code 1 if any task has regressed (newly failing or metrics degraded
beyond tolerance).

## Edge Deployment

Forge can export trained PyTorch policies to ONNX for edge deployment,
with optional TensorRT optimization.

```bash
# Export a trained model to ONNX
python3 scripts/forge_deploy.py --export --model models/bc_policy.pt --output models/bc_policy.onnx

# Validate ONNX model against PyTorch checkpoint
python3 scripts/forge_deploy.py --validate --model models/bc_policy.onnx --checkpoint models/bc_policy.pt

# Benchmark inference latency
python3 scripts/forge_deploy.py --benchmark --model models/bc_policy.onnx --n-iterations 200

# Package model for deployment (ONNX + manifest)
python3 scripts/forge_deploy.py --package --model models/bc_policy.onnx --output deploy/ --name bc_pick_red_cube

# Optimize with TensorRT (requires TensorRT installed)
python3 scripts/forge_deploy.py --optimize --model models/bc_policy.onnx --output models/bc_policy.trt --precision fp16

# Export a model from the registry to ONNX
python3 scripts/forge_deploy.py --export-registry --name bc_pick_red_cube --version 1 --output models/exported.onnx
```
## CI/CD

Forge uses GitHub Actions for continuous integration and regression
evaluation. Workflow files are in `.github/workflows/`.

### Test Workflow (`ci.yml`)

Runs on every push and pull request to `main`.

- **Python matrix**: 3.10, 3.11
- **Headless rendering**: `MUJOCO_GL=egl`
- **PyTorch**: CPU-only variant for faster installs
- **Caching**: pip dependencies cached via `actions/cache@v4`
- **Command**: `python3 -m pytest tests/ -v`

### Regression Evaluation Workflow (`regression.yml`)

Runs on pull requests to `main` and can be triggered manually via
`workflow_dispatch`.

- **Python**: 3.11
- **Command**: `python3 scripts/forge_regression.py --check --suite manipulation-v1`
- **On failure**: uploads `baselines/` directory as a build artifact
  for debugging

### Updating Baselines for CI

When you intentionally change evaluation behavior (new policy, different
task parameters), update the baseline before merging:

```bash
# Run evaluation and save new baseline
python3 scripts/forge_regression.py --update-baseline --suite manipulation-v1

# Commit the updated baseline
git add baselines/manipulation-v1.json
git commit -m "chore: update regression baseline"
```

## Running Tests

```bash
# All tests (272 tests, no ROS needed)
python3 -m pytest tests/ -v

# Specific module
python3 -m pytest tests/unit/test_faults.py -v

# Specific test class
python3 -m pytest tests/unit/test_simulation.py::TestReset -v
```
