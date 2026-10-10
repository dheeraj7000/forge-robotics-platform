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

## Running Tests

```bash
# All tests (88 tests, no ROS needed)
python3 -m pytest tests/ -v

# Specific module
python3 -m pytest tests/unit/test_faults.py -v

# Specific test class
python3 -m pytest tests/unit/test_simulation.py::TestReset -v
```
