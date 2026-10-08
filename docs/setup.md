# Forge — Setup Guide

## Prerequisites

- **OS**: Ubuntu 22.04 or 24.04
- **Python**: 3.10+
- **RAM**: 16 GB (recommended)
- **GPU**: Not required for Phase 1

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
pip install mujoco numpy PyYAML
pip install pytest pytest-timeout  # for testing
```

## Step 3: Verify Installation

### Quick check (no ROS needed)

```bash
cd /root/forge
python3 -m pytest tests/unit/ -v
```

### Health check

```bash
python3 scripts/forge_doctor.py
```

## Step 4: Build ROS 2 Workspace

```bash
cd /root/forge/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Step 5: Run the System

### Option A: Single-process (recommended for development)

```bash
cd /root/forge
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
# Check topics
ros2 topic list

# Echo joint state
ros2 topic echo /forge/joint_states --once

# Echo object state
ros2 topic echo /forge/object_states --once

# Send a joint command
ros2 topic pub /forge/joint_commands sensor_msgs/msg/JointState \
  "{position: [0.5, -0.3, 0.2, -1.5, 0.1, 1.2, 0.3]}" --once

# Reset simulation
ros2 service call /forge/reset std_srvs/srv/Trigger

# Full health check
python3 scripts/forge_doctor.py
```

## Running Tests

```bash
cd /root/forge

# Unit tests (no ROS needed)
python3 -m pytest tests/unit/ -v

# All tests
python3 -m pytest tests/ -v

# Specific test
python3 -m pytest tests/unit/test_simulation.py::TestReset -v
```
