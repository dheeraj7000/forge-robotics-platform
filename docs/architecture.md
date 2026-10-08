# Forge — Architecture (Phase 1)

## Overview

Forge Phase 1 implements the **Robotics Simulation Foundation**: a reproducible ROS 2 system
that controls a simulated Franka Panda arm in MuJoCo and exposes robot/object state through
ROS topics.

## System Diagram

```
                     ┌─────────────────────┐
                     │   ROS 2 Command      │
                     │  /forge/joint_commands│
                     └─────────┬────────────┘
                               │
                               ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ forge_state   │◄───│ forge_simulator   │◄───│ forge_controller  │
│               │    │                  │    │                  │
│ Publishes:    │    │ Owns:            │    │ Receives:        │
│ - JointState  │    │ - MuJoCo model   │    │ - Joint commands │
│ - EE Pose     │    │ - Sim stepping   │    │ Applies:         │
│               │    │ - Object state   │    │ - Actuator ctrl  │
│               │    │ - Reset service  │    │                  │
└──────┬───────┘    └──────┬───────────┘    └──────────────────┘
       │                   │
       ▼                   ▼
  /forge/joint_states  /forge/object_states
  /forge/ee_pose       /forge/reset (service)
```

## Component Responsibilities

### forge_core (Python library, no ROS dependency)

- **config.py**: Loads YAML scenario/configuration files
- **simulation.py**: MuJoCo simulation wrapper — load, step, reset, query state

### forge_simulator (ROS 2 node)

- Owns the MuJoCo simulation lifecycle
- Publishes object ground-truth state (`geometry_msgs/PoseArray`)
- Publishes end-effector pose (`geometry_msgs/PoseStamped`)
- Provides reset service (`std_srvs/Trigger`)

### forge_controller (ROS 2 node)

- Subscribes to `/forge/joint_commands` (`sensor_msgs/JointState`)
- Applies joint position targets to MuJoCo actuators

### forge_state (ROS 2 node)

- Publishes joint state (`sensor_msgs/JointState`) on `/forge/joint_states`
- Publishes end-effector pose on `/forge/ee_pose`

## ROS Topics & Services

| Interface | Type | Direction |
|-----------|------|-----------|
| `/forge/joint_states` | `sensor_msgs/JointState` | Published |
| `/forge/object_states` | `geometry_msgs/PoseArray` | Published |
| `/forge/ee_pose` | `geometry_msgs/PoseStamped` | Published |
| `/forge/joint_commands` | `sensor_msgs/JointState` | Subscribed |
| `/forge/reset` | `std_srvs/Trigger` | Service |

## Message Types

All messages use **standard ROS types**. No custom message types are used in Phase 1.

## Simulation Engine

- **Physics**: MuJoCo with implicit integrator, 0.002s timestep
- **Robot**: Franka Panda 7-DOF arm + 2-finger gripper (simplified MJCF)
- **Control**: Position actuators with PD gains (kp=100, kv=20)
- **Scene**: Tabletop with objects defined by scenario YAML

## Configuration

All scenario parameters are in YAML files under `simulation/scenarios/`.
No hardcoded positions, gains, or model paths.

## Design Decisions

1. **Single-process composition**: All three nodes share one MuJoCo instance in the
   recommended run mode. This avoids inter-process state synchronization.

2. **forge_core has no ROS dependency**: The simulation engine can be tested with
   pytest alone, without launching ROS.

3. **Standard ROS message types only**: Per FR-3/FR-4, no custom msgs.

4. **Position control**: Phase 1 uses position actuators — motion planning is not
   required (FR-5).

5. **Simplified Panda MJCF**: A kinematically accurate but geometrically simplified
   model is used. The joint limits match the real Panda. Can be swapped for the
   MuJoCo Menagerie model.


---

# Phase 2 — Task Execution & Evaluation

## Overview

Phase 2 adds the ability to define manipulation tasks, run them with policies,
and evaluate success/failure with structured metrics.

## Architecture

```
Task YAML
  ↓
TaskSpec (task.py)
  ↓
Runner (runner.py)
  ├── Load scenario → ForgeSimulation
  ├── Reset simulation
  ├── Reset policy
  ├── Loop: policy.act() → sim.set_joint_targets() → sim.step()
  ├── Evaluate criteria → Evaluator (evaluator.py)
  └── Return EvalResult
  ↓
CLI (forge_eval.py)
  ├── --task, --suite, --all
  ├── Summary table
  └── JSON result files (runs/)
```

## New Components

### forge_core/task.py — Task Specification
- Loads YAML task definitions from `tasks/`
- Defines success criteria (object_in_zone, object_above_height, etc.)
- Supports test suites (`tasks/suites/`)

### forge_core/policy.py — Policy Interface
- Abstract `Policy` base class: `reset()`, `act()`, `done`
- `ScriptedPickAndPlace`: deterministic waypoint push policy
- `NullPolicy`: do-nothing baseline
- Policy registry for CLI access

### forge_core/evaluator.py — Evaluation Engine
- Checks success criteria against simulation state
- Returns structured `EvalResult` with per-criterion results and metrics
- Criterion types: `object_in_zone`, `object_above_height`, `object_distance_to`, `robot_at_home`

### forge_core/runner.py — Task Runner
- Orchestrates: load → reset → run policy → evaluate
- Supports single tasks, suites, and result saving

### scripts/forge_eval.py — Benchmark CLI
- `forge_eval.py --task pick_red_cube`
- `forge_eval.py --suite manipulation-v1`
- `forge_eval.py --all --save`
- `forge_eval.py --list`

## Design Decisions

1. **No ROS dependency in task execution**: The runner operates directly on
   `ForgeSimulation`, keeping evaluation fast and testable without ROS.

2. **YAML task definitions**: Tasks are data, not code. New tasks can be added
   without modifying Python.

3. **Policy interface designed for future ML**: The `Policy.act()` interface
   takes the simulation and returns joint targets — same interface a learned
   policy will use.

4. **Scripted policy uses push strategy**: The simplified Panda MJCF model's
   workspace geometry makes grasping unreliable. The scripted policy pushes
   the cube toward the target using arm sweep. A proper Menagerie model or
   IK solver would enable true pick-and-place.

5. **Suite results show PASS/FAIL mix**: The `manipulation-v1` suite produces
   1 PASS (red cube) and 2 FAILs (blue cube, spread scenario), demonstrating
   the evaluator correctly distinguishes success from failure.
