# Implementation Plan — Phase 11: Physical Robot Support

## Summary

Phase 11 adds the software infrastructure to bridge Forge's simulation stack to real physical robots: hardware abstraction layer, safety system, sim-to-real transfer utilities, a physical robot runner, CLI, configuration, tests, and documentation updates. No actual hardware required — SimulatedRobot backed by ForgeSimulation is the primary test vehicle.

## Prerequisites

- All 328 existing tests passing
- Phases 1–10 complete

---

- [ ] 1. Create `forge_core/hardware.py` — Hardware Abstraction Layer
      Define RobotState dataclass (joint_positions, joint_velocities, ee_position, ee_orientation, gripper_state, timestamp) with to_dict/from_dict. Define RobotCommand dataclass (joint_targets, gripper_command, timestamp) with to_dict/from_dict. Define RobotStatus dataclass (connected, status, error_message, uptime, last_heartbeat) with to_dict. Define RobotInterface ABC with methods: connect(), disconnect(), get_state()->RobotState, send_command(cmd), emergency_stop(), get_status()->RobotStatus. Implement SimulatedRobot(RobotInterface) backed by ForgeSimulation — connect creates sim, get_state maps sim state, send_command calls set_joint_targets+step, emergency_stop sends zero targets.
      Files: `forge_core/hardware.py`
      Verify: `python3 -c "from forge_core.hardware import RobotInterface, SimulatedRobot, RobotState, RobotCommand"` succeeds; `python3 -m pytest tests/ -v` — all 328 existing tests still pass.

- [ ] 2. Create `forge_core/safety.py` — Safety Monitoring System
      Define SafetyConfig dataclass with Panda-appropriate defaults (joint limits, velocity limits, workspace bounds). Define SafetyViolation dataclass. Implement load_safety_config() loading from config/safety.yaml. Implement SafetyMonitor wrapping a RobotInterface: check_command validates joint limits, velocity limits, workspace bounds; is_safe returns bool; safe_send_command only sends if safe; emergency_stop delegates; violations are logged.
      Files: `forge_core/safety.py`
      Verify: `python3 -c "from forge_core.safety import SafetyMonitor, SafetyConfig"` succeeds; existing tests still pass.

- [ ] 3. Create `forge_core/sim2real.py` — Sim-to-Real Transfer Utilities
      Define DomainRandomizationConfig dataclass. Implement apply_domain_randomization(sim, config) modifying MuJoCo model params. Define TrajectoryComparison dataclass and compare_trajectories function computing RMSE/correlation. Define CalibrationResult dataclass and calibrate_joint_offsets/calibrate_workspace functions. Implement load_sim2real_config loading from config/robot.yaml.
      Files: `forge_core/sim2real.py`
      Verify: `python3 -c "from forge_core.sim2real import DomainRandomizationConfig, compare_trajectories, calibrate_joint_offsets"` succeeds; existing tests still pass.

- [ ] 4. Create `config/robot.yaml` and `config/safety.yaml`
      robot.yaml: robot type, scenario, steps_per_command, sim2real domain randomization params. safety.yaml: joint limits (Panda-appropriate), velocity limits, workspace bounds, max gripper force.
      Files: `config/robot.yaml`, `config/safety.yaml`
      Verify: `python3 -c "from forge_core.config import load_yaml; c=load_yaml('config/robot.yaml'); print(c['robot']['type'])"` prints "simulated".

- [ ] 5. Add `run_policy_on_robot` function to `forge_core/hardware.py`
      Add a runner function that executes a Policy on any RobotInterface. Reuses the existing Policy ABC. Integrates with SafetyMonitor and TrajectoryBuffer from recorder. Returns summary dict with steps, completed, safety_violations, e_stopped.
      Files: `forge_core/hardware.py` (append)
      Verify: Import succeeds; existing tests still pass.

- [ ] 6. Create `scripts/forge_robot.py` — Physical Robot CLI
      argparse CLI with subcommands: connect (--type simulated, --scenario), run (--policy, --task, --record), safety-check, calibrate, compare-sim, status. Follow forge_fleet.py pattern (sys.path hack, logging, main guard).
      Files: `scripts/forge_robot.py`
      Verify: `python3 scripts/forge_robot.py --help` prints help with subcommands.

- [ ] 7. Create `tests/unit/test_hardware.py` — Hardware abstraction tests
      At least 15 tests: RobotState/RobotCommand/RobotStatus dataclass creation and serialization, SimulatedRobot lifecycle (connect/disconnect/get_state/send_command/emergency_stop/get_status), RobotInterface ABC enforcement, run_policy_on_robot with NullPolicy.
      Files: `tests/unit/test_hardware.py`
      Verify: `python3 -m pytest tests/unit/test_hardware.py -v` — all pass.

- [ ] 8. Create `tests/unit/test_safety.py` — Safety system tests
      At least 12 tests: SafetyConfig defaults and from_dict, load_safety_config, SafetyViolation creation, SafetyMonitor.check_command (safe/joint violation/velocity violation/workspace violation), is_safe, safe_send_command (safe/blocked), emergency_stop delegation, violations log accumulation.
      Files: `tests/unit/test_safety.py`
      Verify: `python3 -m pytest tests/unit/test_safety.py -v` — all pass.

- [ ] 9. Create `tests/unit/test_sim2real.py` — Sim-to-real transfer tests
      At least 10 tests: DomainRandomizationConfig defaults/from_dict, apply_domain_randomization, compare_trajectories (identical/different/different lengths), calibrate_joint_offsets, calibrate_workspace, load_sim2real_config.
      Files: `tests/unit/test_sim2real.py`
      Verify: `python3 -m pytest tests/unit/test_sim2real.py -v` — all pass.

- [ ] 10. Update `docs/architecture.md` — Phase 11 section
       Add Phase 11 section after Phase 10: overview, hardware abstraction, safety system, sim-to-real transfer, configuration, CLI, integration points. Update opening paragraph to "Phases 1–11".
       Files: `docs/architecture.md`
       Verify: `grep 'Phase 11' docs/architecture.md` finds matches.

- [ ] 11. Update `docs/setup.md` — Physical Robot section
       Add Physical Robot Support section with CLI usage examples for all forge_robot.py subcommands. Update test count references.
       Files: `docs/setup.md`
       Verify: `grep 'forge_robot' docs/setup.md` finds matches.

- [ ] 12. Update `README.md` — Final status
       Change status to "Phase 11 — All Phases Complete". Update features list, What's included, Project Structure (add hardware.py, safety.py, sim2real.py, forge_robot.py, robot.yaml, safety.yaml), Phase Roadmap (Phase 11 Done), test count, Quick Start section with Physical Robot CLI examples.
       Files: `README.md`
       Verify: `grep 'All Phases Complete' README.md` finds match; `grep 'Done.*Physical robot' README.md` finds match.

- [ ] 13. Full verification
       Run `python3 -m pytest tests/ -v` — all 328 existing tests + all new Phase 11 tests pass (expect 365+ total). Zero failures.
       Files: (none — verification only)
       Verify: Exit code 0, total test count ≥ 365.
