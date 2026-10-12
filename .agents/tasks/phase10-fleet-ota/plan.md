# Implementation Plan — Phase 10: Fleet & OTA

## Design Decisions

**Module structure:** Two new core modules (forge_core/fleet.py and forge_core/ota.py) following the established Manager-class + JSON-persistence pattern from model_registry.py and experiment.py. The Fleet Manager owns device state; the OTA Manager handles update packages and rollouts, delegating device mutations to the Fleet Manager. This matches the existing separation of concerns (experiment.py tracks experiments, model_registry.py tracks models, they reference each other by ID).

**Persistence layout:** fleet/devices.json for fleet state (single-file, same as registry/registry.json). ota/packages.json and ota/rollouts.json for OTA state, plus ota/history/{device_id}.json for per-device audit logs. All under PROJECT_ROOT, all JSON.

**Rollout simulation:** Since all operations are local/simulated, execute_rollout() walks the target device list, checks compatibility, and records results synchronously. Device status transitions are: online → updating → online (success) or error (failure). This gives realistic interfaces without any networking.

**CLI design:** Single script scripts/forge_fleet.py with three subcommand groups (devices, ota, overview), following forge_registry.py's argparse+subparsers pattern. This keeps fleet + OTA operations in one place rather than splitting across two scripts.

**Integration:** OTA packages reference Phase 9 deployment packages by reading manifest.yaml from the source directory. Device entries track current_model/current_model_version/experiment_id, linking to the model registry and experiment tracker. No import of ROS modules anywhere.

---

- [ ] 1. Create the Fleet Manager module at forge_core/fleet.py.
      DeviceInfo dataclass with device_id, name, device_type, capabilities, status, tags, groups,
      last_seen, registered_at, current_model, current_model_version, experiment_id, metadata.
      FleetManager class with JSON persistence in fleet/devices.json and methods: register_device,
      deregister_device, get_device, update_status, list_devices (filter by status/group/tag),
      add_tag, remove_tag, add_to_group, remove_from_group, update_device_model, get_fleet_summary.
      VALID_STATUSES = ('online', 'offline', 'updating', 'error').
      Files: forge_core/fleet.py
      Verify: `python3 -c "from forge_core.fleet import FleetManager, DeviceInfo, VALID_STATUSES"` — imports succeed.

- [ ] 2. Create the OTA Update System module at forge_core/ota.py.
      UpdatePackage dataclass (package_id, version, model_name, model_version, source_package,
      requirements, created_at, metadata). RolloutResult dataclass (device_id, status, previous_version,
      new_version, message, timestamp). OTAManager class with persistence in ota/ and methods:
      create_update_package, get_update_package, list_update_packages, check_compatibility,
      create_rollout, execute_rollout (simulated), get_rollout, list_rollouts, rollback_rollout,
      get_device_history. Includes parse_version/compare_versions for semver. Reads manifest.yaml
      from Phase 9 deployment package dirs. Integrates with FleetManager for device status/model updates.
      Files: forge_core/ota.py
      Verify: `python3 -c "from forge_core.ota import OTAManager, UpdatePackage, RolloutResult, parse_version"` — imports succeed.

- [ ] 3. Create unit tests for Fleet Manager at tests/unit/test_fleet.py.
      ~28 tests covering: device registration (including duplicates, metadata, tags/groups),
      retrieval (get, list with filters, sorting), status updates (valid/invalid), tags & groups
      (add/remove), model tracking, fleet summary, deregistration. All use tmp_path for isolation.
      Files: tests/unit/test_fleet.py
      Verify: `python3 -m pytest tests/unit/test_fleet.py -v` — all tests pass.

- [ ] 4. Create unit tests for OTA Update System at tests/unit/test_ota.py.
      ~28 tests covering: package creation (with/without manifest, sequential IDs), compatibility
      checking (compatible, missing capabilities, version comparison), rollout creation (immediate,
      canary, staged, by group/tag), rollout execution (simulated deployment, canary partial,
      device model updates, incompatible skips), rollback (revert device model), device history
      (recorded on execute/rollback), version parsing/comparison, rollout listing with filters.
      Files: tests/unit/test_ota.py
      Verify: `python3 -m pytest tests/unit/test_ota.py -v` — all tests pass.

- [ ] 5. Create the Fleet & OTA CLI script at scripts/forge_fleet.py.
      Three subcommand groups: devices (register, deregister, list, status, tag, group),
      ota (create-package, deploy, status, rollback, history, list-packages, list-rollouts),
      overview (fleet summary dashboard). Follows forge_registry.py pattern: argparse, subparsers,
      setup_logging, tabular output with column headers and separators.
      Files: scripts/forge_fleet.py
      Verify: `python3 scripts/forge_fleet.py --help` — prints help without error.

- [ ] 6. Create config/fleet.yaml with default fleet configuration.
      Fields: default_strategy, canary_percentage, auto_rollback, status_check_interval.
      Files: config/fleet.yaml
      Verify: `python3 -c "from forge_core.config import load_yaml; print(load_yaml('config/fleet.yaml'))"` — loads without error.

- [ ] 7. Update documentation: add Phase 10 section to docs/architecture.md,
      update README.md (status, features, structure, test count, roadmap),
      update docs/setup.md (Fleet & OTA section, test count).
      Files: docs/architecture.md, README.md, docs/setup.md
      Verify: `python3 -m pytest tests/ -v` — all tests pass, count updated in docs matches actual.

- [ ] 8. Run full test suite and confirm all tests pass.
      Verify: `python3 -m pytest tests/ -v` — all 272 existing + ~56 new tests pass (total ~328).
