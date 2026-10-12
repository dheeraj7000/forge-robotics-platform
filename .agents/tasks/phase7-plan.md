# Phase 7 Implementation Plan — ML Infrastructure (Experiment Tracking + Model Registry)

## Overview

Phase 7 adds a self-contained, file-based ML infrastructure layer to Forge: an experiment tracker
that records every training run with full metadata, and a local model registry with versioning
and stage management. Both integrate into the existing training/eval pipeline and are managed via
a new CLI script. Zero external ML tracking dependencies (no MLflow/W&B). All data is JSON.

## Design Decisions

1. **One JSON file per experiment** (`experiments/exp_XXXXX.json`) — matches the `runs/run_XXXXX/`
   pattern from recorder.py and keeps experiments human-readable and individually loadable.
   Sequential IDs use the same `get_next_run_id()` scanning pattern from recorder.py.

2. **Single registry file** (`registry/registry.json`) — the model registry is a single JSON
   file containing a list of model entries. Unlike experiments (which can number in hundreds),
   the registry is small and benefits from atomic reads. Each entry tracks name, version, stage,
   checkpoint path, and links to the experiment that produced it.

3. **Stage transitions are explicit and enforced**: training → validated → production → archived,
   with archived → validated as the only back-transition (re-activation). Direct jumps
   (e.g., training → production) are rejected.

4. **Integration via optional parameters, not hard coupling**: `train_bc()` gains optional
   `experiment_tracker` and `experiment_id` parameters. Callers that don't pass them get
   identical behavior to today. The CLI handles the orchestration.

5. **Registry CLI as a separate script** (`scripts/forge_registry.py`) with subcommands,
   not flags bolted onto forge_train.py. This keeps forge_train.py focused on training
   and follows the pattern of specialized CLI scripts (forge_chaos.py, forge_perception.py).

## On-Disk Formats

### Experiment Record (`experiments/exp_00001.json`)

```json
{
  "experiment_id": "exp_00001",
  "name": "bc_pick_red_cube_20240101_120000",
  "timestamp": "2024-01-01T12:00:00",
  "status": "completed",
  "git_commit": "abc123def456...",
  "config": {
    "epochs": 100, "batch_size": 64, "lr": 0.001,
    "validation_split": 0.1, "hidden_layers": [256, 256, 128]
  },
  "dataset_info": {
    "n_samples": 1500,
    "source": "collected",
    "source_runs": [],
    "target_object": "red_cube",
    "task": "pick_red_cube"
  },
  "metrics": {
    "final_loss": 0.001234,
    "final_val_loss": 0.001456,
    "loss_history": [0.1, 0.05, 0.02, "..."],
    "val_history": [0.12, 0.06, 0.03, "..."],
    "best_epoch": 95,
    "best_loss": 0.001200
  },
  "model_path": "models/bc_policy.pt",
  "duration_seconds": 45.3,
  "tags": ["pick_red_cube", "bc"],
  "error": null
}
```

### Registry File (`registry/registry.json`)

```json
{
  "models": [
    {
      "name": "bc_pick_red_cube",
      "version": 1,
      "stage": "training",
      "checkpoint_path": "models/bc_policy.pt",
      "experiment_id": "exp_00001",
      "registered_at": "2024-01-01T12:00:05",
      "description": "BC policy for pick_red_cube",
      "metadata": {
        "final_loss": 0.001234,
        "epochs": 100
      }
    }
  ]
}
```

## Implementation Steps

- [ ] 1. Create the ExperimentTracker class in `forge_core/experiment.py`.
      File-based tracker storing one JSON per experiment under `experiments/`.
      Includes: `_get_git_commit()`, `create_experiment()`, `complete_experiment()`,
      `fail_experiment()`, `get_experiment()`, `list_experiments()`,
      `compare_experiments()`, `delete_experiment()`. Uses sequential IDs
      (exp_00001, exp_00002, ...) following the recorder.py pattern.
      Files: `/root/forge/forge_core/experiment.py`
      Verify: `python3 -c "from forge_core.experiment import ExperimentTracker; print('OK')"`

- [ ] 2. Create the ModelRegistry class in `forge_core/model_registry.py`.
      Single-file registry (`registry/registry.json`) with model entries supporting
      name, version, stage, checkpoint_path, experiment_id. Stage transitions:
      training→validated→production→archived, archived→validated.
      Includes: `register_model()`, `get_model()`, `list_models()`, `promote()`,
      `delete_model()`, `get_latest()`.
      Files: `/root/forge/forge_core/model_registry.py`
      Verify: `python3 -c "from forge_core.model_registry import ModelRegistry; print('OK')"`

- [ ] 3. Create unit tests for ExperimentTracker.
      Test creation, completion, failure, listing (with filters), comparison,
      deletion. All tests use `tmp_path` for isolation.
      Files: `/root/forge/tests/unit/test_experiment.py`
      Verify: `python3 -m pytest tests/unit/test_experiment.py -v` — all tests pass.

- [ ] 4. Create unit tests for ModelRegistry.
      Test registration, version auto-increment, retrieval (by name/version/stage),
      listing (with filters), stage promotion (valid + invalid transitions), deletion.
      Files: `/root/forge/tests/unit/test_model_registry.py`
      Verify: `python3 -m pytest tests/unit/test_model_registry.py -v` — all tests pass.

- [ ] 5. Integrate experiment tracking into `train_bc()` in `forge_core/learned_policy.py`.
      Add optional `experiment_id` and `experiment_tracker` parameters. On completion,
      auto-log metrics. On failure, mark experiment as failed. Add `experiment_id`
      field to `TrainResult` dataclass. No behavior change when parameters are None.
      Files: `/root/forge/forge_core/learned_policy.py`
      Verify: `python3 -m pytest tests/unit/test_learned_policy.py -v` — all 149 existing tests still pass.

- [ ] 6. Integrate auto-logging into `scripts/forge_train.py`.
      On training: create experiment, pass tracker to train_bc(), register model
      in registry on success. Add `--no-track`, `--experiment-name`, `--model-name` flags.
      Wrap tracking calls in try/except so failures don't crash training.
      Files: `/root/forge/scripts/forge_train.py`
      Verify: `python3 scripts/forge_train.py --help` shows new flags. Existing tests pass.

- [ ] 7. Add registry model loading to `scripts/forge_eval.py` and `forge_core/runner.py`.
      forge_eval.py: add `--registry-model` and `--registry-version` args.
      runner.py: add `load_policy_from_registry()` convenience function.
      Files: `/root/forge/scripts/forge_eval.py`, `/root/forge/forge_core/runner.py`
      Verify: `python3 scripts/forge_eval.py --help` shows new flags. Existing tests pass.

- [ ] 8. Create the registry management CLI `scripts/forge_registry.py`.
      Subcommands: `experiments list|show|compare|delete`, `models list|show|promote|delete`.
      Follows forge_chaos.py patterns (argparse, setup_logging, sys.path setup).
      Files: `/root/forge/scripts/forge_registry.py`
      Verify: `python3 scripts/forge_registry.py --help`, `python3 scripts/forge_registry.py experiments list`, `python3 scripts/forge_registry.py models list`

- [ ] 9. Create integration-level unit tests for the tracking + registry pipeline.
      Test train_bc() with tracker → experiment created and completed. Test
      registry registration after training. Test load_policy_from_registry.
      Test CLI argument parsing.
      Files: `/root/forge/tests/unit/test_integration_tracking.py`
      Verify: `python3 -m pytest tests/unit/test_integration_tracking.py -v`

- [ ] 10. Final verification: all tests pass.
       Verify: `python3 -m pytest tests/ -v` — all tests pass (existing 149 + new tests).

## Documentation Touch-Points (for later docs step)

- **docs/architecture.md**: Add Phase 7 section covering experiment tracker design, model registry
  design, on-disk formats, stage transitions, integration points, CLI surface.
- **README.md**: Update status to "Phase 7 — ML Infrastructure", add experiment/registry CLI
  examples in Quick Start, update project structure (add experiments/, registry/, forge_registry.py,
  experiment.py, model_registry.py), update test count, update roadmap table (Phase 7 → Done).
- **docs/setup.md**: Add "Experiment Tracking & Model Registry" section with CLI usage examples.

## File Summary

### New Files
- `/root/forge/forge_core/experiment.py` — ExperimentTracker class
- `/root/forge/forge_core/model_registry.py` — ModelRegistry class
- `/root/forge/scripts/forge_registry.py` — Management CLI
- `/root/forge/tests/unit/test_experiment.py` — ExperimentTracker tests
- `/root/forge/tests/unit/test_model_registry.py` — ModelRegistry tests
- `/root/forge/tests/unit/test_integration_tracking.py` — Integration tests

### Modified Files
- `/root/forge/forge_core/learned_policy.py` — train_bc() experiment tracking integration
- `/root/forge/scripts/forge_train.py` — Auto-log experiments and register models
- `/root/forge/scripts/forge_eval.py` — Registry model loading
- `/root/forge/forge_core/runner.py` — load_policy_from_registry() helper
