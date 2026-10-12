# Phase 8 — CI/CD (Regression Evaluation) Implementation Plan

## Codebase State (verified)

- **Tests**: 215 (6+22+10+8+4+23+6+9+45+5+7+7+24+17+21+1 across 16 test files)
- **Branch**: `main` — no existing stubs for regression, no `.github/`, no `baselines/`
- **Patterns**: File-based JSON storage (see `ExperimentTracker`, `ModelRegistry`); CLI scripts use argparse with `sys.path` injection; tests use `pytest` with `tmp_path` fixtures and class-based grouping

---

- [ ] 1. Create the regression evaluation module `forge_core/regression.py`.
      Implements `RegressionChecker` class following the `ExperimentTracker`/`ModelRegistry` JSON-on-disk pattern:
      - `Baseline` dataclass: stores per-task results (success, metrics, sim_time) plus suite-level metadata (timestamp, git commit, suite name)
      - `RegressionResult` dataclass: per-task comparison (passed/failed, metric deltas, newly_failing flag)
      - `RegressionReport` dataclass: overall pass/fail, list of `RegressionResult`, summary formatting
      - `save_baseline(results: list[EvalResult], suite_name: str, baselines_dir)` → writes a JSON baseline file to `baselines/<suite_name>.json`
      - `load_baseline(suite_name: str, baselines_dir)` → loads baseline JSON
      - `check_regression(current: list[EvalResult], baseline: Baseline, thresholds: dict)` → compares, returns `RegressionReport`
      - Configurable thresholds: `metric_tolerance` (float, default 0.1 = 10%), `allow_new_failures` (bool, default False)
      - Threshold config loadable from YAML (`config/regression.yaml`)
      Files: `forge_core/regression.py`, `config/regression.yaml`
      Verify: `python3 -m pytest tests/unit/test_regression.py -v` — new tests pass (see step 3)

- [ ] 2. Create the regression CLI script `scripts/forge_regression.py`.
      Follows the argparse pattern from `forge_eval.py` and `forge_chaos.py`:
      - `--check --suite <name>`: run evaluation suite and compare against stored baseline, exit 1 on regression
      - `--create-baseline --suite <name>`: run evaluation suite and save results as a new baseline
      - `--update-baseline --suite <name>`: alias for create-baseline (overwrite existing)
      - `--compare <baseline1.json> <baseline2.json>`: compare two baseline files directly
      - `--report --suite <name>`: show last regression report details
      - `--policy` flag (default: `scripted_pick_and_place`) passed through to runner
      - `--threshold-config` flag to specify custom threshold YAML
      - Uses `run_suite` from `forge_core/runner.py` and `load_suite` from `forge_core/task.py`
      - Exit code 0 = no regression, 1 = regression detected
      Files: `scripts/forge_regression.py`
      Verify: `python3 scripts/forge_regression.py --help` prints usage without error

- [ ] 3. Create unit tests for the regression module `tests/unit/test_regression.py`.
      Follow the class-based test structure from `test_experiment.py` and `test_model_registry.py` (use `tmp_path` fixtures, `_project_root` sys.path injection). Test classes:
      - `TestBaselineSaveLoad` (~4 tests): save/load round-trip, missing baseline raises FileNotFoundError, baseline contains expected fields
      - `TestRegressionCheck` (~8 tests): no regression when results match baseline; detect newly failing task; detect metric degradation beyond threshold; pass when degradation within threshold; handle missing task in baseline gracefully; detect improvement (informational); empty baseline/results edge cases
      - `TestRegressionReport` (~3 tests): report summary format, report to_dict serialization, overall pass/fail logic
      - `TestThresholds` (~3 tests): custom metric tolerance, allow_new_failures=True skips failure regression, load thresholds from YAML config
      Target: ~18 new tests (bringing total from 215 to ~233)
      Files: `tests/unit/test_regression.py`
      Verify: `python3 -m pytest tests/unit/test_regression.py -v` — all new tests pass; `python3 -m pytest tests/ -v` — all 233+ tests pass

- [ ] 4. Create GitHub Actions CI workflow `.github/workflows/ci.yml`.
      Test workflow triggered on push/PR to `main`:
      - Matrix: Python 3.10, 3.11
      - Steps: checkout, setup-python, pip cache, install deps (`pip install -e '.[dev]' && pip install torch --index-url https://download.pytorch.org/whl/cpu`), run `python3 -m pytest tests/ -v`
      - Runs headless — no display server needed for unit tests (MuJoCo offscreen via EGL/OSMesa)
      - Set `MUJOCO_GL=egl` env var for headless rendering
      Files: `.github/workflows/ci.yml`
      Verify: YAML syntax valid (checked by Python yaml.safe_load in tests or manual review); file exists at correct path

- [ ] 5. Create GitHub Actions regression evaluation workflow `.github/workflows/regression.yml`.
      Triggered on PR to `main` and manual `workflow_dispatch`:
      - Single Python version (3.11)
      - Steps: checkout, setup-python, pip cache, install deps (same as ci.yml), run `python3 scripts/forge_regression.py --check --suite manipulation-v1`
      - `MUJOCO_GL=egl` env var
      - Uploads regression report as artifact on failure
      - Separate from CI so test suite stays fast; regression eval runs the actual simulation
      Files: `.github/workflows/regression.yml`
      Verify: YAML syntax valid; file references correct CLI command

- [ ] 6. Create an initial baseline file for the manipulation-v1 suite.
      The baseline is a JSON file at `baselines/manipulation-v1.json` that records the expected results for `pick_red_cube`, `pick_blue_cube`, `pick_red_cube_spread` using the scripted policy. Since CI can't run MuJoCo at authoring time, generate it locally or provide a placeholder structure that `forge_regression.py --create-baseline` would produce. Include the expected task result structure (success bool, key metrics, sim_time).
      Files: `baselines/manipulation-v1.json`
      Verify: `python3 -c "import json; json.load(open('baselines/manipulation-v1.json'))"` — valid JSON

- [ ] 7. Update documentation: `docs/architecture.md`, `README.md`, `docs/setup.md`.
      - `docs/architecture.md`: Add Phase 8 section after Phase 7. Cover: regression module design, baseline management, CI/CD workflow design, CLI commands. Follow the existing section pattern (heading, overview, subsections for module/CLI/integration).
      - `README.md`: Change status to "Phase 8 — CI/CD (Regression Evaluation)"; update "What's included" with regression evaluation, CI/CD, baseline management bullet points; update test count (verify actual count after step 3); add `forge_regression.py` CLI examples in Quick Start; add `forge_regression.py` to project structure tree; update Phase 8 row in roadmap table to "Done"; add CI badge placeholder.
      - `docs/setup.md`: Add "Regression Testing" section with CLI examples; add "CI/CD" section explaining the GitHub Actions workflows; add regression CLI to the commands reference area.
      Files: `docs/architecture.md`, `README.md`, `docs/setup.md`
      Verify: All three files updated; test count in README matches actual `python3 -m pytest tests/ -v` output; Phase 8 marked Done in roadmap.
