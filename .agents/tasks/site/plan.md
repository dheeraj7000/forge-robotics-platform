# Implementation Plan — Forge GitHub Pages Website

## Verified Project Data

All stats verified against the actual repo on disk:

- **Tests**: 384
- **Phases**: 11 (all complete)
- **Core modules**: 21 (in forge_core/, excluding __init__.py)
- **CLI scripts**: 12 total (11 forge_-prefixed + run_forge.py)
- **Fault types**: 6 (object_moved, sensor_noise, sensor_delay, actuator_stuck, controller_dropout, gravity_shift)

### CLI Scripts (verified from scripts/ directory)

| Script | Purpose |
|--------|---------|
| `forge_eval.py` | Task evaluation |
| `forge_chaos.py` | Chaos testing |
| `forge_replay.py` | Run replay |
| `forge_perception.py` | Perception pipeline |
| `forge_train.py` | ML training |
| `forge_registry.py` | Experiment & model registry |
| `forge_regression.py` | Regression testing |
| `forge_deploy.py` | Edge deployment |
| `forge_fleet.py` | Fleet & OTA management |
| `forge_robot.py` | Physical robot |
| `forge_doctor.py` | Health check |
| `run_forge.py` | ROS 2 launcher |

### Phase Roadmap (from README)

| Phase | Capability |
|-------|-----------|
| 1 | Simulation foundation |
| 2 | Task execution & evaluation |
| 3 | Telemetry & replay |
| 4 | Failure injection / chaos testing |
| 5 | Perception (camera + OpenCV) |
| 6 | Learned policy (PyTorch) |
| 7 | ML infrastructure (experiment tracking, model registry) |
| 8 | CI/CD (regression evaluation) |
| 9 | Edge deployment (ONNX, TensorRT) |
| 10 | Fleet & OTA |
| 11 | Physical robot support |

### Architecture Diagram (from docs/architecture.md)

```
  Task YAML + Fault Profile
        │
        ▼
  ┌───────────┐     ┌──────────┐
  │  Runner    │────▶│ Recorder │──▶ runs/run_XXXXX/
  │            │     └──────────┘
  │ load       │     ┌──────────┐
  │ reset      │────▶│ Faults   │
  │ inject     │     └──────────┘
  │ loop:      │
  │  policy    │     ┌──────────┐
  │  sim.step  │────▶│Evaluator │──▶ EvalResult
  │ evaluate   │     └──────────┘
  └─────┬──────┘
        │
        ▼
  ForgeSimulation (MuJoCo)
```

---

## Design Decisions

**Site location**: `docs/site/` — keeps site assets out of the project root. Entry point at `docs/site/index.html`. GitHub Pages deploys from this subdirectory via a custom Actions workflow (not the legacy "deploy from branch/docs" mode). This avoids polluting the docs/ folder that already has architecture.md and setup.md.

**File structure**: Single-page site split into index.html + style.css + script.js. One HTML file keeps things simple and fast-loading. CSS is separate for maintainability. JS is separate for the scroll animations and mobile nav toggle.

**GitHub Pages strategy**: Use the modern GitHub Pages Actions deployment (v2 API). The workflow at `.github/workflows/pages.yml` uses `actions/upload-pages-artifact@v3` pointing at `docs/site/` and `actions/deploy-pages@v4`. This requires `pages: write`, `id-token: write` permissions and a deployment environment. Triggers on push to main.

**Design system**: Dark theme with `#0d1117` background (GitHub dark), `#58a6ff` electric blue accent, `#1f6feb` for hover states, `#161b22` for card backgrounds, `#21262d` for borders. Typography: Inter (body) and JetBrains Mono (code) from Google Fonts CDN. No external CSS frameworks.

**Syntax highlighting**: Use Prism.js from CDN (prism-tomorrow theme for dark mode) for code blocks. Lightweight, no build step needed.

**Animations**: CSS-only fade-in-up on scroll via IntersectionObserver in vanilla JS. Hover transforms on cards. No animation library needed.

---

## File Layout

```
docs/site/
├── index.html          # Single-page site with all 8 sections
├── style.css           # All styles (dark theme, responsive, animations)
└── script.js           # Scroll animations (IntersectionObserver), mobile nav toggle

.github/workflows/
└── pages.yml           # GitHub Pages deployment workflow (NEW file)
```

---

## Implementation Plan

- [ ] 1. Create `docs/site/style.css` with the complete dark-theme stylesheet.
      This file contains all CSS: reset/base styles, CSS custom properties for the design tokens (#0d1117 bg, #58a6ff accent, #161b22 cards, #21262d borders), typography (Inter body, JetBrains Mono code), layout grid, responsive breakpoints (mobile < 768px, tablet < 1024px), component styles for each section (hero, features grid, roadmap timeline, architecture diagram, code blocks, CLI showcase, stats counter, footer), hover effects (card lift with box-shadow transition, button color transitions), scroll fade-in animation keyframes (@keyframes fadeInUp), and the `.revealed` class that triggers them. Use CSS Grid for the features cards (auto-fill, minmax 300px) and Flexbox for the nav/hero layout.
      Files: `docs/site/style.css`
      Verify: File exists and is valid CSS (no syntax errors on visual inspection — there is no CSS linter in this project).

- [ ] 2. Create `docs/site/index.html` with the full single-page site content.
      The HTML file includes: doctype, lang="en", meta viewport, Google Fonts link (Inter 400/600/700 + JetBrains Mono 400), Prism.js CSS from CDN (tomorrow theme), link to style.css, and all 8 content sections as described below. Each section gets a semantic `<section>` with an `id` for anchor links. Script tags at the bottom: Prism.js from CDN, then script.js.

      **Section 1 — Hero** (`#hero`):
      - H1: "Forge"
      - Tagline: "Robotics Reliability & Deployment Platform"
      - Description paragraph: "A robotics reliability platform for developing, evaluating, debugging, and deploying ML-powered robotic systems. What happens after a model or robot policy leaves the notebook?"
      - CTA buttons: "View on GitHub" → `https://github.com/dheeraj7000/forge-robotics-platform` (opens new tab), "Quick Start" → `#quickstart` anchor

      **Section 2 — Features** (`#features`):
      - Section heading: "Core Capabilities"
      - Grid of cards, each with emoji icon, title, and 1-line description:
        - 🤖 Simulation — MuJoCo Menagerie Franka Panda with mesh collision and accurate kinematics
        - 📊 Task Evaluation — YAML task definitions with measurable success criteria and structured results
        - 📹 Telemetry & Replay — Per-cycle trajectory recording with numpy storage and step-level replay
        - 💥 Chaos Testing — 6 fault types, composable YAML profiles, baseline comparison mode
        - 📷 Perception — MuJoCo offscreen camera rendering, OpenCV HSV color detection, depth-based 3D estimation
        - 🧠 ML Training — PyTorch behavioral cloning from scripted demonstrations, configurable training pipeline
        - 🔬 Experiment Tracking — File-based experiment tracker with config, metrics, git commit, and tags (zero external deps)
        - 📦 Model Registry — Versioned models with stage lifecycle (training → validated → production → archived)
        - ✅ Regression Testing — Automated baseline comparison to detect performance degradations
        - 🚀 Edge Deployment — ONNX export, optional TensorRT optimization, edge inference runtime
        - 🌐 Fleet & OTA — Device registry, OTA update packages, rollout strategies (immediate, canary, staged)
        - 🔧 Physical Robot — Hardware abstraction layer, safety monitoring, sim-to-real transfer

      **Section 3 — Phase Roadmap** (`#roadmap`):
      - Section heading: "Development Roadmap"
      - Visual vertical timeline with 11 items. Each item shows: phase number in a circle badge, status "Complete" with a green checkmark, capability name, and a short description. All 11 phases marked complete. Use the exact phase names from the README table:
        1. Simulation foundation
        2. Task execution & evaluation
        3. Telemetry & replay
        4. Failure injection / chaos testing
        5. Perception (camera + OpenCV)
        6. Learned policy (PyTorch)
        7. ML infrastructure (experiment tracking, model registry)
        8. CI/CD (regression evaluation)
        9. Edge deployment (ONNX, TensorRT)
        10. Fleet & OTA
        11. Physical robot support

      **Section 4 — Architecture** (`#architecture`):
      - Section heading: "System Architecture"
      - A styled `<pre>` block containing the ASCII pipeline diagram from architecture.md (the Runner → Recorder/Faults/Evaluator → ForgeSimulation diagram). Style this pre block with the card background, colored text for the box borders (accent color for pipe chars), and JetBrains Mono font.
      - Below it, a brief prose description: "The Runner orchestrates the full lifecycle: load scenario, reset simulation, inject faults, run the policy loop, and evaluate results. Recorded runs capture every joint state, action, and evaluation result for replay and regression testing."

      **Section 5 — Quick Start** (`#quickstart`):
      - Section heading: "Quick Start"
      - Sub-sections with Prism.js-highlighted code blocks (`<pre><code class="language-bash">`):
        - "Install" — `pip install mujoco mujoco_menagerie numpy PyYAML opencv-python-headless pytest` and `pip install -e '.[ml]'`
        - "Run Tests" — `python3 -m pytest tests/ -v` with note "384 tests, no ROS needed"
        - "Evaluate a Task" — `python3 scripts/forge_eval.py --task pick_red_cube`
        - "Train a Policy" — `python3 scripts/forge_train.py --task pick_red_cube --episodes 5`
        - "Deploy to Edge" — `python3 scripts/forge_deploy.py --export --model models/bc_policy.pt --output models/bc_policy.onnx`

      **Section 6 — CLI Tools** (`#cli`):
      - Section heading: "CLI Tools"
      - Grid or list of 11 CLI tools, each showing the script name and 1-2 example commands in code blocks. Use the exact commands from the README:
        - `forge_eval` — `--task pick_red_cube`, `--suite manipulation-v1`
        - `forge_chaos` — `--task pick_red_cube --fault object_moved`, `--task pick_red_cube --fault cascading --compare`
        - `forge_perception` — `--scenario basic_workspace --detect --annotate -o detections.png`
        - `forge_train` — `--task pick_red_cube --episodes 5`, `--list-models`
        - `forge_registry` — `experiments list`, `models promote bc_pick_red_cube --version 1 --stage validated`
        - `forge_regression` — `--check --suite manipulation-v1`, `--create-baseline --suite manipulation-v1`
        - `forge_deploy` — `--export --model models/bc_policy.pt --output models/bc_policy.onnx`, `--benchmark --model models/bc_policy.onnx`
        - `forge_fleet` — `devices register --id arm-01 --name "Panda Arm 01" --type panda`, `ota deploy --package-id pkg_00001 --target-group production`
        - `forge_robot` — `connect --type simulated --scenario basic_workspace`, `safety-check`
        - `forge_doctor` — (health check, no subcommands — just `python3 scripts/forge_doctor.py`)
        - `forge_replay` — `--list`, `runs/run_00001/ --step 50`

      **Section 7 — Stats** (`#stats`):
      - Horizontal row of stat cards with large numbers and labels:
        - 384 — Automated Tests
        - 11 — Phases Complete
        - 21 — Core Modules
        - 12 — CLI Tools
        - 6 — Fault Types
      - These numbers are verified from the actual repo.

      **Section 8 — Footer**:
      - MIT License
      - GitHub link: `https://github.com/dheeraj7000/forge-robotics-platform`
      - Links: Documentation (architecture.md), Setup Guide (setup.md)
      - "Built with Forge" tagline

      **Navigation bar** (sticky top):
      - Logo/name "Forge" on the left
      - Nav links: Features, Roadmap, Architecture, Quick Start, CLI, GitHub (external)
      - Mobile: hamburger icon toggling a dropdown menu

      Files: `docs/site/index.html`
      Verify: Open in a browser or verify the HTML is well-formed (check matching tags). Confirm Prism.js CDN links are valid URLs.

- [ ] 3. Create `docs/site/script.js` with scroll animations and mobile nav.
      Vanilla JS with two features:
      (a) IntersectionObserver that watches all elements with class `.fade-in` and adds `.revealed` when they enter the viewport (threshold 0.1). This triggers the CSS fadeInUp animation.
      (b) Mobile nav toggle: hamburger button click toggles `.nav-open` class on the nav element, showing/hiding the mobile menu.
      (c) Smooth scroll for anchor links (CSS `scroll-behavior: smooth` on html, but also a JS handler that closes the mobile menu on nav link click).
      Files: `docs/site/script.js`
      Verify: File exists. No syntax errors (check with `node --check docs/site/script.js` if Node is available, otherwise visual inspection).

- [ ] 4. Create `.github/workflows/pages.yml` for GitHub Pages deployment.
      Valid YAML workflow file with:
      - Name: "Deploy to GitHub Pages"
      - Trigger: push to main branch
      - Permissions: `pages: write`, `id-token: write`
      - Concurrency group: `pages` with cancel-in-progress
      - Single job `deploy` with `environment: github-pages` and `url: ${{ steps.deployment.outputs.page_url }}`
      - Steps:
        1. `actions/checkout@v4`
        2. `actions/configure-pages@v5`
        3. `actions/upload-pages-artifact@v3` with `path: docs/site`
        4. `actions/deploy-pages@v4` with `id: deployment`
      Files: `.github/workflows/pages.yml`
      Verify: Validate YAML syntax — `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/pages.yml'))"` succeeds. Confirm the `path: docs/site` matches the actual site directory.

- [ ] 5. Verify the complete site by checking all files exist and cross-referencing content.
      Run a final check: confirm docs/site/index.html, docs/site/style.css, docs/site/script.js, and .github/workflows/pages.yml all exist. Verify the HTML references style.css and script.js with correct relative paths. Verify all GitHub repo URLs point to `https://github.com/dheeraj7000/forge-robotics-platform`. Verify the stats (384 tests, 11 phases, 21 core modules, 12 CLI tools, 6 fault types) match the actual repo. Verify all 11 phase descriptions match the README exactly.
      Files: all 4 files created above
      Verify: `ls docs/site/index.html docs/site/style.css docs/site/script.js .github/workflows/pages.yml` — all exist. Grep for key content strings in index.html to confirm sections are present.

---

## Design Specifications Reference

### Color Palette
| Token | Value | Usage |
|-------|-------|-------|
| `--bg-primary` | `#0d1117` | Page background |
| `--bg-secondary` | `#161b22` | Card backgrounds, nav |
| `--bg-tertiary` | `#21262d` | Borders, subtle dividers |
| `--text-primary` | `#e6edf3` | Body text |
| `--text-secondary` | `#8b949e` | Muted text, descriptions |
| `--accent` | `#58a6ff` | Links, highlights, accent elements |
| `--accent-hover` | `#1f6feb` | Button/link hover states |
| `--success` | `#3fb950` | Phase complete badges |
| `--gradient-start` | `#58a6ff` | Hero gradient start |
| `--gradient-end` | `#3fb950` | Hero gradient end |

### Typography
| Element | Font | Weight | Size |
|---------|------|--------|------|
| Body | Inter | 400 | 16px base |
| Headings | Inter | 700 | 2.5rem (h1), 2rem (h2), 1.25rem (h3) |
| Code | JetBrains Mono | 400 | 0.875rem |
| Nav links | Inter | 600 | 0.9rem |
| Stat numbers | Inter | 700 | 3rem |

### Responsive Breakpoints
| Breakpoint | Layout Change |
|------------|--------------|
| ≤ 768px | Single column, hamburger nav, smaller headings |
| ≤ 1024px | 2-column feature grid instead of 3 |
| > 1024px | Full 3-column grid, horizontal nav |

### Animation Specs
| Animation | Trigger | Duration | Easing |
|-----------|---------|----------|--------|
| fadeInUp | IntersectionObserver (threshold 0.1) | 0.6s | ease-out |
| Card hover lift | :hover | 0.3s | ease |
| Button hover | :hover | 0.2s | ease |
| Nav background | scroll past hero | 0.3s | ease |

### CDN Resources
| Resource | URL |
|----------|-----|
| Google Fonts (Inter) | `https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap` |
| Google Fonts (JetBrains Mono) | `https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400&display=swap` |
| Prism.js CSS (Tomorrow) | `https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css` |
| Prism.js Core | `https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js` |
| Prism.js Bash | `https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-bash.min.js` |
| Prism.js YAML | `https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-yaml.min.js` |
| Prism.js Python | `https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js` |
