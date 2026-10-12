# Forge GitHub Pages site

Adds a static documentation site under `docs/site/` and a GitHub Pages deployment workflow at `.github/workflows/pages.yml`. The site is a single-page dark-themed layout built with pure HTML, CSS, and vanilla JS (no build step) covering all 11 project phases across 8 content sections. Every CLI command, stat, and phase description is sourced from the README and architecture docs. Watch for: invisible content when JavaScript is disabled (confirmed), missing `aria-expanded` toggle on the mobile nav (confirmed).

**Verdict**: APPROVED

## High-level view

The site ships as three files — `index.html`, `style.css`, `script.js` — plus CDN-loaded Google Fonts and Prism.js for syntax highlighting. No build step, no framework, no bundler. The deployment workflow is a textbook GitHub Pages setup using the modern `upload-pages-artifact` / `deploy-pages` action pair, scoped to `docs/site` and isolated from the existing CI and regression workflows.

All 8 required sections are present and populated with real data. CLI commands were cross-referenced against the README and script source files; every command shown on the site appears verbatim in the project documentation. Stats (384 tests, 11 phases, 21 core modules, 12 CLI tools, 6 fault types) were verified against the README's test count, roadmap table, project structure listing, scripts directory, and fault types table respectively.

The CSS uses a GitHub-flavored dark palette with CSS custom properties, responsive breakpoints at 1024px and 768px, and a collapsing hamburger nav for mobile. Scroll-triggered fade-in animations use IntersectionObserver with a graceful fallback for browsers that don't support it — but no fallback for browsers with JS disabled entirely, which is the main gap.

<details>
<summary>Issues (2)</summary>

1. **No-JS invisible content** — All `.fade-in` elements start with `opacity: 0` in CSS. If JavaScript is disabled or fails to load, most page content is invisible. A `<noscript><style>.fade-in { opacity: 1; transform: none; }</style></noscript>` block would fix this. Non-blocking for a project docs site but worth addressing.
2. **aria-expanded not toggled on hamburger** — The hamburger button declares `aria-expanded="false"` in the HTML but the JS toggle never updates this attribute, so screen readers won't announce menu open/closed state. One line addition: `hamburger.setAttribute('aria-expanded', nav.classList.contains('nav-open'))` after the toggle.

</details>

<details>
<summary>Details</summary>

## Content accuracy and the 11/12 CLI gap

The 12 feature cards, 11-phase roadmap, and architecture diagram all trace back to the README and `docs/architecture.md`. Phase names match word-for-word; the architecture diagram omits the ROS 2 nodes section from the source, consistent with ROS being optional.

The CLI section shows 11 tool cards while the stats section claims "12 CLI Tools." The stat is accurate — there are 12 scripts in `scripts/` (11 `forge_`-prefixed plus `run_forge.py`) — but `run_forge.py` has no card. Not inaccurate, but a reader who counts the cards and reads the stat will notice the mismatch.

## No-JS content visibility

All `.fade-in` elements start at `opacity: 0`. The IntersectionObserver fallback handles browsers without that API, but if JavaScript is entirely disabled, the hero tagline, all section headings, all feature cards, all timeline items, all CLI cards, and the architecture diagram remain invisible. A one-line `<noscript>` style override would eliminate the edge case.

## Accessibility: aria-expanded gap

The hamburger button declares `aria-expanded="false"` but the JS toggle never updates the attribute, so screen readers always report "collapsed" regardless of actual state. Fix: `hamburger.setAttribute('aria-expanded', nav.classList.contains('nav-open'));` after the class toggle.

</details>

<details>
<summary>File map</summary>

| File | Change |
|------|--------|
| `docs/site/index.html` | New — single-page site with all 8 sections |
| `docs/site/style.css` | New — dark-theme responsive stylesheet |
| `docs/site/script.js` | New — scroll animations + mobile nav toggle |
| `.github/workflows/pages.yml` | New — GitHub Pages deployment workflow |

</details>
