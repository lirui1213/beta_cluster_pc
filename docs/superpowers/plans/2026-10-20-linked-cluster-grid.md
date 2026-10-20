# Linked Four-Band Cluster Grid Implementation Plan

> **For agentic workers:** Use subagent-driven-development for the isolated backend task and inline frontend integration. User approved the design in chat.

**Goal:** Add a duration-normalized class summary below Continuous response, matching fourband_cluster_grid.png and following the shared class palette.

**Architecture:** A read-only API returns the existing round's fourband_curve_summary.csv in typed panel arrays. Session metadata supplies exact band limits and median speech duration from the trusted extraction. A focused Plotly module owns the grid and recolors mean/SEM without recomputation. Missing/corrupt saved summaries show a local unavailable message without blocking the existing explorer.

**Tech Stack:** Python/pandas, native browser modules, existing Plotly/Lucide, pytest and Playwright.

## Constraints
- Rows are classes, columns are the existing four bands. Preserve mean, electrode SEM, normalized audio and onset/offset markers.
- No filtering, model refitting, changes to trial selection or modification of saved PNGs. Trial pair and continuous sigma do not change this saved summary.
- Class colors are shared with brain electrodes. Other classes retain their colors.
- No arbitrary paths/API parameters; use current trusted round output only. Desktop four-column layout, mobile stacked band panels per class.
- Keep unrelated user changes and notebooks intact. Save locally, do not push.

## Task 1: Saved Summary API
- [x] Write and observe failing tests for CSV round identity, gaps/nulls, missing/corrupt summaries and both server routes.
- [x] Implement `cluster_grid_payload(request, metadata)`, cached `BrowserSession.cluster_grid()`, `api/cluster-grid` routes, and metadata band limits/median duration.
- [x] Run focused pytest; review changes against data source and isolation requirements.

## Task 2: Browser Grid
- [x] Extend `scripts/verify_fourband_browser_frontend.mjs` with summary fixture and assertions for 2x4 panels, shared palette mean/SEM, unchanged audio, pair/sigma independence, missing summary and mobile layout; observe failure.
- [x] Add `cluster_grid.mjs` and pure Plotly trace helper, markup and styles; initialize once from API and recolor on class-color changes. Keep errors local and late responses closed-session safe.
- [x] Verify tooltips/axes, desktop/mobile screenshots, actual PNG-equivalent data, no blank plots, no additional compute requests on recolor.

## Task 3: Finish
- [x] Run relevant Python and both browser regression workflows; inspect screenshots and request independent code review.
- [x] Update README, record evidence, commit only scoped files and start updated local service. Do not resume/replace patient analyses solely for testing.

## Verification Evidence
- Initial frontend regression failed because summary plots were absent. Backend initial suite had 43 expected missing-feature failures, then passed.
- Independent review found partial-panel truncation and post-close async mutation risks. Both reproduced with red tests, then fixed. Scoped re-review found no issues, including an independent delayed-newPlot close probe.
- Full Python regression: 883 passed, 1 skipped. Subsequent two added panel-grid regressions also pass: final focused adapter/summary run 73 passed (45 summary cases).
- Explorer browser fixture passes: 2x4 grid, color/SEM updates, unchanged other classes/audio, pair/sigma independence, missing-summary isolation, export isolation, desktop/mobile.
- Workspace browser regression: 17 passed. Existing previews, PCA review and navigation preserved.
- `scripts/verify_fourband_grid_saved.mjs` verified actual HS0015/B03 saved round2 without numerical jobs: 9008 CSV rows, 4 classes, 16 panels; time/mean/SEM/audio/count arrays exactly match CSV, rendered curves match, recoloring works, mobile has no overflow. Artifacts: `3-bands/outputs/verification/linked_grid_20261020/actual/`.
- Actual desktop and mobile screenshots inspected. `git diff --check` passes. Live launcher returns HTTP 200 for workspace HTML, grid module, explorer JS and state API, without opening patient records.
