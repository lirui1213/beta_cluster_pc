# Muted Class Presets Implementation Plan

> **For agentic workers:** Use executing-plans and focused TDD. Design approved in chat.

**Goal:** Add five low-saturation class color swatches and default the browser cortex to opaque ivory.

**Architecture:** Five accessible color swatch buttons reuse the existing class-color input event, sharing the palette with brain, continuous curves and class summary. Set consistent missing-value defaults in browser metadata, form and renderer; preserve explicit surface options and saved user adjustments.

**Tech Stack:** Existing native HTML/CSS/JS, Three.js, Plotly, pytest, Playwright.

## Constraints
- Red, green, blue, yellow and white swatches; retain the custom color picker.
- Swatches apply only to the currently selected class; mean and SEM colors follow existing linkage.
- Defaults: cortex #FFFFF0, opacity 1.00. Explicit user appearance remains adjustable and persisted.
- Color buttons use color itself, title/aria-label, selected state and border for soft white.
- Display-only change; preserve notebooks, analysis data and unrelated user files. No GitHub push.

## Tasks
- [x] Add failing browser tests for swatches, selected state, linkage/no computation, default opacity and mobile bounds; add failing backend-default test.
- [x] Implement markup/styles/event reuse and consistent cortex defaults. Leave explicit appearance options intact.
- [x] Run focused Python and existing explorer browser checks; inspect desktop/mobile screenshots and request scoped review.
- [x] Save scoped changes locally, launch the updated workspace and report the usable URL.

## Evidence
- Red tests reproduced old gray defaults (backend lightgray, frontend #c8ccd0).
- `python -m pytest tests/test_fourband_browser_data.py tests/test_fourband_browser_service.py tests/test_fourband_cluster_grid.py tests/test_workspace_service.py -q`: 117 passed.
- Browser fixture passes: all five presets, selected-state reset with custom color, mean/SEM linkage for another class, no additional numerical jobs, preserved other classes, opacity 1.00 initially and manually changed 0.30 after refresh, mobile bounds and existing controls/export flow.
- Desktop and mobile screenshots inspected at `3-bands/outputs/verification/muted_presets_20261028/frontend/`.
- Independent scoped review: no findings. `git diff --check` passes.
- Updated local service serves workspace, viewer assets and state API with HTTP 200; no patient analysis was resumed or replaced for this check.
