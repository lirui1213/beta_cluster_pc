# Single Workspace Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to execute the tested steps.

**Goal:** Retain one resumable full-browser analysis per patient/block.

**Architecture:** A contained retention module owns current discovery, context
leases and garbage collection. WorkspaceSession commits replacement only after
successful extraction, retains current rounds and prunes unreachable task data.
The frontend uses a selection dialog, not an accumulating analysis dropdown.

**Tech Stack:** Python standard library, existing joblib snapshots, JavaScript,
pytest, Node/Playwright, existing local browser service.

## Global Constraints

- Preserve Notebook outputs and raw ECoG/audio/baseline/recon data.
- Keep both current accepted rounds and explicit review gates.
- Do not delete old accepted data before replacement extraction commits.
- Verify containment and ownership before recursive deletion.
- No GitHub push requested for this change.

### Task 1: Backend Retention And Resume

Files: new `workspace_retention.py`, `tests/test_workspace_retention.py`;
modify `workspace_session.py`, `workspace_server.py`, scoped service tests.

- [x] Write failing pytest cases for current discovery, leases, safe cleanup,
  explicit replace_run_id, success-only replacement, and current-round pruning.
- [x] Run `python -m pytest tests/test_workspace_retention.py -q` and confirm red.
- [x] Implement isolated retention helpers and connect start/open/refresh/cancel
  and close paths. Save manifest before pruning any unreachable snapshots.
- [x] Add GET current returning `{analysis: summary|null}` from trusted catalog
  context. Keep legacy read-only APIs available without deleting Notebook data.
- [x] Run retention/service/catalog/action tests and verify all current model
  and artifact references remain readable after repeated actions and refits.

### Task 2: Browser Selection Workflow

Files: `browser_assets/workspace.js`, `workspace.html`, `workspace.css`,
`workspace_core.mjs` if needed; `scripts/verify_workspace_frontend.mjs`.

- [x] Add failing tests for selection dialog, Continue hydration, New detached
  draft with replace_run_id, Cancel without destructive calls, and reload.
- [x] Replace history selection with a compact current-analysis control and
  accessible modal. Use GET current and pass exact replace_run_id on start.
- [x] Preserve the first-round 3D plot and manual K/trial review behavior.
- [x] Run Node fixtures with desktop/mobile screenshots and inspect dialogs.

### Task 3: Integrated Verification And Documentation

- [x] Update README with single-record scope, atomic replacement and Notebook
  preservation; document migration happens on user Continue/replacement.
- [x] Independently review deletion/lifecycle paths and address findings.
- [x] Run full pytest, Node/browser checks and `git diff --check`. Exercise an
  isolated real-data output root, never destructive tests against user outputs.
- [x] Save scoped changes locally and report verification, without pushing.

## Verification

- Full pytest: 840 passed, 1 skipped. Retention/service subset: 44 passed.
- Node/Chrome desktop/mobile fixture: 17 passed, with dialogs, deferred replace,
  failure/cancel, reload, channel/trial review and 3D camera regressions.
- Independent adversarial review identified stale reads before lease acquisition,
  display-loader startup during shutdown, and ownership-probe failures bypassing
  leases. All three were reproduced with failing interleaving tests and fixed.
  Final scoped rereview found no remaining issue in those paths.
- Real ECoG_LR_v1 environment / HS0015 B03: extraction, manual first K=3,
  second K=2, stability, restart/resume and frontend passed. All 126 plotted
  electrodes match the FIRST model; four 1061-sample curves retain audio, cortex
  canvases are nonblank, mobile has no overflow, and browser errors/external
  requests are empty.
- `scripts/verify_single_analysis.py` ran only against `3-bands/outputs/rt19`:
  cancelling replacement preserved the old analysis, successful replacement
  removed it, Continue restored the fresh trial review, and a synthetic Notebook
  sentinel remained byte-identical. Final count: one analysis, one current UI
  metadata file, all 44 retained original trials (not a single ECoG trial).
- Reports/screenshots are under
  `3-bands/outputs/verification/single_analysis_20261019/`. The initial longer
  verification path exceeded Windows CSV path limits, so the dedicated numerical
  test root was shortened; production/raw paths were not changed.
- Existing Notebook/PPT edits and unrelated untracked projects were not staged.
  No production historical records were deleted during verification; browser
  consolidation is triggered by the user's Continue or successful replacement.
