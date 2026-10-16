# Four-Band Browser Analysis Implementation Plan

> **For agentic workers:** Use subagent-driven-development with test-first tasks and scoped reviews.

**Goal:** Run the existing four-band, two-round manually reviewed analysis from a local browser, including patient/block selection and persistent history.

**Architecture:** Keep numerical functions unchanged. Add a validated data catalog, file-backed numerical actions, an owned background task service, and a browser workspace embedding revision-scoped existing explorers. Each layer has separate tests and ownership.

**Tech Stack:** Python stdlib HTTP/subprocess, existing NumPy/MNE/sklearn/joblib pipeline, offline Plotly/Three.js/lucide.

## Global Constraints

- Specification: `docs/superpowers/specs/2026-10-16-fourband-browser-analysis-design.md`, user approved for implementation.
- Local only, no push. Preserve unrelated edits and both existing Notebooks.
- Four existing bands, ECoG-only features, manual K, explicit trial and electrode review; no automatic exclusion or K selection.
- Loopback/token/origin protection, configured-root path containment, owned workers, no GUI scientific backends.
- Results and exports revision-bound, immutable history, actual stage/elapsed/logs and cancellation.

## Task 1: Catalog And Validated Configuration

**Files:** `3-bands/fourband/workspace_catalog.py`, `tests/test_workspace_catalog.py`.
**Interface:** `WorkspaceCatalog(data_root, output_root).patients()`, `.resolve(subject, block, selections=None)`, `.history(subject, block)`; `validate_config(config, resolved)` returns JSON-safe effective settings with source paths and surface options.
- [x] Test discovery, missing files, traversal/junction escapes and validation. Run `python -m pytest tests/test_workspace_catalog.py -q` and observe failure.
- [x] Implement using pathlib and structured JSON, bounded history; rerun tests.
- [x] Review against spec and integrate; add only a backward-compatible companion-CSV opt-out to shared scientific inputs.

## Task 2: File-Backed Numerical Actions

**Files:** `3-bands/fourband/workspace_actions.py`, `tests/test_workspace_actions.py`.
**Interface:** `execute_action(run_dir, action, payload)` reads `config.json`, returns `{stage, data, artifacts, display_request?}`; snapshots remain worker-side. Actions: `extract`, `detect_qc`, `confirm_trials`, `fit_first`, `preview_review`, `fit_second`, `waveform`, `stability`.
- [x] Tests first: explicit empty trial confirmation, mask provenance, first/second manual K, preview/refit parity, raw waveform bounds, excluded audio predictors. Run focused tests and observe failures.
- [x] Reuse `load_fourband`, trial-QC, `fit_joint`, review helpers, workflow exports and stability. Persist distinct reviewed extraction/model snapshots and revision directories. Use Agg.
- [x] Return compact JSON metadata/PCA/preview and trusted request paths for display; no raw arrays/models over HTTP. Re-run tests and review.

## Task 3: Task Service, Security And Launcher

**Files:** `workspace_session.py`, `workspace_server.py`, `workspace_worker.py`, `3-bands/run_browser_analysis.py`, `tests/test_workspace_service.py`.
**Interface:** GET `api/workspace/catalog`, `state`, `history`, `artifact/<id>`; POST `start`, `action`, `cancel`, `open`, `close`. Action requires `run_id`, `revision`, `action`, `payload`. State contains active run/revision, current task/status/data, saved rounds and optional explorer URL.
- [x] Tests first: stale identity, concurrent actions, explicit confirmations, failure/cancel and protected HTTP mutations.
- [x] Implement owned numerical subprocess tasks with JSON files, logs, Windows Job startup gate and cancellation. Atomically save manifest with schema version; recover history without unpickling browser input.
- [x] Serve existing explorers at immutable `explorer/<id>/` routes on same loopback service with contained assets; exact display instance for each revision. Add standalone launcher and bounded lifecycle.
- [x] Re-run API/ownership tests and existing browser regression tests.

## Task 4: Browser Workflow

**Files:** `browser_assets/workspace.html`, `workspace.css`, `workspace.js`, `workspace_core.mjs`, `scripts/verify_workspace_frontend.mjs`.
- [x] Pure state/request tests first, then implement patient/block/history selectors, settings and stages, trial review grid/raw waveform, first PCA/curves, second review/K/preview, optional stability, log/cancel and iframe explorer.
- [x] Server-authoritative revisions; changing context clears stale displays and asks before cancelling work. Empty confirmation must be explicit. Error states visible and no old export available as current.
- [x] Desktop/mobile screenshots, nonblank plot/canvas checks, browser errors/external request checks and actual interactions.

## Task 5: Integration And Real Verification

**Files:** `3-bands/README.md`, `scripts/verify_workspace_analysis.py`, acceptance reports under ignored outputs.
- [x] Full API flow in actual `ECoG_LR_v1`: source validation, extraction, explicit QC confirmation, both manual-K fits, review preview and linked explorer.
- [x] Change patient/block, reopen saved run, cancel running work, inspect failure and optional stability; verify labels/features against direct functions.
- [x] Run full regression suite, syntax/schema checks, `git diff --check`, scoped independent review and fixes.
- [x] Document launch/recovery and limitations; commit only scoped verified files locally, leave service available and report URL.

## Execution Ledger

- Design approved; implementation started 2026-10-16.
- Tasks 1, 2 and 4 can proceed with disjoint files while parent implements Task 3.
- Task 5 follows integration; no completion claim before evidence.
- Implemented catalog, actions, worker/service and offline browser workspace; existing Notebooks/PPT remain untouched by this task.
- Numerical actions: 34 tests passed. Service: 25 tests passed, including Windows rename contention, persistence rollback, round-scoped exports and stability reload.
- Real HS0015/B03 flow used explicit empty manual trial exclusions, K1=3/K2=2, fixed-K stability with two resamples, and saved-run reopening. These are diagnostic settings, not chosen patient-analysis defaults.
- Real cortex and four audio-overlay curves passed pixel checks on desktop/mobile; no external requests or browser errors. First-model PCA review and a patient switch were exercised.
- Scoped independent review findings were fixed with regressions and rereviewed; no remaining load-bearing findings in the final scoped review.
- Final full suite: **821 passed, 1 skipped**. Browser/core fixture: **14 passed**. Compilation and staged whitespace checks passed.
- Final actual lifecycle verification: failed numerical review preserves accepted state; display preparation cancellation took about 0.17 seconds; HS0012/B01 start/cancel and HS0015/B03 history restore passed; stability summary survives reopening its round.
- Acceptance evidence: `3-bands/outputs/verification/workspace_20261016/final_flow`, `accepted_lifecycle`, `3-bands/outputs/verification/workspace_accepted`, and `workspace_frontend` (ignored generated artifacts).
- The small resampling count in acceptance is a software test, not evidence that biological cluster membership is strongly stable.
- No GitHub push. No user Notebook, presentation, high-gamma, prediction, or speech-project edits included.
