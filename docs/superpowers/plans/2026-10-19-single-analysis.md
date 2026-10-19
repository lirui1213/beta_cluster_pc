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

- [ ] Write failing pytest cases for current discovery, leases, safe cleanup,
  explicit replace_run_id, success-only replacement, and current-round pruning.
- [ ] Run `python -m pytest tests/test_workspace_retention.py -q` and confirm red.
- [ ] Implement isolated retention helpers and connect start/open/refresh/cancel
  and close paths. Save manifest before pruning any unreachable snapshots.
- [ ] Add GET current returning `{analysis: summary|null}` from trusted catalog
  context. Keep legacy read-only APIs available without deleting Notebook data.
- [ ] Run retention/service/catalog/action tests and verify all current model
  and artifact references remain readable after repeated actions and refits.

### Task 2: Browser Selection Workflow

Files: `browser_assets/workspace.js`, `workspace.html`, `workspace.css`,
`workspace_core.mjs` if needed; `scripts/verify_workspace_frontend.mjs`.

- [ ] Add failing tests for selection dialog, Continue hydration, New detached
  draft with replace_run_id, Cancel without destructive calls, and reload.
- [ ] Replace history selection with a compact current-analysis control and
  accessible modal. Use GET current and pass exact replace_run_id on start.
- [ ] Preserve the first-round 3D plot and manual K/trial review behavior.
- [ ] Run Node fixtures with desktop/mobile screenshots and inspect dialogs.

### Task 3: Integrated Verification And Documentation

- [ ] Update README with single-record scope, atomic replacement and Notebook
  preservation; document migration happens on user Continue/replacement.
- [ ] Independently review deletion/lifecycle paths and address findings.
- [ ] Run full pytest, Node/browser checks and `git diff --check`. Exercise an
  isolated real-data output root, never destructive tests against user outputs.
- [ ] Save scoped changes locally and report verification, without pushing.
