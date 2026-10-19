# Artifact Preview Implementation Plan

> **For agentic workers:** Use executing-plans to implement this approved plan task-by-task.

**Goal:** Preview exported images in the workspace before explicitly downloading them.

**Architecture:** A focused browser module owns a native modal dialog, image navigation, zoom and loading state. The workspace passes only currently authorized artifact IDs and resets the viewer on analysis/round changes. Existing artifact endpoints and scientific calculations remain unchanged.

**Tech Stack:** Native JavaScript modules, HTML dialog, CSS, local Lucide icons, Playwright.

## Constraints
- User approved in-page preview, previous/next, zoom, close and independent download.
- CSV and JSON retain direct download behavior.
- Preserve unrelated user changes. Do not push to GitHub.
- Desktop/mobile layouts, safe text labels, failed image loads and stale results must be covered.

## Task 1: Preview And Regression Coverage
- [x] Extend `scripts/verify_workspace_frontend.mjs` with multiple images and non-image artifacts; assert opening an image causes no download and shows a dialog.
- [x] Run `node scripts/verify_workspace_frontend.mjs --browser` and observe the missing-preview failure (preview button absent).
- [x] Add `artifact_preview.js`, modal markup/styles and workspace integration. Use `safeLocalURL` for artifact URLs, native focus handling, image load/error states and explicit download links.
- [x] Cover navigation, zoom/reset, Escape/focus restoration, explicit download, CSV/JSON download, thumbnail clicks and round/context invalidation. Inspect desktop/mobile screenshots.
- [x] Run frontend tests (17 passed), relevant Python server tests (149 passed) and `git diff --check`; independent scoped review: no findings.
- [x] Commit only feature files locally and start the existing workspace launcher without opening/replacing patient records.

## Verification Evidence
- Browser fixture screenshots/report: `3-bands/outputs/verification/artifact_preview_20261020/frontend/`.
- Regression verifies polling preserves zoom, last-image keyboard navigation, failed image loads, 320px/390px mobile widths, round invalidation, removed-image invalidation, focus restoration and exactly three explicit downloads (one PNG, CSV, JSON).
- `python -m pytest tests/test_workspace_service.py tests/test_workspace_catalog.py tests/test_workspace_retention.py -q`: 149 passed.
- Live launcher serves HTML, preview module, CSS and state API with HTTP 200. No patient records were resumed or replaced during launch verification.
