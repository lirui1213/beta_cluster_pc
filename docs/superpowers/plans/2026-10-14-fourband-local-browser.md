# Four-Band Local Browser Implementation Plan

> **For agentic workers:** Use subagent-driven-development for independent data/frontend/notebook slices; the controller owns service integration, verification and local commits.

**Goal:** Replace the four-band brain/explorer native windows with a tested offline local browser interface without changing fitted labels or numerical processing.

**Architecture:** Keep the launcher snapshot and owned subprocess tree. A loopback-only standard-library HTTP service serves bundled Three.js/Plotly assets, validates authenticated requests and schedules existing computations in a numerical process. The browser owns all interactive rendering.

**Tech Stack:** Python stdlib HTTP server, existing NumPy/nibabel/joblib/Matplotlib Agg, Three.js, Plotly, locally bundled JavaScript dependencies.

## Global Constraints

- Follow `docs/superpowers/specs/2026-10-14-fourband-local-browser-design.md`.
- Preserve user edits, patient/block, numerical behavior, manual K and labels.
- No native GUI imports/objects in the browser workflow; no runtime CDN calls.
- Bind only 127.0.0.1, random port, session-token path, strict mutation origins.
- Commit only scoped changes locally; no GitHub push or Python environment updates.

## Shared Interfaces

`browser_data.py` provides `session_metadata(request) -> dict`,
`surface_payload(request) -> dict`, `continuous_payload(data, result, channel) -> dict`,
`render_detail_snapshot(request_path, settings) -> bytes` and `json_safe(value)`.

Metadata: subject/block/round, channels (strings), labels (integers), bands
(names), pairs (first_trial_id/second_trial_id/onsets/offsets), settings
(trial_ids, sigma_seconds, pre_seconds, post_seconds, display_sfreq),
surface_options (brain_color, brain_opacity, point_size), surface_enabled,
curves_enabled and warnings. Anatomy payload: meshes [{name, positions (flat
xyz), indices (flat triangles)}], electrodes [{channel, label, position [x,y,z]}],
warnings. Invalid/missing numeric data serialize as null, never NaN.

Continuous payload: times, audio, bands [{name, selected, mean}], events
[{original_trial_id, onset, offset, excluded}] in seconds relative to onset1,
channel, label, trial_ids, baseline_description and metadata. It preserves
source coordinates, fixed cluster membership and display-only QC semantics.

All routes live under `/s/<token>/`. `GET api/meta` and `GET api/surface` return
the above. `POST api/compute` accepts {view: continuous|detail, trial_ids:[a,b],
sigma_seconds, channel, band}; continuous job keys ignore channel/band.
It returns {job_id, state}. `GET api/job/<id>?channel=<name>` returns state,
elapsed_seconds and stage; on ready, data is a continuous payload or
{image_url:'api/image/<id>', channel, band} for a detail PNG. Failed jobs return
an error; superseded jobs never replace a current browser view.

`POST api/clear` clears worker and last-result caches. `POST api/export`
accepts {settings:object, images:{brain|curves:PNG-data-URL}} and writes only
server-generated names under round/browser_exports. `POST api/close` stops
the service and numerical child. Page refresh is not session shutdown.

The launcher gains `backend='browser'`; `backend='native'` explicitly selects
legacy display_worker.py. Browser worker writes `browser_session.json` with
URL/PID and prints the URL for notebook reopening. Existing return contract,
disabled flags, startup gate and Windows Job ownership remain in force.

## Task 1: Data Adapters (Delegated)

Files: new `3-bands/fourband/browser_data.py`, `tests/test_fourband_browser_data.py`.

- [x] Write and run failing tests for channel alignment, missing JSON values,
  geometry alignment/transforms, relative event times and immutable labels.
- [x] Implement the five shared functions using existing numerical helpers.
- [x] Verify no refit, audio predictors, coordinate projection or GUI rendering.
- [x] Run focused tests and report exact changed files and test outcomes.

Example acceptance:
```python
assert json_safe(np.array([1., np.nan, np.inf])) == [1., None, None]
payload = continuous_payload(data, result, 'E1')
assert payload['channel'] == 'E1'
assert payload['trial_ids'] == list(data.trial_ids)
```

## Task 2: Browser Frontend (Delegated)

Files: new `3-bands/fourband/browser_assets/` HTML/CSS/JS, pinned vendors,
license notices, dependency manifest and frontend verification tests/helpers.

- [x] Verify vendor/version availability from primary documentation, and bundle
  dependencies locally without changing the user's Python environment.
- [x] Implement API-driven controls, Three.js anatomy and Plotly four-band
  curves with generation guards, loading/errors, WebGL fallback and exports.
- [x] Retain independent camera/style state during channel/pair changes.
- [x] Verify controls, responsiveness and actual canvas pixels in Playwright;
  no marketing UI, external requests or fabricated physiological curves.

## Task 3: Owned Loopback Service (Controller)

Files: new `browser_session.py`, `browser_server.py`, `browser_worker.py`,
tests; modify `display_process.py`, narrowly extend `display_computation.py`.

- [x] Write failing tests for token/origin/path/body validation, pure GET routes,
  same-settings job sharing, stale-job suppression, export boundaries and close.
- [x] Add generic numerical task submission while preserving old submit API.
- [x] Implement service, spawn bootstrap logging, status persistence, browser
  opening and explicit close without leaving numerical descendants behind.
- [x] Verify child failures remain ordinary parent exceptions; existing legacy
  boundary tests explicitly request native where they are native-specific.

Example acceptance:
```python
assert session.compute(settings)['job_id'] == session.compute(settings)['job_id']
assert bad_origin_post.status == 403
assert unknown_path.status == 404
assert all(path.is_relative_to(export_root) for path in saved_files)
```

## Task 4: Notebook And Generator (Delegated)

Files: modify four-band Notebook/generator/README and focused notebook tests.

- [x] Write failing source-execution tests for DISPLAY_BACKEND='browser',
  backend forwarding in both displays, preservation of guards and settings.
- [x] Apply targeted source-only changes, no output clearing or regeneration
  over user-edited configuration/cells. Retain native as explicit fallback.
- [x] Document local URL, initial computation, refresh, explicit close,
  errors/logs, offline assets and unchanged analysis/classification behavior.
- [x] Validate Notebook schema/compiled cells and focused tests.

## Task 5: Integrated Verification And Local Finish (Controller)

- [x] Run data/service/Notebook tests and inspect agent changes against the spec.
- [x] Replay actual HS0015/B03 snapshot in the ECoG environment; use Playwright
  to inspect real brain/curves, pair5/6 and ECoG_16, labels/colors, channel
  clicks, camera movement, cache clearing, exports, refresh and session close.
- [x] Save desktop/narrow screenshots and nonblank canvas-pixel measurements;
  verify no external network requests and unchanged fitted label arrays.
- [x] Exercise invalid settings, numerical failure and interrupt cleanup.
- [x] Obtain independent scoped code review, repair findings and rerun tests.
- [x] Run the full test suite, compile Notebook cells, check staged diff and
  commit only owned changes locally. Leave user modifications unstaged.

## Progress

Implementation started after user approval on 2026-10-14. Task ownership and
verification outcomes will be recorded here as the work proceeds.

- Data adapters: 28 focused tests; real HS0015/B03 metadata/anatomy have no warnings.
- Notebook/generator: 107 focused tests passed, one skipped; 27 existing cells,
  outputs, metadata and user parameters preserved apart from three backend edits.
- Service and owned-worker tests cover strict origin/token/path boundaries,
  A/B/A in-flight sharing, stopped completion clocks and actual idle-child death.
- Actual browser replay: HS0015/B03, trial pair5/6, ECoG_16; one real lh mesh
  with 115675 vertices, 128 contacts, four curves with 931 samples and audio.
  Camera/picking, labels, class3 styling, detail PNG, export, refresh and cache
  clearing passed. Desktop/narrow PNGs and pixel/network report reside under
  `3-bands/outputs/verification/browser_actual/`; external page requests: zero.
- Initial full-suite run: 648 passed, one skipped, one adapter-test failure
  caused by assuming no unrelated Matplotlib figures exist. The test was
  corrected to verify it preserves the pre-existing figure set instead.
- Backend review fixes: reuse running A during A/B/A navigation, preserve idle
  numerical failures before cleanup. Actual idle-child failure and busy user
  shutdown probes pass; parent classification labels remain unchanged.
- Frontend review fixes: block stale exports after context loss, identify the
  selected pair's gap among context events, include captions/legend in PNGs,
  and retain brain-only cameras across two refreshes. Scoped rereview is clear.
- Final full suite: 654 passed, one skipped in 146.04 seconds. Notebook schema
  and all code cells compile. Final frontend fixture and real-data Playwright
  runs pass with zero browser errors/external requests. Accepted artifacts:
  `3-bands/outputs/verification/frontend_browser_accepted/` and
  `3-bands/outputs/verification/browser_accepted/`.
- Actual browser WebGL canvas: 3112 distinct sampled colors, 12039 nonbackground
  sampled pixels; camera movement changes pixels. Real exports include patient,
  block, round, channel, class, pair, smoothing, baseline and trace legend.
- Browser demo is running as an owned background launcher using the saved
  HS0015/B03 snapshot; its logs/session metadata are in
  `3-bands/outputs/browser_demo/20261014/`. Close Session releases this demo.
- Only the three Notebook backend-source edits are staged; all user parameters,
  saved outputs, unrelated projects and presentations remain outside the commit.
