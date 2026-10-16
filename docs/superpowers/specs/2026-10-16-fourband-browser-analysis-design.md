# Four-Band Browser Analysis Workspace

## Approved Direction

The user selected option B and approved the following workflow on 2026-10-16:
patient/block selection, extraction settings, manual trial review, first-round
classification, manual PCA/electrode review with second-round K, and linked
brain/curve displays with persistent history. This document specifies the
behavior for review before implementation.

The target is the existing four-band project under `3-bands`. Keep the current
Notebook as an alternate entry point. Do not change the root Beta project,
high-gamma project, prediction project, presentations, or unrelated user edits.
Commit scoped work locally only; do not push to GitHub.

## Scope And Scientific Invariants

- Reuse the existing extraction, trial-QC, feature selection, weighted fitting,
  review, stability and export functions rather than executing Notebook cells
  or rewriting scientific algorithms in JavaScript.
- Keep Theta 4-8 Hz, Beta1 13-24 Hz, Beta2 24-40 Hz and High-gamma 70-150 Hz.
  Validate sampling rate; never silently truncate or drop an unsupported band.
- Both rounds use ECoG-only predictors and one joint label per electrode.
  Audio-envelope relationships remain descriptive plots/exports, not predictors.
- Both cluster counts are manually selected. Stability does not choose K or
  mutate accepted assignments. Report collapsed/tiny clusters explicitly.
- Candidate trial artifacts and PCA-distance exclusions are review aids.
  Neither is automatically accepted. Existing raw/baseline bad channels cannot
  be restored through browser controls.
- Never change source recordings. Persist exclusion masks and provenance.
- Preserve physical-time continuous double-trial displays, matched audio,
  existing morphology summaries and actual anatomical coordinates.

## Data Discovery And Configuration

Use a launcher-configured data root, initially `I:\ECoG_data`, and the existing
`3-bands/outputs` result root. Discover patients from their `blocks` directories
and blocks from `blocks/<block>/standardized`. The browser sends catalog IDs,
not arbitrary absolute paths. The backend resolves and validates every source
against the configured roots, including resolved links/junctions.

Resolve the existing standardized layout:

- `ECoG/ecog_TEST_clean_car-raw.fif`
- `events/onset.npy` and `events/offset.npy`
- `audio/analog1.wav`
- Patient `baseline/baseline.fif`, with optional baseline segment CSV.
- Patient `recon/electrodes/electrodes.csv`, `recon/mri/T1.mgz` and FreeSurfer
  surfaces using the existing recon-subject resolution logic.

Display file availability and validation errors before starting a run. A
missing required recording/event/audio input blocks analysis. Missing baseline
blocks file-mode extraction but not a valid pre-onset configuration. Missing
anatomy disables the brain with an explicit message, not the numerical analysis.
Multiple candidate inputs require explicit selection from discovered local
files; do not silently pick a different recording or baseline.

Expose baseline mode, baseline interval, baseline edge exclusion, feature
smoothing, band weights, weight preset/group weights and first-round K. Seed
defaults from documented project settings, not by importing/executing a mutable
Notebook. Show all effective settings before submission and persist them.

File mode remains the default. Pre-onset defaults to [-1.3, -0.3] seconds: a
one-second window ending 0.3 seconds before onset, not a 0.3-second baseline.
Expose validated start/end values and retain the existing warning about short
Theta baselines and temporal leakage. Keep scale-aware baseline validation.

## Workspace And State Model

Use one compact browser workspace, not a landing page. Patient, block and
analysis history are explicit selectors. Analysis/settings, trial review,
classification/review and results are navigable views of the same active run.
Display the active patient, block, run ID and fitted round beside results.

A run advances through these states:

1. Validated configuration.
2. Extracting, followed by trial review required.
3. Trial review confirmed, followed by first-round fitting.
4. First round ready, followed by optional manual second-round review.
5. Second-round fitting, followed by second round ready.

Each numerical operation also has queued/running/completed/failed/cancelled
status and a real stage, elapsed time and log location. Do not invent percentage
progress for operations without measurable totals.

Draft settings never relabel previously fitted data. Changing patient/block
selects a separate context. Changing extraction settings requires new extraction
and trial confirmation. Changing trial-QC settings invalidates confirmation.
Changing first-round K/weights requires a new first-round fit and invalidates
the pending second-round review. Changing only second-round review settings
creates another second-round result after explicit confirmation. Display-only
settings do not invalidate numerical results.

Keep old accepted results visible as identified history, never as the current
draft result. Switching while work is active requires confirmation to cancel
the owned task; do not silently abandon work or start parallel extractions.
Refreshing reconnects to the server's run/job state, not a new analysis.

## Manual Trial Review

Reuse `trial_qc.py` candidate reports and source fingerprints. Provide a
channel-by-original-trial table/grid, reason filters, raw waveform inspection,
manual selection of unflagged entries, select candidates, clear and undo.
Initial selection is empty. Changing thresholds recomputes candidates without
automatically accepting them. Explicitly confirming an empty selection is valid.

Only Confirm Exclusions applies the selected channel/trial pairs across all
four bands and recomputes features. Preserve the source extraction so review
can be revised without accumulating irreversible masks. Cancel/back does not
confirm exclusions and cannot enable a fit from stale review state. Record
thresholds, original trial IDs, channel names and source provenance.

## Classification And Electrode Review

First-round fitting uses `fit_joint` and records all effective feature budgets,
selected columns and quality exclusions. Provide PCA, cluster sizes, four-band
power curves with audio, and existing relationship summaries. The first-round
K is an explicit validated integer, not an automatic recommendation.

Second-round review reuses `pca_review_table`, `review_exclusion_audit` and
`refit_reviewed`. Controls include distance filtering enabled/disabled, positive
distance threshold, global/within-cluster reference, whole-class selection,
explicit channel selection and second-round K. Distance filtering initially
remains disabled. Preview the exclusion union and remaining population using
the same backend rule as the actual refit. Validate K against remaining usable
electrodes. Confirmation is bound to the exact first-round model/revision.

No fit runs until Confirm And Refit. Cancel leaves the accepted first round
unchanged; it cannot resurrect a stale second round. Refit all selection,
imputation, scaling, weighting and clustering steps on the accepted population.

Optional stability checks operate on an explicitly selected accepted round,
keep its electrode population/K fixed and export supported/unsupported
replicates with reasons. Expose cancellation and never substitute labels.
Retain within-class similarity and morphology/amplitude diagnostics already
available in the existing analysis helpers.

## Results And History

Reuse the offline Three.js/Plotly explorer for accepted rounds. Retain class
filter/color, contact picking, channel labels, cortex styling, camera controls,
true adjacent-trial selection, display smoothing, detail views and export.
Both fitted rounds can be selected independently. Audio remains visible.

Every run gets a unique directory under `outputs/<patient>/<block>/<run-id>`;
each fit/review revision has a distinct directory. Save configuration, input
identity, extraction snapshot, QC provenance, trial confirmation, fit settings,
assignments, model, review audit, figures, diagnostics and status. Numerical
completion and export completion are tracked so partial exports never appear
as a fully saved run. Use atomic manifest replacement and explicit schema
versions. Do not overwrite earlier reviews or saved runs.

History includes compatible new runs and existing project outputs. Incomplete
or legacy records are identified and cannot pretend to support unavailable
resume actions. Existing trusted display snapshots may provide read-only
viewing. Resuming calculations requires matching configuration/provenance and
necessary persisted source objects; otherwise offer a new analysis explicitly.
Never automatically unpickle arbitrary browser-provided files.

## Architecture And Process Ownership

Separate responsibilities into catalog/configuration, persistent run state,
numerical task runner, browser API integration and analysis/review UI. Retain
existing display-only launcher behavior for Notebook callers; add a standalone
workspace entry point that does not require a running Notebook kernel.

The HTTP service remains responsive and owns a bounded numerical worker. Heavy
imports, FIF reads, extraction, fitting, QC recomputation, plotting and stability
run outside HTTP callbacks. Load large scientific objects in the worker and
pass compact metadata/progress to the server. Save/reuse snapshots where needed
without returning raw recordings or large pickle payloads to the browser.

Use the same configured Python environment and headless Agg rendering. Create
no Tk/Qt/VTK windows in this workflow. On Windows preserve Job ownership/startup
gating and hidden helper consoles. Task cancellation must terminate only owned
computation, mark partial work cancelled and permit another run using a fresh
worker. Close Session and launcher interruption clean the owned process tree;
closing a tab alone does not cancel the service. Worker failures become logged
task failures, not Notebook/process crashes or automatic native fallback.

## API, Privacy And Race Safety

Retain loopback-only binding, random session token, strict mutation Origin/Host
checks, fixed routes, local pinned frontend assets and no permissive CORS.
No external patient-data requests, shell commands, arbitrary Python execution,
arbitrary path access or arbitrary file download endpoints are permitted.
Bound JSON body sizes, catalog enumeration, history responses and worker/cache
memory. Limit waveform responses to requested validated windows/channels.

All jobs and confirmations carry run/model/revision identity. Validate these
server-side. Stale UI responses cannot replace another patient's brain, curves,
review or labels. Surface, waveform, plots and exports are revision-bound; clear
selection/camera/contact caches on context changes. Export only consistent,
accepted numerical results and completed display state. Invalid requests return
actionable errors without changing the active accepted state.

## Verification And Acceptance

- Unit tests: catalog/path containment, missing/ambiguous inputs, parameter
  validation, state transitions, stale revision rejection, confirmation
  requirements, union exclusions, manual K and atomic/history persistence.
- Numerical parity: the same explicit configuration/trial mask/review produces
  matching features, exclusion masks and deterministic labels through existing
  Python functions and the browser runner. Audio predictors remain excluded.
- Service tests: token/origin/path protection, responsive polling, concurrent
  submission rejection, cancellation, restart, failure and no foreign-process
  termination. Refresh must not duplicate analysis.
- Browser tests: patient/block/run/round navigation, settings, empty/manual trial
  confirmation, PCA review and second K, both rounds, logs/errors, history,
  disabled stale exports, desktop/mobile layout and no external requests.
- Real-data acceptance in `ECoG_LR_v1`: a complete patient/block run, explicit
  trial review and two manual-K fits; a second discovered patient/block switch;
  optional stability; saved-result reload; cancellation/failure cleanup.
  Verify nonblank anatomical and curve rendering with screenshots/pixel checks.
- Run the existing regression suite; report any unavailable data or environment
  limitations honestly. Do not claim an unexecuted end-to-end check passed.

## Design Checklist

- [x] Existing browser, scientific workflow and user edits inspected.
- [x] Display-only versus full-analysis approaches offered; user chose B.
- [x] Full workflow presented and approved by the user.
- [x] Written specification checked for scope, contradictions and ambiguity.
- [ ] User review of this written specification.
- [ ] Detailed implementation plan and test-first execution after review.
