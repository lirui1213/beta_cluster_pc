# Four-band Reliability Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** Add descriptive stability, manually confirmed channel-trial QC, group weighting, detailed curves, and cached linked inspection to the four-band workflow.

**Architecture:** Keep extraction immutable as the original reference; reviewed copies mask only confirmed channel-trial pairs. Independent modules handle QC, stability, weighting, inspection, and signal caching. The notebook orchestrates them and invalidates downstream results on analysis changes.

**Tech Stack:** Python, NumPy, pandas, SciPy, scikit-learn, MNE, Matplotlib, Tk, PyVista, nbformat, pytest.

## Global Constraints

- Follow `docs/superpowers/specs/2026-10-06-fourband-reliability-explorer-design.md`.
- Work locally in the existing non-main development checkout; no push and no unrelated staging/commits.
- Preserve all user parameters, added notebook cells, source data, and existing staged changes.
- Keep manual K, ECoG-only predictors, and fixed electrode population during stability evaluation.
- Candidate artifact flags never automatically exclude trial/channel data.
- New optional shared visualization arguments preserve existing callers.

## Task 1: Trial QC and manual review

**Files:** Create `3-bands/fourband/trial_qc.py`, `3-bands/fourband/trial_review_window.py`, `tests/test_fourband_trial_qc.py`.

**Interfaces:** `detect_trial_quality(raw, extraction, settings=None) -> DataFrame`; `apply_trial_exclusions(extraction, excluded_pairs, *, smoothing_sigma_seconds=.06, slope_method="savgol", max_baseline_invalid_fraction=.25) -> FourBandExtraction`; `TrialQualityReviewWindow(raw, extraction, report, initial_pairs=()).show() -> tuple[(str,int),...] | None`. The reviewed extraction must retain filter kwargs, actual original trial IDs, baseline diagnostics, and expose `trial_exclusions` for display provenance.

- [x] Write tests that spike/flat candidates are flagged, no pairs are excluded implicitly, masking `(E1, 17)` affects only that channel/trial in all bands, original arrays remain unchanged, unknown pairs fail, remaining-trial QC is recomputed, and cancel returns None.
```python
reviewed = apply_trial_exclusions(original, [("E1", 17)])
assert np.isnan(reviewed.epochs["theta"].speech[row17, col1]).all()
assert np.array_equal(original.epochs["theta"].speech, original_copy, equal_nan=True)
```
- [x] Run `python -m pytest tests/test_fourband_trial_qc.py -q` and record intended missing-feature failures.
- [x] Implement robust adjustable raw candidate metrics, baseline-report joins, immutable masking/reaggregation, and Tk waveform/table review with reversible bulk selection and no default selections.
- [x] Run focused tests and record results; parent integrates only this task's files.

## Task 2: Descriptive four-band stability

**Files:** Create `3-bands/fourband/stability.py`, `tests/test_fourband_stability.py`.

**Interfaces:** `evaluate_stability(extraction, result, *, n_resamples=30, random_state=50, fit_kwargs=None, feature_kwargs=None, progress=None, cancelled=None) -> report` with `summary: dict`, `replicates: DataFrame`, `channels: DataFrame`; `save_stability(report, output_dir)` writes JSON/CSV/PNG. Reaggregate epochs via existing beta feature extraction, never resample only electrode feature rows. Use matching draws across all bands.

- [x] Write tests for identity/permuted-label agreement, immutable original data, seeded resampling, common band trial draws, fixed masks including unsupported samples, failed-replicate reporting without raising, cancellation and tiny-cluster accounting.
```python
report = evaluate_stability(extraction, result, n_resamples=3, random_state=12)
assert report.summary["requested_k"] == result.kmeans.n_clusters
assert len(report.channels) == int(result.clean_mask.sum())
```
- [x] Run `python -m pytest tests/test_fourband_stability.py -q` to observe missing implementation failure.
- [x] Implement chronological and random halves, bootstrap trials, independently fitted transforms, ARI and maximum-overlap label mapping, support counts/reasons, progress/cancellation and report exports. Do not union resample QC into permanent masks.
- [x] Run tests and record results; return integration interface to parent.

## Task 3: Cached power and surface channel picking

**Files:** Modify `3-bands/fourband/continuous.py`, `visualization.py`; create `tests/test_fourband_cache.py`, `tests/test_surface_channel_picking.py`.

**Interfaces:** `ContinuousPowerCache(max_bytes=512*1024**2)` with `clear()` and `info()`; optional `cache=None` in both continuous preparation/loading. Optional `on_channel_pick=None` in both surface render functions; callback receives the actual channel string. Cache only raw band power; normalize per current extraction after retrieval.

- [x] Test a real raw change invalidates cached power, filtering runs once across display settings/pair changes, changes to bands/filter/sfreq/order invalidate, limit fallback is correct, and actor-specific point selections map to actual channel names.
```python
first = prepare_continuous_trials(raw, events, audio, extraction, cache=cache)
second = prepare_continuous_trials(raw, events, audio, extraction, cache=cache, sigma_seconds=.15)
assert cache.info()["hits"] > 0
```
- [x] Run focused tests and record the intended failures.
- [x] Add bounded session cache with full data/provenance validation, fallback reporting, explicit clearing and electrode-only picking callbacks with cleanup.
- [x] Run cache, picking, existing continuous and surface tests; parent owns explorer and notebook integration.

## Task 4: Weight budgets and detailed inspection

**Files:** Modify `3-bands/fourband/analysis.py`; create `3-bands/fourband/weights.py`, `3-bands/fourband/inspection.py`, `3-bands/fourband/explorer.py`, `tests/test_fourband_weights.py`, `tests/test_fourband_inspection.py`.

**Interfaces:** optional `weight_preset="legacy", group_weights=None` to `fit_joint`; `feature_group(column) -> str`; `save_inspection(extraction, result, output_dir, *, sigma_seconds=.10, n_bootstrap=100, random_state=50)`; `FourBandExplorer(...).show(block=False)` and `.select_channel(channel)` plus stable callable for surface picks.

- [x] Test legacy equivalence, exact band/group totals, zero/invalid budgets, excluded audio columns, sparse group behavior, correct electrode/trial identifiers, bootstrap dimensions, NaN masks, and unchanged labels during navigation.
```python
fit = fit_joint(table, n_clusters=2, weight_preset="balanced")
assert sum(w for c,w in zip(fit.feature_columns, fit.feature_weights) if c.startswith("theta__")) == pytest.approx(1.)
```
- [x] Run tests to see missing-feature failures.
- [x] Implement mutually exclusive group assignment/budgets, bounded overlays and heatmaps, optional pointwise trial CI, and Tk explorer with adjacent-pair navigation, band/channel/cluster and smoothing controls, labels/SEM/QC markers, export and cleanup. Integrate Task 3's cache without duplicating it.
- [x] Run focused tests and inspect real rendered detail figures.

## Task 5: Notebook integration, provenance and verification

**Files:** Modify `3-bands/create_notebook.py`, `3-bands/ECoG_4band_clustering_pipeline.ipynb`, `3-bands/fourband/workflow.py`, `3-bands/verify_notebook.py`, `3-bands/README.md`; create `tests/test_fourband_reliability_notebook.py`, `tests/fourband_reliability_gui_checks.py`.

- [x] Test semantic migration with added user cells/custom settings, first/second-result invalidation, pending review cancellation, weighting identity, nonblocking stability reports, and opt-in surface callbacks.
```python
before = notebook_parameter_source(nb)
update_reliability_cells(path)
assert notebook_parameter_source(read_notebook(path)) == before
```
- [x] Run intended missing migration failures.
- [x] Add separate settings/QC/stability/inspection cells, preserve original parameter cell byte-for-byte, keep legacy weighting default, run optional seeded stability without selecting K, save confirmed QC and weight budgets, and share a cache/explorer session. Semantic markers make migration repeatable; clear stale outputs only.
- [x] Run full pytest, notebook schema/code compile, `git diff --check`, and real-data verification copies in file/pre_onset modes with GUI disabled only in copies. GUI script checks no default exclusions, cancel/confirm, pair navigation, surface selection and close handling; inspect PNGs.
- [x] Request independent spec/quality review, resolve concrete defects with regression tests, update this ledger with evidence, and commit only owned optimization files locally.

## Execution Ledger

- Baseline: existing four-band implementation and user settings are preserved; prior verified suite had 303 passing tests. Run a fresh baseline before relying on this count.
- Plan self-review: all approved scope has a task; no auto K, auto trial exclusion, audio predictor, intertrial feature, or GitHub operation is introduced.
- TDD: initial missing-feature failures were observed for QC, stability, weights, inspection, cache, explorer and migration. Subsequent regressions covered manual-mask baseline accounting, huge finite group budgets, stale QC settings, disabled surface guards, boundary labels and compact legends.
- Latest complete suite: `OMP_NUM_THREADS=1`, explicit local Tcl/Tk library paths; **447 passed in 78.39 seconds**. Explicit native Tk/VTK checks: **7 passed in 7.03 seconds**, including right-click channel identity, linked event pumping, pair navigation, review confirm/cancel and cleanup.
- Notebook schema validation and AST compilation passed. The original parameter cell matches HEAD byte-for-byte, and the user's extra blank code cell remains. `git diff --check` passed.
- Independent weights/inspection reviews found an overflow with group budgets near `1e308`; max-scaled normalization and regression tests resolve it. Final QC/stability/notebook review found no actionable issues; reviewer ran **68 focused tests**.
- Screenshot checks covered GUI continuous/detail views and the real HS0012/B08 first-round Beta1 detail with K=10, compact legend, bounded overlays, original channel/trial identifiers and nonblank heatmaps.
- Final actual `ECoG_LR_v1` notebook-copy runs passed all execution and numerical assertions in both modes on HS0012/B08 (18 trials, 128 channels, 126 included). File mode used legacy weights and manual K=10/4; pre_onset mode used balanced weights, K=3/2 and the explicit `(ECoG_1, 0)` exclusion. Verification copies alone changed these parameters; the user's notebook parameters remain untouched.
- Evidence: `3-bands/outputs/verification/file/20261006_015917_543359/verification_metrics.json` and `3-bands/outputs/verification/pre_onset/20261006_020254_787958/verification_metrics.json`. Each round had 17/17 supported comparisons for this small functional check (`n_resamples=5`); the notebook default remains 30. The file-mode K=10 run had 16 tiny replicates, which remain reported and included, not discarded. These settings differ between runs and are not evidence that changing weights improves stability.
- In the manually reviewed real run, all four bands retained 17 valid trials for ECoG_1 and 18 for ECoG_2; the manual fraction is separate from baseline invalidity. The rendered trial heatmap leaves original trial 0 blank and explicitly marks its exclusion. Continuous traces retain the original recording with QC markers.
- Delivery: integrate into the existing local branch only. Exclude the user's modified presentation, Office lock file, raw signals, verification outputs and cache arrays; no push.
