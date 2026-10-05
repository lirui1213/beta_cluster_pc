# Four-Band Review Window Implementation Plan

> **For agentic workers:** Execute inline with tests and a scoped review before completion.

**Goal:** Move four-band round-two exclusions and K selection into a default modal window.

**Architecture:** `review_exclusion_audit(first, ...)` is shared by preview and refit. `ReviewSettings` and `FourBandReviewWindow(result, initial=...).show()` live in `fourband/review_window.py`; `show()` returns accepted settings or None. Notebook round two gates fitting/export and optional anatomy on accepted settings and a completed round. Root Beta files are unchanged.

**Tech Stack:** Existing Python, Tk/ttk, Matplotlib TkAgg, NumPy/pandas, nbformat, pytest.

## Constraints

- Work in the current local project, preserving dirty notebook settings and unrelated edits. No GitHub push or automatic commit.
- Empty distance threshold is disabled; no guessed threshold. Raw/QC bads remain excluded.
- Whole cluster/manual/distance exclusions form a union. Reject K<2, nonintegers and K greater than included count before applying.
- Cancel/close does not fit, export or render old round-two anatomy.

## Tasks

- [x] Add failing tests in `tests/test_fourband_review_window.py`: parse thresholds/K, unique counts, disabled thresholds, raw-mask retention, accept/cancel, stale model settings and notebook gating. Initial run: 20 failures for missing module/default popup functionality.
- [x] Refactor only the existing mask rules into `review_exclusion_audit` in `3-bands/fourband/review.py`. Implement frozen settings and a lazily constructed Tk window in `3-bands/fourband/review_window.py`. Preview and confirmation call the same audit; confirmation additionally validates K. Cluster checkboxes, manual entry, numeric K/threshold and readonly reference combobox drive preview, with invalid settings disabling Apply.
- [x] Update review/round-two cell sources in `3-bands/create_notebook.py`. Add `--update-review` to replace only those cells, their markdown, relevant imports and anatomy guard in the existing notebook, preserving the parameter cell. Round two calls `FourBandReviewWindow(...).show()` by default and fits only in the accepted branch. Keep a no-GUI `ReviewSettings` path for the verifier.
- [x] Update `3-bands/verify_notebook.py` to override only its copy with `OPEN_REVIEW_WINDOW=False` and explicit settings. Update `3-bands/README.md` for the new flow and cancellation/repeat semantics.
- [x] Run targeted tests, real Tk callbacks, complete pytest, notebook validation and actual HS0023/B07 copy verification in ECoG_LR_v1. Request read-only scoped review and fix confirmed findings with regression tests. Actual outcomes recorded below.

Minimal behavioral contract:

```python
settings, audit = parse_review_settings(first, manual_channels="E0", n_clusters="2",
    distance_enabled=True, distance_threshold="3", distance_reference="global")
assert settings.n_clusters == 2
assert audit.loc[audit.channel == "E0", "excluded_final"].all()
selection = FourBandReviewWindow(first, initial=settings).show()
if selection is not None:
    second, audit = refit_reviewed(features, first, **selection.refit_kwargs())
```

## Verification Record

- Targeted suite: 46 passed, including a subprocess integration check that executes all 5 native Tk widget tests. Full root suite: 205 passed in 42.15 seconds. No tests skipped for the popup.
- Windows base Python 3.13/Tcl 8.6.14 exhibited init.tcl file-descriptor failures when multiple native windows shared pytest FD capture. Basic Tk and ten sequential popup openings outside pytest passed, and the same widget checks passed with system-stream capture. Native checks now run in `tests/fourband_review_gui_checks.py` in an isolated subprocess with `--capture=sys`. No production environment changes or Tk retries were added.
- Native widget checks cover accepted threshold/reference/manual/cluster/K, invalid K and cancellation, pending timer consumption, axis-label bounds and callback teardown. Additional red-green regressions clear old anatomy readiness before failing guards and reset settings after a changed first model.
- Scoped read-only review found no remaining correctness issues after stale readiness was moved before guards. Root Beta review/render modules were not edited.
- Actual ECoG_LR_v1 Python 3.11/Tcl 8.6.15 window opened visibly: global threshold 3, manual ECoG_1, K=4 showed 119 remaining electrodes; changing the reference combobox to cluster was accepted. Final screenshot `3-bands/outputs/review_window_verification/20261005/review_window_final.png` inspected, axis labels and controls fit. Threshold 3 is verification-only; delivered defaults remain distance-disabled.
- Actual HS0023/B07 no-GUI copy: 38 trials, 122 initially eligible channels, first K=3, second K=2 with manual ECoG_1/global threshold 3 yielded 119 channels in sizes 55/64. Baseline and uniform-grid checks passed. Accepted settings are in the round config and audit under `3-bands/outputs/verification/file/20261005_142530_583802/`.
- Notebook schema and compilation passed. User parameter cell preserved: HS0012/B08, ivory surface, opacity 1 and both anatomy flags True. Unrelated dirty `audio_cross_3band_utils.py`, empty-speech tests and presentation lock file untouched. No commit or GitHub push.
