# Selectable Baseline Implementation Plan

**Approved design:** Two baseline sources for the beta-progress pipeline: a patient-level FIF reference (default in the notebook), or per-trial onset-relative [-1.3, -0.3] seconds. Display windows remain independent. No GitHub push.

**Architecture:** Add `baseline_sources.py` for independently filtered reference segments and channel alignment. Extend the existing epoch/loader APIs and optional diagnostics, without changing the legacy three-band workflow. Preserve current notebook HS0006/B09 and user-selected cluster settings.

1. Write `tests/test_baseline_sources.py` covering onset timing, out-of-recording/overlapping windows, file pooling, reordered/missing channels, segment boundaries, insufficient reference data and metadata mismatch. Run before implementation and verify missing-feature failures.
2. Implement file statistics using the same beta/Hilbert-power preprocessing as task data. Read companion segment CSV or quiet annotations, never filter across joins. Trim half the FIR support plus an adjustable Hilbert-edge margin; report skipped segments. Match channel names and reject mismatched units/types/reference flags/sample rate. Metadata flags cannot prove identical reference electrodes; document this limitation.
3. Extend `extract_beta_progress_epochs` and `build_beta_progress_feature_table_from_files` with `baseline_mode`, `baseline_window`, `baseline_fif`, `baseline_segments_csv`, and `baseline_edge_seconds`. Invalid local windows remain NaN with explicit reasons; overlap checks include duration-rejected speech trials. Export normalized reference checks separately from display arrays.
4. Add notebook parameters, conditional path validation, file-segment/reference reports, and mode logging. Preserve user edits and clear stale outputs mechanically after the targeted source patch. Update README and verification script.
5. Run full pytest, schema/compile checks and the actual ECoG environment for both modes. Inspect numerical reports and generated curves. Record limitations without relaxing stability eligibility or silently falling back to a different baseline.

**Commands:** `python -m pytest tests/test_baseline_sources.py -q`; `python -m pytest -q`; `E:\anaconda\envs\ECoG_LR_v1\python.exe scripts/verify_beta_optimization.py --mode inprocess --baseline-mode file`; repeat with `--baseline-mode pre_onset`.

## Verification Record

- Initial new-source tests failed for the missing baseline API/module, then passed after implementation. Additional regressions reproduced inaccurate low-variance labels for reference bad channels, empty trial previews, and discarded no-eligible-K diagnostics before fixes.
- Full suite: 107 passed in 27.71 seconds. Notebook schema, IPython-transformed cell compilation, and Python module compilation passed.
- Current HS0006/B09 preserved: 44 retained trials, 256 electrodes. Both baseline sources were validated in the actual ECoG_LR_v1 environment, with normalized reference means within 4e-16 of zero and standard deviations approximately one.
- File reference: 5 of 6 segments used, 34.31 seconds after 0.8075-second edge trimming on each side; 13 reference bad channels remain invalid. K=2..6 each had minimum cluster size 1, so there is no eligible automatic K (all resample fits completed). Diagnostics are now exported before raising; no threshold was relaxed. Explicit manual-K=4 plotting verification passed, with ECoG_175 as a singleton; this is not an accepted stable classification.
- Pre-onset reference: five trial windows overlap earlier labeled speech and remain invalid, without truncation. Full analysis/plotting verification passed; eligible K=2 with 99/150 electrodes, 7 original bad channels. Split-half ARI is only 0.0573 and bootstrap ARI 0.2318; an eligible recommendation is not proof of strong repeatability.
- Inspected both rendered curve grids; original notebook saved in ignored outputs before clearing stale results. GUI defaults and unrelated PPT files retained. No GitHub push.
- Independent review identified missing stored highpass/lowpass checks. Two regressions reproduced the issue, then passed after enforcing matching finite preprocessing metadata. Actual HS0006/B09 metadata matched (0/200 Hz); file baseline numerical verification passed again after the fix.
