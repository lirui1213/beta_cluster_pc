# Four-band Continuous Trials Implementation Plan

> **For agentic workers:** Use subagent-driven-development for independent feature and surface tasks; the parent implements continuous data/display and notebook integration. Track steps below.

**Goal:** Cluster electrodes using ECoG-only descriptors and display two truly adjacent trials with their real gap, audio envelope, and optional surface channel labels.

**Architecture:** Apply an explicit ECoG predictor allowlist in four-band analysis. A focused continuous module reads actual power and reuses extraction baseline statistics; plots and exports never concatenate progress-normalized epochs. Add opt-in surface-label controls to existing PyVista renderers and targeted notebook updates preserving user configuration.

**Tech Stack:** Python, MNE, NumPy/SciPy, pandas, scikit-learn, matplotlib, nbformat, PyVista.

## Global Constraints

- Preserve root Beta behavior and existing patient/block/style parameters.
- Both K values remain manual; preserve round-two popup review.
- Use original adjacent event IDs, real seconds, default 1-second pre/post margins.
- File baseline reuses extraction statistics; pre-onset continuous display anchors to first trial for the whole interval.
- Audio is display/descriptive output only, never a predictor.
- No GitHub push; unrelated Office lock files remain untouched.

## Task 1: ECoG-only Predictors

Files: `3-bands/fourband/analysis.py`, `tests/test_fourband_ecog_only.py`.
Interface: `fit_joint` retains signature; no audio-coupling column enters either round.

- [x] Add regression with identical ECoG and modified audio-coupling descriptors, asserting equal weighted predictors/labels across both fits and reviewed refits. Legacy `audio_dur_ecog_cov` is timing-only, not envelope-derived.
- [x] Run scoped pytest and confirm expected failure.
- [x] Add explicit allowed ECoG feature suffixes before within-band redundancy pruning; retain audio descriptors only for existing diagnostic exports. Validate all four band budgets still equal 1.
- [x] Run existing four-band tests and new regressions.

## Task 2: Surface Label Toggle

Files: `visualization.py`, `tests/test_surface_channel_labels.py`.
Interface: both renderers gain `channel_label_toggle=False`, `show_channel_labels=False`; notebook opts in.

- [x] Write callback tests proving labels toggle on/off and selectable labels follow visible cluster with real channel names.
- [x] Confirm failures before implementation.
- [x] Add small checkbox, lazy point-label actor and safe actor replacement. Preserve camera/color/screenshot lifecycle and default behavior.
- [x] Run visualization tests and actual offscreen rendering checks, including HS0012 pial surface/MRI/CSV and second-round labels.

## Task 3: Continuous Data and Figure

Files: create `3-bands/fourband/continuous.py`, `tests/test_fourband_continuous.py`.
Interfaces: `adjacent_trial_pairs(events, retained_events)`, `prepare_continuous_trials(raw, events, audio, extraction, *, trial_ids=None, pre_seconds=1., post_seconds=1., sigma_seconds=.10, display_sfreq=100.)`, `save_continuous_trial_plot(data, result, output_dir)`.

- [x] Synthetic event/continuous tests: full gap is sampled, original IDs retained, two true neighboring events required, missing/invalid IDs rejected.
- [x] Verify missing-interface failures, implement selection and boundary validation.
- [x] Baseline tests: first anchor applied throughout, bad reference stays NaN, reordered channels rejected; verify failures then implement full-record band filtering and Hilbert power with extraction reference reuse.
- [x] Add real-time smoothing/resampling and envelope sampling without filling absent coverage; test spacing and recording-boundary clamps.
- [x] Add wider cluster/band plots with speech spans/onset-offset, audio right axis, cluster SEM and CSV/JSON exports; test numerical outputs and baseline metadata.

## Task 4: Notebook and Documentation

Files: `3-bands/create_notebook.py`, notebook, `3-bands/verify_notebook.py`, `3-bands/README.md`.

- [x] Add structural tests for feature policy, optional continuous display cell, both-round results, display parameters and surface opt-in.
- [x] Implement targeted updater preserving patient/block/baseline/style settings and removing stale outputs; validate schema and compile all code cells.
- [x] New cell uses `TRIAL_PAIR_IDS=None`, `TRIAL_PRE_SECONDS=1.`, `TRIAL_POST_SECONDS=1.`, a selected round, original event table and fresh-config guards. Export to unique display directory under chosen round.
- [x] Update verifier copy to disable only popups, execute continuous display, assert no coupling features and real time spacing.
- [x] README explains ECoG-only grouping, descriptive audio, original IDs, real vs normalized time and first-trial baseline display anchor.

## Task 5: Integration Verification

- [x] Run `python -m pytest -q`, scoped native/offscreen checks, notebook compile, `git diff --check`.
- [x] Run actual ECoG environment no-popup notebook copy; inspect continuous image, output CSV/JSON and both-round model feature lists.
- [x] Independent scoped review; address actionable findings with regressions.
- [x] Mark completed steps and report usage, limitations and evidence. Do not push remote.

## Review Ledger

- Task 1/2 review: no actionable findings; 115 scoped tests including real offscreen PyVista passed.
- Task 3/4 first review: reproduced four issues with failing regressions. Fixes identify notebook cells by unique semantic markers, preserve display assignments using AST, invalidate second-round readiness before first refitting, and retain/reuse custom extraction filter settings.
- Additional regressions: no valid included baseline fails before creating a display directory; changed original event times require reextraction.
- Scoped rereview: all four findings addressed, no fix regressions; 31 scoped tests passed, including independent all-band IIR extraction/display equality checks.
- Final full-suite result: 303 passed in 48.58 seconds. Notebook schema/code and original HS0012/B08 parameters/user-added cell verified against HEAD.
- Post-review actual ECoG-environment notebook copies passed in both modes: `3-bands/outputs/verification/file/20261005_210703_065054/verification_metrics.json` and `3-bands/outputs/verification/pre_onset/20261005_210733_489506/verification_metrics.json`.
- Continuous selection: original trials 0 and 1, true gap 1.9757171781 seconds, grid 0.01 seconds, smoothing sigma 0.10 seconds. Final PNG inspected with no text overlap. Surface labels checked on actual HS0012 reconstruction with unchanged camera and successful class switches.
- Work remains local on the user's existing branch. No remote push and no unrelated Office file changes.
