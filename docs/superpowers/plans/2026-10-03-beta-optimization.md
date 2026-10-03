# Beta Analysis Optimization Implementation Plan

> **For agentic workers:** Execute signal processing and notebook integration locally; delegate the independent stability module using dispatching-parallel-agents. Each scientific behavior requires a regression test before implementation.

**Goal:** Preserve the pre-optimization snapshot on GitHub branch `2026/10/03`, then implement scale-aware baseline Z scores, consistent smoothing, morphology features and reproducible cluster evaluation.

**Architecture:** Keep existing epoch and clustering APIs compatible. Add shared signal utilities and an independent cluster evaluation module. Quality flags exclude invalid trial/channel baselines, never become clustering predictors. Keep the PyVista controls and the user's current subject/block choice (HS0015/B05).

**Tech Stack:** NumPy, SciPy, pandas, MNE, scikit-learn, Matplotlib, pytest, Jupyter.

## Constraints

- Snapshot commit: `3b8d161`; remote branch must remain the pre-optimization version.
- Local implementation branch: `codex/beta-optimization-2026-10-03`.
- Do not upload raw data, generated outputs, unrelated PPT or Office temporary files.
- Default analysis smoothing: 0.06 seconds; display comparison: 0.06, 0.10, 0.15 seconds.
- Every stability refit constructs new imputer, scaler and KMeans; compare labels using ARI.
- Compare K=2 through K=6; do not force three classes or target cluster sizes.

## Task 1: Baseline Quality and Signal Processing

Files: `signal_processing.py`, `audio_cross_3band_utils.py`, `tests/test_beta_optimization.py`.

- [x] Write tests asserting voltage-unit invariance of Z scores and baseline standard deviation equal to one.
- [x] Write tests asserting flat baselines become NaN and are exported as per-trial/channel quality flags.
- [x] Run `python -m pytest tests/test_beta_optimization.py -q` and confirm the expected failures.
- [x] Implement a denominator floor relative to each channel's typical baseline variability, plus a numerical floor relative to baseline power; explicit legacy epsilon is an optional absolute floor.
- [x] Add baseline diagnostics to epoch dataclasses with default values for backward compatibility. Route both beta and three-band normalization through shared baseline statistics.
- [x] Implement Gaussian NaN-aware smoothing and interpolation onto an evenly spaced grid with anti-alias filtering when downsampling.

## Task 2: Analysis Features and Selection

Files: `audio_cross_3band_utils.py`, `clustering_pipeline.py`, `tests/test_beta_optimization.py`.

- [x] Write tests asserting denoised derivatives retain the sign of a known response despite high-frequency noise.
- [x] Add adjustable analysis smoothing and Savitzky-Golay derivative estimation on each trial's physical-time axis.
- [x] Add early/middle/late response means, positive/negative area, signed peaks and positions, dominant peak width, offset rebound and late-to-post change.
- [x] Exclude quality metadata, constant columns and highly correlated feature pairs using good electrodes only; prioritize morphology and interpretable correlation features.
- [x] Add optional feature-group weight normalization so group influence does not grow merely with feature count.

## Task 3: Cluster Stability (Independent Worker)

Files: `cluster_evaluation.py`, `tests/test_cluster_evaluation.py`.

- [x] Test clear synthetic groups, label permutation invariance, reproducibility, bad-channel exclusion and insufficient-trial behavior.
- [x] Implement `evaluate_beta_cluster_stability(epochs, *, k_values=range(2, 7), n_resamples=20, feature_kwargs=None, bad_mask=None, feature_columns=None, feature_weights=None, random_state=50)` returning summary, resamples and recommended_k.
- [x] Re-extract features for randomized disjoint trial halves and bootstrap samples. Refit all preprocessors for every sample; use a common electrode quality mask.
- [x] Report silhouette, split-half ARI, bootstrap ARI with dispersion, minimum cluster size and within-cluster waveform correlation.
- [x] Recommend K with a documented combined ranking of silhouette, split-half and bootstrap stability; flag undersized clusters. Results are descriptive stability diagnostics, not statistical significance tests.

## Task 4: Display and Notebook Integration

Files: `visualization.py`, `ECoG_3band_clustering_pipeline.ipynb`, `README.md`, `tests/test_beta_optimization.py`.

- [x] Test that dense context and coarse speech segments produce the same physical smoothing width.
- [x] Resample the concatenated display to a shared uniform grid; smooth electrode-level trial means before calculating SEM.
- [x] Add optional trial-bootstrap confidence bands conditional on fixed cluster membership.
- [x] Add smoothing comparison and stability figures; keep Cell 7 as the curve output cell and Cell 9/10 as PyVista cells.
- [x] Expose baseline floor, analysis smoothing, feature selection, K evaluation and confidence-band settings in the notebook.
- [x] Clear stale main-notebook outputs after changing the analysis and document the changed Z-score scale.

## Task 5: Verification

- [x] Run full pytest suite and notebook JSON/schema/compile checks.
- [x] Execute the main notebook with popups disabled in a verification copy on the current HS0015/B05 dataset; preserve user's interactive defaults.
- [x] Check baseline mean/std, uniformly spaced display coordinates, feature selection report and K=2..6 evaluation outputs.
- [x] Inspect comparison and stability plots; commit the verified optimization locally without moving the remote snapshot branch.

## Verification Record (2026-10-04)

- Full suite: 87 passed in 27.56 seconds.
- Notebook schema and IPython-transformed code compilation passed.
- Actual ECoG_LR_v1 environment: in-process verification completed on HS0015/B05; GUI popups disabled only in the verification copy.
- 47 trials, 128 channels, 4 original bad channels, 0 invalid baselines; baseline mean max absolute error 4.15e-16, standard deviation approximately 1.
- Uniform display step 0.009993439 seconds; 34 of 41 predictors retained.
- Recommended K=2, cluster sizes 98 and 26; silhouette 0.26754, split-half ARI 0.24072, bootstrap ARI 0.47804. Stability remains modest, not evidence of definitive physiological classes.
- Inspected smoothing comparison and stability figures. Resampling QC now keeps the original electrode population fixed; unsupported samples are reported rather than permanently excluding electrodes.
- Remote snapshot remains 3b8d161d7dfa1aa5a69263355ca447d9d0616ab4. Optimization remains on the local implementation branch.
