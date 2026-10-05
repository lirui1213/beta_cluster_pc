# Four-band ECoG-only clustering and continuous trial display

## Approved Direction

The user requested removal of audio-derived clustering features, display of two
trials, a brain-surface channel-label toggle, a longer time axis, and retention
of the speech envelope. The user selected continuous adjacent trials (option A),
not onset-aligned overlays.

## Scope

- Update the four-band workflow in `3-bands` and its notebook generator.
- Preserve patient/block settings, baseline settings, surface styling, and the
  two-round manual K/PCA review workflow.
- Do not change the root Beta notebook or push anything to GitHub.
- Shared visualization helpers may gain optional label controls; their defaults
  must preserve existing callers.

## Classification

- Both rounds use ECoG-only power, morphology, and power-slope descriptors.
- Explicitly exclude audio correlation/regression/lag, audio slope products,
  audio-related same-sign/crossing descriptors and trialwise audio coupling.
- Retain duration descriptors derived from event timings, not the envelope.
- Filter predictors before redundancy selection, imputation, scaling, and
  balanced per-band weighting, including when given older feature tables.
- Retain audio relationship descriptors only as clearly named descriptive output
  where needed for existing plots; they cannot enter either fitted model.
- Removing audio predictors can change labels. Do not promise unchanged results.

## Continuous Display

- Add an explicit display cell available after each clustering round, with
  configurable original event IDs and pre/post margins (defaults: 1 second).
- Default to the first available pair of truly adjacent original events that
  survived extraction. Show an event table for choosing another pair.
- Display original trial IDs, not silently reindexed retained trial numbers.
- Read and band-filter the continuous ECoG recording, with filtering context,
  then select the requested interval. Never join separate normalized epochs or
  fabricate the gap between trials.
- Use elapsed seconds relative to the first onset; keep actual speech durations
  and the actual inter-trial gap. Mark both onset/offset pairs and speech spans.
- Preserve the envelope throughout the interval on a separate axis. Audio does
  not affect electrode eligibility, normalization, or clustering predictors.
- Display cluster mean and electrode SEM per band from the selected interval;
  smooth each electrode trace in physical seconds before averaging.
- Use a wide figure so the two trials and gap are readable. Keep existing
  duration-normalized average plots as supplemental results, explicitly labeled.
- File baseline: apply the same per-band/channel baseline statistics as
  extraction. Pre-onset baseline: use the first selected trial's statistics for
  the entire display interval and label this choice. Do not switch baseline at
  the second onset. Classification normalization remains unchanged.
- Clamp requested margins at recording boundaries and disclose actual bounds.
  Reject nonadjacent/unavailable event IDs, invalid margins, mismatched channel
  names, insufficient valid baseline, or missing continuous input with clear
  errors. Do not reuse a stale result after parameters change.
- Export trace CSV, figure PNG, and display metadata including input IDs, bounds,
  baseline anchor, smoothing, sampling and cluster labels.

## Surface Labels

- Four-band surface windows have a compact channel-label checkbox, initially off.
- Labels are sourced from the electrode CSV channel names, not positional row
  numbers; show the actual channel identity to avoid off-by-one ambiguity.
- In the selectable-class window, labels follow the currently visible class.
- Toggling labels must not alter colors, visibility of electrode meshes, camera,
  coordinates, or fitted labels. Existing screenshot/show lifecycle is preserved.

## Verification

- Regression tests prove no audio-derived predictor is selected in either round
  and changing only audio descriptors does not change fitted features/labels.
- Synthetic continuous data verify true spacing, gap activity, event IDs,
  baseline anchoring, boundary behavior, smoothing and channel alignment.
- Test surface toggle and class-switch callbacks, using actual offscreen PyVista
  when available plus existing deterministic mocks.
- Validate notebook schema/code and preservation of existing user parameters.
- Execute a no-popup real-data verification copy; inspect the continuous figure.
