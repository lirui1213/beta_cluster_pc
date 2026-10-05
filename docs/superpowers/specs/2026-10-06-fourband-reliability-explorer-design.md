# Four-band reliability and inspection

## Approved direction and boundaries

The user selected stability checks, within-cluster inspection, shape/amplitude
weighting, trial QC, and interaction/performance improvements. For trial QC the
user selected option A: review flags and explicitly confirm exclusions.

- Target the four-band notebook, generator, and focused fourband helpers.
- Keep both cluster counts manual and retain the second-round PCA review.
- Audio remains display-only; do not add audio predictors or intertrial-gap
  predictors. Retain the continuous adjacent-trial display.
- Preserve patient/block, baseline, filter, and surface styling settings.
- Leave the root Beta workflow, unrelated presentations, and GitHub unchanged.
- No stability result automatically removes channels or prevents manual fitting.

## Architecture and processing order

Use separate helpers for trial QC/review, stability, weighting, inspection, and
continuous-power caching. Reuse the existing extraction, feature allowlist,
clustering, export, Tk, and PyVista patterns. Avoid a framework rewrite.

The workflow is extraction -> trial QC review -> feature aggregation -> first
manual-K fit -> electrode PCA review -> second manual-K fit -> diagnostics and
inspection. Changing trial exclusions or weights invalidates downstream fits,
including previous second-round surface labels. Display-only changes do not
refit classification. Export confirmed settings with each result.

## Trial quality and manual confirmation

- Assess raw ECoG for candidate spikes/jumps, flat runs, repeated-extreme/clipping
  candidates, and nonfinite samples; retain existing per-band baseline flags.
  Describe clipping as a candidate unless hardware limits are known.
- Use transparent, adjustable thresholds and save their units, references, and
  detection interval. Default robust thresholds are review aids, not validated
  physiological rejection criteria. Do not add notch filtering in this change.
- Review in a popup table keyed by actual channel and original trial ID, showing
  reasons, measurements, and a selected waveform. Every exclusion starts off.
- Allow checking channel-trial rows, reversible bulk selection, and manual
  selection of unflagged channel-trial rows. Show remaining trials per channel.
- Confirmation records selected pairs and recalculates features using the same
  pair mask in all four bands. This does not mutate source data or bad lists.
- Cancel/close applies no pending change. Never silently apply a previous review
  to different data, events, channels, or QC settings.
- Existing raw-bad and invalid-baseline exclusions remain non-restorable here.
  Recompute insufficient-valid-trial/channel eligibility after confirmation and
  explicitly report any newly ineligible channel. Do not impute discarded trial
  responses back into classification.
- Mask discarded responses in aggregated plots and mark them in individual
  inspection. Continuous raw traces remain inspectable with exclusion markers;
  classification exclusion must not fabricate a gap in the original recording.

## Feature budgets and presets

- Keep per-band distance budgets and ECoG-only predictor selection.
- Expose three exclusive feature-group budgets: morphology, amplitude, slope.
  Assign each selected predictor to exactly one group, disclose membership, and
  normalize within each group to avoid feature-count-driven weight changes.
- Offer balanced (1:1:1), morphology-priority (3:1:1), amplitude-priority
  (1:3:1), and custom nonnegative budgets. Require at least one active usable
  group in every positively weighted band; disclose absent groups and actual
  effective budgets. Validate finite values and preserve band-budget totals.
- Retain an explicit legacy weighting option as the migration default, allowing
  users to reproduce existing results before choosing a new preset.
- This is feature weighting, not amplitude-free waveform clustering. Explain
  that morphology descriptors can still depend on amplitude. No waveform-PCA
  classifier or new scaler is introduced in this iteration.
- Weight changes require refitting both rounds; PCA remains a display/review
  projection of the full weighted-feature model.

## Descriptive stability, not automatic K selection

- Evaluate each confirmed round at its chosen K, with the round's electrode
  exclusions and confirmed channel-trial mask held fixed.
- Compare chronological halves and seeded random half splits; bootstrap original
  trial IDs with replacement. Use the same draw across bands/channels and keep
  duplicated draws as repeated trials. Recompute features and refit preprocessing
  on each draw with the same policy and weights.
- Compute ARI against the full-data partition and between split partitions.
  Match replicate labels to the reference using maximum-overlap assignment for
  per-electrode agreement. This is descriptive agreement, not a probability of
  physiological correctness.
- If any included channel lacks supported features/trials, record a failed
  replicate rather than changing the population. Report attempted, supported,
  collapsed/tiny, and failed counts with reasons. Include supported tiny-cluster
  replicates in the agreement summary; do not hide unfavorable resamples.
- Return reports even when no replicate is supported; use unavailable metrics
  and an explanation, never the previous No eligible K exception.
- Expose seed and repeat count, default 30 bootstrap draws plus half-split
  checks. Provide progress/cancellation without discarding fitted labels.
- Export summary, replicate metrics, and channel agreement/coverage tables.

## Within-cluster inspection and linked interactions

- Add electrode-mean overlays per band: thin individual electrode curves and
  a thick cluster mean. Use bounded channel selection for dense clusters.
- Add an electrode-by-time heatmap and a selected-channel trial-by-time heatmap,
  with original IDs, exclusion markers, and explicit duration-normalized axes.
  Keep real-time continuous plots distinctly labeled in seconds.
- Keep electrode SEM labeled as such. Optionally bootstrap trials for pointwise
  confidence bands conditional on the fixed electrodes/labels; do not present
  SEM or these bands as statistical significance or cluster stability.
- Add a popup explorer with original adjacent-pair selection, previous/next
  controls, band/channel/cluster selectors, and smoothing settings. Audio stays
  on a separate axis and controls cannot change K or exclusions implicitly.
- Opt-in electrode picking in both surface windows updates the explorer's
  selected actual channel and four-band trace. Restrict pickable actors to
  electrodes, resolve identity through the displayed channel mapping, and keep
  labels, colors, camera, and class visibility consistent.
- Keep linked Tk/PyVista windows responsive, avoid blocking reentrant mainloops,
  and safely disconnect callbacks on close. Headless exports remain available.

## Performance and provenance

Cache continuous full-record per-band power in a bounded session cache, with an
explicit clear operation and memory-size report. Reuse across adjacent-pair,
display smoothing, channel, and label changes. Validate input data revision,
channels/order, sample rate, band definitions, and filter configuration before
reuse. Never reuse stale signal processing after changed inputs. Baseline and
trial-QC settings are applied after cached power and validated separately.
If the memory limit is insufficient, disclose the fallback and still produce
correct uncached results. Do not save large derived arrays into the repository.

## Verification and delivery

Use focused regression tests before implementation for masks, original IDs,
weight budgets, invalid inputs, resampling support and label permutations,
cache hits/invalidation, and notebook migration preserving user cells/settings.
Run the complete test suite and real-data notebook copies in both baseline
modes. Exercise popup selection/confirm/cancel, adjacent-pair navigation,
surface channel picking, cleanup, and caching; inspect rendered screenshots.
Keep diagnostics nonblocking and verify audio cannot enter either model.
Document runtime, actual support counts, and limitations without claiming
improved stability merely because curves look cleaner. Deliver locally only.
