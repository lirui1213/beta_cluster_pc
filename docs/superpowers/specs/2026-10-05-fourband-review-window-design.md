# Four-Band Review Window

User approved the design on 2026-10-05: the four-band notebook opens a modal PCA review window whenever round two runs. Controls cover enabling a manual PCA distance threshold, global/cluster reference, whole-cluster checkboxes, manual channel names and the second-round K. Show a PCA plot and live unique exclusion/remaining counts. Apply starts round two; cancel/close never fits or exports another round. Original raw/QC exclusions cannot be undone.

Keep the root Beta/shared review window unchanged. Add `fourband.review_window` using the established Tk/Matplotlib approach and a frozen `ReviewSettings`. Share exclusion audit computation with `refit_reviewed`, not a second implementation of mask rules. Validate finite positive thresholds, known channels/clusters and integer K from 2 to the remaining electrode count. Do not auto-select K or invent a distance threshold.

Retain the last accepted settings only for the same first-round model. Preserve existing notebook patient/block, baseline, surface settings and unrelated outputs during a targeted notebook update; do not regenerate it wholesale. Clear the round-two-ready flag before opening the window, and permit anatomy rendering only after a completed second round. Provide an explicit no-GUI path for the verification script only. Save accepted settings in the existing round configuration and audit.

Verification includes pure parameter/mask tests, real hidden Tk widget/callback tests, notebook accept/cancel branches, notebook schema/compile checks, a real-data no-GUI run and scoped code review. No GitHub push. Spec self-review: scope and cancel semantics explicit, no unresolved choices or placeholders.
