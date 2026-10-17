# First-Round 3D PCA Review

Approved scope: restore the browser first-round review to the native review's
three-dimensional PC1/PC2/PC3 scatter style without restoring native windows.

Use the bundled Plotly scientific plotting library, keeping the current browser
workflow and exclusion controls. Match the existing analysis palette (red,
blue, green, purple, gray, then the extended colors), with outlined circular
electrodes, pale grid planes, and black crosses for exclusion previews. Hover
retains channel identity. Orbit, zoom, and the standard reset view remain
available. Preserve the camera across previews and tab/selected-round changes;
reset it when a different accepted first model or run is loaded.

Always read coordinates and labels from the accepted FIRST model, even when
second-round results are displayed. Preserve PC3 in all supported PCA payload
formats. Missing, null, nonfinite, or nonnumeric coordinates are omitted, never
silently placed at the origin. No numerical fitting, baseline, exclusion-rule,
manual-K, waveform, or brain-rendering changes are included.

Verify trace values and coordinate validation with Node tests, then exercise
desktop/mobile rendering, camera interaction/persistence/reset, exclusion
preview, and the existing workflow in headless Chrome. Inspect screenshots and
nonblank canvas pixels. Check the real accepted HS0015/B03 model if its local
service is available. Commit only scoped files and push the exact user-requested
branch `2026-10-17_showmodify`; preserve unrelated dirty files.
