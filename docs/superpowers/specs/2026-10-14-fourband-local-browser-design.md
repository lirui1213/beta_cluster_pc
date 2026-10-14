# Four-Band Local Browser Display

## Direction And Scope

The user selected option A: replace the unreliable native brain/curve display
with one local browser interface. This specification makes that choice concrete
for review before implementation.

- Target `3-bands/ECoG_4band_clustering_pipeline.ipynb`, its generator, and
  display helpers in `3-bands/fourband`.
- Replace the brain/linked-curve and standalone explorer displays only. Keep
  the existing trial-QC and second-round PCA review workflows unchanged.
- Do not change baseline modes, filtering, feature selection, weights, manual
  cluster counts, trial exclusions, fitted models, or classification labels.
- Preserve the user's current patient/block settings and unrelated edits.
- Leave the root Beta workflow, high-gamma project, prediction project and
  presentations unchanged. Commit only scoped changes locally; do not push.

## Interface And Interaction

Use a single work-focused page: a compact control strip, an unframed brain view
and four frequency-band plots beside it on desktop. Narrow viewports stack the
brain above the plots. The initial screen is the actual explorer, not a landing
page. Display the current patient/block, fitted round and selected channel.

Retain these controls:

- Select adjacent original trial pairs, with previous/next controls.
- Select class and channel; click a visible electrode to select its channel.
- Show all classes or one class, change class colors, and toggle channel labels.
- Adjust cortex color/opacity and electrode size; rotate, pan, zoom and reset
  the brain camera without changing electrode coordinates.
- Set display smoothing in seconds and apply it explicitly.
- Switch between continuous curves and the existing cluster-detail view.
- Clear the display cache and export the current view and display settings.

Continuous plots retain the four bands, selected-channel and class-mean curves,
audio envelope, true seconds, the interval between trials, event boundaries and
QC markers. Missing data remain missing. Audio remains display-only. Class
detail retains the existing summaries/heatmaps and duration-normalized axes;
it may be served as an Agg-rendered image rather than rewritten numerically.

The ordinary and selectable brain modes become one view with visibility
controls, avoiding duplicate windows. A standalone-explorer request hides the
brain pane. A brain-only request does not load or compute continuous curves.

## Rendering And Numerical Boundaries

- Three.js renders the actual FreeSurfer mesh and electrode geometry using
  browser WebGL. Plotly renders interactive continuous 2D curves. All frontend
  libraries, icons, fonts and assets are served locally, with pinned versions
  and their required license notices; no CDN or external runtime requests.
- Reuse the existing geometry loading and scanner-RAS-to-tkRAS transform. Do
  not project, shift, or silently drop electrode coordinates to improve the
  picture. Report unmapped channels, missing anatomy and transform failures.
- The Notebook hands a trusted local extraction/model snapshot to an owned
  display-service subprocess. It creates no Tk, Qt or VTK render window.
- A separate numerical subprocess retains the current continuous processing
  and bounded cache. The service never performs expensive signal processing
  inside an HTTP request handler or a GUI event callback.
- Same-settings requests share an in-flight calculation. Channel, class,
  visibility, color and label changes do not refilter the recording. New pair
  or smoothing requests supersede pending requests; obsolete results cannot
  overwrite the latest view. Cancel queued obsolete work when possible.
- Report actual loading/computing/error stages and elapsed time, without
  inventing percentage progress. Keep controls usable while computing. First
  calculation may still take about a minute; faster processing is not promised
  by changing the display backend.

## Local Service And Lifecycle

The service listens only on `127.0.0.1` using an OS-assigned free port. Each
session has an unpredictable access token and a fixed set of routes/assets;
there is no arbitrary file-reading, path-writing or Python-execution endpoint.
Validate settings, request sizes, session tokens and write-request origins.
Do not enable permissive CORS or send patient data to external services.

Open the session in the default browser and also print its URL in the Notebook
so the user can reopen it. Refreshing the page preserves the session and does
not refit or discard labels. Closing a browser tab alone is not a reliable
shutdown signal: an explicit Close Session command stops the service and its
numerical worker. Interrupting the Notebook cell also cleans up only its owned
process tree. Keep Windows Job ownership and the startup gate.

The display cell remains waiting while the service is active, matching the
current blocking launcher contract. Normal session shutdown returns to the
Notebook. Worker errors become ordinary exceptions with log locations, not
kernel termination. Store session metadata, logs and exports under the existing
round output directory. Save browser-side view settings with exported images.

Keep the native backend as an explicit legacy fallback, not an automatic retry
after a browser failure. Default the updated Notebook to the browser backend;
preserve the existing disabled-display guards and standalone display entry.

## Errors And Acceptance

Missing anatomy, invalid pair/channel, numerical failure and unavailable WebGL
must produce visible errors. A WebGL failure must not hide usable 2D curves.
Do not leave a permanently blank canvas or silently fall back to unsafe native
windows. Preserve classification results on every display failure.

Verification must include:

1. Unit tests for alignment/transforms, JSON missing-value handling, request
   validation, token/origin checks, cache reuse and stale-result suppression.
2. Notebook/generator tests confirming the default browser backend, preserved
   user parameters, display guards, manual class counts and unchanged labels.
3. Desktop and narrow-viewport browser screenshots plus canvas-pixel checks:
   real nonblank mesh, correctly framed anatomy, rendered curves, no overlapping
   controls/text, camera movement, channel selection and working label toggle.
4. A replay of the user's HS0015/B03 saved snapshot: initial progress, completed
   four-band/audio curves, class visibility/colors, electrode linking, export,
   cache clearing, page refresh and explicit shutdown. Check the displayed
   trial-pair 5/6 and ECoG_16 when they are available in the saved result.
5. Numerical and service crash/interrupt tests proving parent survival and
   owned-process cleanup; a complete existing test-suite run and a scoped
   independent code review before local integration.

Success means the actual browser workflow passes these checks. Existing native
GUI tests or an exported plot alone are not sufficient evidence. Report any
remaining browser/GPU compatibility limitation rather than promising that the
new architecture can never fail.
