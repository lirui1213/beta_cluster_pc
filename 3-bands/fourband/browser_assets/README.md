# Four-band browser frontend

Static frontend for the approved token-scoped local service. Serve this directory
at `/s/<token>/`; all module, library, API and detail-image URLs stay inside that
root. There is no build step, CDN, web font, inline script, import map, service
worker or automatic unload shutdown. The parent owns service integration.

## Files and contracts

- `index.html`, `styles.css`: responsive workbench, controls and status.
- `app.js`: API-driven display state, refresh settings, Plotly rendering/export.
- `brain.js`: source geometry, fixed electrode coordinates, OrbitControls,
  raycast selection, label projection, pixel-diameter glyphs and camera export.
- `core.mjs`: same-origin JSON requests, abort and generation guards, null-gap
  traces, true-second event boundaries and scoped detail URLs.

`GET api/meta` and `GET api/surface` initialize the page. `POST api/compute`
sends only `view`, `trial_ids`, `sigma_seconds`, `channel`, `band`.
`GET api/job/<id>?channel=<encoded channel>` polls queued/running jobs until
ready, displaying the actual stage and elapsed seconds without percentages.
The renderer checks the current revision before and after asynchronous work.
`POST api/clear` invalidates the view and reloads the latest selection.
`POST api/export` sends `{settings, images}` with PNG data URLs; all style and
camera settings remain local until this export. `POST api/close` is explicit.

Class changes choose a member channel when necessary; the service's continuous
job key ignores channel/band. Pure style/visibility changes submit no numerical
work. Brain-only sessions do not submit compute, and standalone curve sessions
do not request anatomy. WebGL failures leave 2D curves available.

Point size is screen-pixel diameter, matching the supplied PyVista default 14.
Sphere glyph radii adapt to camera depth; electrode xyz never change. Initial
camera faces the left hemisphere from -x for `lh`/left meshes, or the right
hemisphere from +x for `rh`/right meshes. Other geometry uses contact distribution
to choose a lateral/front view. Panning, orbit, wheel/pinch zoom and reset use
the pinned Three.js OrbitControls implementation.

## Dependency evidence

Official npm registry archives were downloaded at exact versions and verified
against their published SHA-512 integrity values before extraction. See
`dependencies.json` for source URLs, versions and SHA-256 hashes of shipped files.

| Library | Version | License notice |
| --- | --- | --- |
| Three.js + OrbitControls | 0.170.0 | `vendor/three/LICENSE` (MIT) |
| Plotly full browser distribution | 2.35.2 | `vendor/plotly.js-dist-min/LICENSE` (MIT) |
| lucide UMD | 0.468.0 | `vendor/lucide/LICENSE` (ISC) |

The only vendor adaptation is OrbitControls' bare `three` import changed to a
relative local module import, avoiding an inline import map under strict CSP.
License notices are retained intact. System fonts need no bundled font file.

## Frontend verification

From the workspace root:

```powershell
node scripts/verify_fourband_browser_frontend.mjs
$env:PLAYWRIGHT_CHROMIUM_EXECUTABLE = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
node scripts/verify_fourband_browser_frontend.mjs --browser
```

The optional browser check uses an isolated Node HTTP fixture with the service's
strict CSP and same-origin POST requirements. It never connects to or closes a
real notebook service. It writes desktop/mobile/fallback screenshots and a JSON
report under `3-bands/outputs/verification/frontend_browser/` (or
`FRONTEND_ARTIFACTS`). It tests frontend behavior
and synthetic geometry only; the parent must verify actual anatomy/numerics and
owned-process lifecycle on the real saved snapshot.
