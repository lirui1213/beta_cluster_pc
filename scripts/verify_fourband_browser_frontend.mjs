// Run: node scripts/verify_fourband_browser_frontend.mjs [--browser]
// Optional: PLAYWRIGHT_MODULE, PLAYWRIGHT_CHROMIUM_EXECUTABLE, FRONTEND_ARTIFACTS.
import assert from "node:assert/strict";
import { existsSync, readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";
import http from "node:http";
import { createRequire } from "node:module";
import { createHash } from "node:crypto";

const assets = fileURLToPath(
  new URL("../3-bands/fourband/browser_assets/", import.meta.url),
);
assert.ok(
  existsSync(path.join(assets, "core.mjs")),
  "Frontend request and curve module must exist",
);
const manifest = JSON.parse(
  readFileSync(path.join(assets, "dependencies.json"), "utf8"),
);
for (const dependency of manifest.dependencies)
  for (const [file, expected] of Object.entries(dependency.files))
    assert.equal(
      "sha256-" +
        createHash("sha256")
          .update(readFileSync(path.join(assets, file)))
          .digest("hex"),
      expected,
      `Vendor integrity: ${file}`,
    );
const { SessionAPI, curveTraces, eventShapes, safeImageURL } = await import(
  pathToFileURL(path.join(assets, "core.mjs"))
);
const {gridTraces,ClusterGrid} = await import(pathToFileURL(path.join(assets,"cluster_grid.mjs")));
const gapTraces = gridTraces({times_ms:[0,1,2,3,4],mean:[1,2,null,3,4],
  sem:[.1,.1,null,.2,.2],audio:[0,1,null,1,0],counts:[2,2,0,2,2]}, "#d42f66");
assert.equal(gapTraces[1].fill, "toself", "SEM must close each finite region without bridging gaps");
assert.deepEqual(gapTraces[1].x, [0,1,1,0,null,3,4,4,3,null]);
assert.deepEqual(gapTraces[2].y, [1,2,null,3,4]);
const probe = Object.create(ClusterGrid.prototype);
let beginRestyle,releaseRestyle,restyleCalls=0;
const restyleStarted = new Promise(resolve => {beginRestyle=resolve;});
const oldWindow = globalThis.window;
globalThis.window = {Plotly:{restyle:()=>{
  restyleCalls++;
  beginRestyle();
  if (restyleCalls > 1) return Promise.resolve();
  return new Promise(resolve => {releaseRestyle=resolve;});
}}};
Object.assign(probe,{closed:false,queue:Promise.resolve(),controller:new AbortController(),
  colors:()=>({0:"#167d91"}),entries:[{plot:{},label:0,dot:{style:{}}}]});
const recoloring=probe.recolor();
await restyleStarted;
probe.close();
releaseRestyle();
await recoloring;
globalThis.window=oldWindow;
assert.equal(restyleCalls,1,"Closing during SEM update must stop the pending mean update");
const origin = "http://127.0.0.1:1234";
assert.equal(
  safeImageURL("api/image/job1", `${origin}/s/token/`),
  `${origin}/s/token/api/image/job1`,
);
assert.throws(() =>
  safeImageURL("https://example.com/patient.png", `${origin}/s/token/`),
);
assert.throws(() => safeImageURL("../other/image", `${origin}/s/token/`));
const bands = { name: "Beta", selected: [1, null, 3], mean: [null, 2, 4] };
const traces = curveTraces(
  { times: [0, 1, 2], audio: [0.2, null, 0.4] },
  bands,
  "#167d91",
  true,
  true,
);
assert.deepEqual(traces[0].y, [1, null, 3]);
assert.deepEqual(traces[1].y, [null, 2, 4]);
assert.equal(traces[0].connectgaps, false);
assert.equal(traces[2].yaxis, "y2");
assert.deepEqual(
  curveTraces({ times: [0] }, { selected: [] }, "#167d91", false, false)[0].y,
  [],
);
assert.equal(
  eventShapes([{ onset: 0, offset: 1, excluded: true }]).filter(
    (x) => x.type === "line",
  ).length,
  2,
);
const contextEvents = [
  { original_trial_id: 4, onset: -1, offset: -0.5 },
  { original_trial_id: 5, onset: 0, offset: 1 },
  { original_trial_id: 6, onset: 2, offset: 3 },
];
const gaps = (events, ids) =>
  eventShapes(events, ids)
    .filter((shape) => shape.fillcolor === "rgba(93,109,121,0.055)")
    .map((shape) => [shape.x0, shape.x1]);
assert.deepEqual(
  gaps(contextEvents, [5, 6]),
  [[1, 2]],
  "Gap must join selected trials, not preceding context",
);
assert.deepEqual(
  gaps([...contextEvents].reverse(), [5, 6]),
  [[1, 2]],
  "Selected gap must not depend on context ordering",
);
assert.deepEqual(
  gaps(contextEvents, [6, 7]),
  [],
  "Missing selected events must not invent a gap",
);

const calls = [];
let readyChannel = "";
let firstResolve;
const client = new SessionAPI(`${origin}/s/token/`, async (url, options) => {
  calls.push({ url: String(url), options });
  if (String(url).endsWith("api/compute")) {
    const body = JSON.parse(options.body);
    if (body.channel === "slow")
      return new Promise((resolve) => {
        firstResolve = () =>
          resolve(
            new Response(JSON.stringify({ job_id: "old", state: "queued" })),
          );
      });
    return new Response(JSON.stringify({ job_id: "new", state: "queued" }));
  }
  readyChannel = new URL(url).searchParams.get("channel");
  return new Response(
    JSON.stringify({ state: "ready", data: { channel: readyChannel } }),
  );
});
let result = null;
const slow = client.compute(
  { channel: "slow" },
  () => {
    throw new Error("stale progress");
  },
  (value) => {
    result = value;
  },
);
await Promise.resolve();
await client.compute(
  { channel: "E 16" },
  () => {},
  (value) => {
    result = value;
  },
);
firstResolve();
await slow;
assert.equal(
  result.channel,
  "E 16",
  "Old compute response must never replace current channel",
);
assert.equal(readyChannel, "E 16");
assert.equal(calls[0].options.credentials, "same-origin");
assert.equal(calls[0].options.headers["Content-Type"], "application/json");
assert.ok(calls.every((call) => call.url.startsWith(`${origin}/s/token/api/`)));
console.log(
  "PASS core: null gaps, event boundaries, token scope, JSON requests and stale compute suppression",
);

if (!process.argv.includes("--browser")) process.exit(0);
let playwright;
if (process.env.PLAYWRIGHT_MODULE)
  playwright = await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE));
else {
  const require = createRequire(import.meta.url);
  try {
    playwright = require("playwright");
  } catch {
    const bundled = path.join(
      process.env.USERPROFILE || "",
      ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs",
    );
    playwright = await import(pathToFileURL(bundled));
  }
}
const out =
  process.env.FRONTEND_ARTIFACTS ||
  path.resolve(assets, "../../outputs/verification/frontend_browser");
mkdirSync(out, { recursive: true });
const requests = [],
  jobs = new Map();
let exports = [],
  clearCount = 0,
  closeCount = 0,
  nextID = 0;
let heldChannel = null;
let gridAvailable = true;
const channels = ["ECoG_1", "ECoG_16", "ECoG_3"];
const metadata = {
  subject: "Fixture",
  block: "B03",
  round: 2,
  channels,
  labels: [0, 1, 0],
  bands: ["Theta", "Alpha", "Beta", "High gamma"],
  pairs: [
    { first_trial_id: 5, second_trial_id: 6, onsets: [0, 2], offsets: [1, 3] },
    { first_trial_id: 6, second_trial_id: 7, onsets: [0, 2], offsets: [1, 3] },
  ],
  settings: {
    trial_ids: [5, 6],
    sigma_seconds: 0.05,
    pre_seconds: 1,
    post_seconds: 1,
    display_sfreq: 100,
  },
  surface_options: {
    brain_color: "ivory",
    brain_opacity: 0.65,
    point_size: 14,
  },
  surface_enabled: true,
  curves_enabled: true,
  warnings: ["Synthetic frontend fixture; no physiological verification."],
};
const gridPayload = {
  available: true,
  bands: metadata.bands.map((name, i) => ({name, low:[4,13,24,70][i], high:[8,24,40,150][i]})),
  classes: [{label:0,n_electrodes:2}, {label:1,n_electrodes:1}],
  speech_offset_ms: 1500,
  panels: [0,1].flatMap(label => metadata.bands.map((band, i) => ({
    band, label, times_ms:[-1000,0,500,1000,1500,2000,2500],
    mean:[0,i+.2,i+.6,null,i+.3,i-.2,0],
    sem:label === 0 ? [.1,.2,.2,null,.15,.1,.1] : [null,null,null,null,null,null,null],
    audio:[0,.1,.8,null,.3,.1,0], counts:[2,2,2,0,2,2,2],
  }))),
};
// A deterministic asymmetric triangulated ellipsoid tests real WebGL geometry rendering.
const positions = [],
  indices = [];
for (let v = 0; v <= 24; v++)
  for (let u = 0; u <= 36; u++) {
    const phi = (v * Math.PI) / 24,
      theta = (u * 2 * Math.PI) / 36;
    positions.push(
      50 * Math.sin(phi) * Math.cos(theta),
      70 * Math.sin(phi) * Math.sin(theta),
      60 * Math.cos(phi),
    );
  }
for (let v = 0; v < 24; v++)
  for (let u = 0; u < 36; u++) {
    const a = v * 37 + u;
    indices.push(a, a + 1, a + 37, a + 1, a + 38, a + 37);
  }
const surface = {
  meshes: [{ name: "fixture-cortex", positions, indices }],
  electrodes: [
    { channel: "ECoG_1", label: 0, position: [-28, -69, 5] },
    { channel: "ECoG_16", label: 1, position: [0, -73, 5] },
    { channel: "ECoG_3", label: 0, position: [28, -69, 5] },
  ],
  warnings: [],
};
function payload(channel, pair) {
  const times = [-1, 0, 1, 1.5, 2, 3, 4];
  return {
    times,
    audio: [0, 0.2, null, 0.1, 0.5, 0.2, 0],
    bands: metadata.bands.map((name, i) => ({
      name,
      selected: [i, 1 + i, null, 2 + i, 1 + i, 3 + i, i],
      mean: [i, 0.5 + i, null, 1 + i, 2 + i, 1 + i, i],
    })),
    events: [
      {
        original_trial_id: pair[0] - 1,
        onset: -1,
        offset: -0.5,
        excluded: false,
      },
      { original_trial_id: pair[0], onset: 0, offset: 1, excluded: false },
      { original_trial_id: pair[1], onset: 2, offset: 3, excluded: true },
    ],
    channel,
    label: metadata.labels[channels.indexOf(channel)],
    trial_ids: pair,
    baseline_description: "Fixture baseline",
    metadata: {},
  };
}
const server = http.createServer(async (req, res) => {
  res.setHeader(
    "Content-Security-Policy",
    "default-src 'self'; script-src 'self' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; object-src 'none'; frame-ancestors 'none'",
  );
  requests.push({ method: req.method, url: req.url });
  const url = new URL(req.url, "http://127.0.0.1");
  const prefix = "/s/frontend-fixture/";
  if (!url.pathname.startsWith(prefix)) {
    res.writeHead(404).end();
    return;
  }
  const route = url.pathname.slice(prefix.length);
  const json = (value) => {
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify(value));
  };
  if (route === "api/meta") {
    json(metadata);
    return;
  }
  if (route === "api/surface") {
    json(surface);
    return;
  }
  if (route === "api/cluster-grid") {
    json(gridAvailable ? gridPayload : {available:false,message:"Saved class summary is unavailable"});
    return;
  }
  if (req.method === "POST") {
    let body = "";
    for await (const chunk of req) body += chunk;
    const data = JSON.parse(body || "{}");
    assert.equal(req.headers["content-type"], "application/json");
    assert.equal(
      req.headers.origin,
      `http://127.0.0.1:${server.address().port}`,
    );
    if (route === "api/compute") {
      assert.deepEqual(Object.keys(data).sort(), [
        "band",
        "channel",
        "sigma_seconds",
        "trial_ids",
        "view",
      ]);
      const job_id = String(++nextID);
      jobs.set(job_id, { ...data, polls: 0 });
      json({ job_id, state: "queued" });
      return;
    }
    if (route === "api/clear") {
      clearCount++;
      jobs.clear();
      json({ state: "cleared" });
      return;
    }
    if (route === "api/export") {
      exports.push(data);
      json({ files: ["fixture.png", "fixture.json"] });
      return;
    }
    if (route === "api/close") {
      closeCount++;
      json({ state: "closed" });
      return;
    }
  }
  if (route.startsWith("api/job/")) {
    const job = jobs.get(route.split("/").at(-1));
    if (!job) {
      res.writeHead(404);
      json({ error: "Missing fixture job" });
      return;
    }
    if (job.polls++ === 0) {
      json({
        state: "running",
        stage: "fixture numerical work",
        elapsed_seconds: 0.1,
      });
      return;
    }
    if (job.sigma_seconds === 9) {
      json({
        state: "failed",
        error: "Fixture numerical failure",
        elapsed_seconds: 0.2,
      });
      return;
    }
    if (job.channel === heldChannel) {
      json({
        state: "running",
        stage: "fixture held channel",
        elapsed_seconds: 0.2,
      });
      return;
    }
    if (job.view === "detail") {
      json({
        state: "ready",
        data: {
          image_url: "api/image/detail",
          channel: job.channel,
          band: job.band,
        },
      });
      return;
    }
    if (job.trial_ids[0] === 6)
      await new Promise((resolve) => setTimeout(resolve, 300));
    json({
      state: "ready",
      data: payload(url.searchParams.get("channel"), job.trial_ids),
      elapsed_seconds: 0.2,
    });
    return;
  }
  if (route === "api/image/detail") {
    res.setHeader("Content-Type", "image/png");
    res.end(
      Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=",
        "base64",
      ),
    );
    return;
  }
  const rel = route || "index.html";
  const target = path.resolve(assets, rel);
  if (!target.startsWith(assets) || !existsSync(target)) {
    res.writeHead(404).end();
    return;
  }
  const types = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".json": "application/json",
  };
  res.setHeader(
    "Content-Type",
    types[path.extname(target)] || "application/octet-stream",
  );
  res.end(readFileSync(target));
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const base = `http://127.0.0.1:${server.address().port}/s/frontend-fixture/`;
let browser;
try {
  browser = await playwright.chromium.launch({
    headless: true,
    executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || undefined,
    args: ["--enable-unsafe-swiftshader"],
  });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 960 },
    deviceScaleFactor: 1,
  });
  const errors = [],
    external = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("request", (request) => {
    const url = request.url();
    if (
      !url.startsWith(base) &&
      !url.startsWith("data:") &&
      !(url.startsWith("blob:") && new URL(url).origin === new URL(base).origin)
    )
      external.push(url);
  });
  await page.goto(base);
  await page.waitForFunction(
    () => document.querySelector("#status-text")?.textContent === "Ready",
  );
  await page.waitForSelector("#brain canvas");
  await page.waitForFunction(
    () => document.querySelectorAll("#plots .js-plotly-plot").length === 4,
  );
  await page.waitForFunction(() => document.querySelectorAll("#cluster-grid .js-plotly-plot").length === 8, null, {timeout:3000});
  assert.equal(await page.locator("#cluster-grid .cluster-grid-row").count(), 2);
  const gridSnapshot = await page.locator("#cluster-grid .js-plotly-plot").first().evaluate(el => ({
    y:el.data[2].y, audio:el.data[3].y, shapes:el.layout.shapes,
  }));
  assert.deepEqual(gridSnapshot.y, gridPayload.panels[0].mean);
  assert.equal(gridSnapshot.y[3], null, "Missing samples remain gaps");
  assert.deepEqual(gridSnapshot.shapes.map(shape => shape.x0), [0,1500]);
  await page.locator("#cluster-grid-pane").screenshot({path:path.join(out,"cluster-grid-desktop.png")});
  assert.equal(await page.locator("#pair").inputValue(), "0");
  assert.equal(await page.locator("#point-size").inputValue(), "14");
  assert.equal(await page.locator("#brain-color").inputValue(), "#fffff0");
  const pendingPage = await browser.newPage();
  await pendingPage.goto(base);
  await pendingPage.waitForFunction(
    () => document.querySelector("#status-text").textContent === "Ready",
  );
  await pendingPage.locator("#view-detail").click();
  await pendingPage.waitForFunction(
    () =>
      document.querySelector("#status-text").textContent === "Ready" &&
      !document.querySelector("#detail").hidden,
  );
  heldChannel = "ECoG_16";
  await pendingPage.locator("#channel").selectOption("ECoG_16");
  await pendingPage.waitForFunction(
    () =>
      document.querySelector("#status-text").textContent ===
      "fixture held channel",
  );
  await pendingPage.locator("#brain canvas").dispatchEvent("webglcontextlost");
  assert.equal(
    await pendingPage.locator("#export").isDisabled(),
    true,
    "Context loss must preserve pending export lock for retained E1 detail",
  );
  const beforeStaleExport = exports.length;
  await pendingPage.evaluate(() => {
    const button = document.querySelector("#export");
    button.disabled = false;
    button.click();
  });
  await pendingPage.waitForFunction(() =>
    document.querySelector("#notice").textContent.includes("Export failed"),
  );
  assert.equal(
    exports.length,
    beforeStaleExport,
    "Export handler must reject an obsolete rendered revision even if invoked directly",
  );
  heldChannel = null;
  await pendingPage.waitForFunction(
    () => document.querySelector("#status-text").textContent === "Ready",
  );
  await pendingPage.locator("#export").click();
  await pendingPage.waitForFunction(() =>
    document.querySelector("#notice").textContent.includes("Export saved"),
  );
  assert.equal(exports.at(-1).settings.channel, "ECoG_16");
  assert.ok(exports.at(-1).images.curves);
  assert.equal(
    exports.at(-1).images.brain,
    undefined,
    "Context-lost anatomy must not be exported",
  );
  exports.pop();
  await pendingPage.close();
  const initial = await page.evaluate(() => ({
    nullGap: document.querySelector(".js-plotly-plot").data[0].y[2],
    plotCount: document.querySelectorAll("#plots .js-plotly-plot").length,
    selectedGap: document
      .querySelector(".js-plotly-plot")
      .layout.shapes.filter(
        (shape) => shape.fillcolor === "rgba(93,109,121,0.055)",
      )
      .map((shape) => [shape.x0, shape.x1]),
    canvas: (() => {
      const canvas = document.querySelector("#brain canvas");
      const copy = document.createElement("canvas");
      copy.width = canvas.width;
      copy.height = canvas.height;
      const ctx = copy.getContext("2d");
      ctx.drawImage(canvas, 0, 0);
      const pixels = ctx.getImageData(0, 0, copy.width, copy.height).data;
      const colors = new Set();
      let nonBackground = 0;
      for (let i = 0; i < pixels.length; i += 16) {
        colors.add(`${pixels[i]},${pixels[i + 1]},${pixels[i + 2]}`);
        if (
          Math.abs(pixels[i] - pixels[0]) +
            Math.abs(pixels[i + 1] - pixels[1]) +
            Math.abs(pixels[i + 2] - pixels[2]) >
          35
        )
          nonBackground++;
      }
      return { colors: colors.size, nonBackground };
    })(),
  }));
  assert.equal(initial.nullGap, null);
  assert.equal(initial.plotCount, 4);
  assert.deepEqual(
    initial.selectedGap,
    [[1, 2]],
    "Rendered gap must use selected trial IDs with preceding context",
  );
  assert.ok(initial.canvas.colors > 20);
  assert.ok(initial.canvas.nonBackground > 1000);
  await page.screenshot({
    path: path.join(out, "desktop.png"),
    fullPage: true,
  });
  const canvas = page.locator("#brain canvas");
  const before = await canvas.screenshot();
  const box = await canvas.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(
    box.x + box.width / 2 + 90,
    box.y + box.height / 2 + 30,
    { steps: 10 },
  );
  await page.mouse.up();
  await page.waitForTimeout(300);
  assert.notDeepEqual(
    await canvas.screenshot(),
    before,
    "Orbit must change rendered pixels",
  );
  await page.locator("#reset-camera").click();
  await page.locator("#labels").check();
  await page.waitForFunction(() =>
    [...document.querySelectorAll(".electrode-label")].some(
      (node) => !node.hidden,
    ),
  );
  await page.locator("#channel").selectOption("ECoG_16");
  await page.waitForFunction(
    () =>
      document.querySelector("#selection-name").textContent === "ECoG_16" &&
      document.querySelector("#status-text").textContent === "Ready",
  );
  // Click a projected label position: raycasting and labels use the same original xyz.
  const label = page.locator(".electrode-label").filter({ hasText: "ECoG_3" });
  const labelBox = await label.boundingBox();
  assert.ok(
    labelBox,
    "Visible electrode label must have projected coordinates",
  );
  await page.mouse.click(labelBox.x + labelBox.width / 2, labelBox.y - 5);
  await page.waitForFunction(
    () =>
      document.querySelector("#channel").value === "ECoG_3" &&
      document.querySelector("#status-text").textContent === "Ready",
  );
  await page.locator("#class-filter").selectOption("1");
  await page.waitForFunction(
    () => document.querySelector("#status-text").textContent === "Ready",
  );
  const computeCount = requests.filter((req) =>
    req.url.endsWith("api/compute"),
  ).length;
  await page.locator("#class-color").fill("#d42f66");
  await page.waitForFunction(() => [...document.querySelectorAll('#cluster-grid [data-label="1"] .js-plotly-plot')]
    .every(el => el.data[2].line.color === "#d42f66" && el.data[1].fillcolor === "rgba(212,47,102,0.18)"));
  assert.equal(await page.locator('#cluster-grid [data-label="0"] .js-plotly-plot').first().evaluate(el => el.data[2].line.color), "#167d91");
  await page.locator("#brain-opacity").fill("0.3");
  await page.locator("#point-size").fill("5");
  await page.locator("#labels").uncheck();
  assert.equal(
    requests.filter((req) => req.url.endsWith("api/compute")).length,
    computeCount,
    "Style changes must not submit numerical work",
  );
  await page.locator("#next-pair").click();
  await page.locator("#previous-pair").click();
  await page.waitForFunction(
    () =>
      document.querySelector("#status-text").textContent === "Ready" &&
      document.querySelector("#pair").value === "0",
  );
  await page.waitForTimeout(500);
  assert.ok(
    (await page.locator("#pair-description").textContent()).includes("5"),
  );
  await page.locator("#sigma").fill("0.12");
  await page.locator("#apply-sigma").click();
  await page.waitForFunction(
    () => document.querySelector("#status-text").textContent === "Ready",
  );
  assert.deepEqual(await page.locator("#cluster-grid .js-plotly-plot").first().evaluate(el => ({
    y:el.data[2].y,audio:el.data[3].y,shapes:el.layout.shapes,
  })), gridSnapshot, "Pair and continuous sigma do not change saved class summary");
  assert.equal(requests.filter(req => req.url.endsWith("api/cluster-grid")).length, 2, "One saved-summary fetch per page, independent of display changes");
  await page.locator("#view-detail").click();
  await page.waitForSelector("#detail-image:not([hidden])");
  await page.waitForFunction(
    () =>
      document.querySelector("#detail-image").complete &&
      document.querySelector("#detail-image").naturalWidth > 0,
  );
  await page.locator("#view-continuous").click();
  await page.waitForFunction(
    () => document.querySelector("#status-text").textContent === "Ready",
  );
  await page.locator("#export").click();
  await page.locator("#class-color").fill("#317c55");
  await page.waitForFunction(() =>
    document
      .querySelector("#notice")
      .textContent.includes("View changed during export"),
  );
  assert.equal(
    exports.length,
    0,
    "An export must not combine images with different display settings",
  );
  await page.waitForTimeout(150);
  await page.locator("#export").click();
  await page.waitForFunction(() =>
    document.querySelector("#notice").textContent.includes("Export"),
  );
  assert.equal(exports.length, 1);
  assert.ok(exports[0].images.brain.startsWith("data:image/png;base64,"));
  assert.ok(exports[0].images.curves.startsWith("data:image/png;base64,"));
  assert.deepEqual(exports[0].settings.trial_ids, [5, 6]);
  assert.equal(exports[0].settings.sigma_seconds, 0.12);
  assert.ok(exports[0].settings.camera);
  await page.locator("#clear-cache").click();
  await page.waitForFunction(
    () => document.querySelector("#status-text").textContent === "Ready",
  );
  assert.equal(clearCount, 1);
  await page.reload();
  await page.waitForFunction(
    () => document.querySelector("#status-text").textContent === "Ready",
  );
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(300);
  const mobile = await page.evaluate(() => ({
    width: document.documentElement.scrollWidth,
    viewport: innerWidth,
    brain: document
      .querySelector("#brain-pane")
      .getBoundingClientRect()
      .toJSON(),
    curves: document
      .querySelector("#curves-pane")
      .getBoundingClientRect()
      .toJSON(),
    controls: [
      ...document.querySelectorAll(
        "#controls button,#controls select,#controls input",
      ),
    ]
      .filter((node) => node.offsetWidth && node.offsetHeight)
      .map((node) => ({
        id: node.id,
        ...node.getBoundingClientRect().toJSON(),
      })),
  }));
  assert.ok(
    mobile.width <= mobile.viewport + 1,
    "No mobile horizontal overflow",
  );
  assert.ok(mobile.curves.y >= mobile.brain.bottom - 1, "Mobile panes stack");
  for (let i = 0; i < mobile.controls.length; i++)
    for (let j = i + 1; j < mobile.controls.length; j++) {
      const a = mobile.controls[i],
        b = mobile.controls[j];
      assert.ok(
        Math.min(a.right, b.right) - Math.max(a.left, b.left) <= 1 ||
          Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) <= 1,
        `Mobile controls overlap: ${a.id}/${b.id}`,
      );
    }
  await page.screenshot({ path: path.join(out, "mobile.png"), fullPage: true });
  await page.waitForFunction(() => [...document.querySelectorAll("#cluster-grid .js-plotly-plot")].every(el => Math.abs(el._fullLayout.width-el.clientWidth) < 2));
  const mobileGrid = await page.locator("#cluster-grid .js-plotly-plot").evaluateAll(nodes => nodes.map(el => el.getBoundingClientRect().toJSON()));
  assert.ok(mobileGrid[1].top >= mobileGrid[0].bottom, "Band panels stack on mobile");
  await page.locator("#cluster-grid-pane").screenshot({path:path.join(out,"cluster-grid-mobile.png")});
  await page.locator("#sigma").fill("9");
  await page.locator("#apply-sigma").click();
  await page.waitForFunction(() =>
    document
      .querySelector("#status-text")
      .textContent.includes("Fixture numerical failure"),
  );
  await page.locator("#sigma").fill("0.05");
  await page.locator("#apply-sigma").click();
  await page.waitForFunction(
    () => document.querySelector("#status-text").textContent === "Ready",
  );
  await page.locator("#close-session").click();
  await page.waitForFunction(
    () =>
      document.querySelector("#status-text").textContent === "Session closed",
  );
  assert.equal(closeCount, 1);
  const fallback = await browser.newPage();
  await fallback.addInitScript(() => {
    const original = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, ...args) {
      if (
        type === "webgl" ||
        type === "webgl2" ||
        type === "experimental-webgl"
      )
        return null;
      return original.call(this, type, ...args);
    };
  });
  await fallback.goto(base);
  await fallback.waitForFunction(
    () => document.querySelector("#status-text").textContent === "Ready",
  );
  assert.match(await fallback.locator("#brain-error").textContent(), /WebGL/);
  assert.equal(await fallback.locator("#plots .js-plotly-plot").count(), 4);
  await fallback.screenshot({
    path: path.join(out, "webgl-unavailable.png"),
    fullPage: true,
  });
  const captionPage = await browser.newPage({
    viewport: { width: 1440, height: 960 },
  });
  await captionPage.goto(base);
  await captionPage.waitForFunction(
    () => document.querySelector("#status-text").textContent === "Ready",
  );
  // Record text only while calling the native canvas methods; inspect the actual exported PNG below.
  await captionPage.evaluate(() => {
    const texts = new WeakMap(),
      fillText = CanvasRenderingContext2D.prototype.fillText,
      toDataURL = HTMLCanvasElement.prototype.toDataURL;
    window.exportedCanvasText = new Map();
    CanvasRenderingContext2D.prototype.fillText = function (text, ...args) {
      if (!texts.has(this.canvas)) texts.set(this.canvas, []);
      texts.get(this.canvas).push(String(text));
      return fillText.call(this, text, ...args);
    };
    HTMLCanvasElement.prototype.toDataURL = function (...args) {
      const value = toDataURL.apply(this, args);
      window.exportedCanvasText.set(value, texts.get(this) || []);
      return value;
    };
  });
  const labeledExports = [];
  for (const channel of ["ECoG_1", "ECoG_3"]) {
    await captionPage.locator("#channel").selectOption(channel);
    await captionPage.waitForFunction(
      () => document.querySelector("#status-text").textContent === "Ready",
    );
    const count = exports.length;
    await captionPage.locator("#export").click();
    await captionPage.waitForFunction(() =>
      document.querySelector("#notice").textContent.includes("Export saved"),
    );
    assert.equal(exports.length, count + 1);
    const png = exports.at(-1).images.curves;
    const caption = await captionPage.evaluate(async (source) => {
      const image = new Image();
      image.src = source;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = image.naturalWidth;
      canvas.height = 80;
      const context = canvas.getContext("2d");
      context.drawImage(image, 0, 0);
      const pixels = context.getImageData(0, 0, canvas.width, 80).data;
      let ink = 0;
      for (let i = 0; i < pixels.length; i += 4)
        if (Math.min(pixels[i], pixels[i + 1], pixels[i + 2]) < 220) ink++;
      return {
        text: (window.exportedCanvasText.get(source) || []).join("\n"),
        ink,
      };
    }, png);
    assert.ok(
      caption.text.includes(channel),
      `PNG must identify channel ${channel} independently of JSON`,
    );
    for (const text of [
      "Class 0",
      "Fixture baseline",
      "Channel",
      "Class mean",
      "Audio",
      "QC excluded",
    ])
      assert.ok(caption.text.includes(text), `PNG must contain ${text}`);
    assert.ok(
      caption.ink > 100,
      "Export identifying header must have rendered text pixels",
    );
    writeFileSync(
      path.join(out, `continuous-${channel}.png`),
      Buffer.from(png.split(",")[1], "base64"),
    );
    labeledExports.push(png);
  }
  assert.notEqual(
    labeledExports[0],
    labeledExports[1],
    "Identical fixture curves for different channels must have independently labeled PNGs",
  );
  await captionPage.close();
  const brainOnly = await browser.newPage();
  metadata.curves_enabled = false;
  const beforeBrainOnly = requests.filter((req) =>
    req.url.endsWith("api/compute"),
  ).length;
  await brainOnly.goto(base);
  await brainOnly.waitForFunction(
    () => document.querySelector("#status-text").textContent === "Ready",
  );
  assert.equal(
    requests.filter((req) => req.url.endsWith("api/compute")).length,
    beforeBrainOnly,
  );
  assert.equal(await brainOnly.locator("#curves-pane").isVisible(), false);
  const brainOnlyBox = await brainOnly.locator("#brain canvas").boundingBox();
  await brainOnly.mouse.move(
    brainOnlyBox.x + brainOnlyBox.width / 2,
    brainOnlyBox.y + brainOnlyBox.height / 2,
  );
  await brainOnly.mouse.down();
  await brainOnly.mouse.move(
    brainOnlyBox.x + brainOnlyBox.width / 2 + 90,
    brainOnlyBox.y + brainOnlyBox.height / 2 + 30,
    { steps: 6 },
  );
  await brainOnly.mouse.up();
  await brainOnly.mouse.down({ button: "right" });
  await brainOnly.mouse.move(
    brainOnlyBox.x + brainOnlyBox.width / 2 + 115,
    brainOnlyBox.y + brainOnlyBox.height / 2 + 55,
    { steps: 4 },
  );
  await brainOnly.mouse.up({ button: "right" });
  const cameraExport = async () => {
    const count = exports.length;
    await brainOnly.locator("#export").click();
    await brainOnly.waitForFunction(() =>
      document.querySelector("#notice").textContent.includes("Export saved"),
    );
    assert.equal(exports.length, count + 1);
    return exports.at(-1).settings.camera;
  };
  const brainOnlyCamera = await cameraExport();
  for (let refresh = 1; refresh <= 2; refresh++) {
    await brainOnly.reload();
    await brainOnly.waitForFunction(
      () => document.querySelector("#status-text").textContent === "Ready",
    );
    const restored = await cameraExport();
    for (const key of ["position", "target"])
      for (let i = 0; i < 3; i++)
        assert.ok(
          Math.abs(restored[key][i] - brainOnlyCamera[key][i]) < 1e-8,
          `Brain-only ${key} must survive refresh ${refresh}`,
        );
  }
  metadata.curves_enabled = true;
  metadata.surface_enabled = false;
  const standalone = await browser.newPage();
  const surfaceCount = requests.filter((req) =>
    req.url.endsWith("api/surface"),
  ).length;
  await standalone.goto(base);
  await standalone.waitForFunction(
    () => document.querySelector("#status-text").textContent === "Ready",
  );
  assert.equal(await standalone.locator("#brain-pane").isVisible(), false);
  assert.equal(
    requests.filter((req) => req.url.endsWith("api/surface")).length,
    surfaceCount,
  );
  gridAvailable = false;
  const missingSummary = await browser.newPage();
  await missingSummary.goto(base);
  await missingSummary.waitForFunction(() => document.querySelector("#status-text").textContent === "Ready");
  await missingSummary.waitForFunction(() => document.querySelector("#cluster-grid-status").textContent.includes("unavailable"));
  assert.equal(await missingSummary.locator("#plots .js-plotly-plot").count(), 4, "Missing summary must not block continuous response");
  assert.equal(await missingSummary.locator("#cluster-grid .js-plotly-plot").count(), 0);
  await missingSummary.close();
  gridAvailable = true;
  assert.deepEqual(errors, []);
  assert.deepEqual(external, []);
  const report = {
    initial,
    mobile,
    requests: requests.length,
    exportCount: exports.length,
    clearCount,
    closeCount,
    errors,
    external,
    fixtureOnly: true,
    brainOnlyCamera,
  };
  writeFileSync(path.join(out, "report.json"), JSON.stringify(report, null, 2));
  console.log(
    "PASS browser: offline vendors, WebGL pixels, orbit, labels, controls, selected-trial gaps, stale pair, detail, pending/context-loss export guard, independently labeled PNGs, clear, refresh, failure, mobile, close, WebGL fallback, brain-only camera across two refreshes and standalone",
  );
  console.log(`Artifacts: ${out}`);
} finally {
  if (browser) await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
