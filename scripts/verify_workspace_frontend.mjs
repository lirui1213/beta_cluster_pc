// node --test scripts/verify_workspace_frontend.mjs; add --browser for UI fixtures.
import test from "node:test";
import assert from "node:assert/strict";
import {
  WorkspaceAPI,
  StateGate,
  TrialSelection,
  validateConfig,
  reviewPayload,
  isBusy,
  canAction,
  safeLocalURL,
  normalizeQC,
  pcaTraces,
  inputErrors,
  reviewData,
  stabilityText,
  configChanges,
  firstFitPayload,
} from "../3-bands/fourband/browser_assets/workspace_core.mjs";

test('new-analysis context rejects old server state until explicit start/open',()=>{
  const gate=new StateGate();gate.accept({run_id:'old',revision:4},gate.generation);gate.detach();
  assert.equal(gate.accept({run_id:'old',revision:5},gate.generation),false);
  gate.reset();
  assert.equal(gate.accept({run_id:'old',revision:6},gate.generation),false);
  assert.equal(gate.accept({run_id:'new',revision:1},gate.generation,{attach:true}),true);
  assert.equal(gate.accept({run_id:'new',revision:2},gate.generation),true);
});

test('clipping candidate flags remain available to the reason filter',()=>{
  const [row]=normalizeQC({qc_rows:[{channel:'A',original_trial_id:3,flag_clipping_candidate:true}]});
  assert.equal(row.flag_clipping,true);
  assert.equal(row.candidate,true);
});

test('fit-only settings reuse reviewed extraction and invalidate second-round review',()=>{
  const accepted=validateConfig({first_k:3});
  const changed=validateConfig({...accepted,first_k:4,band_weights:{theta:2,beta1:1,beta2:1,high_gamma:1}});
  assert.deepEqual(configChanges(accepted,changed),{extractionDirty:false,fitDirty:true});
  assert.deepEqual(configChanges(accepted,{...changed,feature_sigma_seconds:.08}),{extractionDirty:true,fitDirty:true});
  const state={run_id:'r',stage:'first_ready',first_round_id:'f',selected_round:'f'};
  assert.equal(canAction(state,'fit_first',{fitDirty:true}),true);
  assert.equal(canAction(state,'fit_second',{fitDirty:true}),false);
  assert.equal(canAction(state,'preview_review',{fitDirty:true}),false);
  assert.equal(canAction(state,'fit_first',{extractionDirty:true}),false);
  assert.deepEqual(firstFitPayload(changed),{first_k:4,fit_settings:{band_weights:{theta:2,beta1:1,beta2:1,high_gamma:1},weight_preset:'legacy',group_weights:null}});
});

test("stability toolbar shows summary metrics without replicate/channel arrays", () => {
  const value=stabilityText({round:2,summary:{supported:15,attempted:17,failed:2,ari_full_mean:.91234,collapsed:1,included_channels:Array(100).fill('secret-channel')},channels:Array(200).fill({channel:'secret-channel'}),replicates:Array(200).fill({status:'failed'})});
  assert.match(value,/ARI 0.912/);
  assert.match(value,/15\/17/);
  assert.ok(value.length<150);
  assert.equal(value.includes('secret-channel'),false);
});

test("opening blocks mutations and historical rounds block review", () => {
  const state = {
    run_id: "r",
    revision: 4,
    stage: "first_ready",
    first_round_id: "f",
    selected_round: "f",
    task: { status: "opening" },
  };
  assert.equal(isBusy(state), true);
  assert.equal(canAction(state, "fit_second"), false);
  state.task.status = "cancelling";
  assert.equal(isBusy(state), true);
  assert.equal(canAction(state, "confirm_trials"), false);
  state.read_only = true;
  assert.equal(canAction(state, "detect_qc"), false);
  assert.equal(canAction(state, "fit_first"), false);
  state.task.status = "completed";
  state.read_only = false;
  assert.equal(canAction(state, "fit_second"), true);
  assert.equal(canAction(state, "fit_second", { trialDirty: true }), false);
  assert.equal(canAction(state, "preview_review", { qcDirty: true }), false);
  assert.equal(canAction(state, "fit_first", { configDirty: true }), false);
  assert.equal(canAction(state, "confirm_trials", { qcDirty: true }), false);
  state.selected_round = "old";
  assert.equal(canAction(state, "preview_review"), false);
  assert.equal(canAction(state, "stability"), false);
  state.first_round_id = null;
  state.selected_round = "f";
  assert.equal(canAction(state, "fit_second"), false);
});

test("generation and revision gate rejects stale context and older polls", () => {
  const gate = new StateGate();
  const token = gate.generation;
  assert.equal(gate.accept({ run_id: "a", revision: 7 }, token), true);
  assert.equal(gate.accept({ run_id: "a", revision: 6 }, token), false);
  gate.reset();
  assert.equal(gate.accept({ run_id: "a", revision: 8 }, token), false);
  assert.equal(
    gate.accept({ run_id: "b", revision: 1 }, gate.generation),
    true,
  );
});

test("trial selection retains original ids and supports manual, empty and undo", () => {
  const selection = new TrialSelection();
  selection.toggle("A", 0);
  selection.toggle("B", 9);
  assert.deepEqual(selection.pairs(), [
    ["A", 0],
    ["B", 9],
  ]);
  selection.clear();
  assert.deepEqual(selection.pairs(), []);
  selection.undo();
  assert.deepEqual(selection.pairs(), [
    ["A", 0],
    ["B", 9],
  ]);
  selection.candidates([
    { channel: "C", trial_id: 4, candidate: true },
    { channel: "D", trial_id: 5, candidate: false },
  ]);
  assert.deepEqual(selection.pairs(), [["C", 4]]);
});

test("config rejects invalid intervals, K and weight budgets", () => {
  const config = validateConfig({
    baseline_mode: "pre_onset",
    baseline_window: [-1.3, -0.3],
    feature_sigma_seconds: 0.06,
    first_k: 10,
    second_k: 4,
  });
  assert.equal(config.first_k, 10);
  assert.equal(config.baseline_edge_seconds, 0.3);
  assert.throws(() =>
    validateConfig({
      ...config,
      band_weights: { theta: 0, beta1: 1, beta2: 1, high_gamma: 1 },
    }),
  );
  assert.throws(() =>
    validateConfig({ ...config, baseline_window: [-0.3, -1.3] }),
  );
  assert.throws(() => validateConfig({ ...config, baseline_window: [-1, 0] }));
  assert.throws(() => validateConfig({ ...config, first_k: 2.5 }));
  assert.throws(() =>
    validateConfig({
      ...config,
      band_weights: { theta: 0, beta1: 0, beta2: 0, high_gamma: 0 },
    }),
  );
  assert.throws(() => reviewPayload({ second_k: 0 }));
  assert.deepEqual(
    reviewPayload({
      second_k: 4,
      bad_clusters: [2],
      manual_bad_channels: ["A"],
      distance_enabled: false,
    }),
    {
      second_k: 4,
      bad_clusters: [2],
      manual_bad_channels: ["A"],
      pca_distance_threshold: null,
      distance_reference: "global",
    },
  );
});

test("missing anatomy never blocks analysis, missing baseline blocks only file mode", () => {
  const resolved = {
    errors: [
      "Missing anatomy reference_mri; brain display unavailable",
      "Missing input baseline_fif (required in file baseline mode)",
    ],
  };
  assert.equal(inputErrors(resolved, "file").length, 1);
  assert.deepEqual(inputErrors(resolved, "pre_onset"), []);
  resolved.errors.push("Missing input wav_path");
  assert.equal(inputErrors(resolved, "pre_onset").length, 1);
});

test("API sends identity JSON without inventing revisions, propagates errors", async () => {
  const calls = [];
  const api = new WorkspaceAPI(
    "http://127.0.0.1:1/s/token/",
    async (url, options) => {
      calls.push({ url: String(url), options });
      return new Response(JSON.stringify({ run_id: "r", revision: 9 }), {
        status: 200,
      });
    },
  );
  await api.action({ run_id: "r", revision: 7 }, "confirm_trials", {
    pairs: [],
  });
  assert.deepEqual(JSON.parse(calls[0].options.body), {
    run_id: "r",
    revision: 7,
    action: "confirm_trials",
    payload: { pairs: [] },
  });
  assert.equal(calls[0].options.credentials, "same-origin");
  await api.get("history", { subject: "A & B", block: "b" });
  assert.equal(new URL(calls[1].url).searchParams.get("subject"), "A & B");
  const failing = new WorkspaceAPI(
    "http://127.0.0.1:1/s/t/",
    async () => new Response('{"error":"Stale revision"}', { status: 400 }),
  );
  await assert.rejects(failing.get("state"), /Stale revision/);
});

test("display URLs cannot escape the session or contact another origin", () => {
  const base = "http://127.0.0.1:1/s/t/";
  assert.equal(safeLocalURL("explorer/a/", base), base + "explorer/a/");
  for (const value of [
    "https://example.com/a",
    "../other",
    "/s/other/",
    "javascript:alert(1)",
    "explorer/%2e%2e/%2e%2e/a",
  ])
    assert.throws(() => safeLocalURL(value, base));
});

test("QC normalization preserves unflagged cells and PCA groups actual points", () => {
  const rows = normalizeQC({
    qc_rows: [
      { channel: "A", original_trial_id: 7, candidate: false },
      { channel: "A", trial_id: 9, flag_spike: true },
    ],
  });
  assert.equal(rows[0].trial_id, 7);
  assert.equal(rows[0].candidate, false);
  assert.equal(rows[1].candidate, true);
  const traces = pcaTraces(
    [
      { channel: "A", pc1: 1, pc2: 2, cluster: 0 },
      { channel: "B", pc1: 3, pc2: 4, cluster: 1 },
    ],
    ["A"],
  );
  assert.equal(
    traces.reduce((n, t) => n + t.x.length, 0),
    2,
  );
  assert.equal(
    traces.find((t) => t.customdata.includes("A")).marker.symbol,
    "x",
  );
  const actual = pcaTraces([{ channel: "A", pc1: 1, pc2: 2, first_label: 2 }]);
  assert.equal(actual[0].name, "类 2");
});

test("review coordinates and classes stay bound to accepted FIRST model", () => {
  const state = {
    first_round_id: "first",
    selected_round: "second",
    data: { pca: [{ channel: "A", pc1: 99, cluster: 8 }] },
    rounds: [
      {
        id: "first",
        data: {
          pca: [{ channel: "A", pc1: 1, cluster: 0 }],
          cluster_sizes: { 0: 1 },
        },
      },
      { id: "second", data: { pca: [{ channel: "A", pc1: 99, cluster: 8 }] } },
    ],
  };
  assert.equal(reviewData(state).pca[0].pc1, 1);
  assert.equal(reviewData(state).pca[0].cluster, 0);
  state.selected_round = "superseded";
  assert.equal(reviewData(state).pca[0].pc1, 1);
});

if (process.argv.includes("--browser"))
  test(
    "desktop and mobile browser workflow",
    { timeout: 180000 },
    browserChecks,
  );

async function browserChecks() {
  const { createServer } = await import("node:http");
  const { readFile, mkdir, writeFile } = await import("node:fs/promises");
  const { fileURLToPath, pathToFileURL } = await import("node:url");
  const path = await import("node:path");
  const modulePath =
    process.env.PLAYWRIGHT_MODULE ||
    path.join(
      process.env.USERPROFILE || process.env.HOME,
      ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs",
    );
  const { chromium } = await import(pathToFileURL(modulePath));
  const assets = path.resolve(
    fileURLToPath(
      new URL("../3-bands/fourband/browser_assets/", import.meta.url),
    ),
  );
  const output =
    process.env.FRONTEND_ARTIFACTS ||
    fileURLToPath(
      new URL(
        "../3-bands/outputs/verification/workspace_frontend/",
        import.meta.url,
      ),
    );
  await mkdir(output, { recursive: true });
  let state = {
    run_id: null,
    revision: 0,
    stage: "idle",
    config: {},
    task: null,
    data: {},
    rounds: [],
  };
  const requests = [],
    timers = new Set();
  let serial = 0;
  const clone = (v) => JSON.parse(JSON.stringify(v));
  const qcRows = ["A1", "A2", "A3", "B1", "B2", "C1"].flatMap((channel, c) =>
    [0, 2, 5, 7, 9, 12].map((trial_id, t) => ({
      channel,
      original_trial_id: trial_id,
      candidate: (c + t) % 5 === 0,
      flag_spike: (c + t) % 5 === 0,
      raw_bad: false,
    })),
  );
  const pca = Array.from({ length: 18 }, (_, i) => ({
    channel: `A${i + 1}`,
    first_label: i % 3,
    pc1: Math.sin(i * 1.7) * ((i % 3) + 1),
    pc2: Math.cos(i * 1.2) * ((i % 3) + 1),
  }));
  function later(fn) {
    const timer = setTimeout(() => {
      timers.delete(timer);
      fn();
    }, 100);
    timers.add(timer);
  }
  function finish(action, payload) {
    state.task.status = "completed";
    state.revision++;
    if (action === "extract" || action === "detect_qc") {
      state.stage = "trial_review";
      state.data = {
        qc_rows: qcRows,
        channels: ["A1", "A2", "A3", "B1", "B2", "C1"],
        events: [{ original_trial_id: 0 }],
      };
      state.first_round_id = state.second_round_id = null;
    }
    if (action === "confirm_trials") {
      state.stage = "trial_confirmed";
      state.data.trial_exclusions = payload.pairs;
      state.first_round_id = state.second_round_id = null;
    }
    if (action === "waveform")
      state.data.waveform = {
        ...payload,
        time: Array.from({ length: 500 }, (_, i) => i / 100),
        voltage: Array.from(
          { length: 500 },
          (_, i) => Math.sin(i / 7) * 0.00003,
        ),
        units: "volts",
        onset: 1,
        offset: 2,
      };
    if (action === "fit_first" || action === "fit_second") {
      const round = action === "fit_first" ? 1 : 2,
        id = `round-${++serial}`;
      state.stage = round === 1 ? "first_ready" : "second_ready";
      state.data.pca =
        round === 1
          ? pca
          : pca.map((r) => ({
              ...r,
              pc1: r.pc1 + 100,
              first_label: 8 + r.first_label,
            }));
      state.data.cluster_sizes =
        round === 1 ? { 0: 6, 1: 6, 2: 6 } : { 8: 6, 9: 6, 10: 6 };
      state.rounds.push({
        id,
        round,
        label: `第 ${round} 轮 ${id}`,
        data: clone(state.data),
      });
      state.selected_round = id;
      state[round === 1 ? "first_round_id" : "second_round_id"] = id;
      state.explorer_url = `explorer/${id}/`;
      state.artifacts = [
        { id: "figure", label: "PCA.png", content_type: "image/png" },
      ];
    }
    if (action === "preview_review")
      state.data.preview = {
        remaining: 16,
        excluded: 2,
        excluded_channels: ["A1", "A2"],
        valid_k: payload.second_k <= 16,
      };
    if (action === "stability")
      state.data.stability = {
        summary: { n_resamples: 5, mean_ari: 0.9 },
        round: payload.round,
      };
  }
  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url, "http://127.0.0.1"),
        route = decodeURIComponent(url.pathname.replace(/^\/s\/fixture\//, ""));
      if (route.startsWith("api/workspace/")) {
        const action = route.slice(14);
        let body;
        if (req.method === "POST") {
          let raw = "";
          for await (const chunk of req) raw += chunk;
          body = JSON.parse(raw);
          requests.push({ action, body });
        }
        res.setHeader("Content-Type", "application/json");
        let result;
        if (action === "catalog")
          result = {
            patients: [
              { id: "HS0015", blocks: ["B03", "B04"] },
              { id: "HS0020", blocks: ["B01"] },
            ],
          };
        else if (action === "inputs")
          result = {
            availability: [
              { name: "fif_path", exists: true, path: "local/ecog.fif" },
              {
                name: "baseline_fif",
                exists: true,
                path: "local/baseline.fif",
              },
              { name: "reference_mri", exists: false },
            ],
            errors: [
              "Missing anatomy reference_mri; brain display unavailable",
            ],
          };
        else if (action === "history")
          result = {
            runs: [
              { id: "saved", label: "已保存分析", stage: "second_ready" },
              { id: "legacy", label: "旧版结果", read_only: true },
            ],
          };
        else if (action === "state") result = state;
        else if (action === "start") {
          state = {
            run_id: `run-${++serial}`,
            revision: 1,
            subject: body.subject,
            block: body.block,
            config: body.config,
            stage: "configured",
            data: {},
            rounds: [],
            task: {
              id: `task-${serial}`,
              action: "extract",
              status: "running",
              elapsed_seconds: 0.1,
              stage: "Extracting",
            },
          };
          later(() => finish("extract", {}));
          result = state;
        } else if (action === "action") {
          assert.equal(body.run_id, state.run_id);
          assert.equal(body.revision, state.revision);
          assert.equal(isBusy(state), false);
          state.revision++;
          state.task = {
            id: `task-${++serial}`,
            action: body.action,
            status: "running",
            stage: body.action,
            elapsed_seconds: 0.1,
          };
          later(() => finish(body.action, body.payload));
          result = state;
        } else if (action === "cancel") {
          assert.equal(body.revision, state.revision);
          for (const timer of timers) clearTimeout(timer);
          timers.clear();
          state.task.status = "cancelled";
          state.revision++;
          result = state;
        } else if (action === "select_round") {
          state.selected_round = body.round_id;
          state.explorer_url = `explorer/${body.round_id}/`;
          state.revision++;
          result = state;
        } else if (action === "open") {
          state.run_id = body.run_id;
          state.subject = body.subject;
          state.block = body.block;
          state.read_only = body.run_id === "legacy";
          state.stage = state.read_only ? "historical" : "second_ready";
          state.revision++;
          state.task = {
            id: "opening",
            action: "load_display",
            status: "opening",
            stage: "Loading saved model and anatomy",
          };
          state.explorer_url = null;
          later(() => {
            state.task.status = "completed";
            state.explorer_url = "explorer/saved/";
            state.revision++;
          });
          result = state;
        } else if (action === "close") result = { state: "closed" };
        else if (action === "artifact/figure") {
          res.setHeader("Content-Type", "image/png");
          res.end(
            Buffer.from(
              "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==",
              "base64",
            ),
          );
          return;
        } else throw new Error(`Unknown fixture API ${route}`);
        res.end(JSON.stringify(result));
        return;
      }
      if (route.startsWith("explorer/")) {
        res.setHeader("Content-Type", "text/html");
        res.end(
          '<!doctype html><html lang="zh"><body style="margin:20px;font:14px sans-serif;background:#f0f5f5"><h2>结果 iframe 测试</h2><p>HS0015 / B03</p></body></html>',
        );
        return;
      }
      const filename = path.resolve(assets, route || "workspace.html");
      if (
        !filename.startsWith(assets + path.sep) &&
        filename !== path.join(assets, "workspace.html")
      )
        throw new Error("Path escape");
      const ext = path.extname(filename);
      res.setHeader(
        "Content-Type",
        {
          ".html": "text/html; charset=utf-8",
          ".css": "text/css",
          ".js": "text/javascript",
          ".mjs": "text/javascript",
        }[ext] || "application/octet-stream",
      );
      res.end(await readFile(filename));
    } catch (e) {
      res.statusCode = 500;
      res.end(JSON.stringify({ error: e.message }));
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${server.address().port}/s/fixture/`;
  let browser;
  const errors = [],
    external = [],
    dialogs = [];
  try {
    browser = await chromium.launch({
      headless: true,
      executablePath:
        process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ||
        "C:/Program Files/Google/Chrome/Application/chrome.exe",
    });
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
    });
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("request", (req) => {
      if (
        new URL(req.url()).origin !== new URL(base).origin &&
        !req.url().startsWith("data:")
      )
        external.push(req.url());
    });
    page.on("dialog", async (dialog) => {
      dialogs.push(dialog.message());
      await dialog.accept();
    });
    const enabled = (id) =>
      page.waitForFunction((id) => !document.getElementById(id).disabled, id);
    const click = async (id) => {
      await enabled(id);
      await page.locator(`#${id}`).click();
    };
    const screenshot = async (name) => {
      await page.waitForFunction(() =>
        [...document.querySelectorAll(".plot")]
          .filter((el) => el.clientWidth && el.data)
          .every((el) => Math.abs(el._fullLayout.width - el.clientWidth) < 2),
      );
      await page.screenshot({
        path: path.join(output, name + ".png"),
        fullPage: true,
      });
    };
    const layout = async () =>
      assert.equal(
        await page.evaluate(
          () => document.documentElement.scrollWidth > innerWidth + 1,
        ),
        false,
        "No page overflow",
      );
    await page.goto(base);
    await enabled("start");
    await screenshot("desktop-settings");
    await layout();
    await page.locator("#first-k").fill("3");
    await click("start");
    await enabled("confirm-trials");
    assert.equal(requests.filter((r) => r.action === "start").length, 1);
    await page.reload();
    await enabled("confirm-trials");
    assert.equal(
      requests.filter((r) => r.action === "start").length,
      1,
      "Reload must reconnect without start",
    );
    await page.locator("#tab-trials").click();
    await page.locator(".trial-cell:not(.candidate)").first().click();
    assert.match(
      await page.locator("#selection-count").textContent(),
      /已选 1/,
    );
    await click("undo-trials");
    assert.match(
      await page.locator("#selection-count").textContent(),
      /已选 0/,
    );
    await click("confirm-trials");
    await enabled("fit-first");
    assert.deepEqual(
      requests.find((r) => r.body?.action === "confirm_trials").body.payload,
      { pairs: [] },
    );
    assert.ok(dialogs.some((v) => v.includes("选择为空")));
    await click("waveform");
    await enabled("waveform");
    await page.waitForFunction(
      () =>
        document.getElementById("waveform-plot").data?.[0]?.y?.length === 500,
    );
    await screenshot("desktop-qc-waveform");
    await page.setViewportSize({ width: 390, height: 844 });
    await screenshot("mobile-qc-waveform");
    await layout();
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.locator("#spike-z").fill("12");
    assert.equal(await page.locator("#fit-first").isDisabled(), true);
    await click("detect-qc");
    await enabled("confirm-trials");
    assert.equal(
      requests.find((r) => r.body?.action === "detect_qc").body.payload.settings
        .spike_z,
      12,
    );
    await page.locator(".trial-cell:not(.candidate)").first().click();
    await click("confirm-trials");
    await enabled("fit-first");
    await click("fit-first");
    await enabled("preview-review");
    await page.waitForFunction(() =>
      document.getElementById("pca-plot").data?.some((t) => t.x.length),
    );
    await page.locator('#tab-results').click();
    await page.locator('#tab-review').click();
    await page.waitForFunction(()=>{const el=document.getElementById('pca-plot');return Math.abs(el._fullLayout.width-el.clientWidth)<2;});
    assert.equal(await page.locator("#fit-second").isDisabled(), true);
    await page.locator("#second-k").fill("2");
    await page.locator("#bad-clusters").selectOption("1");
    await click("preview-review");
    await enabled("fit-second");
    assert.match(
      await page.locator("#preview-summary").textContent(),
      /剩余 16/,
    );
    await screenshot("desktop-pca");
    await page.setViewportSize({ width: 390, height: 844 });
    await screenshot("mobile-pca");
    await layout();
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.locator("#second-k").fill("3");
    assert.equal(
      await page.locator("#fit-second").isDisabled(),
      true,
      "Editing K invalidates preview",
    );
    await page.locator("#second-k").fill("2");
    await click("preview-review");
    await enabled("fit-second");
    await click("fit-second");
    await page.waitForFunction(() =>
      document
        .querySelector("#explorer")
        .getAttribute("src")
        ?.includes("round-"),
    );
    assert.equal(
      requests.find((r) => r.body?.action === "fit_second").body.payload
        .second_k,
      2,
    );
    await page.locator("#tab-review").click();
    assert.deepEqual(
      await page
        .locator("#bad-clusters option")
        .evaluateAll((options) => options.map((o) => o.value)),
      ["0", "1", "2"],
    );
    assert.ok(
      await page
        .locator("#pca-plot")
        .evaluate((el) =>
          el.data.every((trace) => trace.x.every((x) => x < 10)),
        ),
    );
    await page.locator("#tab-results").click();
    assert.equal(
      await page.locator("#artifact-panel").getAttribute("open"),
      null,
    );
    assert.equal(
      await page.locator("#artifacts-summary").textContent(),
      "导出文件 (1)",
    );
    await screenshot("desktop-results");
    await page.setViewportSize({ width: 390, height: 844 });
    await screenshot("mobile-results");
    await layout();
    await page.setViewportSize({ width: 1440, height: 1000 });
    await page.locator("#tab-trials").click();
    await click("fit-first");
    await enabled("preview-review");
    await page.locator("#tab-results").click();
    const oldRound = state.rounds[0].id;
    await page.locator("#round").selectOption(oldRound);
    await page.locator("#tab-review").click();
    await page.waitForFunction(() =>
      document.getElementById("round-status").textContent.includes("历史"),
    );
    assert.equal(
      await page.locator("#round option").count(),
      state.rounds.length,
    );
    assert.equal(await page.locator("#preview-review").isDisabled(), true);
    assert.equal(await page.locator("#stability").isDisabled(), true);
    await page.locator("#history").selectOption("legacy");
    await page.waitForFunction(
      () => document.getElementById("task-status").textContent === "已完成",
    );
    await page.locator("#tab-trials").click();
    assert.equal(await page.locator("#detect-qc").isDisabled(), true);
    assert.equal(await page.locator("#confirm-trials").isDisabled(), true);
    state.read_only = false;
    state.task = {
      id: "opening-busy",
      status: "opening",
      stage: "Loading saved model and anatomy",
    };
    state.revision++;
    await page.locator("#refresh").click();
    await enabled("cancel");
    await page.locator("#subject").selectOption("HS0020");
    await page.waitForFunction(
      () => document.getElementById("block").value === "B01",
    );
    assert.equal(requests.at(-1).action, "cancel");
    assert.equal(await page.locator("#explorer").getAttribute("src"), null);
    assert.deepEqual(errors, []);
    assert.deepEqual(external, []);
    await writeFile(
      path.join(output, "report.json"),
      JSON.stringify(
        {
          status: "passed",
          screenshots: 7,
          errors,
          external,
          requests,
          dialogs,
        },
        null,
        2,
      ),
    );
    console.log(`Browser fixture passed; screenshots: ${output}`);
  } finally {
    for (const timer of timers) clearTimeout(timer);
    if (browser) await browser.close();
    await new Promise((resolve) => server.close(resolve));
  }
}
