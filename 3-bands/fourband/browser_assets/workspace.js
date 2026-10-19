import {
  WorkspaceAPI,
  StateGate,
  TrialSelection,
  DEFAULT_CONFIG,
  validateConfig,
  reviewPayload,
  isBusy,
  canAction,
  currentRound,
  safeLocalURL,
  normalizeQC,
  pcaRows,
  pcaTraces,
  inputErrors,
  reviewData,
  stabilityText,
  configChanges,
  firstFitPayload,
} from "./workspace_core.mjs";
import { ArtifactPreview, isPreviewImage } from "./artifact_preview.js";

const $ = (id) => document.getElementById(id);
const api = new WorkspaceAPI(new URL(".", location.href).href),
  gate = new StateGate();
const artifactPreview = new ArtifactPreview(api.base);
let state = null,
  patients = [],
  selection = new TrialSelection(),
  rows = [],
  pending = false,
  closed = false;
let inputsReady = false,
  inputsData = null,
  extractionDirty = false,
  fitDirty = false,
  qcDirty = false,
  trialDirty = false;
let previewBinding = null,
  previewPending = null,
  plotKey = "",
  waveKey = "",
  gridKey = "",
  activePane = "settings";
let replaceRunId = null,
  analysisReady = false,
  contextLoading = false,
  chosenContext = null,
  resolveChoice = null;
let acceptedConfig = null,
  submittedFitConfig = null,
  lastIdentity = "",
  pollTimer = null;
const stages = {
  idle: "空闲",
  configured: "提取中",
  trial_review: "待确认试次",
  trial_confirmed: "试次已确认",
  first_ready: "第一轮就绪",
  second_ready: "第二轮就绪",
  historical: "历史结果",
};
const statuses = {
  running: "运行中",
  opening: "加载模型与解剖",
  cancelling: "取消中",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
  interrupted: "已中断",
};
const text = (id, value) => {
  $(id).textContent = value ?? "";
};
const option = (value, label) => new Option(label, String(value));
const selected = (id) => [...$(id).selectedOptions].map((o) => o.value);
function error(value) {
  $("error").hidden = !value;
  text("error", value?.message ?? value);
}
function context() {
  return { subject: $("subject").value, block: $("block").value };
}
function matches(value) {
  const c = context();
  return value.subject === c.subject && value.block === c.block;
}
function pane(name) {
  activePane = name;
  document
    .querySelectorAll("[data-pane]")
    .forEach((b) =>
      b.setAttribute("aria-selected", String(b.dataset.pane === name)),
    );
  for (const key of ["settings", "trials", "review", "results"])
    $(`pane-${key}`).hidden = key !== name;
  requestAnimationFrame(() => resizePlots());
}
function resizePlots() {
  if (!window.Plotly) return;
  for (const id of ["pca-plot", "waveform-plot"]) {
    const el = $(id);
    if (el.data && el._fullLayout && !el.closest("[hidden]") && el.clientWidth && el.clientHeight &&
      (Math.abs(el._fullLayout.width-el.clientWidth)>1 || Math.abs(el._fullLayout.height-el.clientHeight)>1))
      window.Plotly.relayout(el,{width:el.clientWidth,height:el.clientHeight,autosize:true}).catch(error);
  }
}
function config() {
  return validateConfig({
    baseline_mode: $("baseline-mode").value,
    baseline_window: [$("baseline-start").value, $("baseline-end").value],
    baseline_edge_seconds: $("baseline-edge").value,
    feature_sigma_seconds: $("feature-sigma").value,
    first_k: $("first-k").value,
    second_k: $("second-k").value,
    band_weights: Object.fromEntries(
      ["theta", "beta1", "beta2", "high_gamma"].map((k) => [
        k,
        $(`weight-${k}`).value,
      ]),
    ),
    weight_preset: $("weight-preset").value,
    group_weights: Object.fromEntries(
      ["morphology", "amplitude", "slope"].map((k) => [
        k,
        $(`group-${k}`).value,
      ]),
    ),
  });
}
function hydrate(settings = {}) {
  const c = { ...DEFAULT_CONFIG, ...settings };
  $("baseline-mode").value = c.baseline_mode;
  $("baseline-start").value = c.baseline_window[0];
  $("baseline-end").value = c.baseline_window[1];
  $("baseline-edge").value = c.baseline_edge_seconds;
  $("feature-sigma").value =
    c.feature_sigma_seconds ?? settings.smoothing_sigma_seconds ?? 0.06;
  $("first-k").value = c.first_k;
  $("second-k").value = c.second_k;
  $("weight-preset").value = c.weight_preset;
  for (const k of ["theta", "beta1", "beta2", "high_gamma"])
    $(`weight-${k}`).value = c.band_weights?.[k] ?? 1;
  for (const k of ["morphology", "amplitude", "slope"])
    $(`group-${k}`).value = c.group_weights?.[k] ?? 1;
  acceptedConfig = config();
  extractionDirty = fitDirty = qcDirty = trialDirty = false;
  settingsUI();
}
function settingsUI() {
  const pre = $("baseline-mode").value === "pre_onset";
  $("baseline-start").disabled = !pre;
  $("baseline-end").disabled = !pre;
  $("baseline-warning").hidden = !pre;
  $("group-weights").hidden = $("weight-preset").value !== "custom";
  try {
    text("effective-config", JSON.stringify(config(), null, 2));
  } catch (e) {
    text("effective-config", e.message);
  }
  renderInputs();
}
function review() {
  return reviewPayload({
    bad_clusters: selected("bad-clusters").map((v) =>
      Number.isFinite(Number(v)) ? Number(v) : v,
    ),
    manual_bad_channels: selected("bad-channels"),
    distance_enabled: $("distance-enabled").checked,
    pca_distance_threshold: $("distance-threshold").value,
    distance_reference: $("distance-reference").value,
    second_k: $("second-k").value,
  });
}
function reviewKey() {
  return JSON.stringify({
    run: state?.run_id,
    first: state?.first_round_id,
    payload: review(),
  });
}
function resetReview() {
  previewBinding = previewPending = null;
  $("distance-enabled").checked = false;
  $("distance-threshold").disabled = true;
  for (const id of ["bad-clusters", "bad-channels"])
    for (const o of $(id).options) o.selected = false;
  text("preview-summary", "尚未预览");
  plotKey = "";
}
function clearDisplays({keepReview = false} = {}) {
  artifactPreview.setArtifacts([], "");
  for (const id of keepReview ? ["waveform-plot"] : ["pca-plot", "waveform-plot"]) {
    if (window.Plotly) window.Plotly.purge($(id));
    $(id).replaceChildren();
  }
  $("explorer").removeAttribute("src");
  $("explorer").hidden = true;
  $("artifacts").replaceChildren();
  delete $("artifacts").dataset.key;
  $("artifact-panel").hidden = true;
  $("round").replaceChildren();
  delete $("round").dataset.signature;
  $("trial-grid").replaceChildren();
  $("cluster-sizes").replaceChildren();
  text("result-identity", "");
  text("stability-summary", "");
  text("waveform-plot", "");
  text("round-status", "");
  text("trial-count", "");
  $("result-empty").hidden = false;
  text("result-empty", "尚无已接受结果");
  plotKey = waveKey = gridKey = "";
}
function resetContext() {
  gate.detach();
  state = null;
  lastIdentity = "";
  selection = new TrialSelection();
  rows = [];
  acceptedConfig = null;
  clearDisplays();
  resetReview();
  extractionDirty = fitDirty = qcDirty = trialDirty = false;
  submittedFitConfig=null;
  inputsReady = false;
  inputsData = null;
  replaceRunId = null;
  analysisReady = false;
  error("");
  $("baseline-segments").value = "";
  render();
}
async function permitSwitch() {
  if (contextLoading) return false;
  if (!isBusy(state) && !pending) return true;
  if (pending) return false;
  if (!window.confirm("当前任务仍在运行。取消任务并切换？")) return false;
  await mutate("cancel", api.identity(state));
  return true;
}
function chooseAnalysis(analysis) {
  text("analysis-summary", `${analysis.subject} / ${analysis.block}\n${stages[analysis.stage] ?? analysis.stage} · ${analysis.id}\n${analysis.created_at ?? ""}`);
  return new Promise((resolve) => {
    resolveChoice = resolve;
    $("analysis-dialog").showModal();
  });
}
function finishChoice(choice) {
  $("analysis-dialog").close();
  resolveChoice?.(choice);
  resolveChoice = null;
}
function restoreContext(previous) {
  if (!previous) return;
  $("subject").value = previous.subject;
  fillBlocks(previous.block);
}
async function loadContext({ choose = false, previous = null } = {}) {
  const token = gate.generation,
    c = context(),
    focusTarget = choose ? document.activeElement : null;
  let switched = false;
  contextLoading = true;
  render();
  try {
    const [input, current] = await Promise.allSettled([
      api.get("inputs", c),
      api.get("current", c),
    ]);
    if (token !== gate.generation) return;
    if (current.status !== "fulfilled") throw current.reason;
    const analysis = current.value.analysis;
    const choice = choose ? (analysis ? await chooseAnalysis(analysis) : "new") : null;
    if (choose && choice === "cancel") {
      restoreContext(previous);
      return;
    }
    if (choose) {
      resetContext();
      switched = true;
      hydrate(DEFAULT_CONFIG);
      pane("settings");
      replaceRunId = choice === "new" ? (analysis?.id ?? null) : null;
      analysisReady = true;
    }
    chosenContext = c;
    if (input.status === "fulfilled") {
      inputsData = input.value;
      inputsReady = true;
      renderInputs();
      const options = input.value.selection_options?.baseline_segments_csv ?? [],
        keep = $("baseline-segments").value;
      $("baseline-segments").replaceChildren(
        option("", "不使用片段表"),
        ...options.map((v) => option(v.id, v.label ?? v.id)),
      );
      if (options.some((v) => v.id === keep)) $("baseline-segments").value = keep;
      $("baseline-segments-label").hidden = !options.length;
    } else {
      inputsReady = false;
      inputsData = null;
      text("inputs", input.reason.message);
      error(input.reason);
    }
    if (choice === "continue") {
      const next = await mutate("open", { ...c, run_id: analysis.run_id ?? analysis.id });
      if (next) {
        apply(next, gate.generation, { hydrateConfig: true });
        pane(["first_ready", "second_ready"].includes(next.stage) ? "results" : "trials");
      }
    }
  } catch (e) {
    if (switched) analysisReady = false;
    else restoreContext(previous);
    error(e);
  } finally {
    contextLoading = false;
    render();
    if (focusTarget && focusTarget !== document.body && !focusTarget.disabled)
      focusTarget.focus();
  }
}
function renderInputs() {
  if (!inputsData) return;
  const root = $("inputs");
  root.replaceChildren();
  const availability =
    inputsData.availability ??
    inputsData.files ??
    inputsData.inputs ??
    inputsData.paths ??
    {};
  const table = document.createElement("table");
  const entries = Array.isArray(availability)
    ? availability.map((v, i) => [v.name ?? v.key ?? i, v])
    : Object.entries(availability);
  for (const [key, item] of entries) {
    const tr = document.createElement("tr");
    const available =
      typeof item === "boolean"
        ? item
        : typeof item === "string"
          ? true
          : Boolean(item?.available ?? item?.exists ?? item?.path);
    for (const value of [
      key,
      available ? "可用" : "缺失",
      typeof item === "string" ? item : (item?.path ?? item?.error ?? ""),
    ]) {
      const td = document.createElement("td");
      td.textContent = String(value);
      tr.append(td);
    }
    tr.children[1].className = available ? "input-ok" : "input-missing";
    tr.children[2].className = "input-path";
    table.append(tr);
  }
  root.append(table);
  const problems = inputsData.errors ?? [];
  for (const item of Array.isArray(problems)
    ? problems
    : Object.values(problems)) {
    const el = document.createElement("div");
    el.className = "input-missing";
    el.textContent = typeof item === "string" ? item : JSON.stringify(item);
    root.append(el);
  }
  for (const warning of inputsData.warnings ?? []) {
    const el = document.createElement("div");
    el.className = "warning";
    el.textContent = String(warning);
    root.append(el);
  }
}
function validInputs() {
  return (
    inputsReady &&
    inputErrors(inputsData, $("baseline-mode").value).length === 0
  );
}
function fillBlocks(value) {
  const patient = patients.find((p) => p.id === $("subject").value);
  $("block").replaceChildren(
    ...(patient?.blocks || []).map((b) =>
      option(
        typeof b === "string" ? b : b.id,
        typeof b === "string" ? b : b.id,
      ),
    ),
  );
  if (value && [...$("block").options].some((o) => o.value === value))
    $("block").value = value;
}

function apply(next, token, { hydrateConfig = false, attach=false } = {}) {
  if (!matches(next) || !gate.accept(next, token, {attach})) return false;
  const previous = state;
  state = next;
  if (lastIdentity !== state.run_id) {
    lastIdentity = state.run_id;
    selection = new TrialSelection();
    clearDisplays();
    resetReview();
    hydrateConfig = true;
  }
  if (hydrateConfig) {
    hydrate(state.config);
    acceptedConfig = config();
    selection = new TrialSelection();
    const pairs = state.data?.trial_exclusions ?? [];
    for (const pair of pairs)
      if (Array.isArray(pair) && pair.length === 2)
        selection.toggle(pair[0], pair[1]);
    selection.history = [];
    const settings = state.data?.qc_settings ?? {};
    for (const [id, key] of [
      ["spike-z", "spike_z"],
      ["jump-z", "jump_z"],
      ["flat-run", "flat_run_seconds"],
    ])
      if (settings[key] != null) $(id).value = settings[key];
  }
  const completed =
    (state.task?.status === "completed" &&
      state.task?.id !== previous?.task?.id) ||
    (state.task?.status === "completed" &&
      previous?.task?.status !== state.task.status);
  if (completed) {
    const action = state.task.action;
    if (action === "detect_qc") {
      qcDirty = false;
      trialDirty = true;
      resetReview();
    }
    if (action === "confirm_trials") {
      trialDirty = false;
      resetReview();
    }
    if (action === "preview_review" && previewPending) {
      try {
        if (previewPending === reviewKey()) previewBinding = previewPending;
      } catch {}
      previewPending = null;
    }
    if (action === "fit_first") resetReview();
  }
  if(state.first_round_id&&state.first_round_id!==previous?.first_round_id&&submittedFitConfig){
    const payload=firstFitPayload(submittedFitConfig);
    acceptedConfig={...acceptedConfig,first_k:payload.first_k,...payload.fit_settings};
    fitDirty=false;submittedFitConfig=null;resetReview();
  }
  if(['failed','cancelled'].includes(state.task?.status))submittedFitConfig=null;
  rows = normalizeQC(state.data);
  render();
  return true;
}
async function poll() {
  if (closed) return;
  const token = gate.generation;
  try {
    if (pending || contextLoading || gate.detached) return;
    const next = await api.get("state");
    if (token === gate.generation && !contextLoading) {
      text("connection", "已连接");
      apply(next, token);
    }
  } catch (e) {
    if (token === gate.generation) {
      text("connection", "连接中断");
      error(e);
    }
  } finally {
    if (!closed) pollTimer = setTimeout(poll, isBusy(state) ? 750 : 2200);
  }
}
async function mutate(route, body) {
  if (pending || closed) return null;
  pending = true;
  const token = gate.reset();
  gate.state = state;
  error("");
  render();
  try {
    const next = await api.post(route, body);
    if (token === gate.generation && next && Number.isInteger(next.revision))
      apply(next, token, {attach:['start','open'].includes(route)});
    return next;
  } catch (e) {
    if (token === gate.generation) error(e);
    throw e;
  } finally {
    pending = false;
    render();
  }
}
async function action(name, payload = {}) {
  if (!canAction(state, name, { extractionDirty, fitDirty, qcDirty, trialDirty }) || pending)
    throw new Error("当前状态不可执行该操作");
  return mutate("action", { ...api.identity(state), action: name, payload });
}
function handle(fn) {
  return async (event) => {
    try {
      await fn(event);
    } catch (e) {
      error(e);
    }
  };
}

function render() {
  const busy = isBusy(state) || pending || contextLoading;
  const dirty = extractionDirty || fitDirty || qcDirty || trialDirty;
  const historical = state && !currentRound(state);
  text(
    "stage",
    extractionDirty ? "设置已修改 · 需重新提取" : fitDirty ? '分类设置已修改 · 需重拟合第一轮' : (stages[state?.stage] ?? "空闲"),
  );
  text("task-status", statuses[state?.task?.status] ?? "");
  text("task-stage", state?.task?.stage ?? "");
  text(
    "elapsed",
    state?.task
      ? `${Number(state.task.elapsed_seconds || 0).toFixed(1)} s`
      : "",
  );
  text("run-id", state?.run_id ?? "");
  text("log-tail", state?.task?.log_tail ?? "");
  text("retention-warning", state?.retention_warning ?? "");
  $("retention-warning").hidden = !state?.retention_warning;
  if (state?.task?.error) error(state.task.error);
  $("cancel").disabled = !isBusy(state) || pending;
  $("start").disabled = busy || Boolean(state?.run_id) || !analysisReady || !context().block || !validInputs();
  $("close").disabled = pending || contextLoading;
  $("refresh").disabled = pending || contextLoading;
  for (const id of ["subject", "block", "current-analysis"]) $(id).disabled = pending || contextLoading;
  $("current-analysis").disabled ||= !context().block;
  text("current-analysis-label", state?.run_id ? "当前分析 · " + (stages[state.stage] ?? state.stage) :
    analysisReady ? (replaceRunId ? "新分析 · 待替换" : "新分析") : "选择当前分析");
  for (const button of document.querySelectorAll("[data-action]")) {
    let enabled =
      canAction(state, button.dataset.action, {
        extractionDirty,
        fitDirty,
        qcDirty,
        trialDirty,
      }) && !pending && !contextLoading;
    if (
      ["confirm_trials", "fit_first"].includes(button.dataset.action) &&
      qcDirty
    )
      enabled = false;
    if (button.dataset.action === "fit_first" && trialDirty) enabled = false;
    if (button.dataset.action === "fit_second") {
      try {
        enabled &&=
          previewBinding === reviewKey() &&
          state?.data?.preview?.valid_k !== false;
      } catch {
        enabled = false;
      }
    }
    button.disabled = !enabled;
  }
  for (const id of ["select-candidates", "clear-trials", "undo-trials"])
    $(id).disabled = busy || !rows.length || Boolean(state?.read_only);
  $("undo-trials").disabled ||= selection.history.length === 0;
  text(
    "selection-count",
    `已选 ${selection.pairs().length}${trialDirty ? " · 待确认" : ""}`,
  );
  text(
    "trial-count",
    rows.length ? ` · ${new Set(rows.map((r) => r.trial_id)).size}` : "",
  );
  text(
    "round-status",
    state?.read_only
      ? "历史只读"
      : historical
        ? "历史拟合 · 仅查看"
        : state?.first_round_id
          ? "审查基准 · 当前第一轮"
          : "尚无已接受拟合",
  );
  for (const el of $("config-form").querySelectorAll("input,select"))
    el.disabled = busy;
  $("baseline-start").disabled ||= $("baseline-mode").value !== "pre_onset";
  $("baseline-end").disabled ||= $("baseline-mode").value !== "pre_onset";
  for (const id of ["spike-z", "jump-z", "flat-run"])
    $(id).disabled = busy || Boolean(state?.read_only);
  const reviewDisabled =
    busy || Boolean(state?.read_only) || historical || !state?.first_round_id;
  for (const id of [
    "distance-enabled",
    "distance-reference",
    "distance-threshold",
    "bad-clusters",
    "bad-channels",
    "second-k",
    "reset-review",
    "stability-round",
  ])
    $(id).disabled = reviewDisabled;
  $("distance-threshold").disabled ||= !$("distance-enabled").checked;
  $("round").disabled = busy;
  const key = JSON.stringify([
    rows,
    selection.pairs(),
    $("candidate-filter").value,
    $("channel-filter").value,
    busy,
    state?.read_only,
  ]);
  if (key !== gridKey) {
    gridKey = key;
    renderGrid(busy);
  }
  renderReview();
  renderWaveform();
  renderResults();
  if (dirty && activePane === "results")
    text(
      "result-identity",
      `${state?.subject ?? ""} / ${state?.block ?? ""} · ${state?.run_id ?? ""} · 已保存结果`,
    );
}
function renderGrid(busy) {
  const root = $("trial-grid");
  root.replaceChildren();
  if (!rows.length) {
    text("trial-grid", "尚无试次数据");
    return;
  }
  const filter = $("candidate-filter").value,
    search = $("channel-filter").value.toLowerCase();
  const visible = rows.filter(
    (r) =>
      r.channel.toLowerCase().includes(search) &&
      (filter === "all" ||
        (filter === "candidate" && r.candidate) ||
        (filter === "selected" && selection.has(r.channel, r.trial_id)) ||
        Boolean(r[`flag_${filter}`])),
  );
  const channels = [...new Set(visible.map((r) => r.channel))],
    trials = [...new Set(visible.map((r) => r.trial_id))].sort((a, b) => a - b),
    lookup = new Map(
      visible.map((r) => [JSON.stringify([r.channel, r.trial_id]), r]),
    );
  const table = document.createElement("table");
  table.className = "trial-table";
  const head = document.createElement("thead"),
    header = document.createElement("tr");
  for (const name of ["通道 / 试次", ...trials]) {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = name;
    header.append(th);
  }
  head.append(header);
  table.append(head);
  const body = document.createElement("tbody");
  for (const channel of channels) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.scope = "row";
    th.textContent = channel;
    tr.append(th);
    for (const trial of trials) {
      const td = document.createElement("td");
      const row = lookup.get(JSON.stringify([channel, trial]));
      if (row) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `trial-cell${row.candidate ? " candidate" : ""}${row.raw_bad ? " raw-bad" : ""}`;
        button.dataset.channel = channel;
        button.dataset.trial = trial;
        button.setAttribute(
          "aria-pressed",
          String(selection.has(channel, trial)),
        );
        button.setAttribute(
          "aria-label",
          `${channel} 试次 ${trial}${row.candidate ? " 候选" : ""}`,
        );
        button.title = `${channel} · ${trial}${row.reasons ? " · " + row.reasons : ""}${row.raw_bad ? " · 原始坏通道" : ""}`;
        button.textContent = selection.has(channel, trial)
          ? "×"
          : row.candidate
            ? "!"
            : "·";
        button.disabled = busy || row.raw_bad || Boolean(state?.read_only);
        td.append(button);
      }
      tr.append(td);
    }
    body.append(tr);
  }
  table.append(body);
  root.append(table);
  const oldChannel = $("waveform-channel").value,
    oldTrial = $("waveform-trial").value;
  $("waveform-channel").replaceChildren(
    ...[...new Set(rows.map((r) => r.channel))].map((v) => option(v, v)),
  );
  $("waveform-trial").replaceChildren(
    ...[...new Set(rows.map((r) => r.trial_id))]
      .sort((a, b) => a - b)
      .map((v) => option(v, v)),
  );
  if ([...$("waveform-channel").options].some((o) => o.value === oldChannel))
    $("waveform-channel").value = oldChannel;
  if ([...$("waveform-trial").options].some((o) => o.value === oldTrial))
    $("waveform-trial").value = oldTrial;
}
const plotLayout = {
  paper_bgcolor: "#fff",
  plot_bgcolor: "#fff",
  font: {
    family: "Microsoft YaHei, Segoe UI, sans-serif",
    size: 12,
    color: "#34454d",
  },
  margin: { l: 55, r: 18, t: 24, b: 45 },
  legend: { orientation: "h", y: -0.18 },
  xaxis: {
    title: { text: "PC1" },
    gridcolor: "#e4ebee",
    zerolinecolor: "#cbd5d9",
  },
  yaxis: {
    title: { text: "PC2" },
    gridcolor: "#e4ebee",
    zerolinecolor: "#cbd5d9",
  },
};
function pcaLayout() {
  const axis = (title) => ({
    title: {text: title, font: {size: 12}},
    tickfont: {size: 10},
    nticks: 5,
    showbackground: true,
    backgroundcolor: "#f2f2f2",
    gridcolor: "#fff",
    zerolinecolor: "#b8b8b8",
    showline: true,
    linecolor: "#b8b8b8",
  });
  // Preview/round selection changes the markers, not the first-model viewpoint.
  const revision = JSON.stringify([state?.run_id, state?.first_round_id]);
  return {
    paper_bgcolor: plotLayout.paper_bgcolor,
    font: plotLayout.font,
    margin: {l: 18, r: 18, t: 16, b: 50},
    legend: {orientation: "h", y: 0, yanchor: "top", x: 0},
    uirevision: revision,
    scene: {
      domain: {x: [0, 1], y: [.14, 1]},
      uirevision: revision,
      xaxis: axis("PC1"), yaxis: axis("PC2"), zaxis: axis("PC3"),
      aspectmode: "cube",
      dragmode: "orbit",
      camera: {
        eye: {x: 1.25, y: -2.165, z: 1.443},
        center: {x: 0, y: 0, z: 0},
        up: {x: 0, y: 0, z: 1},
        projection: {type: "perspective"},
      },
    },
  };
}
function plot(id, traces, layout) {
  if (!window.Plotly) {
    text(id, "Plotly 未加载");
    return;
  }
  const el=$(id),sized=structuredClone(layout);
  sized.autosize=true;
  if(el.clientWidth&&el.clientHeight){sized.width=el.clientWidth;sized.height=el.clientHeight;}
  window.Plotly.react(el, traces, sized, {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["sendDataToCloud", "lasso2d", "select2d"],
  }).then(()=>requestAnimationFrame(resizePlots)).catch(error);
}
function renderReview() {
  const data = state?.data ?? {},
    source = reviewData(state),
    points = pcaRows(source);
  const key = JSON.stringify([
    state?.run_id,
    state?.first_round_id,
    state?.selected_round,
    points,
    data.preview,
    data.stability,
    previewBinding,
  ]);
  if (key === plotKey) return;
  plotKey = key;
  const clusters = [...new Set(points.map((r) => r.cluster ?? r.label ?? 0))]
    .filter((v) => Number(v) >= 0)
    .sort((a, b) => Number(a) - Number(b));
  const keepClusters = selected("bad-clusters"),
    keepChannels = selected("bad-channels");
  $("bad-clusters").replaceChildren(
    ...clusters.map((v) => option(v, `类 ${v}`)),
  );
  $("bad-channels").replaceChildren(
    ...[
      ...new Set(points.map((r) => String(r.channel ?? r.channel_name ?? ""))),
    ]
      .filter(Boolean)
      .map((v) => option(v, v)),
  );
  for (const o of $("bad-clusters").options)
    o.selected = keepClusters.includes(o.value);
  for (const o of $("bad-channels").options)
    o.selected = keepChannels.includes(o.value);
  const preview = data.preview ?? {};
  const excluded = previewBinding
    ? (preview.excluded_channels ??
      preview.bad_channels ??
      preview.excluded ??
      [])
    : [];
  plot(
    "pca-plot",
    pcaTraces(points, Array.isArray(excluded) ? excluded : []),
    pcaLayout(),
  );
  const sizes = source.cluster_sizes ?? {};
  const entries = Array.isArray(sizes)
    ? sizes.map((r) => [r.cluster ?? r.label, r.size ?? r.count ?? r.n])
    : Object.entries(sizes);
  $("cluster-sizes").replaceChildren();
  for (const [cluster, n] of entries) {
    const el = document.createElement("span");
    el.className = `cluster-size${Number(n) < 3 ? " small" : ""}`;
    el.textContent = `类 ${cluster} · ${n}${Number(n) < 3 ? " · 小类" : ""}`;
    $("cluster-sizes").append(el);
  }
  if (previewBinding) {
    const count =
      preview.remaining_count ?? preview.n_remaining ?? preview.remaining;
    const removed =
      preview.excluded_count ??
      preview.n_excluded ??
      (typeof preview.excluded === "number"
        ? preview.excluded
        : Array.isArray(excluded)
          ? excluded.length
          : "—");
    text(
      "preview-summary",
      `排除 ${removed} · 剩余 ${count ?? "—"} · K₂ ${$("second-k").value}${preview.valid_k === false ? " · K 无效" : ""}`,
    );
  } else text("preview-summary", "尚未预览");
  const stability = data.stability;
  text("stability-summary",stabilityText(stability));
}
function renderWaveform() {
  const waveform = state?.data?.waveform;
  if (!waveform) return;
  const key = JSON.stringify([state.run_id, waveform]);
  if (key === waveKey) return;
  waveKey = key;
  const times = waveform.times ?? waveform.time ?? waveform.t ?? [],
    values =
      waveform.voltage ??
      waveform.values ??
      waveform.signal ??
      waveform.raw ??
      waveform.ecog ??
      waveform.y ??
      [];
  const traces = [
    {
      type: "scatter",
      mode: "lines",
      name: waveform.channel ?? "原始波形",
      x: times,
      y: values,
      line: { color: "#146c75", width: 1 },
      connectgaps: false,
    },
  ];
  if (waveform.audio)
    traces.push({
      type: "scatter",
      mode: "lines",
      name: "音频",
      x: times,
      y: waveform.audio,
      yaxis: "y2",
      line: { color: "#b16a17", width: 1 },
    });
  plot("waveform-plot", traces, {
    ...plotLayout,
    xaxis: { title: { text: "时间 (s)" }, gridcolor: "#e4ebee" },
    yaxis: {
      title: { text: waveform.units ?? waveform.unit ?? "V" },
      gridcolor: "#e4ebee",
    },
    yaxis2: { overlaying: "y", side: "right", showgrid: false },
    title: {
      text: `${waveform.channel ?? ""} · 原始试次 ${waveform.trial_id ?? waveform.original_trial_id ?? ""}`,
      font: { size: 12 },
    },
    shapes: [waveform.onset, waveform.offset]
      .filter(Number.isFinite)
      .map((x) => ({
        type: "line",
        x0: x,
        x1: x,
        y0: 0,
        y1: 1,
        yref: "paper",
        line: { color: "#a36a18", dash: "dot", width: 1 },
      })),
    margin: { l: 55, r: 40, t: 30, b: 45 },
  });
}
function renderResults() {
  if (!state) return;
  const old = $("round").value,
    rounds = state.rounds ?? [];
  const signature = JSON.stringify(
    rounds.map((r) => [
      r.id,
      r.label,
      state.first_round_id === r.id || state.second_round_id === r.id,
    ]),
  );
  if ($("round").dataset.signature !== signature) {
    $("round").dataset.signature = signature;
    $("round").replaceChildren(
      ...rounds.map((r) =>
        option(
          r.id,
          `${[state.first_round_id, state.second_round_id].includes(r.id) ? "[当前]" : "[历史]"} ${r.label ?? `第 ${r.round} 轮`}`,
        ),
      ),
    );
  }
  $("round").value = state.selected_round ?? old;
  text(
    "result-identity",
    `${state.subject} / ${state.block} · ${state.run_id} · ${currentRound(state) ? "当前" : "历史"} ${rounds.find((r) => r.id === state.selected_round)?.round ?? ""}`,
  );
  const url = state.explorer_url;
  if (url) {
    try {
      const safe = safeLocalURL(url, api.base);
      if ($("explorer").getAttribute("src") !== safe) $("explorer").src = safe;
      $("explorer").hidden = false;
      $("result-empty").hidden = true;
    } catch (e) {
      error(e);
      $("explorer").hidden = true;
      $("explorer").removeAttribute("src");
    }
  } else {
    $("explorer").hidden = true;
    $("explorer").removeAttribute("src");
    $("result-empty").hidden = false;
    text(
      "result-empty",
      state.task?.status === "opening"
        ? "加载模型与解剖…"
        : state.rounds?.length
          ? "暂无脑图；已保存图像如下"
          : "尚无已接受结果",
    );
  }
  const artifacts = state.artifacts ?? [];
  artifactPreview.setArtifacts(artifacts, JSON.stringify([state.run_id, state.selected_round]));
  const key = JSON.stringify([state.run_id, state.selected_round, artifacts, url]);
  if ($("artifacts").dataset.key === key) return;
  $("artifacts").dataset.key = key;
  $("artifacts").replaceChildren();
  $("artifact-panel").hidden = !artifacts.length;
  $("artifact-panel").open = !url;
  text("artifacts-summary", `导出文件 (${artifacts.length})`);
  for (const artifact of artifacts) {
    try {
      const href = safeLocalURL(
          `api/workspace/artifact/${encodeURIComponent(artifact.id)}`,
          api.base,
        ),
        image = isPreviewImage(artifact),
        link = document.createElement(image ? "button" : "a");
      link.textContent = artifact.label ?? artifact.id;
      if (image) {
        link.type = "button";
        link.className = "artifact-trigger";
        link.setAttribute("aria-label", `预览 ${artifact.label ?? artifact.id}`);
        link.setAttribute("aria-haspopup", "dialog");
        link.title = "预览图像";
        link.onclick = () => artifactPreview.open(artifact.id, link);
      } else {
        link.href = href;
        link.download = artifact.label ?? "";
      }
      if (!url && image) {
        const figure = document.createElement("figure"),
          img = document.createElement("img");
        img.src = href;
        img.loading = "lazy";
        img.alt = artifact.label ?? "已保存图像";
        link.prepend(img);
        figure.append(link);
        $("artifacts").append(figure);
      } else $("artifacts").append(link);
    } catch (e) {
      error(e);
    }
  }
}

document
  .querySelectorAll("[data-pane]")
  .forEach((button) =>
    button.addEventListener("click", () => pane(button.dataset.pane)),
  );
document
  .querySelector("[role=tablist]")
  .addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const tabs = [...document.querySelectorAll("[data-pane]")],
      i = tabs.indexOf(document.activeElement),
      next =
        tabs[
          (i + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length
        ];
    next.focus();
    pane(next.dataset.pane);
  });
$("subject").addEventListener(
  "change",
  handle(async () => {
    const old = chosenContext;
    if (!(await permitSwitch())) {
      restoreContext(old);
      return;
    }
    fillBlocks();
    await loadContext({ choose: true, previous: old });
  }),
);
$("block").addEventListener(
  "change",
  handle(async () => {
    const old = chosenContext;
    if (!(await permitSwitch())) {
      restoreContext(old);
      return;
    }
    await loadContext({ choose: true, previous: old });
  }),
);
$("current-analysis").addEventListener(
  "click",
  handle(async () => {
    if (!(await permitSwitch())) return;
    await loadContext({ choose: true, previous: chosenContext });
  }),
);
for (const [id, choice] of [["analysis-continue", "continue"], ["analysis-new", "new"], ["analysis-cancel", "cancel"]])
  $(id).addEventListener("click", () => finishChoice(choice));
$("analysis-dialog").addEventListener("cancel", (event) => {
  event.preventDefault();
  finishChoice("cancel");
});
$("analysis-dialog").addEventListener("keydown", (event) => {
  if (event.key !== "Tab") return;
  const first = $("analysis-continue"), last = $("analysis-cancel");
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
});
$("round").addEventListener(
  "change",
  handle(async () => {
    const id = $("round").value;
    if (!(await permitSwitch())) {
      $("round").value = state?.selected_round ?? "";
      return;
    }
    gate.reset();
    clearDisplays({keepReview: true});
    previewBinding = null;
    await mutate("select_round", { ...api.identity(state), round_id: id });
  }),
);
$("config-form").addEventListener("submit", (e) => e.preventDefault());
$("config-form").addEventListener("input", (event) => {
  settingsUI();
  if(state?.run_id){
    try { ({extractionDirty,fitDirty}=configChanges(acceptedConfig,config())); }
    catch { if(['first-k','weight-preset'].includes(event.target.id)||event.target.id.startsWith('weight-')||event.target.id.startsWith('group-'))fitDirty=true;else extractionDirty=true; }
    if(event.target.id==='baseline-segments')extractionDirty=true;
  }
  previewBinding=previewPending=null;plotKey='';
  render();
});
for (const id of ["spike-z", "jump-z", "flat-run"])
  $(id).addEventListener("input", () => {
    qcDirty = true;
    previewBinding = null;
    render();
  });
for (const id of ["candidate-filter", "channel-filter"])
  $(id).addEventListener("input", () => render());
$("trial-grid").addEventListener("click", (event) => {
  const button = event.target.closest("[data-trial]");
  if (!button || button.disabled) return;
  selection.toggle(button.dataset.channel, Number(button.dataset.trial));
  trialDirty = true;
  previewBinding = null;
  $("waveform-channel").value = button.dataset.channel;
  $("waveform-trial").value = button.dataset.trial;
  render();
});
$("select-candidates").addEventListener("click", () => {
  selection.candidates(rows);
  trialDirty = true;
  render();
});
$("clear-trials").addEventListener("click", () => {
  selection.clear();
  trialDirty = true;
  render();
});
$("undo-trials").addEventListener("click", () => {
  selection.undo();
  trialDirty = true;
  render();
});
$("start").addEventListener(
  "click",
  handle(async () => {
    const effective = config();
    if (state?.run_id || !analysisReady || contextLoading) return;
    const segment = $("baseline-segments").value;
    const next = await mutate("start", {
      ...context(),
      config: effective,
      replace_run_id: replaceRunId,
      ...(segment ? { selections: { baseline_segments_csv: segment } } : {}),
    });
    if (next) {
      apply(next, gate.generation, { hydrateConfig: true });
      pane("trials");
    }
  }),
);
$("detect-qc").addEventListener(
  "click",
  handle(async () => {
    const settings = {
      spike_z: Number($("spike-z").value),
      jump_z: Number($("jump-z").value),
      flat_run_seconds: Number($("flat-run").value),
    };
    if (!Object.values(settings).every((v) => Number.isFinite(v) && v > 0))
      throw new Error("QC 阈值必须为正数");
    previewBinding = null;
    await action("detect_qc", { settings });
  }),
);
$("confirm-trials").addEventListener(
  "click",
  handle(async () => {
    const pairs = selection.pairs();
    if (
      !window.confirm(
        pairs.length
          ? `确认排除 ${pairs.length} 个通道/试次对？`
          : "当前选择为空。确认不排除任何试次？",
      )
    )
      return;
    await action("confirm_trials", { pairs });
  }),
);
$("fit-first").addEventListener(
  "click",
  handle(async () => {
    const effective=config(),payload=firstFitPayload(effective);
    previewBinding = null;
    submittedFitConfig=effective;
    try{await action("fit_first",payload);}catch(e){submittedFitConfig=null;throw e;}
    pane("review");
  }),
);
$("waveform").addEventListener(
  "click",
  handle(async () => {
    const channel = $("waveform-channel").value,
      trial_id = Number($("waveform-trial").value);
    if (!channel || !Number.isInteger(trial_id))
      throw new Error("请选择通道与试次");
    await action("waveform", { channel, trial_id });
  }),
);
for (const id of [
  "distance-enabled",
  "distance-threshold",
  "distance-reference",
  "bad-clusters",
  "bad-channels",
  "second-k",
])
  $(id).addEventListener("input", () => {
    previewBinding = null;
    plotKey = "";
    $("distance-threshold").disabled = !$("distance-enabled").checked;
    render();
  });
$("reset-review").addEventListener("click", () => {
  resetReview();
  render();
});
$("preview-review").addEventListener(
  "click",
  handle(async () => {
    const payload = review();
    previewPending = reviewKey();
    previewBinding = null;
    try {
      await action("preview_review", payload);
    } catch (e) {
      previewPending = null;
      throw e;
    }
  }),
);
$("fit-second").addEventListener(
  "click",
  handle(async () => {
    const payload = review();
    if (previewBinding !== reviewKey())
      throw new Error("请重新预览当前排除设置");
    const preview = state.data?.preview ?? {},
      remaining =
        preview.remaining_count ?? preview.n_remaining ?? preview.remaining;
    if (typeof remaining === "number" && payload.second_k > remaining)
      throw new Error("第二轮 K 超过剩余电极数");
    if (!window.confirm(`确认排除预览并拟合第二轮（K=${payload.second_k}）？`))
      return;
    previewBinding = null;
    await action("fit_second", payload);
    pane("results");
  }),
);
$("stability").addEventListener(
  "click",
  handle(() =>
    action("stability", {
      round: Number($("stability-round").value),
      n_resamples: 5,
    }),
  ),
);
$("cancel").addEventListener(
  "click",
  handle(async () => {
    if (!window.confirm("取消当前任务？")) return;
    previewPending = null;
    await mutate("cancel", api.identity(state));
  }),
);
$("refresh").addEventListener(
  "click",
  handle(async () => {
    const token = gate.generation;
    const next = await api.get("state");
    apply(next, token);
    await loadContext();
  }),
);
$("close").addEventListener(
  "click",
  handle(async () => {
    if (!window.confirm("关闭会话并停止当前任务？")) return;
    await mutate("close", {});
    closed = true;
    clearTimeout(pollTimer);
    clearDisplays();
    state = null;
    render();
    for (const button of document.querySelectorAll("button"))
      button.disabled = true;
    text("connection", "会话已关闭");
  }),
);
window.addEventListener("resize", resizePlots);
const plotObserver=new ResizeObserver(()=>requestAnimationFrame(resizePlots));
for(const id of ['pca-plot','waveform-plot'])plotObserver.observe($(id));
window.addEventListener("pagehide", () => {
  closed = true;
  clearTimeout(pollTimer);
});
async function boot() {
  try {
    const [catalog, initial] = await Promise.all([
      api.get("catalog"),
      api.get("state"),
    ]);
    patients = catalog.patients ?? [];
    $("subject").replaceChildren(...patients.map((p) => option(p.id, p.id)));
    if (initial.subject && patients.some((p) => p.id === initial.subject))
      $("subject").value = initial.subject;
    fillBlocks(initial.block);
    chosenContext = context();
    hydrate(initial.config);
    if (initial.run_id) {
      apply(initial, gate.generation, { hydrateConfig: true });
      pane(
        ["first_ready", "second_ready", "historical"].includes(initial.stage)
          ? "results"
          : "trials",
      );
    } else gate.detach();
    await loadContext({ choose: !initial.run_id });
    text("connection", "已连接");
  } catch (e) {
    error(e);
    text("connection", "连接失败");
  } finally {
    render();
    pollTimer = setTimeout(poll, 750);
  }
}
window.lucide?.createIcons();
boot();
