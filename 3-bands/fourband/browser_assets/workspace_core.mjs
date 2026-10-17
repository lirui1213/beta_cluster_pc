export const DEFAULT_CONFIG = Object.freeze({
  baseline_mode: "file",
  baseline_window: [-1.3, -0.3],
  baseline_edge_seconds: 0.3,
  feature_sigma_seconds: 0.06,
  first_k: 10,
  second_k: 4,
  band_weights: { theta: 1, beta1: 1, beta2: 1, high_gamma: 1 },
  weight_preset: "legacy",
});

export const isBusy = (state) =>
  ["running", "opening", "cancelling"].includes(state?.task?.status);
export function currentRound(state) {
  return (
    !state?.selected_round ||
    [state.first_round_id, state.second_round_id]
      .filter(Boolean)
      .includes(state.selected_round)
  );
}
export function reviewData(state) {
  return (
    state?.rounds?.find((round) => round.id === state.first_round_id)?.data ??
    state?.data ??
    {}
  );
}
export function stabilityText(value) {
  if(!value)return '';
  if(typeof value==='string')return value.slice(0,140);
  const summary=value.summary??value,parts=[];
  if(value.round!=null)parts.push(`第 ${value.round} 轮`);
  if(summary.supported!=null&&summary.attempted!=null)parts.push(`支持 ${summary.supported}/${summary.attempted}`);
  const ari=summary.ari_full_mean??summary.mean_ari;
  if(typeof ari==='number'&&Number.isFinite(ari))parts.push(`ARI ${ari.toFixed(3)}`);
  for(const [key,label]of [['failed','失败'],['collapsed','塌缩'],['tiny','小类']])if(summary[key]!=null)parts.push(`${label} ${summary[key]}`);
  if(summary.cancelled)parts.push('已取消');
  return parts.join(' · ')||'已生成稳定性报告';
}
export function canAction(state, action, draft = {}) {
  if (!state?.run_id || state.read_only || isBusy(state)) return false;
  if (draft.configDirty || draft.extractionDirty) return false;
  if (draft.fitDirty && ['preview_review','fit_second','stability'].includes(action))return false;
  if (draft.qcDirty && action === "confirm_trials") return false;
  if (
    (draft.qcDirty || draft.trialDirty) &&
    ["fit_first", "preview_review", "fit_second", "stability"].includes(action)
  )
    return false;
  const stages = {
    detect_qc: [
      "trial_review",
      "trial_confirmed",
      "first_ready",
      "second_ready",
    ],
    confirm_trials: [
      "trial_review",
      "trial_confirmed",
      "first_ready",
      "second_ready",
    ],
    waveform: [
      "trial_review",
      "trial_confirmed",
      "first_ready",
      "second_ready",
    ],
    fit_first: ["trial_confirmed", "first_ready", "second_ready"],
    preview_review: ["first_ready", "second_ready"],
    fit_second: ["first_ready", "second_ready"],
    stability: ["first_ready", "second_ready"],
  };
  if (!stages[action]?.includes(state.stage)) return false;
  if (["preview_review", "fit_second", "stability"].includes(action))
    return Boolean(state.first_round_id) && currentRound(state);
  return true;
}

export class StateGate {
  generation = 0;
  state = null;
  detached = false;
  detach() { this.reset(); this.detached=true; }
  reset() {
    this.generation++;
    this.state = null;
    return this.generation;
  }
  accept(state, generation, {attach=false}={}) {
    if (
      generation !== this.generation ||
      !state ||
      !Number.isInteger(state.revision)
    )
      return false;
    if(this.detached&&!attach)return false;
    if (
      this.state?.run_id === state.run_id &&
      this.state.revision > state.revision
    )
      return false;
    this.state = state;
    this.detached=false;
    return true;
  }
}

export function firstFitPayload(input) {
  const config=validateConfig(input);
  return {first_k:config.first_k,fit_settings:{band_weights:config.band_weights,weight_preset:config.weight_preset,group_weights:config.group_weights??null}};
}
export function configChanges(accepted,draft) {
  if(!accepted)return {extractionDirty:false,fitDirty:false};
  const extraction=['baseline_mode','baseline_window','baseline_edge_seconds','feature_sigma_seconds'];
  return {extractionDirty:extraction.some(key=>JSON.stringify(accepted[key])!==JSON.stringify(draft[key])),
    fitDirty:JSON.stringify(firstFitPayload(accepted))!==JSON.stringify(firstFitPayload(draft))};
}

export function safeLocalURL(value, base) {
  const root = new URL(".", base);
  const url = new URL(value, root);
  if (
    url.origin !== root.origin ||
    !url.pathname.startsWith(root.pathname) ||
    url.username ||
    url.password ||
    /[\\\u0000]/.test(value)
  )
    throw new Error("拒绝非本地会话地址");
  return url.href;
}
export class WorkspaceAPI {
  constructor(base, fetcher = globalThis.fetch.bind(globalThis)) {
    this.base = new URL(".", base).href;
    this.fetcher = fetcher;
  }
  async request(route, body, query) {
    if (!/^[a-z_]+$/.test(route)) throw new Error("无效 API");
    const url = new URL(`api/workspace/${route}`, this.base);
    for (const [key, value] of Object.entries(query || {}))
      if (value != null) url.searchParams.set(key, value);
    const response = await this.fetcher(url, {
      method: body === undefined ? "GET" : "POST",
      credentials: "same-origin",
      cache: "no-store",
      ...(body === undefined
        ? {}
        : {
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          }),
    });
    let result;
    try {
      result = await response.json();
    } catch {
      throw new Error(`服务响应无效 (${response.status})`);
    }
    if (!response.ok || result.error)
      throw new Error(result.error || `请求失败 (${response.status})`);
    return result;
  }
  get(route, query) {
    return this.request(route, undefined, query);
  }
  post(route, body = {}) {
    return this.request(route, body);
  }
  identity(state) {
    if (!state?.run_id || !Number.isInteger(state.revision))
      throw new Error("没有活动分析");
    return { run_id: state.run_id, revision: state.revision };
  }
  action(state, action, payload = {}) {
    return this.post("action", { ...this.identity(state), action, payload });
  }
}

const pairKey = (channel, trial) =>
  JSON.stringify([String(channel), Number(trial)]);
export class TrialSelection {
  selected = new Map();
  history = [];
  snapshot() {
    this.history.push(new Map(this.selected));
    if (this.history.length > 50) this.history.shift();
  }
  toggle(channel, trial) {
    this.snapshot();
    const key = pairKey(channel, trial);
    if (this.selected.has(key)) this.selected.delete(key);
    else this.selected.set(key, [String(channel), Number(trial)]);
  }
  has(channel, trial) {
    return this.selected.has(pairKey(channel, trial));
  }
  pairs() {
    return [...this.selected.values()];
  }
  clear() {
    this.snapshot();
    this.selected.clear();
  }
  undo() {
    if (this.history.length) this.selected = this.history.pop();
  }
  candidates(rows) {
    this.snapshot();
    this.selected = new Map(
      rows
        .filter((r) => r.candidate && !r.raw_bad)
        .map((r) => [pairKey(r.channel, r.trial_id), [r.channel, r.trial_id]]),
    );
  }
}
export function normalizeQC(data = {}) {
  return (data.qc_rows || data.trial_qc || [])
    .map((row) => ({
      ...row,
      channel: String(row.channel ?? row.channel_name ?? ""),
      trial_id: Number(row.trial_id ?? row.original_trial_id ?? row.trial),
      flag_clipping: Boolean(row.flag_clipping ?? row.flag_clipping_candidate),
      candidate: Boolean(
        row.candidate ??
          (row.flag_spike ||
            row.flag_jump ||
            row.flag_flat ||
            row.flag_clipping || row.flag_clipping_candidate),
      ),
      reasons:
        row.reasons ??
        ["spike", "jump", "flat", "clipping"]
          .filter((k) => row[`flag_${k}`])
          .join(", "),
    }))
    .filter((r) => r.channel && Number.isInteger(r.trial_id));
}
export function inputErrors(resolved, mode) {
  const errors = resolved?.errors ?? [];
  return (Array.isArray(errors) ? errors : Object.values(errors)).filter(
    (error) => {
      const value = typeof error === "string" ? error : JSON.stringify(error);
      return (
        !/Missing anatomy|brain display unavailable/i.test(value) &&
        !(mode === "pre_onset" && /baseline_fif|基线/i.test(value))
      );
    },
  );
}
function number(
  value,
  name,
  { integer = false, min = 0, strict = false } = {},
) {
  if (value === "" || value == null) throw new Error(`${name}不能为空`);
  const n = Number(value);
  if (
    !Number.isFinite(n) ||
    (integer && !Number.isInteger(n)) ||
    (strict ? n <= min : n < min)
  )
    throw new Error(`${name}无效`);
  return n;
}
export function validateConfig(input = {}) {
  const config = { ...DEFAULT_CONFIG, ...input };
  if (!["file", "pre_onset"].includes(config.baseline_mode))
    throw new Error("基线模式无效");
  const interval = config.baseline_window.map(Number);
  if (
    interval.length !== 2 ||
    !interval.every(Number.isFinite) ||
    interval[0] >= interval[1] ||
    interval[1] >= 0
  )
    throw new Error("基线窗口需满足 开始 < 结束 < 0");
  config.baseline_window = interval;
  config.baseline_edge_seconds = number(
    config.baseline_edge_seconds,
    "基线边缘",
  );
  config.feature_sigma_seconds = number(
    config.feature_sigma_seconds,
    "特征平滑",
  );
  for (const k of ["first_k", "second_k"])
    config[k] = number(config[k], k, { integer: true, min: 2 });
  const weights = { ...config.band_weights };
  for (const band of ["theta", "beta1", "beta2", "high_gamma"])
    weights[band] = number(weights[band], `${band}权重`, { strict: true });
  if (!Object.values(weights).some((v) => v > 0))
    throw new Error("频带权重不能全为零");
  config.band_weights = weights;
  if (
    !["legacy", "balanced", "morphology", "amplitude", "custom"].includes(
      config.weight_preset,
    )
  )
    throw new Error("权重预设无效");
  if (config.weight_preset === "custom") {
    config.group_weights = Object.fromEntries(
      ["morphology", "amplitude", "slope"].map((k) => [
        k,
        number(config.group_weights?.[k] ?? 1, k),
      ]),
    );
    if (!Object.values(config.group_weights).some((v) => v > 0))
      throw new Error("组权重不能全为零");
  } else delete config.group_weights;
  return config;
}
export function reviewPayload(input) {
  const distance =
    input.distance_enabled ?? input.pca_distance_threshold != null;
  if (!["global", "cluster", undefined].includes(input.distance_reference))
    throw new Error("距离参照无效");
  return {
    bad_clusters: [...new Set(input.bad_clusters || [])],
    manual_bad_channels: [...new Set(input.manual_bad_channels || [])],
    pca_distance_threshold: distance
      ? number(input.pca_distance_threshold, "PCA距离阈值", { strict: true })
      : null,
    distance_reference: input.distance_reference || "global",
    second_k: number(input.second_k, "第二轮 K", { integer: true, min: 2 }),
  };
}
export function pcaRows(data) {
  const pca = data?.pca ?? [];
  const normalize = (rows) =>
    rows.map((row) => ({
      ...row,
      cluster: row.cluster ?? row.first_label ?? row.label ?? 0,
    }));
  if (Array.isArray(pca)) return normalize(pca);
  if (Array.isArray(pca.rows)) return normalize(pca.rows);
  if (
    Array.isArray(pca.points) &&
    pca.points[0] &&
    !Array.isArray(pca.points[0])
  )
    return normalize(pca.points);
  const points = pca.scores || pca.points || [];
  return points.map((p, i) => ({
    channel: pca.channels?.[i] ?? data.channels?.[i],
    pc1: p[0],
    pc2: p[1],
    pc3: p[2],
    cluster: pca.labels?.[i] ?? 0,
  }));
}
export function pcaTraces(rows, excluded = []) {
  const groups = new Map(),
    bad = new Set(excluded.map(String));
  const colors = [
    "red", "blue", "green", "purple", "gray",
    "orange", "cyan", "magenta", "gold", "brown",
  ];
  for (const row of rows) {
    const coordinates = [
      row.pc1 ?? row.PC1 ?? row.x,
      row.pc2 ?? row.PC2 ?? row.y,
      row.pc3 ?? row.PC3 ?? row.z,
    ];
    if (!coordinates.every((v) =>
      (typeof v === "number" || (typeof v === "string" && v.trim() !== "")) &&
      Number.isFinite(Number(v)))) continue;
    const [x, y, z] = coordinates.map(Number);
    const channel = String(row.channel ?? row.channel_name ?? ""),
      cluster = row.cluster ?? row.first_label ?? row.label ?? 0;
    const removed = bad.has(channel),
      key = JSON.stringify([cluster, removed]);
    if (!groups.has(key))
      groups.set(key, {
        type: "scatter3d",
        mode: "markers",
        name: `类 ${cluster}${removed ? " · 排除预览" : ""}`,
        x: [],
        y: [],
        z: [],
        text: [],
        customdata: [],
        marker: {
          size: removed ? 6 : 5,
          color: removed
            ? "black"
            : Number(cluster) < 0 ? "white" : colors[Math.abs(Number(cluster) || 0) % colors.length],
          symbol: removed ? "x" : "circle",
          line: {color: "black", width: .5},
        },
        hovertemplate: "%{text}<extra>%{fullData.name}</extra>",
      });
    const trace = groups.get(key);
    trace.x.push(x);
    trace.y.push(y);
    trace.z.push(z);
    trace.text.push(channel);
    trace.customdata.push(channel);
  }
  return [...groups.values()];
}
