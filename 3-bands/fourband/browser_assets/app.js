import { SessionAPI, curveTraces, eventShapes, safeImageURL } from "./core.mjs";
import { BrainView } from "./brain.js";
import { ClusterGrid } from "./cluster_grid.mjs";

const $ = (id) => document.getElementById(id);
const base = new URL("./", window.location.href);
const api = new SessionAPI(base);
const palette = [
  "#167d91",
  "#c45277",
  "#719342",
  "#8a65ab",
  "#c47f34",
  "#367bb8",
  "#af5d45",
  "#527d72",
];
const state = {
  meta: null,
  view: "continuous",
  sigma: 0.05,
  pair: 0,
  channel: "",
  classFilter: "all",
  colors: {},
  band: "",
  data: null,
  detail: null,
  revision: 0,
  renderedRevision: null,
  visualRevision: 0,
  closed: false,
  clearing: false,
  surfaceWarnings: [],
};
const clusterGrid = new ClusterGrid($("cluster-grid-pane"), () => state.colors);
let brain = null,
  plotQueue = Promise.resolve(),
  clock = null,
  started = 0,
  serverElapsed = 0,
  serverAt = 0;
const storageKey = `fourband-view:${base.pathname}`;
let saved = {};
try {
  saved = JSON.parse(sessionStorage.getItem(storageKey) || "{}") || {};
} catch {
  /* Storage may be disabled by browser policy. */
}

function setBusy(text, elapsed) {
  $("status-text").textContent = text;
  $("status-dot").className = "status-dot busy";
  $("retry").hidden = true;
  if (!clock) {
    started = performance.now();
    serverElapsed = 0;
    serverAt = started;
    clock = setInterval(() => {
      const now = performance.now();
      const value =
        serverElapsed > 0
          ? serverElapsed + (now - serverAt) / 1000
          : (now - started) / 1000;
      $("elapsed").textContent = `${value.toFixed(1)} s`;
    }, 200);
  }
  if (Number.isFinite(elapsed)) {
    serverElapsed = elapsed;
    serverAt = performance.now();
    $("elapsed").textContent = `${elapsed.toFixed(1)} s`;
  }
}

function stopClock() {
  clearInterval(clock);
  clock = null;
}
function setReady(text = "Ready") {
  stopClock();
  $("status-text").textContent = text;
  $("status-dot").className = "status-dot";
  $("retry").hidden = true;
}
function fail(error) {
  if (state.closed) return;
  stopClock();
  $("status-text").textContent = error.message || String(error);
  $("status-dot").className = "status-dot error";
  $("retry").hidden = false;
  if (state.meta?.curves_enabled) {
    $("plots").hidden = true;
    $("detail").hidden = true;
    $("curves-empty").hidden = false;
    $("curves-empty").textContent =
      `Display unavailable: ${error.message || error}`;
  }
  $("export").disabled = true;
}
function notice(text) {
  $("notice").textContent = text;
  $("notice").hidden = !text;
}
function warnings(messages) {
  const list = [
    ...new Set(
      [...(state.meta?.warnings || []), ...state.surfaceWarnings, ...messages]
        .filter(Boolean)
        .map(String),
    ),
  ];
  $("warnings").textContent = list.join(" | ");
  $("warnings").hidden = !list.length;
}
function colorHex(value, fallback) {
  if (typeof value !== "string" || !CSS.supports("color", value))
    return fallback;
  const context = document.createElement("canvas").getContext("2d");
  context.fillStyle = value;
  return /^#[\da-f]{6}$/i.test(context.fillStyle)
    ? context.fillStyle
    : fallback;
}
function option(select, value, text) {
  const element = document.createElement("option");
  element.value = String(value);
  element.textContent = text;
  select.append(element);
}
function pairIDs() {
  const pair = state.meta.pairs[state.pair];
  return pair ? [pair.first_trial_id, pair.second_trial_id] : [];
}
function classOf(channel = state.channel) {
  return state.meta?.labels[state.meta.channels.indexOf(channel)];
}
function selectedColor() {
  return state.colors[classOf()] || palette[0];
}
function style() {
  return {
    brain_color: $("brain-color").value,
    brain_opacity: Number($("brain-opacity").value),
    point_size: Number($("point-size").value),
    electrodes: $("electrodes").checked,
    labels: $("labels").checked,
    class_filter: state.classFilter,
    channel: state.channel,
    colors: { ...state.colors },
  };
}
function settings() {
  return {
    ...state.meta.settings,
    view: state.view,
    trial_ids: pairIDs(),
    sigma_seconds: state.sigma,
    channel: state.channel,
    band: state.band,
    ...style(),
    show_selected: $("show-selected").checked,
    show_mean: $("show-mean").checked,
    show_audio: $("show-audio").checked,
    camera: brain?.cameraState() || saved.camera || null,
    subject: state.meta.subject,
    block: state.meta.block,
    round: state.meta.round,
  };
}
function canExport() {
  if (!state.meta || state.closed || state.clearing) return false;
  if (!state.meta.curves_enabled)
    return Boolean(brain && $("brain-error").hidden);
  return (
    state.renderedRevision === state.revision &&
    Boolean(state.view === "detail" ? state.detail : state.data)
  );
}
function persist() {
  if (!state.meta) return;
  try {
    sessionStorage.setItem(storageKey, JSON.stringify(settings()));
  } catch {
    /* The live session does not require storage. */
  }
}
function updateSelection() {
  state.visualRevision++;
  $("selection-name").textContent = state.channel || "No channel selected";
  $("selection-class").textContent =
    classOf() === undefined ? "" : `Class ${classOf()}`;
  $("selection-dot").style.background = selectedColor();
  $("class-color").value = selectedColor();
  document.documentElement.style.setProperty("--accent", selectedColor());
  $("previous-pair").disabled = state.closed || state.pair <= 0;
  $("next-pair").disabled =
    state.closed || state.pair >= state.meta.pairs.length - 1;
  const ids = pairIDs();
  $("pair-description").textContent = ids.length
    ? `Trials ${ids[0]} / ${ids[1]}`
    : "No adjacent trial pair";
  $("opacity-value").value = Number($("brain-opacity").value).toFixed(2);
  $("size-value").value = $("point-size").value;
  brain?.update(style());
  persist();
}
function setChannel(channel) {
  if (!state.meta.channels.includes(channel) || state.closed) return;
  state.channel = channel;
  $("channel").value = channel;
  if (state.classFilter !== "all" && String(classOf()) !== state.classFilter) {
    state.classFilter = "all";
    $("class-filter").value = "all";
  }
  updateSelection();
  requestDisplay();
}

async function renderContinuous(data, revision) {
  if (!Array.isArray(data.times) || !Array.isArray(data.bands))
    throw new Error("Invalid continuous payload");
  const names = state.meta.bands;
  if (data.channel && data.channel !== state.channel)
    throw new Error("Display channel does not match current selection");
  const task = async () => {
    if (revision !== state.revision || state.closed) return;
    const color = selectedColor();
    for (let i = 0; i < names.length; i++) {
      if (revision !== state.revision || state.closed) return;
      const band = data.bands.find((value) => value.name === names[i]) || {
        name: names[i],
        selected: [],
        mean: [],
      };
      const host = $(`plot-${i}`);
      const events = data.events || [];
      const annotations = events
        .filter((event) => Number.isFinite(event.onset))
        .map((event) => ({
          x: event.onset,
          y: 1,
          xref: "x",
          yref: "paper",
          yanchor: "bottom",
          xanchor: "left",
          showarrow: false,
          text: `${event.original_trial_id}${event.excluded ? " · QC" : ""}`,
          font: { size: 9, color: event.excluded ? "#b74e64" : "#7a858e" },
        }));
      await window.Plotly.react(
        host,
        curveTraces(
          data,
          band,
          color,
          $("show-mean").checked,
          $("show-audio").checked,
          $("show-selected").checked,
        ),
        {
          autosize: true,
          margin: { l: 56, r: $("show-audio").checked ? 58 : 15, t: 48, b: 43 },
          paper_bgcolor: "#ffffff",
          plot_bgcolor: "#ffffff",
          title: {
            text: band.name,
            font: { family: "Segoe UI, Arial", size: 12, color: "#43535e" },
            x: 0.045,
            xanchor: "left",
            y: 0.96,
          },
          font: { family: "Segoe UI, Arial", size: 10, color: "#64727d" },
          showlegend: false,
          hovermode: "x unified",
          xaxis: {
            title: { text: "Time (s)", font: { size: 10 } },
            zeroline: false,
            gridcolor: "#edf0f2",
            automargin: true,
            range: data.times.length
              ? [data.times[0], data.times.at(-1)]
              : undefined,
          },
          yaxis: {
            title: { text: "Power Z", font: { size: 10 } },
            zerolinecolor: "#dce3e8",
            gridcolor: "#edf0f2",
            automargin: true,
          },
          yaxis2: {
            title: {
              text: "Audio / speech peak",
              font: { size: 9, color: "#b85776" },
            },
            overlaying: "y",
            side: "right",
            showgrid: false,
            zeroline: false,
            showticklabels: $("show-audio").checked,
            visible: $("show-audio").checked,
            tickfont: { size: 9, color: "#b85776" },
            tickformat: ".2g",
            nticks: 4,
            automargin: true,
          },
          shapes: eventShapes(events, data.trial_ids),
          annotations,
          uirevision: JSON.stringify([
            data.trial_ids,
            state.sigma,
            state.channel,
          ]),
        },
        {
          responsive: true,
          displaylogo: false,
          displayModeBar: "hover",
          scrollZoom: false,
          modeBarButtonsToRemove: [
            "toImage",
            "lasso2d",
            "select2d",
            "autoScale2d",
          ],
        },
      );
    }
    if (revision !== state.revision || state.closed) return;
    state.data = data;
    state.detail = null;
    state.renderedRevision = revision;
    $("plots").classList.remove("pending");
    $("plots").hidden = false;
    $("plots").setAttribute("aria-busy", "false");
    $("curves-empty").hidden = true;
    $("detail").hidden = true;
    $("data-channel").textContent =
      `${data.channel || state.channel} · Class ${data.label ?? classOf()}`;
    $("baseline").textContent = data.baseline_description || "";
    $("curve-caption").textContent = "Seconds relative to first trial onset";
    $("curves-heading").textContent = "Continuous response";
    warnings(data.metadata?.warnings || []);
    resizePlots();
  };
  plotQueue = plotQueue.catch(() => {}).then(task);
  await plotQueue;
}

async function renderDetail(data, revision) {
  const source = safeImageURL(data.image_url, base);
  const image = new Image();
  image.src = source;
  await image.decode();
  if (revision !== state.revision || state.closed) return;
  const target = $("detail-image");
  target.src = source;
  target.hidden = false;
  state.detail = image;
  state.data = null;
  state.renderedRevision = revision;
  $("detail").hidden = false;
  $("plots").hidden = true;
  $("curves-empty").hidden = true;
  $("curves-heading").textContent = "Class detail";
  $("baseline").textContent = data.band || state.band;
  $("curve-caption").textContent = "Duration-normalized summary";
  $("data-channel").textContent = data.channel || state.channel;
}

async function requestDisplay() {
  const revision = ++state.revision;
  api.cancel();
  $("export").disabled = true;
  if (!state.meta || state.closed || state.clearing) return;
  if (!state.meta.curves_enabled) {
    setReady();
    $("export").disabled = !brain;
    return;
  }
  const trial_ids = pairIDs();
  if (trial_ids.length !== 2 || !state.channel) {
    fail(new Error("No valid adjacent pair or channel is available"));
    return;
  }
  const payload = {
    view: state.view,
    trial_ids,
    sigma_seconds: state.sigma,
    channel: state.channel,
    band: state.band,
  };
  stopClock();
  setBusy("Submitting display request");
  $("plots").classList.add("pending");
  $("plots").setAttribute("aria-busy", "true");
  $("plots").hidden = true;
  $("detail").hidden = true;
  $("curves-empty").hidden = false;
  $("curves-empty").textContent = "Waiting for display data";
  try {
    await api.compute(
      payload,
      (status) => {
        if (revision !== state.revision) return;
        setBusy(
          `${status.stage || status.state || "Computing"}`,
          status.elapsed_seconds,
        );
        $("curves-empty").textContent =
          status.stage || "Computing display data";
      },
      async (data) => {
        if (revision !== state.revision) return;
        setBusy(
          state.view === "detail"
            ? "Loading class detail image"
            : "Rendering four-band curves",
        );
        if (state.view === "detail") await renderDetail(data, revision);
        else await renderContinuous(data, revision);
        if (revision !== state.revision || state.closed) return;
        setReady();
        $("export").disabled = false;
        persist();
      },
    );
  } catch (error) {
    if (revision === state.revision) fail(error);
  }
}

function resizePlots() {
  if (!$("plots").hidden && window.Plotly)
    document
      .querySelectorAll("#plots .js-plotly-plot")
      .forEach((host) => window.Plotly.Plots.resize(host).catch(() => {}));
}
async function refreshPlotStyle() {
  updateSelection();
  if (
    state.data &&
    state.view === "continuous" &&
    $("status-text").textContent === "Ready"
  ) {
    const revision = state.revision;
    try {
      await renderContinuous(state.data, revision);
    } catch (error) {
      if (revision === state.revision) fail(error);
    }
  }
}

function configure(meta) {
  if (
    !Array.isArray(meta.channels) ||
    !Array.isArray(meta.labels) ||
    meta.channels.length !== meta.labels.length ||
    !Array.isArray(meta.bands) ||
    !Array.isArray(meta.pairs)
  )
    throw new Error("Invalid session metadata");
  state.meta = meta;
  $("session-name").textContent =
    `${meta.subject ?? "Subject"} / ${meta.block ?? "Block"} · Round ${meta.round ?? ""}`;
  document.title = `${meta.subject ?? ""} ${meta.block ?? ""} | Four-band explorer`;
  for (const [i, pair] of meta.pairs.entries())
    option($("pair"), i, `${pair.first_trial_id} / ${pair.second_trial_id}`);
  const requestedPair = saved.trial_ids || meta.settings?.trial_ids;
  state.pair = Math.max(
    0,
    meta.pairs.findIndex(
      (pair) =>
        pair.first_trial_id === requestedPair?.[0] &&
        pair.second_trial_id === requestedPair?.[1],
    ),
  );
  $("pair").value = String(state.pair);
  for (const channel of meta.channels) option($("channel"), channel, channel);
  state.channel = meta.channels.includes(saved.channel)
    ? saved.channel
    : meta.channels[0] || "";
  $("channel").value = state.channel;
  const classes = [...new Set(meta.labels)].sort((a, b) => a - b);
  option($("class-filter"), "all", "All classes");
  for (const [index, label] of classes.entries()) {
    option($("class-filter"), label, `Class ${label}`);
    state.colors[label] = colorHex(
      saved.colors?.[label],
      palette[index % palette.length],
    );
  }
  state.classFilter = classes.some(
    (label) => String(label) === saved.class_filter,
  )
    ? saved.class_filter
    : "all";
  if (state.classFilter !== "all" && String(classOf()) !== state.classFilter)
    state.classFilter = "all";
  $("class-filter").value = state.classFilter;
  state.sigma =
    Number.isFinite(saved.sigma_seconds) &&
    saved.sigma_seconds >= 0 &&
    saved.sigma_seconds <= 10
      ? saved.sigma_seconds
      : (meta.settings?.sigma_seconds ?? 0.05);
  $("sigma").value = state.sigma;
  for (const band of meta.bands) option($("detail-band"), band, band);
  state.band = meta.bands.includes(saved.band)
    ? saved.band
    : meta.bands[0] || "";
  $("detail-band").value = state.band;
  state.view = saved.view === "detail" ? "detail" : "continuous";
  const surface = meta.surface_options || {};
  $("brain-color").value = colorHex(
    saved.brain_color || surface.brain_color,
    "#c8ccd0",
  );
  $("brain-opacity").value = Number.isFinite(saved.brain_opacity)
    ? saved.brain_opacity
    : (surface.brain_opacity ?? 0.65);
  const pointSize = Number.isFinite(saved.point_size)
    ? saved.point_size
    : (surface.point_size ?? 14);
  $("point-size").max = String(Math.max(30, pointSize));
  $("point-size").value = pointSize;
  for (const [id, key, fallback] of [
    ["labels", "labels", false],
    ["electrodes", "electrodes", true],
    ["show-selected", "show_selected", true],
    ["show-mean", "show_mean", true],
    ["show-audio", "show_audio", true],
  ])
    $(id).checked = typeof saved[key] === "boolean" ? saved[key] : fallback;
  document
    .querySelectorAll(
      "#controls button, #controls select, #controls input, .appearance input, #clear-cache",
    )
    .forEach((element) => {
      element.disabled = false;
    });
  $("brain-pane").hidden = !meta.surface_enabled;
  $("curves-pane").hidden = !meta.curves_enabled;
  $("cluster-grid-pane").hidden = !meta.curves_enabled;
  $("workspace").classList.toggle("brain-only", !meta.curves_enabled);
  $("workspace").classList.toggle("curves-only", !meta.surface_enabled);
  document.querySelectorAll(".brain-control").forEach((element) => {
    element.hidden = !meta.surface_enabled;
  });
  document.querySelectorAll(".curve-control").forEach((element) => {
    element.hidden = !meta.curves_enabled;
  });
  $("clear-cache").disabled = !meta.curves_enabled;
  $("view-continuous").setAttribute(
    "aria-pressed",
    String(state.view === "continuous"),
  );
  $("view-detail").setAttribute(
    "aria-pressed",
    String(state.view === "detail"),
  );
  document.querySelector(".detail-control").hidden =
    !meta.curves_enabled || state.view !== "detail";
  $("plots").replaceChildren();
  for (const [i, band] of meta.bands.entries()) {
    const wrapper = document.createElement("div");
    wrapper.className = "band-plot";
    const plot = document.createElement("div");
    plot.id = `plot-${i}`;
    plot.className = "plot";
    plot.setAttribute("aria-label", `${band} response`);
    wrapper.append(plot);
    $("plots").append(wrapper);
  }
  $("plots").hidden = true;
  warnings([]);
  updateSelection();
}

async function loadSurface() {
  try {
    const payload = await api.request("api/surface");
    if (state.closed) return;
    const surfaceWarnings = payload.warnings || [];
    try {
      brain = new BrainView(
        $("brain"),
        $("electrode-labels"),
        setChannel,
        (message) => {
          $("brain-error").textContent = message;
          $("brain-error").hidden = false;
          if (!canExport()) $("export").disabled = true;
        },
      );
    } catch (error) {
      throw new Error(
        `WebGL unavailable: ${error.message}. Continuous curves remain available.`,
      );
    }
    const info = brain.load(payload);
    brain.update(style());
    brain.restoreCamera(saved.camera);
    persist();
    brain.controls.addEventListener("end", persist);
    brain.controls.addEventListener("change", () => {
      state.visualRevision++;
    });
    $("brain-loading").hidden = true;
    $("reset-camera").disabled = false;
    $("anatomy-caption").textContent =
      (payload.meshes || []).map((mesh) => mesh.name).join(" · ") ||
      "Electrode coordinates";
    $("mapped-count").textContent = `${info.mapped} mapped contacts`;
    const mapped = new Set(
      (payload.electrodes || [])
        .filter(
          (item) =>
            item.position?.length === 3 && item.position.every(Number.isFinite),
        )
        .map((item) => item.channel),
    );
    const unmapped = state.meta.channels.filter(
      (channel) => !mapped.has(channel),
    );
    state.surfaceWarnings = [
      ...surfaceWarnings,
      ...info.warnings,
      ...unmapped.map((channel) => `Unmapped channel: ${channel}`),
    ];
    warnings([]);
    if (!state.meta.curves_enabled) {
      setReady();
      $("export").disabled = false;
    }
  } catch (error) {
    $("brain-loading").hidden = true;
    $("brain-error").textContent = error.message;
    $("brain-error").hidden = false;
    if (!state.meta.curves_enabled) fail(error);
  }
}

async function exportView() {
  const revision = state.revision;
  const visualRevision = state.visualRevision;
  $("export").disabled = true;
  if (!canExport()) {
    notice("Export failed: Current view is not ready for export.");
    return;
  }
  notice("Preparing export");
  try {
    const snapshot = settings(),
      images = {};
    if (brain && $("brain-error").hidden) images.brain = brain.exportPNG();
    if (state.meta.curves_enabled) {
      if (state.view === "detail" && state.detail) {
        const canvas = document.createElement("canvas");
        canvas.width = state.detail.naturalWidth;
        canvas.height = state.detail.naturalHeight;
        canvas.getContext("2d").drawImage(state.detail, 0, 0);
        images.curves = canvas.toDataURL("image/png");
      } else if (state.data) {
        const renderedData = state.data;
        await plotQueue;
        const bounds = $("plots").getBoundingClientRect(),
          scale = 2;
        const canvas = document.createElement("canvas");
        canvas.width = Math.round(bounds.width * scale);
        const context = canvas.getContext("2d");
        const captionWidth = bounds.width - 28;
        const captionRows = [];
        const addCaption = (text, font, color) => {
          context.font = font;
          let line = "";
          for (let word of text.split(/\s+/)) {
            const joined = line ? `${line} ${word}` : word;
            if (context.measureText(joined).width <= captionWidth) {
              line = joined;
              continue;
            }
            if (line) captionRows.push({ text: line, font, color });
            while (
              context.measureText(word).width > captionWidth &&
              word.length > 1
            ) {
              let end = word.length - 1;
              while (
                end > 1 &&
                context.measureText(word.slice(0, end)).width > captionWidth
              )
                end--;
              captionRows.push({ text: word.slice(0, end), font, color });
              word = word.slice(end);
            }
            line = word;
          }
          if (line) captionRows.push({ text: line, font, color });
        };
        addCaption(
          `${snapshot.subject} / ${snapshot.block} - Round ${snapshot.round}`,
          "600 13px Segoe UI, Arial",
          "#28333c",
        );
        addCaption(
          `${snapshot.channel} - Class ${renderedData.label ?? classOf()} - Trials ${snapshot.trial_ids.join(" / ")} - Smoothing ${snapshot.sigma_seconds} s`,
          "12px Segoe UI, Arial",
          "#354650",
        );
        addCaption(
          `Baseline: ${renderedData.baseline_description || "Unavailable"}`,
          "11px Segoe UI, Arial",
          "#64727d",
        );
        const legend = [
          ...(snapshot.show_selected
            ? [
                {
                  text: "Channel",
                  color: snapshot.colors[renderedData.label] || palette[0],
                  dash: [],
                },
              ]
            : []),
          ...(snapshot.show_mean
            ? [{ text: "Class mean", color: "#596775", dash: [3, 3] }]
            : []),
          ...(snapshot.show_audio
            ? [{ text: "Audio / speech peak", color: "#b85776", dash: [] }]
            : []),
          { text: "QC excluded", color: "#b74e64", dash: [2, 3] },
        ];
        context.font = "11px Segoe UI, Arial";
        let legendX = 14,
          legendY = 14 + captionRows.length * 18;
        for (const entry of legend) {
          const width = context.measureText(entry.text).width + 40;
          if (legendX > 14 && legendX + width > bounds.width - 14) {
            legendX = 14;
            legendY += 20;
          }
          entry.x = legendX;
          entry.y = legendY;
          legendX += width;
        }
        const headerHeight = legendY + 28;
        canvas.height = Math.round((bounds.height + headerHeight) * scale);
        context.fillStyle = "#fff";
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.save();
        context.scale(scale, scale);
        context.textBaseline = "top";
        captionRows.forEach((row, index) => {
          context.font = row.font;
          context.fillStyle = row.color;
          context.fillText(row.text, 14, 12 + index * 18);
        });
        context.font = "11px Segoe UI, Arial";
        for (const entry of legend) {
          context.strokeStyle = entry.color;
          context.lineWidth = 1.6;
          context.setLineDash(entry.dash);
          context.beginPath();
          context.moveTo(entry.x, entry.y + 6);
          context.lineTo(entry.x + 16, entry.y + 6);
          context.stroke();
          context.fillStyle = "#596974";
          context.fillText(entry.text, entry.x + 22, entry.y);
        }
        context.restore();
        for (const host of document.querySelectorAll("#plots .js-plotly-plot")) {
          const rect = host.getBoundingClientRect();
          const source = await window.Plotly.toImage(host, {
            format: "png",
            width: Math.round(rect.width),
            height: Math.round(rect.height),
            scale,
          });
          const image = new Image();
          image.src = source;
          await image.decode();
          context.drawImage(
            image,
            (rect.x - bounds.x) * scale,
            (rect.y - bounds.y + headerHeight) * scale,
            rect.width * scale,
            rect.height * scale,
          );
        }
        images.curves = canvas.toDataURL("image/png");
      }
    }
    if (
      revision !== state.revision ||
      !canExport() ||
      visualRevision !== state.visualRevision ||
      state.closed
    )
      throw new Error(
        "View changed during export. Export the current view again.",
      );
    if (!Object.keys(images).length)
      throw new Error("No rendered image is available for export");
    const result = await api.request("api/export", {
      settings: snapshot,
      images,
    });
    if (!state.closed)
      notice(
        `Export saved${Array.isArray(result.files) ? ": " + result.files.join(", ") : ""}`,
      );
  } catch (error) {
    if (!state.closed) notice(`Export failed: ${error.message}`);
  } finally {
    if (!state.closed && revision === state.revision)
      $("export").disabled = !canExport();
  }
}

$("channel").addEventListener("change", (event) =>
  setChannel(event.target.value),
);
$("pair").addEventListener("change", (event) => {
  state.pair = Number(event.target.value);
  updateSelection();
  requestDisplay();
});
for (const [id, delta] of [
  ["previous-pair", -1],
  ["next-pair", 1],
])
  $(id).addEventListener("click", () => {
    state.pair = Math.max(
      0,
      Math.min(state.meta.pairs.length - 1, state.pair + delta),
    );
    $("pair").value = state.pair;
    updateSelection();
    requestDisplay();
  });
$("class-filter").addEventListener("change", () => {
  state.classFilter = $("class-filter").value;
  if (state.classFilter !== "all" && String(classOf()) !== state.classFilter) {
    const index = state.meta.labels.findIndex(
      (label) => String(label) === state.classFilter,
    );
    state.channel = state.meta.channels[index];
    $("channel").value = state.channel;
    updateSelection();
    requestDisplay();
  } else updateSelection();
});
$("class-color").addEventListener("input", () => {
  state.colors[classOf()] = $("class-color").value;
  refreshPlotStyle();
  clusterGrid.recolor().catch(error => notice(`Class summary color update failed: ${error.message}`));
});
for (const id of [
  "brain-color",
  "brain-opacity",
  "point-size",
  "electrodes",
  "labels",
])
  $(id).addEventListener("input", updateSelection);
for (const id of ["show-selected", "show-mean", "show-audio"])
  $(id).addEventListener("change", refreshPlotStyle);
$("reset-camera").addEventListener("click", () => {
  brain?.reset();
  persist();
});
$("apply-sigma").addEventListener("click", () => {
  if (!$("sigma").checkValidity() || $("sigma").value === "") {
    notice("Smoothing must be between 0 and 10 seconds.");
    return;
  }
  state.sigma = Number($("sigma").value);
  notice("");
  persist();
  requestDisplay();
});
$("sigma").addEventListener("keydown", (event) => {
  if (event.key === "Enter") $("apply-sigma").click();
});
for (const view of ["continuous", "detail"])
  $(`view-${view}`).addEventListener("click", () => {
    state.view = view;
    for (const item of ["continuous", "detail"])
      $(`view-${item}`).setAttribute("aria-pressed", String(item === view));
    document.querySelector(".detail-control").hidden = view !== "detail";
    persist();
    requestDisplay();
  });
$("detail-band").addEventListener("change", () => {
  state.band = $("detail-band").value;
  persist();
  requestDisplay();
});
$("clear-cache").addEventListener("click", async () => {
  state.clearing = true;
  state.revision++;
  api.cancel();
  state.data = null;
  state.detail = null;
  $("clear-cache").disabled = true;
  $("export").disabled = true;
  setBusy("Clearing display cache");
  try {
    await api.request("api/clear", {});
    notice("Display cache cleared");
    state.clearing = false;
    if (!state.closed) await requestDisplay();
  } catch (error) {
    state.clearing = false;
    fail(error);
  } finally {
    if (!state.closed) $("clear-cache").disabled = false;
  }
});
$("export").addEventListener("click", exportView);
$("retry").addEventListener("click", () => {
  if (state.meta) requestDisplay();
  else boot();
});
$("close-session").addEventListener("click", async () => {
  $("close-session").disabled = true;
  api.cancel();
  state.revision++;
  setBusy("Closing session");
  try {
    await api.request("api/close", {});
    state.closed = true;
    setReady("Session closed");
    notice("");
    clusterGrid.close();
    document.body.classList.add("closed");
    document.querySelectorAll("button,input,select").forEach((element) => {
      element.disabled = true;
    });
  } catch (error) {
    $("close-session").disabled = false;
    fail(error);
  }
});
window.addEventListener("resize", resizePlots);
new ResizeObserver(resizePlots).observe($("curves-pane"));

async function boot() {
  setBusy("Loading session metadata");
  try {
    if (!window.Plotly || !window.lucide)
      throw new Error(
        "A local frontend dependency failed to load. Reload this session.",
      );
    const meta = await api.request("api/meta");
    if (state.closed) return;
    configure(meta);
    const tasks = [];
    if (meta.surface_enabled) tasks.push(loadSurface());
    if (meta.curves_enabled) tasks.push(requestDisplay(), clusterGrid.load(api));
    if (!meta.surface_enabled && !meta.curves_enabled)
      throw new Error("This session has no enabled display");
    await Promise.all(tasks);
  } catch (error) {
    fail(error);
  }
}
window.lucide?.createIcons();
boot();
