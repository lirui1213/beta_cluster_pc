export class SessionAPI {
  constructor(base, fetcher = globalThis.fetch.bind(globalThis)) {
    this.base = new URL("./", base);
    this.fetcher = fetcher;
    this.generation = 0;
    this.controller = null;
  }

  cancel() {
    this.generation++;
    this.controller?.abort();
    this.controller = null;
  }

  async request(route, body, signal) {
    const url = new URL(route, this.base);
    if (
      url.origin !== this.base.origin ||
      !url.pathname.startsWith(this.base.pathname + "api/")
    ) {
      throw new Error("Request outside this session");
    }
    const response = await this.fetcher(url, {
      method: body === undefined ? "GET" : "POST",
      credentials: "same-origin",
      cache: "no-store",
      signal,
      headers:
        body === undefined
          ? { Accept: "application/json" }
          : { "Content-Type": "application/json", Accept: "application/json" },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
    let result;
    try {
      result = await response.json();
    } catch {
      throw new Error(`Invalid service response (HTTP ${response.status})`);
    }
    if (!response.ok)
      throw new Error(
        result.error || `Service request failed (HTTP ${response.status})`,
      );
    return result;
  }

  async compute(settings, progress, ready) {
    this.cancel();
    const generation = this.generation;
    const controller = new AbortController();
    this.controller = controller;
    const current = () =>
      generation === this.generation && !controller.signal.aborted;
    try {
      let status = await this.request(
        "api/compute",
        settings,
        controller.signal,
      );
      if (!current()) return;
      if (!status.job_id) throw new Error("Service did not return a job ID");
      const route = `api/job/${encodeURIComponent(status.job_id)}?channel=${encodeURIComponent(settings.channel)}`;
      while (current()) {
        status = await this.request(route, undefined, controller.signal);
        if (!current()) return;
        progress(status);
        if (status.state === "ready") {
          if (!status.data)
            throw new Error("Ready job contains no display data");
          await ready(status.data, generation);
          return;
        }
        if (
          [
            "failed",
            "error",
            "cancelled",
            "canceled",
            "superseded",
            "closed",
          ].includes(status.state)
        ) {
          throw new Error(status.error || `Display job ${status.state}`);
        }
        await new Promise((resolve, reject) => {
          const abort = () => {
            clearTimeout(timer);
            reject(new DOMException("Aborted", "AbortError"));
          };
          const timer = setTimeout(() => {
            controller.signal.removeEventListener("abort", abort);
            resolve();
          }, 600);
          controller.signal.addEventListener("abort", abort, { once: true });
        });
      }
    } catch (error) {
      if (current() && error.name !== "AbortError") throw error;
    }
  }
}

export function safeImageURL(path, base) {
  const root = new URL("./", base);
  const result = new URL(path, root);
  if (
    result.origin !== root.origin ||
    !result.pathname.startsWith(root.pathname + "api/image/")
  ) {
    throw new Error("Detail image is outside this session");
  }
  return result.href;
}

export function curveTraces(
  data,
  band,
  color,
  showMean,
  showAudio,
  showSelected = true,
) {
  const trace = (name, values, line, extra = {}) => ({
    type: "scatter",
    mode: "lines",
    name,
    x: data.times || [],
    y: values || [],
    connectgaps: false,
    line,
    hovertemplate: "%{x:.3f} s<br>%{y:.4g}<extra>%{fullData.name}</extra>",
    ...extra,
  });
  const result = [];
  if (showSelected)
    result.push(
      trace("Selected channel", band.selected, { color, width: 1.8 }),
    );
  if (showMean)
    result.push(
      trace("Class mean", band.mean, {
        color: "#596775",
        width: 1.4,
        dash: "dot",
      }),
    );
  if (showAudio)
    result.push(
      trace(
        "Audio envelope",
        data.audio,
        { color: "#b85776", width: 1 },
        { yaxis: "y2", opacity: 0.65 },
      ),
    );
  return result;
}

export function eventShapes(events = [], trial_ids = []) {
  const result = [];
  for (const event of events) {
    if (Number.isFinite(event.onset) && Number.isFinite(event.offset)) {
      result.push({
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: event.onset,
        x1: event.offset,
        y0: 0,
        y1: 1,
        fillcolor: event.excluded
          ? "rgba(183,78,100,0.09)"
          : "rgba(22,125,145,0.035)",
        line: { width: 0 },
        layer: "below",
      });
    }
    for (const value of [event.onset, event.offset]) {
      if (Number.isFinite(value))
        result.push({
          type: "line",
          xref: "x",
          yref: "paper",
          x0: value,
          x1: value,
          y0: 0,
          y1: 1,
          line: {
            color: event.excluded ? "#b74e64" : "#a1adb8",
            width: 1,
            dash: "dot",
          },
        });
    }
  }
  const first = events.find(
    (event) => event.original_trial_id === trial_ids[0],
  );
  const second = events.find(
    (event) => event.original_trial_id === trial_ids[1],
  );
  if (
    trial_ids.length === 2 &&
    Number.isFinite(first?.offset) &&
    Number.isFinite(second?.onset) &&
    second.onset > first.offset
  ) {
    result.push({
      type: "rect",
      xref: "x",
      yref: "paper",
      x0: first.offset,
      x1: second.onset,
      y0: 0,
      y1: 1,
      fillcolor: "rgba(93,109,121,0.055)",
      line: { width: 0 },
      layer: "below",
    });
  }
  return result;
}
