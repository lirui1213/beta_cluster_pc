const rgba = (hex) => {
  const rgb = /^#[0-9a-f]{6}$/i.test(hex) ? hex.slice(1) : "167d91";
  return `rgba(${[0,2,4].map(i => parseInt(rgb.slice(i,i+2),16)).join(",")},0.18)`;
};
const escapeText = (text) => String(text).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");

export function gridTraces(panel, color) {
  const boundary = (sign) => panel.mean.map((value, i) =>
    Number.isFinite(value) && Number.isFinite(panel.sem[i]) ? value + sign * panel.sem[i] : null);
  const lower = boundary(-1), upper = boundary(1), areaX = [], areaY = [];
  // Separate closed polygons keep uncertainty shading out of missing intervals.
  for (let start = 0; start < lower.length;) {
    if (lower[start] === null) { start++; continue; }
    let end = start + 1;
    while (end < lower.length && lower[end] !== null) end++;
    if (end - start > 1) {
      areaX.push(...panel.times_ms.slice(start,end), ...panel.times_ms.slice(start,end).reverse(), null);
      areaY.push(...upper.slice(start,end), ...lower.slice(start,end).reverse(), null);
    }
    start = end;
  }
  const trace = (name, y, line, extra = {}) => ({
    type:"scatter", mode:"lines", name, x:panel.times_ms, y,
    line, connectgaps:false, ...extra,
  });
  return [
    trace("Mean − SEM", lower, {width:0}, {hoverinfo:"skip"}),
    trace("Mean ± SEM", areaY, {width:0}, {x:areaX, fill:"toself", fillcolor:rgba(color), hoverinfo:"skip"}),
    trace("Class mean", panel.mean, {color, width:1.7}, {
      customdata:panel.counts,
      hovertemplate:"%{x:.0f} ms<br>Power Z: %{y:.4g}<br>n = %{customdata}<extra>Class mean</extra>",
    }),
    trace("Audio envelope", panel.audio, {color:"#222222", width:1, dash:"dash"}, {
      yaxis:"y2", opacity:.7,
      hovertemplate:"%{x:.0f} ms<br>Audio / speech peak: %{y:.4g}<extra>Audio envelope</extra>",
    }),
  ];
}

export class ClusterGrid {
  constructor(section, colors) {
    this.section = section;
    this.host = section.querySelector("#cluster-grid");
    this.status = section.querySelector("#cluster-grid-status");
    this.colors = colors;
    this.entries = [];
    this.closed = false;
    this.queue = Promise.resolve();
    this.controller = new AbortController();
    new ResizeObserver(() => this.resize()).observe(this.host);
  }

  async load(api) {
    try {
      const data = await api.request("api/cluster-grid", undefined, this.controller.signal);
      if (this.closed) return;
      if (!data.available) {
        this.status.textContent = data.message || "Saved class summary is unavailable";
        return;
      }
      if (!Array.isArray(data.classes) || !data.classes.length || data.bands?.length !== 4 || !Array.isArray(data.panels))
        throw new Error("Invalid saved class summary");
      this.queue = this.queue.then(async () => {
        for (const group of data.classes) {
          if (this.closed) return;
          const row = document.createElement("div");
          row.className = "cluster-grid-row";
          row.dataset.label = String(group.label);
          const heading = document.createElement("h3"), dot = document.createElement("b");
          dot.className = "cluster-grid-dot";
          heading.append(dot, document.createTextNode(`Class ${group.label} · n = ${group.n_electrodes}`));
          row.append(heading);
          this.host.append(row);
          for (const band of data.bands) {
            if (this.closed) return;
            const panel = data.panels.find(value => value.label === group.label && value.band === band.name);
            if (!panel || !panel.times_ms?.length) throw new Error("Incomplete saved class summary");
            const plot = document.createElement("div");
            plot.className = "cluster-grid-plot";
            plot.setAttribute("aria-label", `Class ${group.label} ${band.name} summary`);
            row.append(plot);
            const entry = {plot, label:group.label, dot};
            this.entries.push(entry);
            const limits = Number.isFinite(band.low) && Number.isFinite(band.high) ? ` ${band.low}–${band.high} Hz` : "";
            const markers = [0, data.speech_offset_ms].filter(Number.isFinite);
            await window.Plotly.newPlot(plot, gridTraces(panel, this.colors()[group.label] || "#167d91"), {
              autosize:true, margin:{l:53,r:45,t:40,b:50},
              title:{text:escapeText(band.name + limits), font:{size:12},x:.05,xanchor:"left"},
              font:{family:"Segoe UI, Arial",size:10,color:"#64727d"},
              paper_bgcolor:"#ffffff",plot_bgcolor:"#ffffff",showlegend:false,hovermode:"x unified",
              xaxis:{title:{text:"Duration-normalized time (ms)",font:{size:10}},
                range:[panel.times_ms[0],panel.times_ms.at(-1)],gridcolor:"#edf0f2",automargin:true},
              yaxis:{title:{text:"Power Z",font:{size:10}},gridcolor:"#edf0f2",zerolinecolor:"#dce3e8",automargin:true},
              yaxis2:{title:{text:"Audio / speech peak",font:{size:9}},overlaying:"y",side:"right",
                range:[0,1.05],showgrid:false,zeroline:false,nticks:3,automargin:true},
              shapes:markers.map(value => ({type:"line",xref:"x",yref:"paper",x0:value,x1:value,y0:0,y1:1,
                line:{color:"#a1adb8",dash:"dash",width:1}})),
            }, {responsive:true,displaylogo:false,displayModeBar:"hover",scrollZoom:false,
              modeBarButtonsToRemove:["toImage","lasso2d","select2d"]});
            if (this.closed) return;
          }
        }
        if (!this.closed) this.status.textContent = "";
      });
      await this.queue;
      await this.recolor();
    } catch (error) {
      if (!this.closed) {
        for (const {plot} of this.entries) window.Plotly.purge(plot);
        this.entries = [];
        this.host.replaceChildren();
        this.status.textContent = `Class summary unavailable: ${error.message || error}`;
        this.queue = Promise.resolve();
      }
    }
  }

  recolor() {
    this.queue = this.queue.catch(() => {}).then(async () => {
      for (const {plot,label,dot} of this.entries) {
        if (this.closed) return;
        const color = this.colors()[label] || "#167d91";
        dot.style.backgroundColor = color;
        await window.Plotly.restyle(plot, {fillcolor:rgba(color)}, [1]);
        if (this.closed) return;
        await window.Plotly.restyle(plot, {"line.color":color}, [2]);
      }
    });
    return this.queue;
  }

  resize() {
    if (this.closed) return;
    for (const {plot} of this.entries) {
      if (plot._fullLayout && plot.clientWidth && Math.abs(plot._fullLayout.width-plot.clientWidth) > 1)
        window.Plotly.relayout(plot, {width:plot.clientWidth,height:plot.clientHeight,autosize:true}).catch(() => {});
    }
  }

  close() {
    this.closed = true;
    this.controller.abort();
  }
}
