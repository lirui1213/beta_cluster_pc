"""Modal four-band PCA review with one preview/refit exclusion contract."""
from dataclasses import dataclass
import re

import numpy as np

from .review import review_exclusion_audit


@dataclass(frozen=True)
class ReviewSettings:
    n_clusters: int = 4
    bad_clusters: tuple[int, ...] = ()
    manual_bad_channels: tuple[str, ...] = ()
    pca_distance_threshold: float | None = None
    distance_reference: str = "global"

    def refit_kwargs(self):
        return dict(n_clusters=self.n_clusters, bad_clusters=self.bad_clusters,
            manual_bad_channels=self.manual_bad_channels, pca_distance_threshold=self.pca_distance_threshold,
            distance_reference=self.distance_reference)


def _selection_inputs(*, bad_clusters=(), manual_channels="", distance_enabled=False,
                      distance_threshold="", distance_reference="global"):
    names = tuple(dict.fromkeys(name.strip() for name in re.split(r"[,;，；\s]+", manual_channels) if name.strip()))
    threshold = None
    if distance_enabled:
        try:
            threshold = float(distance_threshold)
        except (TypeError, ValueError) as error:
            raise ValueError("PCA 距离阈值必须为正数。") from error
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("PCA 距离阈值必须为有限正数。")
    return dict(bad_clusters=tuple(sorted(set(bad_clusters))), manual_bad_channels=names,
                pca_distance_threshold=threshold, distance_reference=distance_reference)


def _parse_k(value, remaining):
    text = str(value).strip()
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError("第二轮类别数 K 必须为整数。")
    k = int(text)
    if not 2 <= k <= remaining:
        raise ValueError(f"第二轮类别数 K 必须在 2 与剩余电极数 {remaining} 之间。")
    return k


def parse_review_settings(first, *, n_clusters="4", **options):
    inputs = _selection_inputs(**options)
    audit = review_exclusion_audit(first, **inputs)
    k = _parse_k(n_clusters, int((~audit.excluded_final).sum()))
    return ReviewSettings(n_clusters=k, **inputs), audit


class FourBandReviewWindow:
    """Construct Tk only on show; cancelling never returns accepted settings."""

    def __init__(self, result, *, initial=None):
        self.result = result
        self.initial = initial if initial is not None else ReviewSettings()
        self.selection = None
        self.preview_audit = None
        self.variables = {}
        self.cluster_variables = {}
        self.widgets = {}
        self._pending_preview = None
        self._variable_traces = []

    def show(self):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        self.selection = None
        self.root = tk.Tk()
        self._closed = False
        self.root.title("四频带 PCA 复审与第二轮聚类")
        self.root.geometry("1060x740")
        self.root.minsize(860, 640)
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        plot_frame = ttk.Frame(outer)
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        controls = ttk.Frame(outer, width=320)
        controls.grid(row=0, column=1, sticky="ns")
        controls.columnconfigure(0, weight=1)
        self.figure = Figure(figsize=(6.8, 6.2))
        self.axis = self.figure.add_subplot(111, projection="3d")
        # 3D labels extend beyond the axes box; reserve a right-hand margin.
        self.figure.subplots_adjust(left=.03, right=.84, bottom=.05, top=.97)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, plot_frame)

        self.variables = dict(
            manual=tk.StringVar(value=", ".join(self.initial.manual_bad_channels)),
            distance_enabled=tk.BooleanVar(value=self.initial.pca_distance_threshold is not None),
            distance_threshold=tk.StringVar(value="" if self.initial.pca_distance_threshold is None else str(self.initial.pca_distance_threshold)),
            distance_reference=tk.StringVar(value=self.initial.distance_reference),
            n_clusters=tk.StringVar(value=str(self.initial.n_clusters)),
            counts=tk.StringVar(), error=tk.StringVar(),
        )
        distance = ttk.LabelFrame(controls, text="PCA 距离剔除", padding=8)
        distance.grid(row=0, column=0, sticky="ew")
        distance.columnconfigure(1, weight=1)
        ttk.Checkbutton(distance, text="启用距离阈值", variable=self.variables["distance_enabled"]).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(distance, text="阈值").grid(row=1, column=0, sticky="w", pady=4)
        self.widgets["distance_threshold"] = ttk.Entry(distance, textvariable=self.variables["distance_threshold"], width=14)
        self.widgets["distance_threshold"].grid(row=1, column=1, sticky="ew")
        ttk.Label(distance, text="参考").grid(row=2, column=0, sticky="w")
        reference_label = tk.StringVar(value={"global": "全局距离", "cluster": "簇内距离"}.get(self.initial.distance_reference, "全局距离"))
        self.widgets["distance_reference"] = ttk.Combobox(distance, textvariable=reference_label,
            values=("全局距离", "簇内距离"), state="readonly", width=14)
        self.widgets["distance_reference"].grid(row=2, column=1, sticky="ew")
        trace = reference_label.trace_add("write", lambda *_: self.variables["distance_reference"].set(
            "global" if reference_label.get() == "全局距离" else "cluster"))
        self._variable_traces.append((reference_label, trace))

        cluster_frame = ttk.LabelFrame(controls, text="剔除整类", padding=8)
        cluster_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        cluster_canvas = tk.Canvas(cluster_frame, height=180, width=285, highlightthickness=0)
        scrollbar = ttk.Scrollbar(cluster_frame, orient="vertical", command=cluster_canvas.yview)
        cluster_canvas.configure(yscrollcommand=scrollbar.set)
        cluster_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        cluster_list = ttk.Frame(cluster_canvas)
        cluster_canvas.create_window((0, 0), window=cluster_list, anchor="nw")
        cluster_list.bind("<Configure>", lambda event: cluster_canvas.configure(scrollregion=cluster_canvas.bbox("all")))
        for row, label in enumerate(sorted(set(self.result.labels) - {-1})):
            variable = tk.BooleanVar(value=int(label) in self.initial.bad_clusters)
            self.cluster_variables[int(label)] = variable
            count = int((self.result.labels == label).sum())
            ttk.Checkbutton(cluster_list, text=f"Cluster {label}  ({count} 电极)", variable=variable).grid(row=row, column=0, sticky="w")

        ttk.Label(controls, text="指定剔除电极").grid(row=2, column=0, sticky="w", pady=(10, 3))
        self.widgets["manual"] = ttk.Entry(controls, textvariable=self.variables["manual"], width=34)
        self.widgets["manual"].grid(row=3, column=0, sticky="ew")
        k_frame = ttk.Frame(controls)
        k_frame.grid(row=4, column=0, sticky="ew", pady=(12, 8))
        ttk.Label(k_frame, text="第二轮类别数 K").pack(side="left")
        self.widgets["n_clusters"] = ttk.Spinbox(k_frame, from_=2, to=int(self.result.clean_mask.sum()),
            textvariable=self.variables["n_clusters"], width=6)
        self.widgets["n_clusters"].pack(side="right")
        ttk.Label(controls, textvariable=self.variables["counts"], wraplength=310).grid(row=5, column=0, sticky="ew", pady=4)
        ttk.Label(controls, textvariable=self.variables["error"], foreground="firebrick", wraplength=310).grid(row=6, column=0, sticky="ew", pady=4)
        buttons = ttk.Frame(controls)
        buttons.grid(row=7, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(buttons, text="取消", command=self.cancel).pack(side="left")
        self.widgets["apply"] = ttk.Button(buttons, text="确认并重新聚类", command=self.apply)
        self.widgets["apply"].pack(side="right")
        for variable in (*self.cluster_variables.values(), *(self.variables[name] for name in
                ("manual", "distance_enabled", "distance_threshold", "distance_reference", "n_clusters"))):
            trace = variable.trace_add("write", self._schedule_preview)
            self._variable_traces.append((variable, trace))
        self.update_preview()
        self.root.grab_set()
        try:
            self.root.mainloop()
        finally:
            from matplotlib import pyplot as plt
            plt.close(self.figure)
            if not self._closed:
                self.cancel()
        return self.selection

    def _options(self):
        return dict(bad_clusters=tuple(k for k, v in self.cluster_variables.items() if v.get()),
            manual_channels=self.variables["manual"].get(),
            distance_enabled=self.variables["distance_enabled"].get(),
            distance_threshold=self.variables["distance_threshold"].get(),
            distance_reference=self.variables["distance_reference"].get())

    def _schedule_preview(self, *_):
        if self._pending_preview is not None:
            self.root.after_cancel(self._pending_preview)
        self._pending_preview = self.root.after(150, self.update_preview)

    def update_preview(self):
        from visualization import plot_pca_clusters
        if self._pending_preview is not None:
            self.root.after_cancel(self._pending_preview)
        self._pending_preview = None
        enabled = self.variables["distance_enabled"].get()
        self.widgets["distance_threshold"].configure(state="normal" if enabled else "disabled")
        self.widgets["distance_reference"].configure(state="readonly" if enabled else "disabled")
        try:
            self.preview_audit = review_exclusion_audit(self.result, **_selection_inputs(**self._options()))
        except ValueError as error:
            self.preview_audit = None
            self.variables["counts"].set("参数无效，无法计算剔除及剩余数。")
            self.variables["error"].set(str(error))
            self.widgets["apply"].state(["disabled"])
            return
        remaining = int((~self.preview_audit.excluded_final).sum())
        initial = int(self.preview_audit.excluded_initial.sum())
        total = int(self.preview_audit.excluded_final.sum())
        self.variables["counts"].set(f"原始/QC 坏道：{initial}\n新增剔除：{total - initial}   合计：{total}\n剩余电极：{remaining}")
        self.widgets["n_clusters"].configure(to=max(2, remaining))
        try:
            _parse_k(self.variables["n_clusters"].get(), remaining)
        except ValueError as error:
            self.variables["error"].set(str(error))
            self.widgets["apply"].state(["disabled"])
        else:
            self.variables["error"].set("")
            self.widgets["apply"].state(["!disabled"])
        self.axis.clear()
        plot_pca_clusters(self.result.pca_scores, self.result.labels, channel_names=self.result.channels,
                          ax=self.axis, annotate=False)
        removed = self.preview_audit.excluded_final.to_numpy() & self.result.clean_mask
        if removed.any():
            points = self.preview_audit.loc[removed, ["pc1", "pc2", "pc3"]].to_numpy()
            self.axis.scatter(*points.T, c="black", marker="x", s=65, linewidths=1.7, label="Excluded")
            self.axis.legend(fontsize=8, loc="upper left")
        self.canvas.draw_idle()

    def apply(self):
        self.update_preview()
        try:
            settings, _ = parse_review_settings(self.result, n_clusters=self.variables["n_clusters"].get(), **self._options())
        except ValueError as error:
            self.variables["error"].set(str(error))
            return
        self.selection = settings
        self.cancel()

    def cancel(self):
        if self._closed:
            return
        if self._pending_preview is not None:
            self.root.after_cancel(self._pending_preview)
            self._pending_preview = None
        for variable, trace in self._variable_traces:
            variable.trace_remove("write", trace)
        self._variable_traces.clear()
        self.root.destroy()
        self._closed = True
