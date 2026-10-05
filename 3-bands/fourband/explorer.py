"""A display-only Tk explorer for real adjacent trials and cluster detail."""
from pathlib import Path

import numpy as np

from signal_processing import finite_mean
from visualization import cluster_color
from .continuous import adjacent_trial_pairs, prepare_continuous_trials


class ExplorerState:
    def __init__(self, raw, events, audio, extraction, result, *, cache=None,
                 trial_ids=None, pre_seconds=1., post_seconds=1., sigma_seconds=.10,
                 display_sfreq=100.):
        self.raw, self.events, self.audio = raw, events, audio
        self.extraction, self.result, self.cache = extraction, result, cache
        reference = next(iter(extraction.epochs.values()))
        if tuple(result.channels) != tuple(reference.channel_names):
            raise ValueError("Explorer channel order must match extraction.")
        self.pairs = adjacent_trial_pairs(events, reference.events)
        if self.pairs.empty:
            raise ValueError("No adjacent retained trials available for inspection.")
        self.pair_index = 0
        self.band = next(iter(extraction.bands))
        self.channel = result.channels[0]
        self.pre_seconds, self.post_seconds = pre_seconds, post_seconds
        self.sigma_seconds, self.display_sfreq = sigma_seconds, display_sfreq
        included = np.flatnonzero(np.asarray(result.labels) >= 0)
        if len(included):
            self.channel = result.channels[int(included[0])]
        if trial_ids is not None:
            self.select_pair(trial_ids)

    @property
    def trial_ids(self):
        pair = self.pairs.iloc[self.pair_index]
        return int(pair.first_trial_id), int(pair.second_trial_id)

    @property
    def label(self):
        return int(self.result.labels[self.result.channels.index(self.channel)])

    @property
    def excluded_trial_ids(self):
        return tuple(int(trial) for channel, trial in getattr(self.extraction, "trial_exclusions", ())
                     if channel == self.channel)

    def navigate(self, step):
        index = self.pair_index + int(step)
        if not 0 <= index < len(self.pairs):
            return False
        self.pair_index = index
        return True

    def select_pair(self, pair):
        matches = np.flatnonzero((self.pairs.first_trial_id == pair[0]) & (self.pairs.second_trial_id == pair[1]))
        if len(matches) != 1:
            raise ValueError("Select a pair of adjacent retained original trials.")
        self.pair_index = int(matches[0])

    def select_channel(self, channel):
        if channel not in self.result.channels:
            raise ValueError(f"Unknown channel: {channel}")
        self.channel = channel

    def select_band(self, band):
        if band not in self.extraction.bands:
            raise ValueError(f"Unknown band: {band}")
        self.band = band

    def continuous(self):
        kwargs = dict(trial_ids=self.trial_ids, pre_seconds=self.pre_seconds, post_seconds=self.post_seconds,
                      sigma_seconds=self.sigma_seconds, display_sfreq=self.display_sfreq)
        if self.cache is not None:
            kwargs["cache"] = self.cache
        return prepare_continuous_trials(self.raw, self.events, self.audio, self.extraction, **kwargs)


class FourBandExplorer:
    def __init__(self, raw, events, audio, extraction, result, *, output_dir=None, **settings):
        self.state = ExplorerState(raw, events, audio, extraction, result, **settings)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.root = None
        self.closed = True
        self._drawing = False
        self._pending = None

    def show(self, block=True):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        if not self.closed:
            self.root.lift()
            if block:
                self.root.mainloop()
            return self
        self.root = tk.Tk()
        self.closed = False
        self.root.title("Four-band trial and electrode explorer")
        self.root.geometry("1250x820")
        self.root.minsize(900, 650)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        controls = ttk.Frame(self.root, padding=8)
        controls.pack(fill="x")
        tools = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        tools.pack(fill="x")
        self.variables = {"channel": tk.StringVar(value=self.state.channel),
            "band": tk.StringVar(value=self.state.band), "cluster": tk.StringVar(value=str(self.state.label)),
            "sigma": tk.StringVar(value=str(self.state.sigma_seconds)), "pair": tk.StringVar(),
            "view": tk.StringVar(value="Continuous"), "status": tk.StringVar()}
        self.previous = ttk.Button(controls, text="<", width=3, command=lambda: self.navigate(-1))
        self.previous.pack(side="left")
        self.next = ttk.Button(controls, text=">", width=3, command=lambda: self.navigate(1))
        self.next.pack(side="left", padx=(3, 8))
        self.pair_labels = [f"{int(p.first_trial_id)} -> {int(p.second_trial_id)}" for p in self.state.pairs.itertuples()]
        for name, values, width in (("pair", self.pair_labels, 15),
                ("band", tuple(self.state.extraction.bands), 12),
                ("cluster", tuple(map(str, np.unique(self.state.result.labels))), 7),
                ("channel", self.state.result.channels, 12), ("view", ("Continuous", "Cluster detail"), 15)):
            widget = ttk.Combobox(controls, textvariable=self.variables[name], values=values, state="readonly", width=width)
            widget.pack(side="left", padx=4)
            widget.bind("<<ComboboxSelected>>", lambda event, key=name: self._changed(key))
        ttk.Label(tools, text="Sigma (s)").pack(side="left", padx=(0, 2))
        entry = ttk.Entry(tools, textvariable=self.variables["sigma"], width=6)
        entry.pack(side="left")
        entry.bind("<Return>", lambda event: self._changed("sigma"))
        ttk.Button(tools, text="Apply", command=lambda: self._changed("sigma")).pack(side="left", padx=4)
        ttk.Button(tools, text="Save", command=self.save).pack(side="left", padx=4)
        ttk.Button(tools, text="Clear cache", command=self.clear_cache).pack(side="left", padx=4)
        self.figure = Figure(figsize=(12, 7), layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, self.root)
        ttk.Label(self.root, textvariable=self.variables["status"], anchor="w").pack(fill="x", padx=8, pady=4)
        self._sync()
        self.redraw()
        if block:
            self.root.mainloop()
        else:
            self.pump()
        return self

    def _sync(self):
        self.variables["pair"].set(self.pair_labels[self.state.pair_index])
        self.variables["channel"].set(self.state.channel)
        self.variables["cluster"].set(str(self.state.label))
        self.previous.configure(state="normal" if self.state.pair_index > 0 else "disabled")
        self.next.configure(state="normal" if self.state.pair_index < len(self.pair_labels) - 1 else "disabled")

    def navigate(self, step):
        if self.state.navigate(step):
            self._sync()
            self.redraw()

    def select_channel(self, channel):
        self.state.select_channel(channel)
        if self.closed:
            self.show(block=False)
        else:
            self._sync()
            self._schedule_redraw()

    def _schedule_redraw(self):
        if not self.closed:
            if self._pending is not None:
                self.root.after_cancel(self._pending)
            self._pending = self.root.after_idle(self.redraw)

    def _changed(self, key):
        try:
            if key == "pair":
                self.state.pair_index = self.pair_labels.index(self.variables[key].get())
            elif key == "channel":
                self.state.select_channel(self.variables[key].get())
            elif key == "band":
                self.state.select_band(self.variables[key].get())
            elif key == "cluster":
                label = int(self.variables[key].get())
                index = np.flatnonzero(self.state.result.labels == label)[0]
                self.state.select_channel(self.state.result.channels[int(index)])
            sigma = float(self.variables["sigma"].get())
            if not np.isfinite(sigma) or sigma < 0:
                raise ValueError("Sigma must be finite and nonnegative.")
            self.state.sigma_seconds = sigma
            self._sync()
            self.redraw()
        except (ValueError, IndexError) as error:
            self.variables["status"].set(str(error))

    def redraw(self):
        self._pending = None
        if self.closed or self._drawing:
            return
        self._drawing = True
        try:
            self.variables["status"].set("Rendering...")
            self.root.update_idletasks()
            self.figure.clear()
            if self.variables["view"].get() == "Cluster detail":
                from .inspection import plot_inspection
                plot_inspection(self.state.extraction, self.state.result, self.state.band,
                    channel=self.state.channel, label=self.state.label, sigma_seconds=self.state.sigma_seconds,
                    n_bootstrap=0, figure=self.figure)
            else:
                self._draw_continuous()
            self.canvas.draw_idle()
            cache_info = self.state.cache.info() if self.state.cache is not None else {"enabled": False}
            self.variables["status"].set(f"Channel {self.state.channel} | Cluster {self.state.label} | "
                f"Trials {self.state.trial_ids} | QC excluded: {self.state.excluded_trial_ids} | Cache {cache_info}")
        except Exception as error:
            self.variables["status"].set(f"Display unavailable: {error}")
        finally:
            self._drawing = False

    def _draw_continuous(self):
        data = self.state.continuous()
        axes = self.figure.subplots(2, 2, squeeze=False)
        channel_index = data.channels.index(self.state.channel)
        members = self.state.result.labels == self.state.label
        origin = data.metadata["first_onset_seconds"]
        for axis, (band, curves) in zip(axes.flat, data.power.items()):
            axis.plot(data.times, finite_mean(curves[members]), color=cluster_color(self.state.label),
                linewidth=1., alpha=.5, label="Cluster mean")
            axis.plot(data.times, curves[channel_index], color="#177b72", linewidth=1.6, label=self.state.channel)
            right = axis.twinx()
            right.plot(data.times, data.audio, color="black", linestyle="--", linewidth=.8)
            right.set_ylabel("Audio / speech peak", fontsize=8)
            for event in data.events.itertuples():
                excluded = event.original_trial_id in self.state.excluded_trial_ids
                start, stop = event.onset - origin, event.offset - origin
                axis.axvspan(start, stop, color="red" if excluded else "green", alpha=.09)
                axis.axvline(start, color="gray", linestyle="--", linewidth=.7)
                axis.axvline(stop, color="gray", linestyle=":", linewidth=.7)
                left, bound = max(start, data.times[0]), min(stop, data.times[-1])
                if bound > left:
                    label_x = np.clip((left + bound) / 2, data.times[0] + .05 * np.ptp(data.times),
                                      data.times[-1] - .05 * np.ptp(data.times))
                    axis.text(label_x, .98, f"{event.original_trial_id}" + (" (QC)" if excluded else ""),
                        transform=axis.get_xaxis_transform(), va="top", ha="center", fontsize=8)
            axis.set(title=band, xlabel="Real time from first onset (s)", ylabel="Power Z",
                     xlim=(data.times[0], data.times[-1]))
            axis.legend(fontsize=8, loc="lower left")
        self.figure.suptitle(f"{self.state.channel} | Cluster {self.state.label} | Trials {data.trial_ids}\n"
            + data.metadata["baseline_description"], fontsize=11)

    def save(self):
        if self.output_dir is None:
            self.variables["status"].set("No export directory configured.")
            return
        from datetime import datetime
        import json
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.output_dir / f"explorer_{stamp}.png"
        self.figure.savefig(path, dpi=140)
        metadata = dict(channel=self.state.channel, cluster=self.state.label, band=self.state.band,
            trial_ids=self.state.trial_ids, sigma_seconds=self.state.sigma_seconds,
            excluded_trial_ids=self.state.excluded_trial_ids, view=self.variables["view"].get())
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.variables["status"].set(str(path))

    def clear_cache(self):
        if self.state.cache is not None:
            self.state.cache.clear()
        self.variables["status"].set("Cache cleared.")

    def pump(self):
        if not self.closed:
            try:
                self.root.update_idletasks()
                self.root.update()
            except Exception:
                self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self._pending is not None:
            try:
                self.root.after_cancel(self._pending)
            except Exception:
                pass
            self._pending = None
        self.figure.clear()
        try:
            self.root.destroy()
        except Exception:
            pass
