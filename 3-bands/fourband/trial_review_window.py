"""Modal channel-trial review with explicit confirmation and raw waveforms."""
from copy import deepcopy

import numpy as np

from .trial_qc import (TrialExclusions, _aligned, _source, _validated_pairs,
                       validate_trial_report)


class TrialQualityReviewWindow:
    """Defer Tk construction to show(); cancellation returns None, keep-all ()."""

    def __init__(self, raw, extraction, report, initial_pairs=()):
        provenance = validate_trial_report(raw, extraction, report)
        pairs = _validated_pairs(extraction, initial_pairs)
        initial_provenance = getattr(initial_pairs, "provenance", None)
        if ((pairs and initial_provenance is None)
                or (initial_provenance is not None and initial_provenance != provenance)):
            raise ValueError("Initial trial review provenance is missing or changed; start a fresh review.")
        self.raw = raw
        self.extraction = extraction
        self.report = report.copy(deep=True)
        self.provenance = deepcopy(provenance)
        self.selected_pairs = set(pairs)
        self.selection = None
        self.pair_variables = {}
        self.widgets = {}
        self._history = []
        self._traces = []
        self._syncing = False
        self._closed = False
        self._rows = {(row.channel, int(row.original_trial_id)): row for row in self.report.itertuples()}
        self._current_pair = None

    def _set_selection(self, pairs):
        pairs = set(_validated_pairs(self.extraction, pairs))
        if pairs != self.selected_pairs:
            self._history.append(self.selected_pairs.copy())
            self.selected_pairs = pairs
            self._refresh_selection()

    def set_pair(self, pair, selected=True):
        pair = _validated_pairs(self.extraction, [pair])[0]
        pairs = self.selected_pairs.copy()
        if selected:
            pairs.add(pair)
        else:
            pairs.discard(pair)
        self._set_selection(pairs)

    def select_candidates(self):
        self._set_selection(self.selected_pairs | {pair for pair, row in self._rows.items() if row.candidate})

    def clear_selection(self):
        self._set_selection(())

    def undo_selection(self):
        if self._history:
            self.selected_pairs = self._history.pop()
            self._refresh_selection()

    def remaining_counts(self):
        source, reference = _aligned(self.extraction)
        valid = np.logical_and.reduce([np.isfinite(ep.speech).all(axis=-1) for ep in source.epochs.values()])
        for trial, trial_id in enumerate(reference.events.original_trial_id):
            for column, channel in enumerate(reference.channel_names):
                valid[trial, column] &= (channel, int(trial_id)) not in self.selected_pairs
        return dict(zip(reference.channel_names, map(int, valid.sum(axis=0))))

    def _refresh_selection(self):
        if not self.pair_variables or self._closed:
            return
        self._syncing = True
        try:
            for pair, variable in self.pair_variables.items():
                variable.set(pair in self.selected_pairs)
                item = self._items[pair]
                values = list(self.table.item(item, "values"))
                values[0] = "[x]" if pair in self.selected_pairs else "[ ]"
                self.table.item(item, values=values)
        finally:
            self._syncing = False
        counts = self.remaining_counts()
        self.counts_variable.set("Remaining valid trials (all bands): " + "; ".join(f"{ch}: {count}" for ch, count in counts.items()))
        self.widgets["undo"].configure(state="normal" if self._history else "disabled")

    def _variable_changed(self, pair):
        if not self._syncing and not self._closed:
            self.set_pair(pair, self.pair_variables[pair].get())

    def _table_click(self, event):
        item = self.table.identify_row(event.y)
        if item and self.table.identify_column(event.x) == "#1":
            pair = self._item_pairs[item]
            self.set_pair(pair, pair not in self.selected_pairs)

    def _table_selected(self, _event=None):
        selected = self.table.selection()
        if selected:
            self.display_pair(self._item_pairs[selected[0]])

    def display_pair(self, pair):
        pair = _validated_pairs(self.extraction, [pair])[0]
        self._current_pair = pair
        if not hasattr(self, "axis") or self._closed:
            return
        row = self._rows[pair]
        times = np.asarray(self.raw.times)
        mask = (times >= row.detection_start_seconds) & (times <= row.detection_stop_seconds)
        signal = self.raw.get_data()[self.raw.ch_names.index(pair[0]), mask]
        self.axis.clear()
        self.axis.plot(times[mask] - row.onset, signal * 1e6, color="black", linewidth=.8)
        self.axis.axvspan(0, row.offset - row.onset, color="seagreen", alpha=.12)
        self.axis.axvline(0, color="seagreen", linewidth=1)
        self.axis.axvline(row.offset - row.onset, color="firebrick", linewidth=1)
        self.axis.set(xlabel="Seconds from onset", ylabel="Raw ECoG (microvolts)",
                      title=f"{pair[0]} | Original trial {pair[1]}")
        self.axis.grid(alpha=.15)
        self.widgets["pair_checkbox"].configure(text=f"Exclude {pair[0]} / {pair[1]}",
                                                 variable=self.pair_variables[pair])
        self.details_variable.set(row.reasons or "No candidate flags")
        self.canvas.draw_idle()

    def confirm(self):
        validate_trial_report(self.raw, self.extraction, self.report)
        pairs = _validated_pairs(self.extraction, self.selected_pairs)
        self.selection = TrialExclusions(pairs, provenance=self.provenance)
        self._close()

    def _confirm_command(self):
        try:
            self.confirm()
        except ValueError as error:
            self.error_variable.set(str(error))

    def cancel(self):
        self.selection = None
        self.selected_pairs.clear()
        self._close()

    def _close(self):
        if self._closed:
            return
        self._closed = True
        for variable, token in self._traces:
            variable.trace_remove("write", token)
        self._traces.clear()
        if hasattr(self, "canvas"):
            widget = self.canvas.get_tk_widget()
            for name in ("_idle_draw_id", "_event_loop_id"):
                timer = getattr(self.canvas, name, None)
                if timer is not None:
                    widget.after_cancel(timer)
                    setattr(self.canvas, name, None)
            widget.destroy()
            self.figure.clear()
        if hasattr(self, "root"):
            self.root.grab_release()
            self.root.destroy()

    def show(self) -> tuple[tuple[str, int], ...] | None:
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        validate_trial_report(self.raw, self.extraction, self.report)
        self.selection = None
        self._closed = False
        parent = tk._default_root
        self.root = tk.Toplevel(parent) if parent is not None else tk.Tk()
        self.root.title("Four-band trial quality review")
        self.root.geometry("1280x850")
        self.root.minsize(900, 650)
        self.root.protocol("WM_DELETE_WINDOW", self.cancel)
        if parent is not None:
            self.root.transient(parent)
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=3)
        outer.rowconfigure(2, weight=2)
        buttons = ttk.Frame(outer)
        buttons.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(buttons, text="Select candidates", command=self.select_candidates).pack(side="left")
        ttk.Button(buttons, text="Clear", command=self.clear_selection).pack(side="left", padx=6)
        self.widgets["undo"] = ttk.Button(buttons, text="Undo", command=self.undo_selection)
        self.widgets["undo"].pack(side="left")
        ttk.Button(buttons, text="Cancel", command=self.cancel).pack(side="right")
        self.widgets["confirm"] = ttk.Button(buttons, text="Confirm exclusions", command=self._confirm_command)
        self.widgets["confirm"].pack(side="right", padx=6)
        frame = ttk.Frame(outer)
        frame.grid(row=1, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        columns = ("exclude", "channel", "trial", "candidate", "raw_bad", "spike", "jump", "flat", "clip", "nonfinite", "reasons")
        self.table = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        for name, label, width in zip(columns,
                ("Exclude", "Channel", "Original ID", "Candidate", "Raw bad", "Spike (MAD)", "Jump (MAD)",
                 "Flat (s)", "Extreme (%)", "Nonfinite", "Reasons / baseline flags"),
                (65, 80, 85, 75, 65, 100, 100, 80, 95, 80, 420)):
            self.table.heading(name, text=label)
            self.table.column(name, width=width, minwidth=width, stretch=name == "reasons")
        self.table.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(frame, orient="vertical", command=self.table.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(frame, orient="horizontal", command=self.table.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.table.tag_configure("candidate", foreground="firebrick")
        self._items, self._item_pairs = {}, {}
        self.pair_variables = {}
        for index, (pair, row) in enumerate(self._rows.items()):
            item = str(index)
            self._items[pair] = item
            self._item_pairs[item] = pair
            self.table.insert("", "end", iid=item, values=("[ ]", pair[0], pair[1],
                "yes" if row.candidate else "", "yes" if row.raw_bad else "", f"{row.spike_z:.2f}",
                f"{row.jump_z:.2f}", f"{row.flat_run_seconds:.3f}", f"{100 * row.clipping_fraction:.1f}",
                row.nonfinite_count, row.reasons), tags=("candidate",) if row.candidate else ())
            variable = tk.BooleanVar(master=self.root, value=pair in self.selected_pairs)
            self.pair_variables[pair] = variable
            token = variable.trace_add("write", lambda *_, p=pair: self._variable_changed(p))
            self._traces.append((variable, token))
        self.table.bind("<Button-1>", self._table_click)
        self.table.bind("<<TreeviewSelect>>", self._table_selected)
        plot = ttk.Frame(outer)
        plot.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        self.figure = Figure(figsize=(10.5, 3.2))
        self.figure.subplots_adjust(left=.09, right=.98, bottom=.22, top=.86)
        self.axis = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, plot)
        self.widgets["pair_checkbox"] = ttk.Checkbutton(outer)
        self.widgets["pair_checkbox"].grid(row=3, column=0, sticky="w", pady=4)
        self.details_variable = tk.StringVar(master=self.root)
        ttk.Label(outer, textvariable=self.details_variable, wraplength=850).grid(row=4, column=0, sticky="ew")
        self.counts_variable = tk.StringVar(master=self.root)
        ttk.Label(outer, textvariable=self.counts_variable, wraplength=850).grid(row=5, column=0, sticky="ew", pady=4)
        self.error_variable = tk.StringVar(master=self.root)
        ttk.Label(outer, textvariable=self.error_variable, foreground="firebrick", wraplength=850).grid(row=6, column=0, sticky="ew")
        self._refresh_selection()
        first_pair = next(iter(self._rows))
        self.table.selection_set(self._items[first_pair])
        self.display_pair(first_pair)
        self.root.grab_set()
        try:
            if parent is None:
                self.root.mainloop()
            else:
                self.root.wait_window()
        finally:
            if not self._closed:
                self.cancel()
        return self.selection
