"""Notebook state guards and optional cancellable diagnostic progress."""
import json


def ensure_fit_settings(current, fitted):
    if json.loads(json.dumps(current)) != json.loads(json.dumps(fitted)):
        raise ValueError("Classification settings changed; rerun round one before review, round two or display.")


def ensure_trial_review(current, reviewed):
    if json.loads(json.dumps(current)) != json.loads(json.dumps(reviewed)):
        raise ValueError("Changed trial quality settings require a fresh manual confirmation and refit.")


def close_explorers(explorers):
    for explorer in explorers.values():
        explorer.close()
    explorers.clear()


def run_stability(extraction, result, *, show_progress=True, **kwargs):
    from .stability import evaluate_stability
    if not show_progress:
        return evaluate_stability(extraction, result, **kwargs)
    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk()
    root.title("Cluster stability")
    root.geometry("440x140")
    root.resizable(False, False)
    cancelled = [False]
    status = tk.StringVar(value="Preparing trial resampling...")
    ttk.Label(root, textvariable=status, wraplength=410).pack(fill="x", padx=12, pady=12)
    bar = ttk.Progressbar(root, mode="indeterminate")
    bar.pack(fill="x", padx=12)
    bar.start(50)
    def cancel():
        cancelled[0] = True
        status.set("Cancelling after current fit...")
    ttk.Button(root, text="Cancel", command=cancel).pack(pady=10)
    root.protocol("WM_DELETE_WINDOW", cancel)
    def progress(*args):
        if not cancelled[0]:
            status.set("Resampling: " + " ".join(map(str, args)))
        root.update_idletasks()
        root.update()
    root.update()
    try:
        return evaluate_stability(extraction, result, progress=progress, cancelled=lambda: cancelled[0], **kwargs)
    finally:
        bar.stop()
        root.destroy()
