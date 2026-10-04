"""Build the actual Tk review window hidden and exercise its Apply callback."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def collect_destroyed_tk_interpreters():
    # Hidden test event-loop closures keep a destroyed root in a cycle.
    # Collect it outside creation of the next Tcl interpreter.
    import gc
    gc.collect()
    yield
    gc.collect()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "all-channel"))


def test_real_popup_applies_manual_cluster_sample_and_k_after_auto_exclusion(monkeypatch):
    import tkinter as tk
    import pipeline
    from auto_prune import AutomaticPruning, show_pruned_review_window
    features = pd.DataFrame({"subject_id":["A"]*12, "block_id":["B01"]*12,
                             "channel":[f"E{i}" for i in range(12)], "sample_id":[f"A:B01:E{i}" for i in range(12)],
                             "is_bad":[False]*12, "beta_progress__early_mean":[-10,-9,-11,-10,0,1,-1,0,20,19,21,20]})
    first, _ = pipeline.fit_global(features,n_clusters=3)
    removed = int(first.labels[0])
    mask = first.labels == removed
    automatic = AutomaticPruning((removed,),mask,pd.DataFrame(),pd.DataFrame(),{})
    real_tk = tk.Tk
    observed = {}

    def create_hidden_root():
        root = real_tk()
        root.withdraw()
        root.grab_set = lambda: None
        def walk(widget):
            yield widget
            for child in widget.winfo_children():
                yield from walk(child)
        def apply_review():
            widgets = list(walk(root))
            checks = [w for w in widgets if w.winfo_class() == "TCheckbutton"]
            observed['cluster_texts'] = [w.cget('text') for w in checks]
            observed['manual_label'] = int(checks[0].cget('text').split()[1])
            checks[0].invoke()
            entry = next(w for w in widgets if w.winfo_class() == "TEntry")
            observed['manual_sample'] = features.sample_id[~mask].iloc[0]
            entry.insert(0, observed['manual_sample'])
            spin = next(w for w in widgets if w.winfo_class() == "TSpinbox")
            spin.delete(0,'end')
            spin.insert(0,'2')
            next(w for w in widgets if w.winfo_class() == "TButton" and w.cget('text')=='Apply').invoke()
        root.mainloop = apply_review
        return root

    monkeypatch.setattr(tk,'Tk',create_hidden_root)
    review = show_pruned_review_window(first,automatic,initial_second_k=3)
    assert len(observed['cluster_texts']) == 2
    assert not any(s.startswith(f"Cluster {removed} ") for s in observed['cluster_texts'])
    assert review.bad_clusters == (observed['manual_label'],)
    assert review.manual_bads == (observed['manual_sample'],)
    assert review.n_clusters == 2
    assert (first.labels[mask] == removed).all()


@pytest.mark.parametrize("apply",[True,False])
def test_notebook_runs_auto_screen_before_default_popup_and_blocks_cancelled_review(tmp_path,monkeypatch,apply):
    import tkinter as tk
    import nbformat
    import json
    from datetime import datetime
    import pipeline
    from auto_prune import screen_small_high_variance_clusters, show_pruned_review_window
    features = pd.DataFrame({"subject_id":["A"]*8, "block_id":["B01"]*8,
                             "channel":[f"E{i}" for i in range(8)], "sample_id":[f"A:B01:E{i}" for i in range(8)],
                             "is_bad":[False]*8, "beta_progress__early_mean":[-3,-2.9,-3.1,-3,3,2.9,3.1,3]})
    first,labels = pipeline.fit_global(features,n_clusters=2)
    analysis = pipeline.GlobalAnalysisResult(features,labels,first,tmp_path)
    real_tk = tk.Tk
    opened = []
    def create_hidden_root():
        root = real_tk()
        root.withdraw()
        root.grab_set = lambda: None
        def walk(w):
            yield w
            for child in w.winfo_children(): yield from walk(child)
        def finish():
            # Reports must already exist before the manual window event loop.
            assert list(tmp_path.glob('review_*/auto_cluster_screening.csv'))
            opened.append(True)
            if apply:
                next(w for w in walk(root) if w.winfo_class()=='TButton' and w.cget('text')=='Apply').invoke()
            else:
                root.destroy()
        root.mainloop = finish
        return root
    monkeypatch.setattr(tk,'Tk',create_hidden_root)
    nb = nbformat.read(Path(__file__).resolve().parents[1]/'all-channel/all_channel_pipeline.ipynb',as_version=4)
    review_cell = next(c.source for c in nb.cells if c.cell_type=='code' and '# 一次自动筛除' in c.source)
    second_cell = next(c.source for c in nb.cells if c.cell_type=='code' and 'SECOND_RUN_DIR =' in c.source)
    state={'first_analysis':analysis, 'RUN_DIR':tmp_path, 'datetime':datetime, 'json':json,
           'display':lambda value:None,'SECOND_N_CLUSTERS':2,
           'screen_small_high_variance_clusters':screen_small_high_variance_clusters,
           'show_pruned_review_window':show_pruned_review_window}
    if apply:
        exec(compile(review_cell,'notebook_review','exec'),state)
        assert state['review_ready']
        assert state['automatic_pruning'].policy['enabled']
        assert state['selected_second_k']==2
    else:
        with pytest.raises(RuntimeError,match='Apply'):
            exec(compile(review_cell,'notebook_review','exec'),state)
        assert not state['review_ready']
        with pytest.raises(RuntimeError):
            exec(compile(second_cell,'notebook_second','exec'),state)
        assert not list(tmp_path.glob('round2_*'))
    assert opened == [True]
