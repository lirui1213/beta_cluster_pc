"""Execute a verification copy of the notebook without opening GUI windows."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-mode", choices=("file", "pre_onset"), default="file")
    parser.add_argument("--subject")
    parser.add_argument("--block")
    parser.add_argument("--first-k", type=int)
    parser.add_argument("--second-k", type=int)
    parser.add_argument("--manual-bad", action="append", default=[])
    parser.add_argument("--distance-threshold", type=float)
    parser.add_argument("--stability-resamples", type=int, default=5)
    parser.add_argument("--weight-preset", choices=("legacy", "balanced", "morphology", "amplitude"), default="legacy")
    parser.add_argument("--exclude-trial", action="append", default=[], metavar="CHANNEL:ID")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    project = folder.parent
    os.chdir(project)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(project))
    notebook = json.loads((folder / "ECoG_4band_clustering_pipeline.ipynb").read_text(encoding="utf-8"))
    output = folder / "outputs" / "verification" / args.baseline_mode / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        cell["execution_count"] = None
        cell["outputs"] = []
        if "SUBJECT_ID =" in source:
            overrides = {"BASELINE_MODE": args.baseline_mode, "RUN_3D_RENDER": False,
                         "RUN_SELECTABLE_CLUSTER_RENDER": False}
            for key, value in (("SUBJECT_ID", args.subject), ("BLOCK_ID", args.block),
                               ("FIRST_N_CLUSTERS", args.first_k), ("SECOND_N_CLUSTERS", args.second_k)):
                if value is not None:
                    overrides[key] = value
            # Replace only named assignments in the parameter cell, before dependent paths.
            lines = source.splitlines()
            for i, line in enumerate(lines):
                key = line.partition(" =")[0]
                if key in overrides:
                    lines[i] = f"{key} = {overrides[key]!r}"
            source = "\n".join(lines)
            source += f"\nOUTPUT_ROOT = Path({str(output)!r})\n"
        if "OPEN_REVIEW_WINDOW =" in source:
            source += ("\nOPEN_REVIEW_WINDOW = False\n"
                       f"review_settings = ReviewSettings(n_clusters=SECOND_N_CLUSTERS, manual_bad_channels={tuple(args.manual_bad)!r}, "
                       f"pca_distance_threshold={args.distance_threshold!r})\nreview_model = first_result\n")
        if "WEIGHT_PRESET =" in source:
            source += (f"\nWEIGHT_PRESET = {args.weight_preset!r}\nOPEN_TRIAL_QC_WINDOW = False\n"
                "OPEN_EXPLORER = False\nSTABILITY_PROGRESS_WINDOW = False\n"
                f"STABILITY_RESAMPLES = {args.stability_resamples}\nINSPECTION_BOOTSTRAP_DRAWS = 20\n")
        if "selected_trial_pairs =" in source and args.exclude_trial:
            pairs = [(item.rsplit(":", 1)[0], int(item.rsplit(":", 1)[1])) for item in args.exclude_trial]
            source = source.replace("selected_trial_pairs = ()", f"selected_trial_pairs = {tuple(pairs)!r}")
        cell["source"] = source
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    import matplotlib
    matplotlib.use("Agg")
    shell = InteractiveShell.instance()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile(shell.input_transformer_manager.transform_cell(cell["source"]), f"cell_{index}", "exec")
    print(f"Verification artifacts: {output}", flush=True)
    try:
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            print(f"Running notebook index {index}", flush=True)
            with capture_output() as captured:
                result = shell.run_cell(cell["source"], store_history=True)
            cell["execution_count"] = result.execution_count
            for name, value in (("stdout", captured.stdout), ("stderr", captured.stderr)):
                if value:
                    cell["outputs"].append({"output_type": "stream", "name": name, "text": value})
                    print(value, end="", flush=True)
            # Preserve text tables; large figure PNGs already live in the round directories.
            for rich in captured.outputs:
                if "text/plain" in rich.data:
                    cell["outputs"].append({"output_type": "display_data", "data": {"text/plain": rich.data["text/plain"]}, "metadata": {}})
            result.raise_error()
        ns = shell.user_ns
        import numpy as np
        from visualization import _concat_beta_progress_display
        extraction = ns["extraction"]
        first, second = ns["first_result"], ns["second_result"]
        metrics = {"subject": ns["SUBJECT_ID"], "block": ns["BLOCK_ID"], "baseline_mode": args.baseline_mode,
                   "bands": extraction.bands, "baseline": {}, "rounds": {}, "run_dir": str(ns["RUN_DIR"])}
        assert set(extraction.bands) == {"theta", "beta1", "beta2", "high_gamma"}
        for band, epochs in extraction.epochs.items():
            reference = epochs.baseline_reference
            valid = reference.loc[reference.valid]
            assert len(valid), f"No valid {band} baselines"
            mean_error = float(valid.normalized_mean.abs().max())
            std_error = float((valid.normalized_std - 1).abs().max())
            assert mean_error < 1e-8 and std_error < 1e-8
            axis, _, _, _ = _concat_beta_progress_display(epochs, display_sfreq=ns["DISPLAY_SFREQ"])
            assert np.allclose(np.diff(axis), np.diff(axis)[0], atol=1e-12)
            metrics["baseline"][band] = dict(trials=len(epochs.events), mean_error=mean_error, std_error=std_error,
                invalid_pairs=int((~epochs.baseline_diagnostics.valid).sum()), display_step_seconds=float(np.diff(axis)[0]))
        assert first.scaler is not second.scaler
        assert np.all(second.labels[ns["review_audit"].excluded_final] == -1)
        for name, fitted, directory, k in (("first", first, ns["ROUND1_DIR"], ns["FIRST_N_CLUSTERS"]),
                                           ("second", second, ns["ROUND2_DIR"], ns["selected_second_k"])):
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            assert summary["requested_k"] == k and not summary["stability_evaluated"]
            assert set(summary["band_weight_budgets"]) == set(extraction.bands)
            for band, budget in summary["band_weight_budgets"].items():
                assert np.isclose(budget, ns["BAND_WEIGHTS"][band])
                assert (directory / f"{band}_audio_dynamics.png").stat().st_size > 1000
            assert not any(c.startswith("qc_") for c in fitted.feature_columns)
            from fourband.analysis import ECOG_FEATURE_SUFFIXES
            assert all(c.split("__", 1)[1] in ECOG_FEATURE_SUFFIXES for c in fitted.feature_columns)
            metrics["rounds"][name] = dict(summary, output=str(directory))
        continuous = ns["continuous_trials"]
        assert np.allclose(np.diff(continuous.times), 1 / ns["DISPLAY_SFREQ"], atol=1e-12)
        assert continuous.metadata["gap_seconds"] >= 0
        assert len(continuous.trial_ids) == 2
        for name, directory in ns["continuous_display_outputs"].items():
            assert (directory / "fourband_continuous_trials.png").stat().st_size > 1000
            display_metadata = json.loads((directory / "continuous_trial_metadata.json").read_text())
            assert display_metadata["trial_ids"] == list(continuous.trial_ids)
        metrics["continuous"] = dict(continuous.metadata,
            output={name: str(path) for name, path in ns["continuous_display_outputs"].items()})
        metrics["trial_qc"] = dict(candidates=int(ns["trial_quality_report"].candidate.sum()),
            confirmed=list(extraction.trial_exclusions), ready=ns["TRIAL_QC_READY"])
        metrics["diagnostics"] = {}
        for name, directory in ns["diagnostic_outputs"].items():
            summary = json.loads((directory / "stability_summary.json").read_text(encoding="utf-8"))
            assert summary["requested_k"] == (first if name == "first" else second).kmeans.n_clusters
            assert summary["n_included"] == int((first if name == "first" else second).clean_mask.sum())
            metrics["diagnostics"][name] = dict(summary, output=str(directory))
        metrics["cache"] = ns["POWER_CACHE"].info()
        (output / "verification_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2), flush=True)
        print("Notebook execution and numerical checks passed.", flush=True)
    finally:
        (output / "verified_pipeline.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
