"""Run the analysis notebook without popups and export verification artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="ecog_lr_v1")
    parser.add_argument("--mode", choices=("inprocess", "kernel"), default="inprocess")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    notebook = json.loads((project / "ECoG_3band_clustering_pipeline.ipynb").read_text(encoding="utf-8"))
    output = project / "outputs" / "optimization_verification"
    output.mkdir(parents=True, exist_ok=True)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
            cell["source"] = "".join(cell["source"])
            cell["source"] = cell["source"].replace("OPEN_REVIEW_WINDOW = True", "OPEN_REVIEW_WINDOW = False")
            cell["source"] = cell["source"].replace("RUN_3D_RENDER = True", "RUN_3D_RENDER = False")
            cell["source"] = cell["source"].replace("RUN_SELECTABLE_CLUSTER_RENDER = True", "RUN_SELECTABLE_CLUSTER_RENDER = False")
            if args.mode == "inprocess":
                cell["source"] = cell["source"].replace("%matplotlib inline", "import matplotlib\nmatplotlib.use('Agg')")
            if "SUBJECT_ID =" in cell["source"]:
                cell["source"] += f"\nOUTPUT_DIR = Path({str(output)!r}) / SUBJECT_ID / BLOCK_ID\nOUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    notebook["cells"].append({"cell_type": "code", "id": "verification", "metadata": {}, "execution_count": None, "outputs": [], "source": """
import json
from visualization import _concat_beta_progress_display
valid_baselines = baseline_quality['valid'].to_numpy().reshape(beta_epochs.pre.shape[:2])
means = np.mean(beta_epochs.pre, axis=-1)[valid_baselines]
stds = np.std(beta_epochs.pre, axis=-1)[valid_baselines]
assert np.max(np.abs(means)) < 1e-8
assert np.allclose(stds, 1.0, atol=1e-8)
axis, _, _, _ = _concat_beta_progress_display(beta_epochs, display_sfreq=DISPLAY_SFREQ)
assert np.allclose(np.diff(axis), np.diff(axis)[0], atol=1e-12)
assert np.all(second_result.labels[combined_bad_mask] == -1)
assert {'baseline_invalid_fraction', 'valid_trial_count'}.isdisjoint(second_result.feature_columns)
if stability is not None:
    assert np.array_equal(stability.clean_mask, second_result.clean_mask)
    assert selected_second_k == stability.recommended_k if AUTO_SELECT_K else True
metrics = {
    'subject': SUBJECT_ID, 'block': BLOCK_ID, 'trials': len(beta_epochs.events),
    'channels': len(beta_epochs.channel_names),
    'bad_channels': int(combined_bad_mask.sum()),
    'invalid_baselines': int((~baseline_quality['valid']).sum()),
    'baseline_mean_max_abs': float(np.max(np.abs(means))),
    'baseline_std_min': float(stds.min()), 'baseline_std_max': float(stds.max()),
    'display_step_seconds': float(np.diff(axis)[0]),
    'selected_features': len(second_result.feature_columns),
    'selected_k': int(selected_second_k),
    'cluster_sizes': {str(int(k)): int(v) for k, v in second_assignments.label.value_counts().items()},
}
(OUTPUT_DIR / 'verification_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
fig_ci, _ = plot_beta_cluster_grid(
    beta_epochs, second_result.labels, sigma_seconds=BETA_CURVE_SMOOTH_SIGMA_SECONDS,
    display_sfreq=DISPLAY_SFREQ, uncertainty='trial_bootstrap',
    bootstrap_iterations=CURVE_BOOTSTRAP_ITERATIONS, random_state=RANDOM_STATE,
)
fig_ci.savefig(OUTPUT_DIR / 'beta_trial_bootstrap_ci.png', dpi=200, bbox_inches='tight')
print(json.dumps(metrics, indent=2))
"""})
    try:
        import nbformat
    except ImportError:
        nbformat = None
    if nbformat is not None:
        nbformat.validate(notebook)
    from IPython.core.interactiveshell import InteractiveShell
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            transformed = InteractiveShell.instance().input_transformer_manager.transform_cell(cell["source"])
            compile(transformed, f"cell_{index}", "exec")
    print(f"Executing notebook in {args.mode} mode; artifacts: {output}", flush=True)
    try:
        if args.mode == "kernel":
            from nbclient import NotebookClient
            notebook = nbformat.from_dict(notebook)
            client = NotebookClient(notebook, timeout=1200, kernel_name=args.kernel, resources={"metadata": {"path": str(project)}})
            client.execute()
        else:
            from IPython.utils.capture import capture_output
            shell = InteractiveShell.instance()
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                print(f"Running code cell at notebook index {index}", flush=True)
                with capture_output() as captured:
                    result = shell.run_cell(cell["source"], store_history=True)
                cell["execution_count"] = result.execution_count
                if captured.stdout:
                    cell["outputs"].append({"output_type": "stream", "name": "stdout", "text": captured.stdout})
                if captured.stderr:
                    cell["outputs"].append({"output_type": "stream", "name": "stderr", "text": captured.stderr})
                for rich in captured.outputs:
                    cell["outputs"].append({"output_type": "display_data", "data": rich.data, "metadata": rich.metadata})
                result.raise_error()
    finally:
        (output / "verified_pipeline.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            for block in cell["outputs"]:
                if block["output_type"] == "stream":
                    print(block["text"], end="")
    print("Notebook execution and numerical assertions passed.", flush=True)


if __name__ == "__main__":
    main()
