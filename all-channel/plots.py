"""Descriptive plots for pooled block/channel samples, without stability claims."""
import numpy as np
from matplotlib import pyplot as plt
from visualization import cluster_color


def save_global_plots(assignments, speech_curves, progress, output):
    labels = sorted(assignments.loc[assignments.label >= 0, "label"].unique())
    fig, axes = plt.subplots(len(labels), 1, figsize=(9, 3 * len(labels)), squeeze=False)
    stats = []
    for ax, label in zip(axes[:, 0], labels):
        curves = speech_curves[assignments.label.to_numpy() == label]
        n = np.isfinite(curves).sum(axis=0)
        mean = np.divide(np.nansum(curves, axis=0), n, out=np.full(len(progress), np.nan), where=n > 0)
        centered = np.where(np.isfinite(curves), curves - mean, 0)
        variance = np.divide((centered ** 2).sum(axis=0), n - 1, out=np.full(len(progress), np.nan), where=n > 1)
        sem = np.sqrt(variance / np.maximum(n, 1))
        color = cluster_color(int(label))
        ax.plot(progress, mean, color=color, label=f"Cluster {label} (n={len(curves)})")
        ax.fill_between(progress, mean - sem, mean + sem, alpha=.2, color=color)
        ax.axhline(0, color="gray", linewidth=.8)
        ax.set(xlabel="Speech progress (%)", ylabel="Normalized beta change", xlim=(0, 100))
        ax.legend()
        stats.append((label, mean, sem, n))
    fig.suptitle("Global beta curves: mean +/- sample SEM (descriptive)")
    fig.tight_layout()
    fig.savefig(output / "global_beta_curves.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in labels:
        part = assignments[assignments.label == label]
        ax.scatter(part.pc1, part.pc2, s=12, alpha=.65, color=cluster_color(int(label)), label=f"Cluster {label}")
    ax.set(xlabel="PC1", ylabel="PC2", title="Global weighted feature PCA")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "pca_clusters.png", dpi=180)
    plt.close(fig)
    clean = assignments[assignments.label >= 0].copy()
    clean["subject_block"] = clean.subject_id + "/" + clean.block_id
    for name, filename in [("subject_id", "cluster_subject_counts"), ("subject_block", "cluster_block_counts")]:
        counts = clean.groupby([name, "label"]).size().unstack(fill_value=0)
        counts.to_csv(output / (filename + ".csv"), encoding="utf-8-sig")
        fig, ax = plt.subplots(figsize=(10, max(4, len(counts) * .3)))
        counts.plot.barh(stacked=True, ax=ax, color=[cluster_color(int(c)) for c in counts.columns])
        ax.set(xlabel="Block/channel samples", ylabel=name)
        fig.tight_layout()
        fig.savefig(output / (filename + ".png"), dpi=180)
        plt.close(fig)
    return stats
