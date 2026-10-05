"""Exclusive ECoG feature groups and auditable distance budgets."""
import numpy as np
import pandas as pd

GROUPS = ("morphology", "amplitude", "slope")
PRESETS = {"balanced": (1., 1., 1.), "morphology": (3., 1., 1.), "amplitude": (1., 3., 1.)}


def feature_group(column):
    suffix = column.split("__")[-1]
    if "slope" in suffix:
        return "slope"
    if suffix.startswith("energy_") or suffix in {
            "pre_mean", "post_mean", "post_minus_pre_mean", "post_std", "post_auc",
            "post_peak_abs", "positive_peak_value", "negative_peak_value", "articulation_mean"}:
        return "amplitude"
    return "morphology"


def resolve_group_budgets(preset="legacy", custom=None):
    if preset == "legacy":
        if custom is not None:
            raise ValueError("Legacy weights cannot be combined with custom group weights.")
        return None
    if preset in PRESETS:
        if custom is not None:
            raise ValueError("Choose custom preset to supply group_weights.")
        return dict(zip(GROUPS, PRESETS[preset]))
    if preset != "custom":
        raise ValueError("weight_preset must be legacy, balanced, morphology, amplitude or custom.")
    if custom is None or set(custom) - set(GROUPS):
        raise ValueError("Custom group weights need morphology/amplitude/slope only.")
    budgets = dict.fromkeys(GROUPS, 1.)
    budgets.update(custom)
    values = np.asarray(list(budgets.values()), dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not (values > 0).any():
        raise ValueError("Group weights must be finite, nonnegative, and not all zero.")
    return dict(zip(GROUPS, map(float, values)))


def group_feature_weights(columns, budgets, band_budget):
    present = {group: [c for c in columns if feature_group(c) == group] for group in GROUPS}
    scale = max((budgets[group] for group in GROUPS if present[group]), default=0.)
    if scale <= 0:
        raise ValueError("No active usable feature group in this band.")
    normalized = {group: budgets[group] / scale for group in GROUPS}
    active_total = sum(normalized[group] for group in GROUPS if present[group])
    if active_total <= 0:
        raise ValueError("No active usable feature group in this band.")
    return {c: float(band_budget * (normalized[group] / active_total / len(members)))
        for group, members in present.items() for c in members}


def weight_audit(result):
    records = []
    for band in dict.fromkeys(c.split("__")[0] for c in result.feature_columns):
        for group in GROUPS:
            members = [(c, w) for c, w in zip(result.feature_columns, result.feature_weights)
                if c.startswith(band + "__") and feature_group(c) == group]
            records.append(dict(band=band, group=group, feature_count=len(members),
                effective_budget=sum(w for _, w in members), features=";".join(c for c, _ in members)))
    return pd.DataFrame(records)
