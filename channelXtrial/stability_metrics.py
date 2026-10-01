"""Channel-level stability summaries for trial-wise beta pattern labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _switch_count(labels: list[str]) -> int:
    if len(labels) <= 1:
        return 0
    return int(sum(left != right for left, right in zip(labels[:-1], labels[1:], strict=True)))


def _type1_fraction(labels: pd.Series) -> float:
    certain = labels[labels.isin(["Type1", "Type2"])]
    if certain.empty:
        return np.nan
    return float(np.mean(certain == "Type1"))


def compute_channel_stability(classified_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether each channel keeps or switches its trial-wise type."""

    required = {"channel", "trial_index", "trial_type", "confidence", "is_bad"}
    missing = sorted(required - set(classified_table.columns))
    if missing:
        raise ValueError(f"Missing required classified columns: {missing}")

    rows: list[dict[str, object]] = []
    for channel, group in classified_table.groupby("channel", sort=False):
        ordered = group.sort_values("trial_index").reset_index(drop=True)
        is_bad = bool(ordered["is_bad"].fillna(False).iloc[0])
        labels = ordered["trial_type"].astype(str)
        certain = labels[labels.isin(["Type1", "Type2"])]
        counts = certain.value_counts()
        if counts.empty:
            dominant_type = "Bad" if is_bad else "Uncertain"
            dominant_count = 0
        else:
            dominant_type = str(counts.idxmax())
            dominant_count = int(counts.max())
        n_trials = int(len(ordered))
        n_certain = int(len(certain))
        uncertain_count = int(np.count_nonzero(labels == "Uncertain"))
        type1_count = int(np.count_nonzero(labels == "Type1"))
        type2_count = int(np.count_nonzero(labels == "Type2"))
        certain_sequence = certain.tolist()
        switches = _switch_count(certain_sequence)
        midpoint = n_trials // 2
        early_type1 = _type1_fraction(labels.iloc[:midpoint])
        late_type1 = _type1_fraction(labels.iloc[midpoint:])
        rows.append(
            {
                "channel": channel,
                "is_bad": is_bad,
                "n_valid_trials": n_trials,
                "n_certain_trials": n_certain,
                "type1_count": type1_count,
                "type2_count": type2_count,
                "uncertain_count": uncertain_count,
                "dominant_type": dominant_type,
                "stability_rate": dominant_count / n_trials if n_trials else np.nan,
                "stability_rate_certain_only": (
                    dominant_count / n_certain if n_certain else np.nan
                ),
                "switch_count": switches,
                "switch_rate": switches / max(n_certain - 1, 1),
                "mean_confidence": float(
                    pd.to_numeric(ordered["confidence"], errors="coerce").mean()
                ),
                "median_confidence": float(
                    pd.to_numeric(ordered["confidence"], errors="coerce").median()
                ),
                "uncertain_rate": uncertain_count / n_trials if n_trials else np.nan,
                "early_type1_fraction": early_type1,
                "late_type1_fraction": late_type1,
                "early_late_type1_shift": late_type1 - early_type1,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["is_bad", "dominant_type", "stability_rate", "switch_count"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


__all__ = ["compute_channel_stability"]
