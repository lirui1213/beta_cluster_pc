"""Rule-based Type 1/Type 2 scoring for channel x trial beta patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TrialTypeClassifierConfig:
    """Weights and thresholds for Type 1/Type 2 morphology classification."""

    onset_drop_weight: float = 0.20
    rebound_weight: float = 1.30
    sustained_low_weight: float = 1.30
    offset_drop_weight: float = 0.45
    post_recovery_weight: float = 0.35
    slope_audio_weight: float = 0.15
    decision_margin: float = 0.35
    min_common_support: float = -1.25
    min_finite_fraction: float = 0.85
    probability_temperature: float = 1.0


def _robust_z(values: pd.Series, mask: NDArray[np.bool_]) -> FloatArray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    result = np.full_like(numeric, np.nan, dtype=float)
    reference = numeric[mask & np.isfinite(numeric)]
    if reference.size == 0:
        return result
    median = float(np.median(reference))
    mad = float(np.median(np.abs(reference - median)))
    if mad > np.finfo(float).eps:
        scale = 1.4826 * mad
    else:
        std = float(np.std(reference, ddof=0))
        scale = std if std > np.finfo(float).eps else 1.0
    result[np.isfinite(numeric)] = (numeric[np.isfinite(numeric)] - median) / scale
    return result


def _sigmoid(values: FloatArray) -> FloatArray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def classify_trial_types(
    feature_table: pd.DataFrame,
    *,
    config: TrialTypeClassifierConfig | None = None,
) -> pd.DataFrame:
    """Classify every channel-trial row as Type1, Type2, or Uncertain.

    The classifier is intentionally morphology-based. It uses robustly scaled
    features so the decision focuses on shape differences within the block
    rather than absolute beta-power magnitude.
    """

    cfg = config or TrialTypeClassifierConfig()
    required = {
        "channel",
        "is_bad",
        "trial_index",
        "onset_drop",
        "mid_speech_rebound",
        "speech_low_maintenance",
        "offset_drop",
        "post_offset_recovery",
        "beta_slope_audio_slope_corr",
        "finite_fraction",
    }
    missing = sorted(required - set(feature_table.columns))
    if missing:
        raise ValueError(f"Missing required trial type features: {missing}")
    if cfg.probability_temperature <= 0:
        raise ValueError("probability_temperature must be positive.")

    result = feature_table.copy()
    usable = ~result["is_bad"].fillna(False).to_numpy(dtype=bool)
    z_columns: Mapping[str, str] = {
        "onset_drop": "z_onset_drop",
        "mid_speech_rebound": "z_mid_speech_rebound",
        "speech_low_maintenance": "z_speech_low_maintenance",
        "offset_drop": "z_offset_drop",
        "post_offset_recovery": "z_post_offset_recovery",
        "beta_slope_audio_slope_corr": "z_beta_slope_audio_slope_corr",
    }
    for raw_column, z_column in z_columns.items():
        result[z_column] = _robust_z(result[raw_column], usable)

    slope_audio = result["z_beta_slope_audio_slope_corr"].to_numpy(float)
    slope_audio = np.nan_to_num(slope_audio, nan=0.0)
    z_onset = result["z_onset_drop"].to_numpy(float, copy=True)
    z_rebound = result["z_mid_speech_rebound"].to_numpy(float, copy=True)
    z_low = result["z_speech_low_maintenance"].to_numpy(float, copy=True)
    z_offset = result["z_offset_drop"].to_numpy(float, copy=True)
    z_recovery = result["z_post_offset_recovery"].to_numpy(float, copy=True)
    for values in (z_onset, z_rebound, z_low, z_offset, z_recovery):
        values[~np.isfinite(values)] = 0.0

    common_support = 0.5 * z_onset + 0.5 * z_recovery
    type1_score = (
        cfg.onset_drop_weight * z_onset
        + cfg.rebound_weight * z_rebound
        - 0.55 * cfg.sustained_low_weight * z_low
        + cfg.offset_drop_weight * z_offset
        + cfg.post_recovery_weight * z_recovery
        + cfg.slope_audio_weight * slope_audio
    )
    type2_score = (
        cfg.onset_drop_weight * z_onset
        - 0.75 * cfg.rebound_weight * z_rebound
        + cfg.sustained_low_weight * z_low
        - 0.20 * cfg.offset_drop_weight * z_offset
        + cfg.post_recovery_weight * z_recovery
        - 0.25 * cfg.slope_audio_weight * slope_audio
    )
    delta = type1_score - type2_score
    confidence = np.abs(delta)
    type1_probability = _sigmoid(delta / cfg.probability_temperature)

    labels = np.where(delta >= 0.0, "Type1", "Type2").astype(object)
    uncertain_reason = np.full(len(result), "", dtype=object)
    finite_fraction = pd.to_numeric(result["finite_fraction"], errors="coerce").to_numpy(float)
    low_quality = finite_fraction < cfg.min_finite_fraction
    low_margin = confidence < cfg.decision_margin
    low_common = common_support < cfg.min_common_support
    labels[low_margin | low_quality | low_common] = "Uncertain"
    labels[result["is_bad"].fillna(False).to_numpy(dtype=bool)] = "Bad"
    uncertain_reason[low_margin] = "low_margin"
    uncertain_reason[low_quality] = "low_finite_fraction"
    uncertain_reason[low_common] = "low_common_type_support"
    uncertain_reason[labels == "Bad"] = "bad_channel"

    result["type1_score"] = type1_score
    result["type2_score"] = type2_score
    result["type_delta"] = delta
    result["type1_probability"] = type1_probability
    result["confidence"] = confidence
    result["common_type_support"] = common_support
    result["trial_type"] = labels
    result["uncertain_reason"] = uncertain_reason
    result["type_code"] = result["trial_type"].map(
        {"Bad": -1, "Uncertain": 0, "Type1": 1, "Type2": 2}
    ).astype(int)
    return result


__all__ = ["TrialTypeClassifierConfig", "classify_trial_types"]
