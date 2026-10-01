"""End-to-end runner for channel x trial beta Type 1/Type 2 analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from audio_cross_3band_utils import BetaProgressEpochs, build_beta_progress_feature_table_from_files

from .stability_metrics import compute_channel_stability
from .trial_type_classifier import TrialTypeClassifierConfig, classify_trial_types
from .trial_type_features import TrialTypeFeatureConfig, extract_trial_type_features
from .visualization import (
    plot_block_type_proportions,
    plot_channel_trial_curves,
    plot_channel_trial_heatmap,
    plot_channel_type_sequence,
)


@dataclass(frozen=True)
class ChannelXTrialAnalysisResult:
    """Tables and epochs produced by the channel x trial analysis."""

    beta_epochs: BetaProgressEpochs
    trial_features: pd.DataFrame
    classified_trials: pd.DataFrame
    channel_stability: pd.DataFrame
    output_dir: Path
    example_channel: str | int | None


def _choose_example_channel(stability: pd.DataFrame) -> str | None:
    if stability.empty:
        return None
    usable = stability[~stability["is_bad"].fillna(False)].copy()
    if usable.empty:
        return None
    usable = usable.sort_values(
        ["switch_count", "uncertain_rate", "mean_confidence"],
        ascending=[False, True, False],
    )
    return str(usable.iloc[0]["channel"])


def _resolve_example_channel(
    example_channel: str | int | None,
    channel_names: tuple[str, ...],
    stability: pd.DataFrame,
) -> str | None:
    if example_channel is None or str(example_channel).strip() == "":
        return _choose_example_channel(stability)
    raw = str(example_channel).strip()
    candidates = [raw]
    if raw.isdigit():
        candidates.append(f"ECoG_{int(raw)}")
        candidates.append(f"E{int(raw)}")
    for candidate in candidates:
        if candidate in channel_names:
            return candidate
    raise ValueError(
        "example_channel must be an existing channel name such as 'ECoG_124', "
        "or a numeric channel id such as 124."
    )


def run_channelxtrial_analysis(
    *,
    fif_path: str | Path,
    event_source: str | Path,
    wav_path: str | Path,
    offset_source: str | Path | None = None,
    output_dir: str | Path,
    picks: Any = "data",
    progress_points: int = 101,
    duration_mad_threshold: float = 2.5,
    min_trials_after_filter: int = 3,
    pre_post_window: float | None = None,
    feature_config: TrialTypeFeatureConfig | None = None,
    classifier_config: TrialTypeClassifierConfig | None = None,
    example_channel: str | int | None = None,
    max_display_trials: int | None = None,
) -> ChannelXTrialAnalysisResult:
    """Run the full Type 1/Type 2 channel x trial analysis and save outputs."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _, beta_epochs = build_beta_progress_feature_table_from_files(
        fif_path=fif_path,
        event_source=event_source,
        offset_source=offset_source,
        wav_path=wav_path,
        picks=picks,
        progress_points=progress_points,
        duration_mad_threshold=duration_mad_threshold,
        min_trials_after_filter=min_trials_after_filter,
        pre_post_window=pre_post_window,
    )
    trial_features = extract_trial_type_features(beta_epochs, config=feature_config)
    classified = classify_trial_types(trial_features, config=classifier_config)
    stability = compute_channel_stability(classified)

    trial_features.to_csv(output_path / "channel_trial_type_features.csv", index=False, encoding="utf-8-sig")
    classified.to_csv(output_path / "channel_trial_type_labels.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(output_path / "channel_type_stability.csv", index=False, encoding="utf-8-sig")
    beta_epochs.events.to_csv(output_path / "valid_trials.csv", index=False, encoding="utf-8-sig")
    beta_epochs.dropped_events.to_csv(output_path / "dropped_trials.csv", index=False, encoding="utf-8-sig")

    heatmap_figure, _ = plot_channel_trial_heatmap(classified, stability_table=stability)
    heatmap_figure.savefig(output_path / "channel_trial_type_heatmap.png", dpi=300, bbox_inches="tight")
    proportion_figure, _ = plot_block_type_proportions(classified)
    proportion_figure.savefig(output_path / "block_type_proportions.png", dpi=300, bbox_inches="tight")

    chosen_channel = _resolve_example_channel(
        example_channel,
        beta_epochs.channel_names,
        stability,
    )
    if chosen_channel is not None:
        sequence_figure, _ = plot_channel_type_sequence(classified, chosen_channel)
        sequence_figure.savefig(
            output_path / f"{chosen_channel}_type_sequence.png",
            dpi=300,
            bbox_inches="tight",
        )
        curve_figure, _ = plot_channel_trial_curves(
            beta_epochs,
            classified,
            chosen_channel,
            max_trials=len(beta_epochs.events) if max_display_trials is None else max_display_trials,
        )
        curve_figure.savefig(
            output_path / f"{chosen_channel}_trial_curves.png",
            dpi=300,
            bbox_inches="tight",
        )

    return ChannelXTrialAnalysisResult(
        beta_epochs=beta_epochs,
        trial_features=trial_features,
        classified_trials=classified,
        channel_stability=stability,
        output_dir=output_path,
        example_channel=chosen_channel,
    )


__all__ = ["ChannelXTrialAnalysisResult", "run_channelxtrial_analysis"]
