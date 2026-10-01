"""Channel x trial beta-pattern stability analysis."""

from .stability_metrics import compute_channel_stability
from .trial_type_classifier import TrialTypeClassifierConfig, classify_trial_types
from .trial_type_features import TrialTypeFeatureConfig, extract_trial_type_features
from .pipeline import ChannelXTrialAnalysisResult, run_channelxtrial_analysis

__all__ = [
    "ChannelXTrialAnalysisResult",
    "TrialTypeClassifierConfig",
    "TrialTypeFeatureConfig",
    "classify_trial_types",
    "compute_channel_stability",
    "extract_trial_type_features",
    "run_channelxtrial_analysis",
]
