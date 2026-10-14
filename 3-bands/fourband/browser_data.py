"""Read-only adapters for the local browser and its numerical worker.

No renderer or fitted estimator is constructed here. Detail images are rendered
with Agg in the numerical process; continuous curves use its existing result.
"""
from collections.abc import Mapping
from io import BytesIO
import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["session_metadata", "surface_payload", "continuous_payload",
           "render_detail_snapshot", "json_safe"]


def json_safe(value):
    """Return JSON-native copies, preserving nonfinite/missing values as null."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, pd.Series)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Real):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported browser JSON value: {type(value).__name__}")


def _aligned_result(result, channels):
    channels = tuple(channels)
    labels = np.asarray(result.labels)
    if (tuple(result.channels) != channels or len(set(channels)) != len(channels)
            or not channels or any(not isinstance(name, str) for name in channels)
            or labels.shape != (len(channels),) or labels.dtype.kind not in "iuf"
            or not np.isfinite(labels).all() or np.any(labels != np.round(labels))):
        raise ValueError("Result channel order and integer labels must align with data.")
    return channels, labels


def _aligned_extraction(request):
    extraction, result = request["extraction"], request["result"]
    if not extraction.epochs or set(extraction.epochs) != set(extraction.bands):
        raise ValueError("Extracted bands and epochs must align.")
    reference = next(iter(extraction.epochs.values()))
    channels, labels = _aligned_result(result, reference.channel_names)
    for band, epoch in extraction.epochs.items():
        if tuple(epoch.channel_names) != channels or not epoch.events.equals(reference.events):
            raise ValueError(f"{band} channel/trial alignment differs from extraction.")
    return extraction, channels, labels, reference


def _enabled(request):
    surface = not request.get("explorer_only", False) and bool(
        request.get("show_surface", False) or request.get("show_selectable", False))
    curves = bool(request.get("explorer_only", False) or request.get("link_explorer", True))
    return surface, curves


def _display_settings(request):
    supplied = request.get("explorer_settings") or {}
    settings = dict(trial_ids=supplied.get("trial_ids"), sigma_seconds=supplied.get("sigma_seconds", .10),
                    pre_seconds=supplied.get("pre_seconds", 1.), post_seconds=supplied.get("post_seconds", 1.),
                    display_sfreq=supplied.get("display_sfreq", 100.))
    for key in ("sigma_seconds", "pre_seconds", "post_seconds", "display_sfreq"):
        value = settings[key]
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0 or (key == "display_sfreq" and value == 0)):
            raise ValueError(f"Invalid display setting: {key}")
        settings[key] = float(value)
    pair = settings["trial_ids"]
    if pair is not None:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(
                isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral) for item in pair)):
            raise ValueError("trial_ids must be two integer original trial IDs.")
        settings["trial_ids"] = list(map(int, pair))
    return settings


def _identity(request, warnings):
    output = Path(request["output_dir"]) if request.get("output_dir") else None
    run_config, round_config = {}, {}
    if output is not None:
        for path, config in ((output.parent / "run_config.json", run_config),
                             (output / "round_config.json", round_config)):
            if path.is_file():
                try:
                    config.update(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, ValueError, TypeError) as error:
                    warnings.append(f"Cannot read display identity from {path.name}: {error}")
    source_identity = {}
    for source in (request.get("sources") or {}).values():
        if not isinstance(source, (str, Path)):
            continue
        parts = Path(source).parts
        for index, part in enumerate(parts):
            if part.lower() == "blocks" and 0 < index < len(parts) - 1:
                source_identity.update(subject=parts[index - 1], block=parts[index + 1])
                break
        if source_identity:
            break
    if output is not None:
        # Existing round outputs are subject/block/run/roundN_timestamp.
        name = output.name.split("_", 1)[0]
        if name in {"round1", "round2"}:
            source_identity["round"] = int(name[-1])
            if len(output.parents) >= 3:
                source_identity.setdefault("subject", output.parents[2].name)
                source_identity.setdefault("block", output.parents[1].name)
    identity = dict(source_identity, **run_config)
    identity.update(round_config)
    identity.update(request.get("metadata") or {})
    for key in ("subject", "block", "round"):
        if request.get(key) is not None:
            identity[key] = request[key]
    for key in ("subject", "block", "round"):
        if identity.get(key) is None:
            warnings.append(f"Display {key} is unavailable in request, saved configuration or source paths.")
    return {key: identity.get(key) for key in ("subject", "block", "round")}


def session_metadata(request):
    """Describe the trusted snapshot without loading recordings or fitting models."""
    extraction, channels, labels, reference = _aligned_extraction(request)
    surface, curves = _enabled(request)
    settings = _display_settings(request)
    warnings = []
    pairs = []
    if curves:
        from audio_cross_3band_utils import load_speech_events
        from .continuous import adjacent_trial_pairs

        sources = request["sources"]
        events = load_speech_events(sources["onset_path"], sources.get("offset_path"))
        pairs = adjacent_trial_pairs(events, reference.events).to_dict("records")
        if not pairs:
            raise ValueError("No adjacent retained original trial pairs available for inspection.")
        if settings["trial_ids"] is None:
            settings["trial_ids"] = [pairs[0]["first_trial_id"], pairs[0]["second_trial_id"]]
        if not any(settings["trial_ids"] == [pair["first_trial_id"], pair["second_trial_id"]] for pair in pairs):
            raise ValueError("Select a pair of adjacent retained original trial IDs.")
    elif settings["trial_ids"] is None:
        settings["trial_ids"] = []
    options = request.get("surface_options") or {}
    identity = _identity(request, warnings)
    return json_safe(dict(identity, channels=channels, labels=[int(label) for label in labels], bands=list(extraction.bands),
        pairs=pairs, settings=settings, surface_options=dict(brain_color=options.get("brain_color", "lightgray"),
        brain_opacity=options.get("brain_opacity", .38), point_size=options.get("point_size", 14.)),
        surface_enabled=surface, curves_enabled=curves, warnings=warnings))


def surface_payload(request):
    """Read real FreeSurfer geometry and aligned electrodes without VTK objects.

    A failed transform suppresses electrodes, with a warning, so scanner RAS
    points can never be presented against a tkRAS mesh by accident.
    """
    payload = dict(meshes=[], electrodes=[], warnings=[])
    if not _enabled(request)[0]:
        return payload
    from nibabel.freesurfer import read_geometry
    from visualization import load_electrode_coordinates, scanner_ras_to_tkras

    channels, labels = _aligned_result(request["result"], request["result"].channels)
    options = request.get("surface_options") or {}
    warnings = payload["warnings"]
    subjects_dir, subject = options.get("subjects_dir"), options.get("subject")
    root = Path(subjects_dir) / str(subject) if subjects_dir is not None and subject is not None else None
    if (subjects_dir is None) != (subject is None):
        warnings.append("Anatomy requires both subjects_dir and subject.")
    if root is None:
        warnings.append("FreeSurfer anatomy is unavailable; electrodes use Scanner RAS unless reference_mri is supplied.")
    else:
        for hemisphere in options.get("hemispheres", ("lh", "rh")):
            if hemisphere not in {"lh", "rh"}:
                raise ValueError("hemispheres may contain only 'lh' and 'rh'.")
            path = root / "surf" / f"{hemisphere}.pial"
            try:
                positions, indices = read_geometry(str(path))
                positions, indices = np.asarray(positions), np.asarray(indices)
                if (positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all()
                        or indices.ndim != 2 or indices.shape[1] != 3 or indices.dtype.kind not in "iu"
                        or np.any(indices < 0) or np.any(indices >= len(positions))):
                    raise ValueError("Invalid surface vertices or triangle indices.")
                payload["meshes"].append(dict(name=hemisphere, positions=positions.ravel(), indices=indices.ravel()))
            except (OSError, ValueError) as error:
                warnings.append(f"Cannot load anatomy {hemisphere}: {error}")
    csv = options.get("electrode_csv")
    if csv is None:
        warnings.append("Electrode coordinates are unavailable: no electrode_csv supplied.")
        return json_safe(payload)
    try:
        electrodes = load_electrode_coordinates(csv).set_index("channel")
    except (OSError, ValueError, KeyError) as error:
        warnings.append(f"Cannot load electrode coordinates: {error}")
        return json_safe(payload)
    for channel in electrodes.index:
        if channel not in channels:
            warnings.append(f"Unmapped coordinate channel {channel}: absent from fitted result.")
    valid = []
    for channel, label in zip(channels, labels):
        if channel not in electrodes.index:
            warnings.append(f"Missing electrode coordinates for {channel}.")
            continue
        position = electrodes.loc[channel, ["x", "y", "z"]].to_numpy(dtype=float)
        if not np.isfinite(position).all():
            warnings.append(f"Invalid nonfinite electrode coordinates for {channel}.")
            continue
        valid.append((channel, int(label), position))
    reference = options.get("reference_mri")
    if reference is None and root is not None:
        reference = root / "mri" / "orig.mgz"
    coordinates = np.array([item[2] for item in valid], dtype=float).reshape(-1, 3)
    if reference is not None:
        try:
            coordinates = scanner_ras_to_tkras(coordinates, reference)
            if not np.isfinite(coordinates).all():
                raise ValueError("Transformed coordinates contain nonfinite values.")
        except (OSError, ValueError, np.linalg.LinAlgError) as error:
            warnings.append(f"Electrode MRI transform failed: {error}")
            return json_safe(payload)
    for (channel, label, _), position in zip(valid, coordinates):
        payload["electrodes"].append(dict(channel=channel, label=label, position=position))
    return json_safe(payload)


def continuous_payload(data, result, channel):
    """Select a channel from completed numerical data using fixed explorer labels.

    QC flags are channel-specific annotations only. They never mask raw curves
    or change the original class population, including the excluded class -1.
    """
    from signal_processing import finite_mean

    channels, labels = _aligned_result(result, data.channels)
    if channel not in channels:
        raise ValueError(f"Unknown channel: {channel}")
    times, audio = np.asarray(data.times), np.asarray(data.audio)
    if times.ndim != 1 or audio.shape != times.shape or set(data.bands) != set(data.power):
        raise ValueError("Continuous band/time/audio alignment differs from data.")
    index = channels.index(channel)
    label = int(labels[index])
    members = labels == label
    bands = []
    for name in data.bands:
        curves = np.asarray(data.power[name])
        if curves.shape != (len(channels), len(times)):
            raise ValueError(f"{name} continuous channel/time alignment differs from data.")
        bands.append(dict(name=name, selected=curves[index], mean=finite_mean(curves[members])))
    excluded = {int(trial) for name, trial in data.metadata.get("trial_exclusions", ()) if name == channel}
    origin = float(data.metadata["first_onset_seconds"])
    events = [dict(original_trial_id=int(event.original_trial_id), onset=float(event.onset) - origin,
                   offset=float(event.offset) - origin, excluded=int(event.original_trial_id) in excluded)
              for event in data.events.itertuples()]
    return json_safe(dict(times=times, audio=audio, bands=bands, events=events, channel=channel,
        label=label, trial_ids=data.trial_ids, baseline_description=data.metadata["baseline_description"],
        metadata=data.metadata))


def render_detail_snapshot(request_path, settings):
    """Render the existing inspection plot to PNG bytes inside the numerical worker."""
    import joblib
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib import pyplot as plt
    from .inspection import plot_inspection

    request = joblib.load(request_path)
    extraction, channels, labels, _ = _aligned_extraction(request)
    channel = settings["channel"]
    band = settings["band"]
    if channel not in channels:
        raise ValueError(f"Unknown channel: {channel}")
    if band not in extraction.bands:
        raise ValueError(f"Unknown inspection band: {band}")
    figure = Figure(figsize=(12, 10), layout="constrained")
    FigureCanvasAgg(figure)
    try:
        plot_inspection(extraction, request["result"], band, channel=channel,
            label=int(labels[channels.index(channel)]), sigma_seconds=settings["sigma_seconds"],
            n_bootstrap=0, figure=figure)
        output = BytesIO()
        figure.savefig(output, format="png", dpi=130)
        return output.getvalue()
    finally:
        plt.close(figure)
        figure.clear()
