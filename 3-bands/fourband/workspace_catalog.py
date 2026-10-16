"""Contained workspace discovery and JSON-only analysis configuration.

Only the documented standardized recording/event/audio/baseline FIF names are
supported. Alternative recording paths are never guessed. Optional baseline
segment CSVs are discovered directly in baseline/ and require explicit IDs via
resolve(..., selections={"baseline_segments_csv": <selection_options ID>}).
History returns at most 100 shallow summaries, never models or arbitrary paths.
Saved output contexts remain discoverable when source data is offline.
"""
from copy import deepcopy
from itertools import islice
import json
import math
import os
from pathlib import Path
import re
from stat import S_ISREG


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,179}\Z")
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)),
             *(f"LPT{i}" for i in range(10))}
_BANDS = ("theta", "beta1", "beta2", "high_gamma")
_GROUPS = ("morphology", "amplitude", "slope")
_MAX_ENTRIES = 10000
_MAX_HISTORY = 100
_MAX_JSON_BYTES = 1024 * 1024
_DEFAULTS = dict(
    baseline_mode="file", baseline_window=[-1.3, -.3],
    baseline_edge_seconds=.3, baseline_relative_floor=.001,
    feature_sigma_seconds=.06, first_k=10, second_k=4,
    band_weights=dict.fromkeys(_BANDS, 1.), weight_preset="legacy", group_weights=None,
    correlation_threshold=.95, random_state=50, duration_mad_threshold=2.5,
    min_trials_after_filter=3, progress_points=101, pre_post_window=None,
    slope_method="savgol", max_baseline_invalid_fraction=.25,
    display_sigma_seconds=.1, display_sfreq=100.,
)


def _identifier(value, name):
    if not isinstance(value, str) or not _ID.fullmatch(value) or value.upper() in _RESERVED:
        raise ValueError(f"Invalid {name} catalog ID.")
    return value


def _contained(path, root):
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as error:
        raise ValueError("Cannot resolve path inside configured root.") from error
    if not resolved.is_relative_to(root):
        raise ValueError("Path is outside the configured root (link/junction escape).")
    return resolved


def _children(path):
    try:
        with os.scandir(path) as entries:
            return sorted((Path(item.path) for item in islice(entries, _MAX_ENTRIES)),
                          key=lambda item: item.name)
    except OSError:
        return []


def _directories(path, root):
    for child in _children(path):
        try:
            _identifier(child.name, "directory")
            if _contained(child, root).is_dir():
                yield child
        except ValueError:
            continue


class WorkspaceCatalog:
    def __init__(self, data_root, output_root):
        self.data_root = Path(data_root).resolve()
        self.output_root = Path(output_root).resolve()

    def _blocks(self, patient):
        blocks = _contained(patient / "blocks", self.data_root)
        result = []
        for directory in _directories(blocks, self.data_root):
            try:
                if _contained(directory / "standardized", self.data_root).is_dir():
                    result.append(directory.name)
            except ValueError:
                continue
        return result

    def patients(self):
        """Return sorted IDs from standardized data and known saved outputs."""
        contexts = {}
        for directory in _directories(self.data_root, self.data_root):
            try:
                blocks = self._blocks(directory)
            except ValueError:
                continue
            if blocks:
                contexts[directory.name] = set(blocks)
        for patient in _directories(self.output_root, self.output_root):
            for block in _directories(patient, self.output_root):
                try:
                    if self._history_summaries(patient.name, block.name, limit=1):
                        contexts.setdefault(patient.name, set()).add(block.name)
                except ValueError:
                    continue
        return [dict(id=subject, blocks=sorted(blocks)) for subject, blocks in sorted(contexts.items())]

    def _data_context(self, subject, block):
        # Match directory names exactly, even on case-insensitive Windows.
        patient = next((p for p in _directories(self.data_root, self.data_root)
                        if p.name == subject), None)
        if patient is None or block not in self._blocks(patient):
            raise ValueError("Unknown subject or standardized block catalog ID.")

    def context(self, subject, block, require_data=False):
        """Validate known catalog IDs and return only {subject, block}.

        Saved manifests or recognized legacy outputs establish a context
        independently of source availability. require_data=True additionally
        requires an actual standardized directory, but does not validate its
        input files. For history opening, check the run ID against history()
        after this call; context validation alone does not authorize any run.
        """
        _identifier(subject, "subject")
        _identifier(block, "block")
        if type(require_data) is not bool:
            raise ValueError("require_data must be a boolean.")
        _contained(self.output_root / subject / block, self.output_root)
        if not require_data:
            patient = next((p for p in _directories(self.output_root, self.output_root)
                            if p.name == subject), None)
            if patient is not None:
                known_block = any(p.name == block for p in _directories(patient, self.output_root))
                if known_block and self._history_summaries(subject, block, limit=1):
                    return dict(subject=subject, block=block)
        self._data_context(subject, block)
        return dict(subject=subject, block=block)

    def resolve(self, subject, block, selections=None):
        """Resolve trusted paths; missing files remain visible with error messages.

        Invalid IDs, unknown selections and root escapes raise ValueError.
        selection_options contains filename IDs, not selectable absolute paths.
        """
        self.context(subject, block)
        patient = _contained(self.data_root / subject, self.data_root)
        standardized = _contained(patient / "blocks" / block / "standardized", self.data_root)
        if selections is None:
            selections = {}
        if not isinstance(selections, dict) or set(selections) - {"baseline_segments_csv"}:
            raise ValueError("Unknown selections; only discovered baseline_segments_csv IDs are allowed.")
        source_paths = dict(
            fif_path=standardized / "ECoG/ecog_TEST_clean_car-raw.fif",
            onset_path=standardized / "events/onset.npy",
            offset_path=standardized / "events/offset.npy",
            wav_path=standardized / "audio/analog1.wav",
            baseline_fif=patient / "baseline/baseline.fif",
        )
        baseline_dir = _contained(patient / "baseline", self.data_root)
        candidates = {}
        for path in _children(baseline_dir):
            if (path.suffix.lower() == ".csv" and "segments" in path.stem.lower()
                    and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}\.csv", path.name)):
                safe = _contained(path, self.data_root)
                if safe.is_file():
                    candidates[path.name] = safe
        selected = selections.get("baseline_segments_csv")
        if selected is not None:
            if not isinstance(selected, str) or selected not in candidates:
                raise ValueError("baseline_segments_csv must be an explicitly discovered selection ID.")
            source_paths["baseline_segments_csv"] = candidates[selected]

        sources, availability, errors = {}, [], []

        def available(name, path, directory=False):
            safe = _contained(path, self.data_root)
            exists = safe.is_dir() if directory else safe.is_file()
            availability.append(dict(name=name, path=str(safe), exists=exists))
            return safe, exists

        for name, path in source_paths.items():
            safe, exists = available(name, path)
            sources[name] = str(safe)
            if not exists:
                qualifier = " (required in file baseline mode)" if name == "baseline_fif" else ""
                errors.append(f"Missing input {name}{qualifier}: {safe}")
        sources.setdefault("baseline_segments_csv", None)
        surface = dict(subject="recon")
        for name, path in dict(electrode_csv=patient / "recon/electrodes/electrodes.csv",
                               subjects_dir=patient, reference_mri=patient / "recon/mri/T1.mgz").items():
            safe, exists = available(name, path, directory=name == "subjects_dir")
            surface[name] = str(safe)
            if not exists:
                errors.append(f"Missing anatomy {name}; brain display unavailable: {safe}")
        surface["hemispheres"] = []
        for hemisphere in ("lh", "rh"):
            _, exists = available(f"{hemisphere}.pial", patient / "recon/surf" / f"{hemisphere}.pial")
            if exists:
                surface["hemispheres"].append(hemisphere)
        if not surface["hemispheres"]:
            errors.append("Missing anatomy pial surfaces; brain display unavailable.")
        return dict(subject=subject, block=block, sources=sources, surface_options=surface,
                    availability=availability, errors=errors,
                    selection_options={"baseline_segments_csv": [dict(id=name, label=name) for name in candidates]})

    def history(self, subject, block):
        """Return at most 100 newest folder IDs and bounded manifest summaries.

        Compatible workspace manifests have read_only=False. Legacy outputs,
        incomplete JSON and unsupported schemas are explicitly read-only;
        this flag alone does not promise that calculation snapshots can resume.
        """
        self.context(subject, block)
        return self._history_summaries(subject, block)

    def _history_summaries(self, subject, block, *, limit=_MAX_HISTORY):
        parent = _contained(self.output_root / subject / block, self.output_root)
        result = []
        for directory in reversed(list(_directories(parent, self.output_root))):
            summary = dict(id=directory.name, run_id=directory.name, subject=subject, block=block)
            try:
                manifest = _contained(directory / "workspace_manifest.json", self.output_root)
                if manifest.is_file():
                    try:
                        saved = _read_json(manifest)
                        if not isinstance(saved, dict):
                            raise ValueError("Invalid manifest object")
                    except (OSError, ValueError, RecursionError):
                        summary.update(kind="workspace", read_only=True, status="incomplete", stage="unknown")
                    else:
                        if (saved.get("subject") != subject or saved.get("block") != block
                                or saved.get("run_id") != directory.name):
                            continue
                        schema = saved.get("schema_version")
                        compatible = type(schema) is int and schema == 1
                        revision = saved.get("revision")
                        stage = saved.get("stage")
                        compatible = compatible and type(revision) is int and 0 <= revision <= 2**53
                        compatible = compatible and stage in (
                            "configured", "trial_review", "trial_confirmed", "first_ready", "second_ready")
                        summary.update(kind="workspace", read_only=not compatible,
                            schema_version=schema if type(schema) is int and 0 <= schema <= 2**31 - 1 else None,
                            stage=stage if compatible else "unknown",
                            revision=revision if compatible else None,
                            created_at=_short_string(saved.get("created_at"), 80))
                        task = saved.get("task")
                        status = task.get("status") if isinstance(task, dict) else None
                        summary["status"] = (status if status in ("queued", "running", "opening", "completed", "failed", "cancelled")
                                             else "unknown") if compatible else "incomplete"
                elif self._legacy(directory):
                    summary.update(kind="legacy", read_only=True, stage="legacy", status="legacy")
                else:
                    continue
            except ValueError:
                continue
            result.append(summary)
            if len(result) == limit:
                break
        return result

    def _legacy(self, directory):
        for name in ("run_config.json", "summary.json"):
            if _contained(directory / name, self.output_root).is_file():
                return True
        for child in _directories(directory, self.output_root):
            if child.name.startswith(("round1_", "round2_")):
                if _contained(child / "summary.json", self.output_root).is_file():
                    return True
        return False


def _short_string(value, maximum):
    return value[:maximum] if isinstance(value, str) else None


def _read_json(path):
    with path.open("rb") as stream:
        contents = stream.read(_MAX_JSON_BYTES + 1)
    if len(contents) > _MAX_JSON_BYTES:
        raise ValueError("Manifest exceeds size limit")

    def reject_constant(value):
        raise ValueError("Nonfinite JSON value")

    return json.loads(contents.decode("utf-8"), parse_constant=reject_constant)


def _number(value, name, *, minimum=0, maximum=None, positive=False):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number.")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if (not finite or value < minimum or (positive and value <= minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} is outside its finite valid range.")
    return value


def _integer(value, name, minimum, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer >= {minimum}" +
                         (f" and <= {maximum}." if maximum is not None else "."))


def validate_config(config, resolved):
    """Return detached JSON-safe defaults plus validated browser settings.

    config accepts only keys in _DEFAULTS. Identity, sources and anatomy are
    copied exclusively from the trusted catalog resolution. Missing anatomy
    is nonfatal; recording, events and audio are always required, and baseline
    FIF is required only for file mode. input_identity records server-observed
    size and nanosecond mtime for each existing source file, without reading
    file contents. Invalid configuration raises ValueError.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object.")
    unknown = set(config) - set(_DEFAULTS)
    if unknown:
        raise ValueError("Unknown or server-owned config keys: " + ", ".join(sorted(map(str, unknown))))
    effective = deepcopy(_DEFAULTS)
    effective.update(deepcopy(config))
    for name in ("first_k", "second_k"):
        _integer(effective[name], name, 2)
    _integer(effective["random_state"], "random_state", 0, 2**32 - 1)
    _integer(effective["min_trials_after_filter"], "min_trials_after_filter", 1)
    _integer(effective["progress_points"], "progress_points", 3)
    for name in ("baseline_edge_seconds", "feature_sigma_seconds", "display_sigma_seconds"):
        _number(effective[name], name)
    for name in ("baseline_relative_floor", "duration_mad_threshold", "display_sfreq"):
        _number(effective[name], name, positive=True)
    _number(effective["correlation_threshold"], "correlation_threshold", positive=True, maximum=1)
    _number(effective["max_baseline_invalid_fraction"], "max_baseline_invalid_fraction", maximum=1)
    if effective["pre_post_window"] is not None:
        _number(effective["pre_post_window"], "pre_post_window", positive=True)
    if effective["baseline_mode"] not in ("file", "pre_onset"):
        raise ValueError("baseline_mode must be file or pre_onset.")
    window = effective["baseline_window"]
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        raise ValueError("baseline_window must contain two finite numbers.")
    for value in window:
        _number(value, "baseline_window", minimum=-float("inf"))
    if not window[0] < window[1] < 0:
        raise ValueError("baseline_window must satisfy start < end < 0.")
    effective["baseline_window"] = list(window)
    if effective["slope_method"] not in ("savgol", "gradient"):
        raise ValueError("slope_method must be savgol or gradient.")
    weights = effective["band_weights"]
    if not isinstance(weights, dict) or set(weights) != set(_BANDS):
        raise ValueError("band_weights must specify exactly theta, beta1, beta2 and high_gamma.")
    for value in weights.values():
        _number(value, "band_weights", positive=True)
    preset, groups = effective["weight_preset"], effective["group_weights"]
    if preset not in ("legacy", "balanced", "morphology", "amplitude", "custom"):
        raise ValueError("Invalid weight_preset.")
    if preset != "custom" and groups is not None:
        raise ValueError("group_weights requires the custom weight_preset.")
    if preset == "custom":
        if not isinstance(groups, dict) or set(groups) - set(_GROUPS):
            raise ValueError("group_weights must contain only morphology, amplitude and slope.")
        budgets = dict.fromkeys(_GROUPS, 1.)
        budgets.update(groups)
        for value in budgets.values():
            _number(value, "group_weights")
        if not any(value > 0 for value in budgets.values()):
            raise ValueError("group_weights must have at least one positive weight.")
        effective["group_weights"] = budgets
    effective.update(subject=resolved["subject"], block=resolved["block"],
                     sources=deepcopy(resolved["sources"]), surface_options=deepcopy(resolved["surface_options"]))
    # Recheck resolved paths before scheduling: files may disappear or a
    # previously safe directory may be replaced with an external junction.
    root = Path(effective["surface_options"]["subjects_dir"]).parent
    required = ["fif_path", "onset_path", "offset_path", "wav_path"]
    if effective["baseline_mode"] == "file":
        required.append("baseline_fif")
        if effective["sources"].get("baseline_segments_csv") is not None:
            required.append("baseline_segments_csv")
    for name, path in effective["sources"].items():
        if path is not None:
            _contained(Path(path), root)
    surface = effective["surface_options"]
    for name in ("subjects_dir", "electrode_csv", "reference_mri"):
        _contained(Path(surface[name]), root)
    for hemisphere in surface["hemispheres"]:
        _contained(Path(surface["subjects_dir"]) / surface["subject"] / "surf" / f"{hemisphere}.pial", root)
    identity = {}
    for name, path in effective["sources"].items():
        if path is None:
            continue
        try:
            metadata = _contained(Path(path), root).stat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"Cannot stat source {name}.") from error
        if S_ISREG(metadata.st_mode):
            identity[name] = dict(size=metadata.st_size, mtime_ns=metadata.st_mtime_ns)
    for name in required:
        if name not in identity:
            raise ValueError(f"Missing required source {name}.")
    effective["input_identity"] = identity
    try:
        return json.loads(json.dumps(effective, allow_nan=False))
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("config must contain finite JSON-safe values.") from error
