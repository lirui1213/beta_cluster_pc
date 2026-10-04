"""Resolve explicit patient/block selections without selecting data implicitly."""
from collections.abc import Mapping
from pathlib import Path
import re

import pandas as pd


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"Invalid patient/block identifier: {value!r}")
    return value


def build_manifest(data_root, included_blocks, *, baseline_mode="file"):
    """Return one row per selected block; never auto-include any block."""
    if baseline_mode not in {"file", "pre_onset"}:
        raise ValueError("baseline_mode must be file or pre_onset")
    if not isinstance(included_blocks, Mapping) or not included_blocks:
        raise ValueError("INCLUDED_BLOCKS is empty. Select patients and blocks first.")
    root = Path(data_root).resolve()
    rows = []
    for subject, blocks in included_blocks.items():
        _identifier(subject)
        if not isinstance(blocks, (list, tuple)) or not blocks:
            raise ValueError(f"{subject}: blocks must be a nonempty list.")
        seen = set()
        for block in blocks:
            _identifier(block)
            if block.casefold() in seen:
                raise ValueError(f"Duplicate block: {subject}/{block}")
            seen.add(block.casefold())
            base = root / subject / "blocks" / block / "standardized"
            paths = {
                "fif_path": base / "ECoG/ecog_TEST_clean_car-raw.fif",
                "onset_path": base / "events/onset.npy",
                "offset_path": base / "events/offset.npy",
                "wav_path": base / "audio/analog1.wav",
                "baseline_fif": root / subject / "baseline/baseline.fif",
            }
            for path in paths.values():
                if not path.resolve().is_relative_to(root):
                    raise ValueError(f"Input resolves outside DATA_ROOT: {path}")
            rows.append({"subject_id": subject, "block_id": block,
                         "baseline_mode": baseline_mode,
                         **{key: str(path) for key, path in paths.items()}})
    table = pd.DataFrame(rows)
    if table[["subject_id", "block_id"]].apply(lambda c: c.str.casefold()).duplicated().any():
        raise ValueError("Duplicate patient/block selection")
    return table


def inspect_manifest(manifest):
    """Show all missing required files without opening signal recordings."""
    if manifest.empty:
        raise ValueError("Manifest is empty")
    required = {"subject_id", "block_id", "baseline_mode", "fif_path", "onset_path", "offset_path", "wav_path", "baseline_fif"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Missing manifest columns: {sorted(required - set(manifest.columns))}")
    if manifest[["subject_id", "block_id"]].duplicated().any():
        raise ValueError("Duplicate patient/block selection")
    rows = []
    for row in manifest.to_dict("records"):
        if row["baseline_mode"] not in {"file", "pre_onset"}:
            raise ValueError("Invalid baseline_mode")
        fields = ["fif_path", "onset_path", "offset_path", "wav_path"]
        if row["baseline_mode"] == "file":
            fields.append("baseline_fif")
        missing = [f"{key}: {row[key]}" for key in fields if not Path(row[key]).is_file()]
        rows.append({**row, "ready": not missing, "missing_files": "\n".join(missing)})
    return pd.DataFrame(rows)


def validate_manifest(manifest):
    report = inspect_manifest(manifest)
    failed = report.loc[~report.ready]
    if not failed.empty:
        details = "\n".join(f"{r.subject_id}/{r.block_id}\n{r.missing_files}" for r in failed.itertuples())
        raise FileNotFoundError("Missing selected input files:\n" + details)
    if manifest.baseline_mode.nunique() != 1:
        raise ValueError("All blocks must use the same baseline_mode")
    return report


def list_available_blocks(data_root):
    """Inventory standardized blocks; does not select any of them."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = []
    for subject in sorted(root.glob("HS*")):
        if not subject.is_dir():
            continue
        for block in sorted((subject / "blocks").glob("*")):
            if block.is_dir() and (block / "standardized").is_dir():
                manifest = build_manifest(root, {subject.name: [block.name]})
                rows.append(inspect_manifest(manifest).iloc[0].to_dict())
    return pd.DataFrame(rows, columns=["subject_id", "block_id", "baseline_mode", "fif_path", "onset_path", "offset_path", "wav_path", "baseline_fif", "ready", "missing_files"])
