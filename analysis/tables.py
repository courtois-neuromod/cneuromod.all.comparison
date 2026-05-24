import yaml
import pandas as pd
from pathlib import Path


COLUMN_GROUPS_PER_SUBJECT = [
    ("Brain", "#4472C4", [
        ("fMRI",  "neuroimaging.fmri.per_subject_h",  "h"),
    ]),
    ("Tasks", "#538135", [
        ("Images",     "tasks.images.per_subject_unique",           "#img"),
        ("Video",      "tasks.video.per_subject_unique",            "h"),
        ("Audio",      "tasks.audio.per_subject_unique",            "h"),
        ("Speech",     "tasks.speech_listening.per_subject_unique", "h"),
        ("Text",       "tasks.text_reading.per_subject_unique",     "h"),
        ("Rest",       "tasks.resting_state.per_subject_h",        "h"),
        ("Controlled", "tasks.controlled.per_subject_h",           "h"),
        ("Games",      "tasks.game.per_subject_h",                 "h"),
        ("Contrasts",  "tasks.contrasts.per_subject",              "#"),
    ]),
    ("Physiology", "#7030A0", [
        ("ECG",   "physiology.ecg.per_subject_h",            "h"),
        ("Resp.", "physiology.respiration.per_subject_h",    "h"),
        ("PPG",   "physiology.plethysmograph.per_subject_h", "h"),
        ("EDA",   "physiology.eda.per_subject_h",            "h"),
        ("Eye",   "physiology.eye_tracking.per_subject_h",   "h"),
    ]),
]

COLUMN_GROUPS_TOTAL = [
    ("Brain recordings", "#4472C4", [
        ("fMRI",  "neuroimaging.fmri.total_h",  "h"),
        ("EEG",   "neuroimaging.eeg.total_h",   "h"),
        ("MEG",   "neuroimaging.meg.total_h",   "h"),
        ("iEEG",  "neuroimaging.ieeg.total_h",  "h"),
        ("Ca²⁺",  "neuroimaging.calcium_imaging.total_h", "h"),
    ]),
    ("Tasks", "#538135", [
        ("Images", "tasks.images.total_unique",           "#img"),
        ("Video",  "tasks.video.total_unique",            "h"),
        ("Audio",  "tasks.audio.total_unique",            "h"),
        ("Speech", "tasks.speech_listening.total_unique", "h"),
        ("Text",   "tasks.text_reading.total_unique",     "h"),
        ("Rest",   "tasks.resting_state.total_h",         "h"),
        ("Controlled", "tasks.controlled.total_h",        "h"),
        ("Games",      "tasks.game.total_h",              "h"),
        ("Contrasts",  "tasks.contrasts.total",               "#"),
    ]),
    ("Physiology", "#7030A0", [
        ("ECG",   "physiology.ecg.total_h",            "h"),
        ("Resp.", "physiology.respiration.total_h",    "h"),
        ("PPG",   "physiology.plethysmograph.total_h", "h"),
        ("EDA",   "physiology.eda.total_h",            "h"),
        ("Eye",   "physiology.eye_tracking.total_h",   "h"),
    ]),
]


def _get_nested(d, path):
    parts = path.split(".")
    for i, key in enumerate(parts):
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
        if not isinstance(d, dict) and i < len(parts) - 1:
            # Scalar where sub-key expected: treat it as both total and per_subject
            if parts[i + 1] in ("total", "per_subject") and isinstance(d, (int, float)):
                return d
            return None
    return d


def _build_rows(datasets: list, column_groups: list) -> list:
    rows = []
    for ds in datasets:
        for group_name, group_color, fields in column_groups:
            for label, dotpath, unit in fields:
                value = _get_nested(ds, dotpath)
                rows.append({
                    "dataset": ds.get("short_name", ds.get("name", "?")),
                    "group": group_name,
                    "group_color": group_color,
                    "modality": label,
                    "dotpath": dotpath,
                    "unit": unit,
                    "value": value,
                })
    return rows


def build_tidy_table(source_dir: Path, column_groups: list, extra_yaml_files=None) -> pd.DataFrame:
    """Load all dataset YAMLs and return a tidy long-format DataFrame."""
    datasets = []
    for yaml_file in sorted(Path(source_dir).glob("*.yaml")):
        with open(yaml_file) as f:
            datasets.append(yaml.safe_load(f))
    for yaml_file in (extra_yaml_files or []):
        with open(yaml_file) as f:
            datasets.append(yaml.safe_load(f))
    return pd.DataFrame(_build_rows(datasets, column_groups))


def aggregate_cneuromod_yaml(cneuromod_dir: Path) -> dict:
    """Aggregate all cneuromod dataset_info.yaml stats into one schema-compatible dict."""
    raw = []
    for info_file in sorted(Path(cneuromod_dir).glob("*/dataset_info.yaml")):
        with open(info_file) as f:
            raw.append(yaml.safe_load(f).get("stats", {}))

    def _sum_dicts(dicts):
        result = {}
        for key in set(k for d in dicts for k in d):
            values = [d[key] for d in dicts if key in d]
            if all(isinstance(v, dict) for v in values):
                result[key] = _sum_dicts(values)
            elif all(isinstance(v, (int, float)) for v in values):
                result[key] = sum(values)
        return result

    combined = _sum_dicts([{k: v for k, v in d.items() if k != "subjects_n"} for d in raw])
    combined["subjects_n"] = raw[0].get("subjects_n", 6) if raw else 6
    combined["name"] = "CNeuroMod"
    return combined


def load_cneuromod_datasets(cneuromod_dir: Path) -> list:
    """Load datasets from cneuromod.all submodule dataset_info.yaml files.

    Each file's `stats` block is used as the dataset dict; the folder name
    becomes the `name` field.
    """
    datasets = []
    for info_file in sorted(Path(cneuromod_dir).glob("*/dataset_info.yaml")):
        with open(info_file) as f:
            data = yaml.safe_load(f)
        stats = data.get("stats", {})
        stats["name"] = info_file.parent.name
        datasets.append(stats)
    return datasets


def build_cneuromod_tidy_table(cneuromod_dir: Path, column_groups: list) -> pd.DataFrame:
    """Load cneuromod datasets and return a tidy long-format DataFrame."""
    return pd.DataFrame(_build_rows(load_cneuromod_datasets(cneuromod_dir), column_groups))
