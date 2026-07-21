# Experiment manifest and self-enforcing output mirror paths.

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# CLI aliases -> canonical model id (for loaders / manifest.model)
MODEL_ALIASES = {
    "llama": "llama7b",
    "mistral": "mistral7b-v02",
    "qwen": "qwen7b",
}

# Canonical id -> short token used in output filenames
FILENAME_MODEL = {
    "llama7b": "llama",
    "mistral7b-v02": "mistral",
    "qwen7b": "qwen",
    "phi3": "phi3",
}


@dataclass
class RunConfig:
    model: str
    quantizer: str = "aqlm2"
    variant: str | None = None
    output_stem: str | None = None
    omit_quantizer: bool = False
    extra: dict[str, Any] = field(default_factory = dict)

    def __post_init__(self):
        """
        Map short CLI model aliases onto canonical model ids.

        Inputs:
            - None

        Outputs:
            - None
        """
        # Resolve short aliases (llama -> llama7b) when present
        self.model = MODEL_ALIASES.get(self.model, self.model)

    def filename_model(self):
        """
        Short model token used in output JSON filenames.

        Inputs:
            - None

        Outputs:
            - short (str): Filename model token
        """
        # Fall back to the full model id when no short token is registered
        return FILENAME_MODEL.get(self.model, self.model)

    def to_dict(self):
        """
        Serialize the config, dropping empty optional fields.

        Inputs:
            - None

        Outputs:
            - d (dict): Manifest-ready dictionary
        """
        # Convert the dataclass to a plain dict
        d = asdict(self)

        # Drop unset optional keys so manifests stay compact
        for k in ("variant", "output_stem", "omit_quantizer"):
            if not d.get(k):
                d.pop(k, None)
        return d


def _git_sha():
    """
    Resolve the current repository HEAD SHA, or None if unavailable.

    Inputs:
        - None

    Outputs:
        - git_sha (str | None): Current HEAD, or None
    """
    # Ask git for HEAD; tolerate missing git or non-repo checkouts
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr = subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _build_stem(calling_file, cfg):
    """
    Build the basename stem for a result JSON filename.

    Inputs:
        - calling_file (Path): Experiment script path
        - cfg (RunConfig): Run configuration

    Outputs:
        - name (str): Stem without directory or .json suffix
    """
    # Prefer an explicit output_stem when the script requests a custom name
    stem = cfg.output_stem or Path(calling_file).stem
    parts = [stem, cfg.filename_model()]

    # Include quantizer unless the experiment opts out (e.g. fp32-only)
    if not cfg.omit_quantizer:
        parts.append(cfg.quantizer)
    name = "_".join(parts)

    # Append variant suffix when present (read, L11-22, etc.)
    if cfg.variant:
        name += f"_{cfg.variant}"
    return name


def result_path(calling_file, cfg):
    """
    Derive outputs/<experiment_folder>/<stem>_<model>_<quant>[_variant].json.

    Inputs:
        - calling_file (Path | str): Experiment script path
        - cfg (RunConfig): Run configuration

    Outputs:
        - path (Path): Canonical result JSON path
    """
    # Mirror the claim folder name under outputs/
    calling_file = Path(calling_file)
    exp_dir = calling_file.parent.name
    return Path("outputs") / exp_dir / f"{_build_stem(calling_file, cfg)}.json"


def legacy_result_path(calling_file, cfg):
    """
    Pre-short-name layout: long model id and script stem (no output_stem).

    Inputs:
        - calling_file (Path | str): Experiment script path
        - cfg (RunConfig): Run configuration

    Outputs:
        - path (Path): Legacy result JSON path
    """
    # Rebuild the older naming scheme for find_result fall-backs
    calling_file = Path(calling_file)
    exp_dir = calling_file.parent.name
    parts = [calling_file.stem, cfg.model]

    # Include quantizer unless the experiment opts out
    if not cfg.omit_quantizer:
        parts.append(cfg.quantizer)
    name = "_".join(parts)

    # Append variant suffix when present
    if cfg.variant:
        name += f"_{cfg.variant}"
    return Path("outputs") / exp_dir / f"{name}.json"


def find_result(calling_file, cfg):
    """
    Return the first existing path among current and legacy naming schemes.

    Inputs:
        - calling_file (Path | str): Experiment script path
        - cfg (RunConfig): Run configuration

    Outputs:
        - path (Path | None): Existing result file, or None
    """
    # Prefer the current short-name layout, then the legacy folder layout
    for builder in (result_path, legacy_result_path):
        path = builder(calling_file, cfg)
        if path.exists():
            return path

    # Last fall-back: flat pre-mirror name still under outputs/
    flat_legacy = Path("outputs") / legacy_result_path(calling_file, cfg).name
    if flat_legacy.exists():
        return flat_legacy
    return None


def save_result(calling_file, cfg, data, raw = False):
    """
    Write a result JSON with an embedded run manifest and git SHA.

    Inputs:
        - calling_file (Path | str): Experiment script path
        - cfg (RunConfig): Run configuration
        - data (dict): Experiment payload to merge into the JSON
        - raw (bool): If True, write a *_raw.json sidecar instead

    Outputs:
        - path (Path): Path of the written JSON file
    """
    # Resolve the canonical output path for this script + config
    path = result_path(calling_file, cfg)

    # Raw sidecars keep large dumps out of the primary artifact name
    if raw:
        path = path.with_name(path.stem + "_raw" + path.suffix)
    path.parent.mkdir(parents = True, exist_ok = True)
    payload = {
        "manifest": {**cfg.to_dict(), **cfg.extra},
        "git_sha": _git_sha(),
        **data,
    }
    path.write_text(json.dumps(payload, indent = 2, default = str))
    return path
