"""Reads experiment settings from configs/*.yaml.

Keeping every number (image size, learning rate, epochs...) in one YAML file
means a result in the report can always be traced back to the exact settings
that produced it.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: dict | None = None) -> dict[str, Any]:
    """Load a YAML config and optionally replace some values.

    Example:
        cfg = load_config("configs/baseline.yaml", {"train": {"epochs": 2}})
    """
    config = yaml.safe_load(Path(path).read_text())
    if overrides:
        config = merge_dicts(config, overrides)
    return config


def merge_dicts(base: dict, updates: dict) -> dict:
    """Return a copy of `base` where nested keys from `updates` replace old values."""
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
