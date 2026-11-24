"""Read-only checks on the raw downloads (notebook 01). Nothing is changed or moved."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
TABLE_SUFFIXES = {".csv", ".json", ".xlsx"}


def inventory_files(raw_dir: Path, datasets: list[str]) -> pd.DataFrame:
    """One row per file under data/raw/<dataset>: path, type, size, and its top two folders."""
    rows = []
    for dataset in datasets:
        root = raw_dir / dataset
        for path in sorted(root.rglob("*")):
            if not path.is_file() or any(part in {".ipynb_checkpoints", "__MACOSX"} for part in path.parts):
                continue
            relative = path.relative_to(root)
            suffix = path.suffix.lower()
            rows.append({
                "dataset": dataset,
                "rel_path": path.relative_to(raw_dir).as_posix(),
                "subfolder": "/".join(relative.parts[:2]) if len(relative.parts) > 2 else relative.parts[0],
                "kind": "image" if suffix in IMAGE_SUFFIXES else "table" if suffix in TABLE_SUFFIXES else "other",
                "suffix": suffix,
                "size_bytes": path.stat().st_size,
            })
    return pd.DataFrame(rows)


def summarise_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    """Images, tables, zero-byte files and extensions per dataset."""
    images = inventory[inventory["kind"] == "image"]
    summary = images.groupby("dataset").agg(
        images=("rel_path", "size"),
        zero_byte_images=("size_bytes", lambda s: int((s == 0).sum())),
        total_gb=("size_bytes", lambda s: round(s.sum() / 1e9, 2)),
        extensions=("suffix", lambda s: dict(Counter(s))),
    )
    summary["tables"] = inventory[inventory["kind"] == "table"].groupby("dataset").size()
    return summary.fillna({"tables": 0}).reset_index()


def skindisnet_name_key(rel_path: str) -> tuple[str, int] | None:
    """'.../AD_01.jpg' and '.../AD (1).jpg' both -> ('AD', 1), so the two copies can be matched."""
    match = re.search(r"/([A-Za-z]+)(?:_|\s\()(\d+)\)?\.\w+$", rel_path)
    return (match.group(1).upper(), int(match.group(2))) if match else None


def skindisnet_mirror_check(inventory: pd.DataFrame) -> dict:
    """Are skindisnet1/Preprocessed and skindisnet2/Preprocessed the same photos under different names?"""
    sd = inventory[(inventory["dataset"] == "skindisnet") & (inventory["kind"] == "image")]
    keys = {}
    for folder in ("skindisnet1/Preprocessed", "skindisnet2/Preprocessed"):
        in_folder = sd[sd["rel_path"].str.contains(folder + "/", regex=False)]
        keys[folder] = {k for k in in_folder["rel_path"].map(skindisnet_name_key) if k}
    first, second = keys.values()
    return {"skindisnet1_originals": len(first), "skindisnet2_originals": len(second),
            "matching_names": len(first & second)}
