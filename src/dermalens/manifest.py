"""Builds the project's image table ("manifest") straight from the raw datasets.

A manifest is one row per image with everything later steps need:

    image_id      unique name, e.g. "dermaconin:IMG_0045.jpg"
    source        dermaconin | skindisnet | mcsi
    source_label  the label exactly as the source dataset wrote it
    label         one of the six target classes (see labels.py)
    group_id      the patient (or case) the photo belongs to; used for splitting
    rel_path      file path relative to data/raw
    fst, mst      Fitzpatrick / Monk skin tone where the source provides it

Nothing here modifies the raw data. Every function returns a new table.
"""

from __future__ import annotations

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import SOURCE_LABEL_MAP

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

MANIFEST_COLUMNS = [
    "image_id", "source", "source_label", "label", "group_id", "rel_path",
    "fst", "mst", "tone_ld", "age", "sex", "body_site", "label_confidence",
]


# ---------------------------------------------------------------------------
# 1. One builder per source dataset
# ---------------------------------------------------------------------------
def _strip_text(table: pd.DataFrame) -> pd.DataFrame:
    """Remove stray spaces at the start/end of every text column ('Eczema ' -> 'Eczema')."""
    table = table.copy()
    for column in table.columns:
        if pd.api.types.is_object_dtype(table[column]) or pd.api.types.is_string_dtype(table[column]):
            table[column] = table[column].astype("string").str.strip()
    return table


def _number_from_text(series: pd.Series) -> pd.Series:
    """Pull the first number out of text like 'FST 3' or 'MST 10' -> 3.0 / 10.0."""
    return pd.to_numeric(series.astype("string").str.extract(r"(\d+)")[0], errors="coerce")


def build_dermaconin_manifest(raw_dir: Path) -> pd.DataFrame:
    """DermaCon-IN: every metadata row names its own image file (column Image_name).

    Rows the dermatologists marked as too poor to diagnose are removed.
    """
    root = raw_dir / "dermaconin"
    meta = _strip_text(pd.read_csv(root / "METADATA" / "Skin_Metadata-1.csv", dtype={"Subject_ID": str}))

    # Find every image once, then look each metadata row up by file name.
    image_files = {p.name: p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
    missing = sorted(set(meta["Image_name"]) - set(image_files))
    if missing:
        raise FileNotFoundError(f"{len(missing)} DermaCon-IN images listed in metadata are missing, "
                                f"e.g. {missing[:3]}")

    # Quality says "No, the quality is insufficient ..." for unusable photos.
    unusable = meta["Quality"].astype("string").str.startswith("No,", na=False)

    table = pd.DataFrame({
        "image_id": "dermaconin:" + meta["Image_name"],
        "source": "dermaconin",
        "source_label": meta["Disease_label"],
        "group_id": "DC_" + meta["Subject_ID"].astype("string"),
        "rel_path": meta["Image_name"].map(lambda name: image_files[name].relative_to(raw_dir).as_posix()),
        "fst": _number_from_text(meta["Fitzpatrick"]),
        "mst": _number_from_text(meta["Monk_skin_tone"]),
        "tone_ld": pd.NA,
        "age": meta["Age"],
        "sex": meta["Sex"],
        "body_site": meta["Body_part"],
        "label_confidence": pd.to_numeric(meta["Confidence"], errors="coerce"),
    })
    return table.loc[~unusable].reset_index(drop=True)


def build_skindisnet_manifest(raw_dir: Path) -> pd.DataFrame:
    """SkinDisNet: only the 1,710 originals in skindisnet2/Preprocessed are used.

    The file name is the metadata's Image_id plus '.jpg', inside the class
    folder named by Folder_name, e.g. 'Contact Dermatitis (CD)/CD (20).jpg'.
    The augmented folders and the skindisnet1 mirror are never read.
    """
    root = raw_dir / "skindisnet" / "skindisnet2"
    meta = _strip_text(pd.read_csv(root / "SkinDisNet_Metadata.csv", dtype=str))

    rel_paths = ("skindisnet/skindisnet2/Preprocessed/" + meta["Folder_name"]
                 + "/" + meta["Image_id"] + ".jpg")
    missing = [p for p in rel_paths if not (raw_dir / p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} SkinDisNet images not found, e.g. {missing[:3]}")

    return pd.DataFrame({
        "image_id": "skindisnet:" + meta["Image_id"],
        "source": "skindisnet",
        "source_label": meta["Diagnosis"],
        "group_id": "SD_" + meta["Patient_id"],
        "rel_path": rel_paths,
        "fst": np.nan,                 # SkinDisNet has no skin-tone labels
        "mst": np.nan,
        "tone_ld": pd.NA,
        "age": meta["Age"],
        "sex": meta["Sex"],
        "body_site": meta["Leision_location"],   # the source spells it this way
        "label_confidence": np.nan,
    })


def build_mcsi_manifest(raw_dir: Path) -> pd.DataFrame:
    """MCSI (CC BY 4.0): only the images whose diagnostic is 'normal' are used (Healthy).

    MCSI has no patient IDs, so every image is its own group.
    """
    root = raw_dir / "mcsi" / "extracted"
    meta = pd.read_csv(root / "metadata.csv", dtype=str)
    tone_path = root / "skin_tone_labels.csv"
    tones = (pd.read_csv(tone_path, dtype=str).set_index("Filename")["Type"]
             if tone_path.exists() else pd.Series(dtype="string"))

    # metadata.csv names some images without their extension ("acne_lvl1_32"); match every id to the
    # real file by full name or by name without extension, and drop rows whose image is not on disk.
    on_disk = {}
    for file in (root / "images").rglob("*"):
        if file.suffix.lower() in IMAGE_SUFFIXES:
            rel = file.relative_to(raw_dir).as_posix()
            on_disk.setdefault(file.name, rel)
            on_disk.setdefault(file.stem, rel)
    rel_paths = meta["img_id"].map(on_disk)
    missing = rel_paths.isna()
    if missing.any():
        print(f"MCSI: {missing.sum()} metadata rows have no image file and are skipped "
              f"(e.g. {', '.join(meta.loc[missing, 'img_id'].head(3))})")
    meta, rel_paths = meta[~missing].reset_index(drop=True), rel_paths[~missing].reset_index(drop=True)
    return pd.DataFrame({
        "image_id": "mcsi:" + meta["img_id"],
        "source": "mcsi",
        "source_label": meta["diagnostic"].str.strip(),
        "group_id": "MC_" + meta["img_id"],
        "rel_path": rel_paths,
        "fst": np.nan,
        "mst": np.nan,
        "tone_ld": meta["img_id"].map(tones),   # 'light' / 'dark'
        "age": pd.NA,
        "sex": pd.NA,
        "body_site": pd.NA,
        "label_confidence": np.nan,
    })


SOURCE_BUILDERS = {
    "dermaconin": build_dermaconin_manifest,
    "skindisnet": build_skindisnet_manifest,
    "mcsi": build_mcsi_manifest,
}


def apply_label_map(table: pd.DataFrame, label_map: dict[str, dict[str, str]] = SOURCE_LABEL_MAP) -> pd.DataFrame:
    """Add the target 'label' column and keep only rows whose source label is mapped."""
    table = table.copy()
    table["label"] = [label_map.get(src, {}).get(lbl)
                      for src, lbl in zip(table["source"], table["source_label"], strict=False)]
    return table.dropna(subset=["label"]).reset_index(drop=True)


def select_other_examples(table: pd.DataFrame, cap_per_diagnosis: int, seed: int) -> pd.DataFrame:
    """Rows of non-target diseases to use as the 'Other' class (open-set training).

    Diagnoses close to a target class (any tinea, eczema, dermatitis, psoriasis,
    scabies) are never used. Each remaining diagnosis contributes at most
    `cap_per_diagnosis` images, so no single disease (e.g. 601 Vitiligo photos)
    dominates the class.
    """
    from .labels import OTHER_CLASS, is_other_candidate

    unmapped = table[table["label"].isna() & table["source_label"].map(is_other_candidate)]
    sampled = (unmapped.groupby("source_label", group_keys=False)
               .apply(lambda part: part.sample(min(len(part), cap_per_diagnosis), random_state=seed)))
    return sampled.assign(label=OTHER_CLASS)


def build_manifest(raw_dir: Path, sources: tuple[str, ...] = ("dermaconin", "skindisnet", "mcsi"),
                   label_map: dict[str, dict[str, str]] = SOURCE_LABEL_MAP, open_set: bool = False,
                   other_cap_per_diagnosis: int = 25, other_sources: tuple[str, ...] = ("dermaconin", "mcsi"),
                   seed: int = 42) -> pd.DataFrame:
    """Build every source, map labels, and stack them into one table.

    open_set=True also adds an 'Other' class made of non-target diseases from
    `other_sources` (see select_other_examples). Metadata columns meta_age,
    meta_sex and meta_regions are always added (see metadata.py).
    """
    from .metadata import add_metadata_columns

    parts = []
    for name in sources:
        table = SOURCE_BUILDERS[name](raw_dir)
        table["label"] = [label_map.get(src, {}).get(lbl)
                          for src, lbl in zip(table["source"], table["source_label"], strict=False)]
        parts.append(table.dropna(subset=["label"]))
        if open_set and name in other_sources:
            parts.append(select_other_examples(table, other_cap_per_diagnosis, seed))
    manifest = pd.concat(parts, ignore_index=True)[MANIFEST_COLUMNS]
    manifest = add_metadata_columns(manifest)
    assert manifest["image_id"].is_unique, "image_id must be unique"
    assert manifest["rel_path"].is_unique, "the same file is listed twice"
    return manifest


# ---------------------------------------------------------------------------
# 2. Exact duplicates: identical bytes = identical fingerprint (SHA-256)
# ---------------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    """SHA-256 fingerprint of a file's bytes, read in 1 MB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_file_hashes(manifest: pd.DataFrame, raw_dir: Path, cache_csv: Path | None = None,
                    workers: int = 16) -> pd.DataFrame:
    """Add a 'sha256' column. Hashes already in `cache_csv` are reused (Drive is slow)."""
    cache = {}
    if cache_csv is not None and Path(cache_csv).exists():
        cache = pd.read_csv(cache_csv, dtype=str).set_index("rel_path")["sha256"].to_dict()

    todo = [p for p in manifest["rel_path"] if p not in cache]
    with ThreadPoolExecutor(max_workers=workers) as pool:          # reading files is I/O-bound
        for rel_path, digest in zip(todo, pool.map(lambda p: file_sha256(raw_dir / p), todo), strict=False):
            cache[rel_path] = digest

    if cache_csv is not None:
        pd.DataFrame({"rel_path": list(cache), "sha256": list(cache.values())}).to_csv(cache_csv, index=False)

    manifest = manifest.copy()
    manifest["sha256"] = manifest["rel_path"].map(cache)
    return manifest


def drop_exact_duplicates(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep one copy of each identical file.

    Returns (clean manifest, table of every duplicate group for the report).
    Raises an error if two identical files carry different labels: that is a
    labelling mistake a human must resolve.
    """
    # An 'Other' photo that is identical to a target-class photo is simply dropped:
    # the target label is the more specific one.
    target_hashes = set(manifest.loc[manifest["label"] != "Other", "sha256"])
    manifest = manifest[~((manifest["label"] == "Other") & manifest["sha256"].isin(target_hashes))]

    copies = manifest[manifest.duplicated("sha256", keep=False)].sort_values(["sha256", "image_id"])
    conflicts = copies.groupby("sha256")["label"].nunique()
    if (conflicts > 1).any():
        bad = copies[copies["sha256"].isin(conflicts[conflicts > 1].index)]
        raise ValueError("Identical images with different labels:\n"
                         + bad[["image_id", "label", "rel_path"]].to_string())
    clean = manifest.drop_duplicates("sha256", keep="first").reset_index(drop=True)
    return clean, copies.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Decode every image once: size, colour statistics, perceptual hash
# ---------------------------------------------------------------------------
def _image_properties(path: Path) -> dict:
    """Open one image and measure it. Failures are recorded, never raised."""
    import imagehash
    from PIL import Image

    try:
        with Image.open(path) as image:
            width, height = image.size              # original size, before any reduction
            file_format, mode = image.format, image.mode
            image.draft("RGB", (512, 512))          # fast JPEG decode at reduced size
            rgb = image.convert("RGB")
            pixels = np.asarray(rgb, dtype=np.float32)
            return {
                "readable": True, "error": "",
                "width": width, "height": height, "format": file_format, "mode": mode,
                "mean_r": pixels[..., 0].mean(), "mean_g": pixels[..., 1].mean(),
                "mean_b": pixels[..., 2].mean(), "brightness": pixels.mean(),
                "contrast": pixels.std(),
                "phash": str(imagehash.phash(rgb)),  # 64-bit "looks alike" fingerprint (hex)
            }
    except Exception as error:  # a broken file must not stop the audit
        return {"readable": False, "error": f"{type(error).__name__}: {error}"}


def add_image_properties(manifest: pd.DataFrame, raw_dir: Path, cache_csv: Path | None = None,
                         workers: int = 16) -> pd.DataFrame:
    """Decode each image and add its size, colour stats and perceptual hash as columns."""
    has_cache = cache_csv is not None and Path(cache_csv).exists()
    cached = pd.read_csv(cache_csv, dtype={"rel_path": str, "phash": str}) if has_cache else pd.DataFrame()
    done = set(cached["rel_path"]) if not cached.empty else set()
    todo = [p for p in manifest["rel_path"] if p not in done]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        new_rows = [{"rel_path": p, **props}
                    for p, props in zip(todo, pool.map(lambda p: _image_properties(raw_dir / p), todo), strict=False)]
    properties = pd.concat([cached, pd.DataFrame(new_rows)], ignore_index=True).drop_duplicates("rel_path")

    if cache_csv is not None:
        properties.to_csv(cache_csv, index=False)
    columns = [c for c in properties.columns if c not in manifest.columns or c == "rel_path"]
    return manifest.merge(properties[columns], on="rel_path", how="left", validate="one_to_one")


# ---------------------------------------------------------------------------
# 4. Near duplicates: same photo resized / re-saved -> keep in the same split
# ---------------------------------------------------------------------------
def _popcount64(values: np.ndarray) -> np.ndarray:
    """Count the 1-bits in each uint64 (= Hamming distance after XOR)."""
    if hasattr(np, "bitwise_count"):                      # NumPy 2.0+
        return np.bitwise_count(values)
    as_bytes = values.view(np.uint8).reshape(-1, 8)       # older NumPy fallback
    return np.unpackbits(as_bytes, axis=1).sum(axis=1)


def find_near_duplicates(manifest: pd.DataFrame, max_distance: int = 4) -> pd.DataFrame:
    """List pairs of images whose perceptual hashes differ by at most `max_distance` bits.

    0 bits = visually identical; up to ~4 bits = the same photo resized,
    re-compressed or slightly cropped. Returns one row per pair.
    """
    usable = manifest.dropna(subset=["phash"]).reset_index(drop=True)
    hashes = np.array([int(h, 16) for h in usable["phash"]], dtype=np.uint64)
    pairs = []
    for i in range(len(hashes) - 1):
        distances = _popcount64(hashes[i] ^ hashes[i + 1:])
        for offset in np.flatnonzero(distances <= max_distance):
            j = i + 1 + offset
            pairs.append({
                "image_a": usable.at[i, "image_id"], "image_b": usable.at[j, "image_id"],
                "label_a": usable.at[i, "label"], "label_b": usable.at[j, "label"],
                "group_a": usable.at[i, "group_id"], "group_b": usable.at[j, "group_id"],
                "distance": int(distances[offset]),
            })
    return pd.DataFrame(pairs, columns=["image_a", "image_b", "label_a", "label_b",
                                        "group_a", "group_b", "distance"])


def merge_groups_of_near_duplicates(manifest: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Give near-duplicate images the same group_id so they always land in the same split.

    Uses union-find: if A~B and B~C, then A, B and C all share one group.
    The original group is kept in 'source_group_id' for traceability.
    """
    parent: dict[str, str] = {}

    def find(group: str) -> str:
        parent.setdefault(group, group)
        while parent[group] != group:
            parent[group] = parent[parent[group]]   # shorten the path as we go
            group = parent[group]
        return group

    for a, b in zip(pairs["group_a"], pairs["group_b"], strict=False):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)   # deterministic choice

    manifest = manifest.copy()
    manifest["source_group_id"] = manifest["group_id"]
    manifest["group_id"] = manifest["group_id"].map(find)
    return manifest


# ---------------------------------------------------------------------------
# 5. Small text fix for SCIN (reference dataset only)
# ---------------------------------------------------------------------------
SCIENTIFIC_NOTATION = re.compile(r"^-?\d+(\.\d+)?[eE][+-]?\d+$")


def count_damaged_ids(series: pd.Series) -> int:
    """How many IDs were turned into scientific notation (e.g. '-1.00E+18') by a spreadsheet."""
    return int(series.astype("string").str.match(SCIENTIFIC_NOTATION, na=False).sum())
