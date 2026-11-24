"""Patient metadata (age group, sex, body region): cleaning and encoding.

DermaCon-IN and SkinDisNet record age, sex and lesion location in different
formats. This module maps both to one small shared vocabulary that a person
can also enter in the web app, then encodes it as a fixed-length vector the
model reads next to the image.

Vector layout (META_DIM = 21):
    age group one-hot (5) + age missing (1)
    sex one-hot (2)       + sex missing (1)
    body regions multi-hot (11) + region missing (1)

Missing values are explicit, so a photo without metadata (e.g. MCSI healthy
skin, or a user who skips the form) is still valid input.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

AGE_GROUPS = ["0-10", "10-20", "20-40", "40-60", "60+"]
AGE_LABELS = ["0–10 years", "10–20 years", "20–40 years", "40–60 years", "60 years and above"]
SEXES = ["female", "male"]

# key -> (label shown in the app, keywords found in source descriptions)
BODY_REGIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "face": ("Face", ("face", "cheek", "forehead", "nose", "chin", "lip", "temple", "ear", "eyelid", "periorbital", "oral", "tongue", "mouth")),  # noqa: E501
    "scalp": ("Scalp", ("scalp", "head")),
    "neck": ("Neck", ("neck",)),
    "chest_abdomen": ("Chest / abdomen", ("chest", "thorax", "abdomen", "belly", "breast", "umbilic")),
    "back": ("Back", ("back", "lowerback", "lumb", "posterior")),
    "buttocks": ("Buttocks", ("buttock", "gluteal")),
    "groin_genital": ("Groin / genital area", ("groin", "inguinal", "genital", "pubic", "scrot", "penis", "vulva")),
    "arm": ("Arm / elbow", ("arm", "brachium", "elbow", "axilla", "shoulder")),
    "hand": ("Hand / fingers / palm", ("hand", "manus", "finger", "palm", "wrist", "nail")),
    "leg": ("Thigh / knee / leg", ("thigh", "femoral", "knee", "patellar", "leg", "crural", "shin", "calf", "calves", "sural")),  # noqa: E501
    "foot": ("Foot / toes / sole", ("foot", "pedal", "toe", "sole", "plantar", "heel", "ankle", "tarsal")),
}
REGION_KEYS = list(BODY_REGIONS)
META_DIM = len(AGE_GROUPS) + 1 + len(SEXES) + 1 + len(REGION_KEYS) + 1


# ---------------------------------------------------------------------------
# Cleaning source values
# ---------------------------------------------------------------------------
def age_group_from_years(years) -> str | None:
    """Numeric age (SkinDisNet) -> age-group key."""
    try:
        value = float(years)
    except (TypeError, ValueError):
        return None
    if np.isnan(value):
        return None
    for upper, key in ((10, "0-10"), (20, "10-20"), (40, "20-40"), (60, "40-60")):
        if value < upper:
            return key
    return "60+"


def age_group_from_text(text) -> str | None:
    """'20 - 40' (DermaCon-IN) -> '20-40'; '60 - 80' and '80 - 100' -> '60+'."""
    numbers = re.findall(r"\d+", str(text))
    if not numbers:
        return None
    low = int(numbers[0])
    return "60+" if low >= 60 else age_group_from_years(low + 0.5)


def sex_key(text) -> str | None:
    """'Female' / 'F' / 'male' -> 'female' / 'male'."""
    value = str(text).strip().lower()
    if value.startswith("f"):
        return "female"
    if value.startswith("m"):
        return "male"
    return None


def regions_from_text(text) -> list[str]:
    """Free-text lesion location -> list of region keys (several are allowed).

    Words are matched whole (prefix match), so 'forearm' is not read as 'ear'
    and 'Head Cheeks' is the face, not the scalp.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    words = re.findall(r"[a-z]+", str(text).lower())
    found = set()
    for key, (_, keywords) in BODY_REGIONS.items():
        if key == "scalp":
            continue
        if any(word.startswith(keyword) for word in words for keyword in keywords):
            found.add(key)
    # 'head' alone (or 'scalp') means the scalp; 'Head Cheeks' etc. already matched the face.
    if "scalp" in words or ("head" in words and "face" not in found):
        found.add("scalp")
    # 'forearm' / 'underarm' start with other letters, so look for 'arm' inside words too.
    if any(word.rstrip("s").endswith("arm") for word in words):
        found.add("arm")
    if "posterior" in words or "lowerback" in words:   # 'Abdomen posterior' = lower back
        found.discard("chest_abdomen")
    return [key for key in REGION_KEYS if key in found]


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def encode(age_group: str | None, sex: str | None, regions) -> np.ndarray:
    """One metadata record -> float32 vector of length META_DIM."""
    vector = np.zeros(META_DIM, dtype=np.float32)
    position = 0
    if age_group in AGE_GROUPS:
        vector[position + AGE_GROUPS.index(age_group)] = 1
    else:
        vector[position + len(AGE_GROUPS)] = 1           # age missing
    position += len(AGE_GROUPS) + 1
    if sex in SEXES:
        vector[position + SEXES.index(sex)] = 1
    else:
        vector[position + len(SEXES)] = 1                # sex missing
    position += len(SEXES) + 1
    region_list = [r for r in (regions or []) if r in REGION_KEYS]
    if region_list:
        for region in region_list:
            vector[position + REGION_KEYS.index(region)] = 1
    else:
        vector[position + len(REGION_KEYS)] = 1          # region missing
    return vector


def encode_row(row: pd.Series) -> np.ndarray:
    """Encode a manifest row (columns meta_age, meta_sex, meta_regions)."""
    regions = str(row.get("meta_regions") or "")
    region_list = [r for r in regions.split("|") if r] if regions and regions != "nan" else []
    age = row.get("meta_age")
    sex = row.get("meta_sex")
    return encode(age if isinstance(age, str) else None, sex if isinstance(sex, str) else None, region_list)


def drop_fields(vector: np.ndarray, rng: np.random.Generator, p: float) -> np.ndarray:
    """Training augmentation: hide each field with probability p, so the model
    also learns to work when the user leaves fields empty."""
    if p <= 0:
        return vector
    vector = vector.copy()
    spans = [(0, len(AGE_GROUPS)), (len(AGE_GROUPS) + 1, len(AGE_GROUPS) + 1 + len(SEXES)),
             (len(AGE_GROUPS) + len(SEXES) + 2, META_DIM - 1)]
    missing_flags = [len(AGE_GROUPS), len(AGE_GROUPS) + len(SEXES) + 1, META_DIM - 1]
    for (start, end), flag in zip(spans, missing_flags, strict=False):
        if rng.random() < p:
            vector[start:end] = 0
            vector[flag] = 1
    return vector


def add_metadata_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Add meta_age, meta_sex, meta_regions to a manifest built by manifest.py.

    DermaCon-IN 'age' is text like '20 - 40'; SkinDisNet 'age' is in years.
    """
    table = table.copy()
    ages = []
    for source, age in zip(table["source"], table["age"], strict=False):
        ages.append(age_group_from_text(age) if source == "dermaconin" else age_group_from_years(age))
    table["meta_age"] = ages
    table["meta_sex"] = [sex_key(s) if isinstance(s, str) else None for s in table["sex"]]
    table["meta_regions"] = ["|".join(regions_from_text(site)) for site in table["body_site"]]
    return table


def options_for_app() -> dict:
    """The choices the web form offers (same keys as the model was trained on)."""
    return {
        "age_groups": [{"key": k, "label": label} for k, label in zip(AGE_GROUPS, AGE_LABELS, strict=False)],
        "sexes": [{"key": "female", "label": "Female"}, {"key": "male", "label": "Male"}],
        "body_regions": [{"key": k, "label": v[0]} for k, v in BODY_REGIONS.items()],
    }
