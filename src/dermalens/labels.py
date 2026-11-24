"""The target classes and how each dataset's own labels map onto them.

This file is the single source of truth for the label taxonomy. Notebook 02
explains *why* these classes were chosen; every later step imports them from
here, so the class list can never differ between notebooks again.
"""

from __future__ import annotations

# The six classes the model predicts, in a fixed alphabetical order.
# The order matters: class index 0 is always "Contact Dermatitis", and so on.
TARGET_CLASSES: list[str] = [
    "Contact Dermatitis",
    "Eczema",
    "Healthy",
    "Psoriasis",
    "Scabies",
    "Tinea Corporis",
]

# For each source dataset: {exact label as written in the source -> target class}.
# A source label that is not listed here is simply not used.
SOURCE_LABEL_MAP: dict[str, dict[str, str]] = {
    "dermaconin": {                       # column: Disease_label
        "Contact Dermatitis": "Contact Dermatitis",
        "Eczema": "Eczema",
        "Psoriasis": "Psoriasis",
        "Scabies": "Scabies",
        "Tinea Corporis": "Tinea Corporis",
    },
    "skindisnet": {                       # column: Diagnosis
        "Contact Dermatitis": "Contact Dermatitis",
        "Eczema": "Eczema",
        "Scabies": "Scabies",
        "Tinea Corporis": "Tinea Corporis",
        # "Atopic Dermatitis" and "Seborrheic Dermatitis" are not target classes.
    },
    "mcsi": {                             # column: diagnostic
        "normal": "Healthy",
    },
}

# Optional, NOT active: labels that could be merged into existing classes after
# the supervisor agrees. Merging the tinea variants would add ~680 images.
CANDIDATE_MERGES: dict[str, dict[str, str]] = {
    "dermaconin": {
        "Tinea Cruris": "Tinea Corporis (as 'Tinea')",
        "Tinea Faciei": "Tinea Corporis (as 'Tinea')",
        "Infected Eczema": "Eczema",
    },
}


def class_to_index(classes: list[str] | None = None) -> dict[str, int]:
    """Map each class name to its integer index, e.g. {'Contact Dermatitis': 0, ...}."""
    classes = classes or TARGET_CLASSES
    return {name: index for index, name in enumerate(classes)}


# ---------------------------------------------------------------------------
# v3: open-set "Other" class and grouped taxonomy
# ---------------------------------------------------------------------------
# Photos of skin diseases that are NOT one of the target classes. The model
# learns an extra "Other" output from them, so that at inference it can say
# "not identified" instead of forcing a guess (outlier exposure).
OTHER_CLASS = "Other"

# Diagnoses that are too close to a target class to be called "Other" (they
# would teach the model contradictory things). Matched case-insensitively.
OTHER_EXCLUDE_PATTERNS = [
    "tinea", "eczema", "dermatitis", "psoria", "scabies", "no definite", "normal", "healthy",
]

# Grouped taxonomy (experiment, notebook 13): Contact Dermatitis is a subtype
# of eczema/dermatitis and the two are often confused clinically.
GROUPED_LABELS = {
    "Contact Dermatitis": "Eczema / Dermatitis",
    "Eczema": "Eczema / Dermatitis",
}


def is_other_candidate(diagnosis: str) -> bool:
    """True when a source diagnosis may be used as an 'Other' example."""
    text = str(diagnosis).lower()
    return not any(pattern in text for pattern in OTHER_EXCLUDE_PATTERNS)


def model_classes(open_set: bool = True, classes: list[str] | None = None) -> list[str]:
    """Output classes of the model: the target classes, plus 'Other' when open-set is on."""
    classes = list(classes or TARGET_CLASSES)
    return classes + [OTHER_CLASS] if open_set else classes


def group_label(label: str) -> str:
    """Map a fine label to the grouped taxonomy (unchanged when not grouped)."""
    return GROUPED_LABELS.get(label, label)
