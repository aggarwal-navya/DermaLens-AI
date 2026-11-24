"""Leakage-safe train / validation / test splits and class weights.

Rule 1: every photo of one patient (one group_id) goes into exactly one split.
Rule 2: each split should hold roughly the same mix of classes.

StratifiedGroupKFold satisfies rule 1 exactly and rule 2 approximately; the
quality of rule 2 depends on the random seed, so we try many seeds and keep
the most balanced one.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_grouped_split(manifest: pd.DataFrame, seed: int, n_folds: int = 7,
                       label_col: str = "label", group_col: str = "group_id") -> pd.DataFrame:
    """Return a copy of the manifest with a 'split' column (train / val / test).

    The data is cut into `n_folds` folds of about equal size (7 folds -> ~14% each).
    Fold 0 becomes validation, fold 1 becomes test, the remaining folds train.
    """
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [held_out for _, held_out in
             splitter.split(manifest, manifest[label_col], manifest[group_col])]

    result = manifest.copy()
    result["split"] = "train"
    result.loc[result.index[folds[0]], "split"] = "val"
    result.loc[result.index[folds[1]], "split"] = "test"
    return result


def split_balance_score(split_manifest: pd.DataFrame, label_col: str = "label") -> float:
    """How far val and test drift from train's class mix, in percentage points.

    0 would mean every class has exactly the same share in all three splits.
    Lower is better.
    """
    shares = pd.crosstab(split_manifest[label_col], split_manifest["split"], normalize="columns") * 100
    drift_val = (shares["val"] - shares["train"]).abs().max()
    drift_test = (shares["test"] - shares["train"]).abs().max()
    return float(max(drift_val, drift_test))


def choose_split_seed(manifest: pd.DataFrame, seeds=range(50), n_folds: int = 7,
                      min_per_class: int = 10) -> tuple[int, pd.DataFrame]:
    """Try many seeds and return the most balanced one plus a table of every seed's score.

    A seed is rejected if any class has fewer than `min_per_class` images in val or test.
    """
    rows = []
    for seed in seeds:
        candidate = make_grouped_split(manifest, seed=seed, n_folds=n_folds)
        counts = pd.crosstab(candidate["label"], candidate["split"])
        smallest_eval_class = int(counts[["val", "test"]].min().min())
        rows.append({
            "seed": seed,
            "max_drift_pp": round(split_balance_score(candidate), 2),
            "smallest_val_or_test_class": smallest_eval_class,
            "valid": smallest_eval_class >= min_per_class,
        })
    scores = pd.DataFrame(rows).sort_values(["valid", "max_drift_pp"], ascending=[False, True])
    best_seed = int(scores.iloc[0]["seed"])
    return best_seed, scores.reset_index(drop=True)


def check_no_leakage(split_manifest: pd.DataFrame, group_col: str = "group_id") -> None:
    """Stop with an error if any patient (group) appears in more than one split,
    or if the same file appears twice."""
    groups = split_manifest.groupby("split")[group_col].agg(set)
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = groups[first] & groups[second]
        assert not shared, f"{len(shared)} patients appear in both {first} and {second}"
    assert split_manifest["rel_path"].is_unique, "a file appears more than once"
    if "sha256" in split_manifest:
        assert split_manifest["sha256"].is_unique, "two identical files survived de-duplication"


def compute_class_weights(train: pd.DataFrame, classes: list[str], label_col: str = "label") -> dict[str, float]:
    """Inverse-frequency weights: w_c = N / (C * n_c).

    A class with half the average number of images gets weight 2, so mistakes
    on rare classes cost more in the loss.
    """
    counts = train[label_col].value_counts().reindex(classes).fillna(0)
    if (counts == 0).any():
        raise ValueError(f"Classes missing from train: {list(counts[counts == 0].index)}")
    weights = len(train) / (len(classes) * counts)
    return {name: round(float(weight), 4) for name, weight in weights.items()}


def split_summary(split_manifest: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Images per class and split, with totals; the table that goes in the report."""
    table = pd.crosstab(split_manifest[label_col], split_manifest["split"], margins=True, margins_name="Total")
    return table[[c for c in ("train", "val", "test", "Total") if c in table.columns]]
