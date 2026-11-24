"""Splits must never share a patient, and class weights must follow the formula."""

import pytest

from dermalens import splits as sp
from dermalens.labels import TARGET_CLASSES


def test_split_has_no_patient_leakage(clean_manifest):
    split = sp.make_grouped_split(clean_manifest, seed=0)
    sp.check_no_leakage(split)
    assert set(split["split"]) == {"train", "val", "test"}


def test_leakage_check_catches_a_shared_patient(clean_manifest):
    split = sp.make_grouped_split(clean_manifest, seed=0)
    patient = split.loc[split["split"] == "train", "group_id"].iloc[0]
    split.loc[split.index[split["split"] == "test"][0], "group_id"] = patient
    with pytest.raises(AssertionError):
        sp.check_no_leakage(split)


def test_seed_search_returns_the_most_balanced_seed(clean_manifest):
    best, scores = sp.choose_split_seed(clean_manifest, seeds=range(5), min_per_class=1)
    assert best == scores.iloc[0]["seed"]
    assert scores["max_drift_pp"].iloc[0] == scores.loc[scores["valid"], "max_drift_pp"].min()


def test_class_weights_formula(clean_manifest):
    train = sp.make_grouped_split(clean_manifest, seed=0).query("split == 'train'")
    weights = sp.compute_class_weights(train, TARGET_CLASSES)
    counts = train["label"].value_counts()
    for name in TARGET_CLASSES:
        assert weights[name] == pytest.approx(len(train) / (len(TARGET_CLASSES) * counts[name]), rel=1e-3)
