"""The data pipeline must catch every problem planted in the synthetic project."""

import pandas as pd

from dermalens import manifest as mf
from dermalens.labels import TARGET_CLASSES


def test_every_row_has_a_target_label_and_existing_file(fake_paths, clean_manifest):
    assert clean_manifest["label"].isin(TARGET_CLASSES).all()
    assert all((fake_paths.raw / p).is_file() for p in clean_manifest["rel_path"])


def test_only_skindisnet2_originals_are_used(clean_manifest):
    skin = clean_manifest[clean_manifest["source"] == "skindisnet"]
    assert skin["rel_path"].str.contains("skindisnet2/Preprocessed/").all()
    assert not clean_manifest["rel_path"].str.contains("Augmented|skindisnet1").any()


def test_skindisnet_file_matches_its_image_id(clean_manifest):
    skin = clean_manifest[clean_manifest["source"] == "skindisnet"]
    stems = skin["rel_path"].str.split("/").str[-1].str.removesuffix(".jpg")
    assert (("skindisnet:" + stems) == skin["image_id"]).all()


def test_poor_quality_dermaconin_row_is_dropped(fake_paths):
    table = mf.build_dermaconin_manifest(fake_paths.raw)
    assert "dermaconin:IMG_0001.jpg" not in set(table["image_id"])


def test_exact_duplicate_is_removed(fake_paths):
    table = mf.add_file_hashes(mf.build_manifest(fake_paths.raw), fake_paths.raw)
    clean, copies = mf.drop_exact_duplicates(table)
    assert len(copies) == 2 and len(clean) == len(table) - 1
    assert clean["sha256"].is_unique


def test_conflicting_duplicate_labels_raise(fake_paths):
    table = mf.add_file_hashes(mf.build_manifest(fake_paths.raw), fake_paths.raw)
    copy_rows = table[table.duplicated("sha256", keep=False)].index
    table.loc[copy_rows[0], "label"] = "Eczema"          # make the twins disagree
    try:
        mf.drop_exact_duplicates(table)
    except ValueError:
        return
    raise AssertionError("conflicting labels were not detected")


def test_near_duplicates_share_a_group(clean_manifest):
    groups = clean_manifest.set_index("image_id")["group_id"]
    assert groups["skindisnet:EC (2)"] == groups["skindisnet:EC (24)"]


def test_damaged_scin_ids_are_counted():
    assert mf.count_damaged_ids(pd.Series(["-1.00E+18", "4.55E+17", "-3205742176803893704"])) == 2
