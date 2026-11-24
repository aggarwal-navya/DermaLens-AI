"""Shared test fixture: a small fake project built once per test session."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


@pytest.fixture(scope="session")
def fake_paths(tmp_path_factory):
    """A ProjectPaths pointing at a freshly generated synthetic project."""
    from make_synthetic_project import make_project

    from dermalens.paths import ProjectPaths

    root = make_project(tmp_path_factory.mktemp("project") / "DermaLens_AI")
    return ProjectPaths(root).ensure()


@pytest.fixture(scope="session")
def clean_manifest(fake_paths):
    """The manifest after hashing, de-duplication, decoding and near-duplicate grouping."""
    from dermalens import manifest as mf

    table = mf.build_manifest(fake_paths.raw)
    table = mf.add_file_hashes(table, fake_paths.raw)
    table, _ = mf.drop_exact_duplicates(table)
    table = mf.add_image_properties(table, fake_paths.raw)
    pairs = mf.find_near_duplicates(table, max_distance=6)
    return mf.merge_groups_of_near_duplicates(table, pairs)
