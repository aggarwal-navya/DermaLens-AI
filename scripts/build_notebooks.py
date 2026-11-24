"""Regenerate every notebook in notebooks/ from the sources in scripts/nbsrc/.

Usage:  python scripts/build_notebooks.py
Editing the .py sources and rebuilding keeps all notebooks consistent
(same setup cell, same header layout).
"""

from pathlib import Path

from nbsrc.common import build
from nbsrc.nb00_03 import nb00, nb01, nb02, nb03
from nbsrc.nb04_06 import nb04, nb05, nb06
from nbsrc.nb07_14 import nb07, nb08, nb09, nb10, nb11, nb12, nb13, nb14

NOTEBOOKS = {
    "00_setup_and_environment_check.ipynb": nb00,
    "01_data_audit.ipynb": nb01,
    "02_label_taxonomy.ipynb": nb02,
    "03_build_clean_manifest.ipynb": nb03,
    "04_eda_and_data_quality.ipynb": nb04,
    "05_splits_and_class_weights.ipynb": nb05,
    "06_dataset_and_transforms.ipynb": nb06,
    "07_training_recipe.ipynb": nb07,
    "08_model_comparison_and_metadata_ablation.ipynb": nb08,
    "09_ensemble_and_test_evaluation.ipynb": nb09,
    "10_open_set_evaluation.ipynb": nb10,
    "11_explainability_gradcam.ipynb": nb11,
    "12_source_shortcut_test.ipynb": nb12,
    "13_grouped_taxonomy.ipynb": nb13,
    "14_export_for_web_app.ipynb": nb14,
}

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "notebooks"
    out_dir.mkdir(exist_ok=True)
    for name, make_cells in NOTEBOOKS.items():
        build(out_dir / name, make_cells())
        print("built", name)
