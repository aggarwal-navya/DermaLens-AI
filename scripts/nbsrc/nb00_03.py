"""Notebooks 00-03: setup, raw-data audit, label taxonomy, clean manifest."""

from .common import SETUP_CODE, SETUP_MD, code, header, md


def nb00():
    return [
        header("00", "Setup & Environment Check",
               "Confirm Colab, Drive, libraries and every raw dataset are in place before any analysis.",
               "`data/raw/*`, `configs/baseline.yaml`", "nothing (optional: archives v1 outputs)",
               "~1 min", "01 · Data Audit"),
        md("""
        ### Project layout
        ```
        DermaLens_AI/                  <- this Drive folder is also the Git repository
        ├── configs/baseline.yaml      every setting (image size, learning rate, ...)
        ├── src/dermalens/             shared code, imported by every notebook
        ├── notebooks/                 00 → 12, run in order
        ├── data/raw/                  original downloads, never modified
        ├── data/processed/            clean tables built from raw
        ├── data/splits/               train / val / test CSVs + class_info.json
        ├── data/cache/                small resized copies of the images
        ├── models/                    checkpoints + the web-app bundle
        ├── reports/                   audit CSVs, figures, results
        └── app/                       FastAPI web prototype
        ```
        """),
        SETUP_MD, SETUP_CODE,
        md("## 1. Hardware and library versions\nTraining needs a GPU: *Runtime → Change runtime type → T4 GPU*."),
        code('''
        import platform
        import importlib

        print("Python", platform.python_version())
        for package in ["numpy", "pandas", "sklearn", "PIL", "torch", "torchvision", "timm", "imagehash"]:
            try:
                module = importlib.import_module(package)
                print(f"{package:<12} {getattr(module, '__version__', 'installed')}")
            except ImportError:
                print(f"{package:<12} MISSING  -> run: pip install -r requirements.txt")

        import torch
        print("\\nGPU available:", torch.cuda.is_available(),
              f"({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "- training will be slow")
        '''),
        md("## 2. Are all raw datasets where the code expects them?"),
        code('''
        import pandas as pd

        # Each source dataset and the one metadata file the pipeline depends on.
        expected = {
            "dermaconin": paths.raw / "dermaconin" / "METADATA" / "Skin_Metadata-1.csv",
            "skindisnet": paths.raw / "skindisnet" / "skindisnet2" / "SkinDisNet_Metadata.csv",
            "mcsi":       paths.raw / "mcsi" / "extracted" / "metadata.csv",
            "scin":       paths.raw / "scin" / "dataset_scin_cases.csv",
        }
        check = pd.DataFrame([
            {"dataset": name, "metadata_file": str(path.relative_to(paths.raw)), "found": path.is_file()}
            for name, path in expected.items()
        ])
        display(check)
        assert check.loc[check["dataset"] != "scin", "found"].all(), "A required dataset is missing: fix before continuing."
        '''),
        md("""
        ## 3. One-time clean-up of version-1 outputs (optional)

        Version 1 left an older split set in `data/processed/` (`train.csv`, `val.csv`, `test.csv`,
        `class_map.json`). They are **moved, not deleted**, to `data/processed/_archive_v1/`, so nobody
        trains on the wrong split by mistake. Set the flag to `True` once, after the team agrees.
        """),
        code('''
        ARCHIVE_V1_OUTPUTS = False

        stale_files = ["train.csv", "val.csv", "test.csv", "class_map.json",
                       "dermaconin_manifest.csv", "skindisnet_manifest.csv"]
        archive_dir = paths.processed / "_archive_v1"
        for name in stale_files:
            source = paths.processed / name
            if not source.exists():
                continue
            if ARCHIVE_V1_OUTPUTS:
                archive_dir.mkdir(exist_ok=True)
                source.rename(archive_dir / name)
                print("moved   ", name, "->", archive_dir.name)
            else:
                print("would move", name, "(set ARCHIVE_V1_OUTPUTS = True)")
        '''),
        md("**Done.** If every check above passed, continue with notebook 01."),
    ]


def nb01():
    return [
        header("01", "Data Audit",
               "Count and describe everything in `data/raw` before using it: files, metadata, labels, duplicates.",
               "`data/raw/*`", "`reports/dataset_audit/inventory.csv`, `inventory_summary.csv`, `metadata_overview.csv`",
               "5-10 min (lists ~43,000 files on Drive)", "02 · Label Taxonomy"),
        SETUP_MD, SETUP_CODE,
        md("## 1. File inventory\nEvery file under `data/raw`, with its type and size. Nothing is opened yet."),
        code('''
        import pandas as pd
        from dermalens.audit import inventory_files, summarise_inventory, skindisnet_mirror_check

        DATASETS = [d.name for d in sorted(paths.raw.iterdir()) if d.is_dir()]
        inventory = inventory_files(paths.raw, DATASETS)
        inventory.to_csv(paths.audit / "inventory.csv", index=False)

        summary = summarise_inventory(inventory)
        summary.to_csv(paths.audit / "inventory_summary.csv", index=False)
        display(summary)
        '''),
        md("Images per sub-folder. SkinDisNet's sub-folders matter: only one of them holds originals."),
        code('''
        images = inventory[inventory["kind"] == "image"]
        display(images.groupby(["dataset", "subfolder"]).size().rename("images").reset_index())
        '''),
        md("""
        ## 2. SkinDisNet: how many photos are really unique?

        `skindisnet1/Preprocessed` names files `AD_01.jpg`, `skindisnet2/Preprocessed` names the same photos
        `AD (1).jpg`. The two `Augmented` folders are rotated/flipped copies. If both folder names match
        one-to-one, the 26,943 files hold only **1,710** distinct photos.
        """),
        code('''
        mirror = skindisnet_mirror_check(inventory)
        print(mirror)
        if mirror["matching_names"] == mirror["skindisnet2_originals"]:
            print("-> skindisnet1/Preprocessed mirrors skindisnet2/Preprocessed. Use skindisnet2 only;"
                  " augmented folders are never used for evaluation.")
        '''),
        md("## 3. Metadata files: shape and columns\nIDs are read as text so long numbers are not rounded."),
        code('''
        overview = []
        for table_path in sorted(inventory.loc[inventory["suffix"] == ".csv", "rel_path"]):
            table = pd.read_csv(paths.raw / table_path, dtype=str, nrows=None)
            overview.append({"file": table_path, "rows": len(table), "columns": len(table.columns),
                             "column_names": ", ".join(table.columns[:12]) + (" ..." if len(table.columns) > 12 else "")})
        overview = pd.DataFrame(overview)
        overview.to_csv(paths.audit / "metadata_overview.csv", index=False)
        display(overview)
        '''),
        md("## 4. DermaCon-IN: labels, subjects, quality"),
        code('''
        derma = pd.read_csv(paths.raw / "dermaconin" / "METADATA" / "Skin_Metadata-1.csv", dtype={"Subject_ID": str})

        label_counts = derma["Disease_label"].str.strip().value_counts()
        print(f"{len(derma):,} images | {derma['Subject_ID'].nunique():,} subjects | {label_counts.size} diagnoses")
        print(f"Diagnoses with fewer than 20 images: {(label_counts < 20).sum()} "
              f"(the long tail) | top 10 cover {label_counts.head(10).sum() / len(derma):.0%} of images")
        print("Every Image_name unique:", derma["Image_name"].is_unique)
        display(label_counts.head(15).rename("images").to_frame())
        display(derma["Quality"].value_counts().rename("images").to_frame())
        '''),
        md("## 5. SkinDisNet: labels and patients"),
        code('''
        skin = pd.read_csv(paths.raw / "skindisnet" / "skindisnet2" / "SkinDisNet_Metadata.csv", dtype=str)
        print(f"{len(skin):,} images | {skin['Patient_id'].nunique()} patients")
        print("Patients with more than one diagnosis:", int((skin.groupby("Patient_id")["Diagnosis"].nunique() > 1).sum()))
        display(skin["Diagnosis"].value_counts().rename("images").to_frame())
        '''),
        md("## 6. MCSI: the healthy-skin source"),
        code('''
        mcsi = pd.read_csv(paths.raw / "mcsi" / "extracted" / "metadata.csv", dtype=str)
        display(mcsi["diagnostic"].value_counts().rename("images").to_frame())
        '''),
        md("""
        ## 7. SCIN: damaged case IDs (reference dataset only)

        In version 1, SCIN reported 2,987 "duplicate" case IDs. The real cause: the CSV was opened and
        re-saved in a spreadsheet tool, which turned 19-digit IDs into text like `-1.00E+18`. Many
        different IDs collapsed into the same text. **Fix:** download `dataset_scin_cases.csv` and
        `dataset_scin_labels.csv` again from the SCIN GitHub page and never open them in Excel/Sheets.
        """),
        code('''
        from dermalens.manifest import count_damaged_ids

        scin_cases_path = paths.raw / "scin" / "dataset_scin_cases.csv"
        if scin_cases_path.exists():
            scin_cases = pd.read_csv(scin_cases_path, dtype={"case_id": str})
            damaged = count_damaged_ids(scin_cases["case_id"])
            print(f"SCIN rows: {len(scin_cases):,} | IDs in scientific notation (damaged): {damaged:,}")
            if damaged:
                print("-> Re-download the SCIN CSVs before using SCIN for anything.")
        else:
            print("SCIN not present - fine, it is reference-only.")
        '''),
        md("""
        ## 8. What this audit tells us

        | Finding | Consequence for the next notebooks |
        |---|---|
        | SkinDisNet's 26,943 files are 1,710 photos | Use `skindisnet2/Preprocessed` only (notebook 03) |
        | DermaCon-IN has a long tail of rare diagnoses | Pick a few well-populated classes (notebook 02) |
        | No source has a healthy class except MCSI | Healthy comes from one source: a shortcut risk (notebooks 04, 11) |
        | DermaCon-IN groups photos by `Subject_ID`, SkinDisNet by `Patient_id` | Split by patient (notebook 05) |
        | SCIN IDs are damaged | Re-download before any SCIN use |
        """),
    ]


def nb02():
    return [
        header("02", "Label Taxonomy",
               "Decide, with evidence, which six classes the model predicts and which source labels feed each class.",
               "raw metadata, `src/dermalens/labels.py`", "`reports/dataset_audit/label_candidates.csv`, `label_mapping.csv`",
               "<1 min", "03 · Build Clean Manifest"),
        SETUP_MD, SETUP_CODE,
        md("""
        ## 1. Selection criteria

        A disease becomes a target class when it is:
        1. **common in Indian primary care** (infectious and inflammatory conditions, not cancer),
        2. **well represented in DermaCon-IN** (at least ~140 images, so the test set has enough examples),
        3. ideally **also present in SkinDisNet**, to add South Asian smartphone photos.

        Healthy skin is the sixth class because a screening tool must be able to say "nothing found".
        """),
        code('''
        import pandas as pd

        derma = pd.read_csv(paths.raw / "dermaconin" / "METADATA" / "Skin_Metadata-1.csv")
        skin = pd.read_csv(paths.raw / "skindisnet" / "skindisnet2" / "SkinDisNet_Metadata.csv")
        derma_counts = derma["Disease_label"].str.strip().value_counts()
        skin_counts = skin["Diagnosis"].str.strip().value_counts()

        # The 12 most common DermaCon-IN diagnoses, side by side with SkinDisNet.
        candidates = pd.DataFrame({
            "dermaconin_images": derma_counts.head(12),
            "dermaconin_subjects": derma.groupby(derma["Disease_label"].str.strip())["Subject_ID"].nunique(),
            "skindisnet_images": skin_counts,
        }).dropna(subset=["dermaconin_images"]).fillna(0).astype(int).sort_values("dermaconin_images", ascending=False)
        candidates.to_csv(paths.audit / "label_candidates.csv")
        display(candidates)
        '''),
        md("""
        ## 2. The decision

        | Class | Why it is in | Sources |
        |---|---|---|
        | Tinea Corporis | most common fungal infection; 323 DermaCon-IN + 275 SkinDisNet | both |
        | Scabies | common parasitic infection | both |
        | Eczema | common inflammatory condition | both |
        | Contact Dermatitis | common; frequently confused with eczema (a hard, useful pair) | both |
        | Psoriasis | common chronic condition, 159 images | DermaCon-IN only |
        | Healthy | the "nothing found" answer | MCSI only |

        **Not chosen:** Vitiligo and Tinea Cruris are larger but would change the project's scope (pigment
        disorder; groin-site tinea). Acne needs different photos (face close-ups). These are listed in
        `CANDIDATE_MERGES` in `labels.py` for discussion with the supervisor, but are **not active**.

        The mapping below is read from `src/dermalens/labels.py`, the single place where it is defined.
        """),
        code('''
        from dermalens.labels import CANDIDATE_MERGES, SOURCE_LABEL_MAP, TARGET_CLASSES

        source_counts = {"dermaconin": derma_counts, "skindisnet": skin_counts}
        rows = []
        for source, mapping in SOURCE_LABEL_MAP.items():
            for source_label, target in mapping.items():
                images = source_counts.get(source, pd.Series(dtype=int)).get(source_label)
                rows.append({"source": source, "source_label": source_label, "target_class": target,
                             "images_in_source": images})
        mapping_table = pd.DataFrame(rows)
        mapping_table.to_csv(paths.audit / "label_mapping.csv", index=False)
        display(mapping_table)
        print("Target classes (model output order):", TARGET_CLASSES)
        '''),
        md("## 3. How sure were the dermatologists? (DermaCon-IN `Confidence`, 1 = unsure, 5 = certain)"),
        code('''
        chosen = derma[derma["Disease_label"].str.strip().isin(SOURCE_LABEL_MAP["dermaconin"])]
        confidence = pd.crosstab(chosen["Disease_label"].str.strip(), chosen["Confidence"])
        display(confidence)
        print(f"Share of chosen images labelled with confidence >= 4: {(chosen['Confidence'] >= 4).mean():.0%}")
        '''),
        md("## 4. What merging the candidate labels would add (not applied)"),
        code('''
        for source, merges in CANDIDATE_MERGES.items():
            for source_label, idea in merges.items():
                print(f"{source}: '{source_label}' -> {idea}: +{int(derma_counts.get(source_label, 0))} images")
        '''),
        md("**Done.** The taxonomy is fixed in `labels.py`; notebook 03 applies it."),
    ]


def nb03():
    return [
        header("03", "Build Clean Manifest",
               "Turn the raw folders into one clean table: one row per unique, readable image with label and patient.",
               "`data/raw/*`, `labels.py`",
               "`data/processed/manifest_clean.csv`, `reports/dataset_audit/{cleaning_log, exact_duplicates, near_duplicate_pairs}.csv`",
               "20-40 min the first time (every image is read once); seconds afterwards (cached)",
               "04 · EDA & Data Quality"),
        SETUP_MD, SETUP_CODE,
        md("""
        ## 1. Build the image table from raw metadata

        Each source has its own builder in `src/dermalens/manifest.py`:

        * **DermaCon-IN**: `Image_name` in the metadata *is* the file name. Rows marked "quality insufficient" are dropped.
        * **SkinDisNet**: `Image_id` + `.jpg` inside `skindisnet2/Preprocessed/<Folder_name>/` (no row-order guessing).
        * **MCSI**: rows with `diagnostic == "normal"` become **Healthy**; each image is its own group.

        Every builder checks that each file it lists exists, and stops with a clear error if not.
        """),
        code('''
        import pandas as pd
        from dermalens import manifest as mf
        from dermalens.labels import model_classes

        CLASSES = model_classes(open_set=CONFIG["data"]["open_set"])   # 6 targets (+ "Other")
        cleaning_log = []                                  # one line per cleaning step, for the report

        def log_step(step, table, note=""):
            cleaning_log.append({"step": step, "images": len(table), "patients": table["group_id"].nunique(), "note": note})
            print(f"{step:<40} {len(table):>6,} images  {table['group_id'].nunique():>5,} patients  {note}")

        # Raw size of each source before any label mapping, for the funnel.
        for name, builder in mf.SOURCE_BUILDERS.items():
            log_step(f"{name}: all usable rows", builder(paths.raw))

        manifest = mf.build_manifest(paths.raw, open_set=CONFIG["data"]["open_set"],
                                     other_cap_per_diagnosis=CONFIG["data"]["other_cap_per_diagnosis"])
        log_step(f"after label mapping ({len(CLASSES)} classes)", manifest)
        display(manifest.groupby(["label", "source"]).size().unstack(fill_value=0))
        '''),
        md("""
        ### v3: the "Other" class (open-set recognition)

        A model trained only on 6 classes must pick one of them for *any* photo, so a photo of vitiligo or
        acne gets a confident wrong answer. We add a 7th output, **Other**, trained on up to
        `other_cap_per_diagnosis` photos of each *non-target* diagnosis (vitiligo, acne, urticaria, ...).
        Diagnoses close to a target class (any tinea, eczema, dermatitis, psoriasis, scabies) are excluded
        so the model is never taught contradictions. In the app, "Other" is shown as **Not identified**.
        This technique is called *outlier exposure* (Hendrycks et al., 2019).
        """),
        code('''
        other = manifest[manifest["label"] == "Other"]
        print(f"Other: {len(other)} images from {other['source_label'].nunique()} diagnoses")
        display(other["source_label"].value_counts().head(25).to_frame("images"))
        '''),
        md("""
        ### v3: patient metadata (age group, sex, body region)

        Dermatologists never diagnose from a picture alone: *where* the rash is and *who* has it matter
        (scabies between the fingers, tinea in the groin, eczema in children's skin folds). Each row now has
        `meta_age`, `meta_sex` and `meta_regions` (11 body regions, pipe-separated), mapped by
        `src/dermalens/metadata.py`. Missing values stay missing - the model learns to cope with that.
        """),
        code('''
        coverage = manifest.assign(has_age=manifest["meta_age"].notna(), has_sex=manifest["meta_sex"].notna(),
                                   has_region=manifest["meta_regions"].fillna("").ne(""))
        display((coverage.groupby("source")[["has_age", "has_sex", "has_region"]].mean() * 100).round(1))
        '''),
        md("""
        ## 2. Exact duplicates (SHA-256)

        A SHA-256 fingerprint is computed from each file's bytes; identical files have identical
        fingerprints. Only the ~2,600 manifest images are hashed (not all 43,000 raw files), and results are
        cached in `reports/dataset_audit/sha256_cache.csv`, so a re-run takes seconds.
        If two identical files ever had **different** labels, the function stops: a person must decide.
        """),
        code('''
        manifest = mf.add_file_hashes(manifest, paths.raw, cache_csv=paths.audit / "sha256_cache.csv")
        manifest, exact_copies = mf.drop_exact_duplicates(manifest)
        exact_copies.to_csv(paths.audit / "exact_duplicates.csv", index=False)
        log_step("after removing exact duplicates", manifest, f"{len(exact_copies)} files were in duplicate groups")
        display(exact_copies[["sha256", "image_id", "label", "group_id"]].head(20))
        '''),
        md("""
        ## 3. Decode every image once

        Opening each image proves it is readable and measures it: width, height, average colour,
        brightness, contrast and a **perceptual hash** (a fingerprint of how the picture *looks*).
        Results are cached in `image_properties_cache.csv`. Unreadable files are removed.
        """),
        code('''
        manifest = mf.add_image_properties(manifest, paths.raw, cache_csv=paths.audit / "image_properties_cache.csv")
        unreadable = manifest[~manifest["readable"].astype(bool)]
        unreadable.to_csv(paths.audit / "unreadable_images.csv", index=False)
        manifest = manifest[manifest["readable"].astype(bool)].reset_index(drop=True)
        log_step("after removing unreadable images", manifest, f"{len(unreadable)} unreadable")
        display(unreadable[["image_id", "error"]].head())
        '''),
        md("""
        ## 4. Near duplicates (same photo, re-saved or resized)

        Two files can differ in bytes yet show the same photo (resized, re-compressed). Their perceptual
        hashes then differ by only a few of 64 bits. We **do not delete** them; we give them the **same
        patient group**, so both land in the same split and cannot leak between train and test.
        **Review the pairs table** before trusting it: a pair with two different labels deserves a look.
        """),
        code('''
        MAX_BITS = CONFIG["data"]["near_duplicate_bits"]
        pairs = mf.find_near_duplicates(manifest, max_distance=MAX_BITS)
        pairs.to_csv(paths.audit / "near_duplicate_pairs.csv", index=False)
        print(f"Near-duplicate pairs (<= {MAX_BITS} bits apart): {len(pairs)}")
        print("Pairs crossing patient groups:", int((pairs["group_a"] != pairs["group_b"]).sum()))
        print("Pairs with different labels (check by eye!):", int((pairs["label_a"] != pairs["label_b"]).sum()))
        display(pairs.sort_values("distance").head(20))

        manifest = mf.merge_groups_of_near_duplicates(manifest, pairs)
        largest_group = manifest["group_id"].value_counts().max()
        log_step("after merging near-duplicate groups", manifest, f"largest group = {largest_group} images")
        '''),
        md("## 5. Final checks and save"),
        code('''
        assert manifest["image_id"].is_unique
        assert manifest["sha256"].is_unique, "exact duplicates remain"
        assert manifest["label"].isin(CLASSES).all()

        manifest.to_csv(paths.processed / "manifest_clean.csv", index=False)
        pd.DataFrame(cleaning_log).to_csv(paths.audit / "cleaning_log.csv", index=False)

        print(f"Saved {len(manifest):,} images from {manifest['group_id'].nunique():,} patient groups.")
        display(pd.DataFrame(cleaning_log))
        '''),
        md("""
        **Done.** `manifest_clean.csv` is the only image table later notebooks use. The cleaning log above is
        the "data funnel" table for the report: how many images survived each step and why.
        """),
    ]
