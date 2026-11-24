"""Notebooks 04-06: exploratory analysis, splits, dataset and transforms."""

from .common import SETUP_CODE, SETUP_MD, code, header, md


def nb04():
    return [
        header("04", "EDA & Data Quality Analysis",
               "Understand the clean data before any model sees it: balance, patients, image properties, "
               "skin tone, and how easily the *source* can be guessed (shortcut risk).",
               "`data/processed/manifest_clean.csv`", "`reports/figures/eda_*.png`, `reports/results/eda_summary.json`",
               "2-3 min", "05 · Splits & Class Weights"),
        SETUP_MD, SETUP_CODE,
        code('''
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from dermalens.labels import model_classes

        TARGET_CLASSES = model_classes(open_set=CONFIG["data"]["open_set"])
        from dermalens.utils import save_json

        sns.set_theme(style="whitegrid")
        data = pd.read_csv(paths.processed / "manifest_clean.csv", dtype={"phash": str})
        findings = {}                                      # numbers quoted in the report

        def save_figure(name):
            plt.tight_layout()
            plt.savefig(paths.figures / f"eda_{name}.png", dpi=150)
            plt.show()

        print(f"{len(data):,} images | {data['group_id'].nunique():,} patient groups | {data['source'].nunique()} sources")
        '''),
        md("## 1. Class balance\nA large gap between classes means the loss needs class weights (notebook 05)."),
        code('''
        per_class = pd.crosstab(data["label"], data["source"]).reindex(TARGET_CLASSES)
        per_class.plot.barh(stacked=True, figsize=(9, 4), title="Images per class, coloured by source")
        plt.xlabel("images")
        save_figure("class_balance")

        totals = per_class.sum(axis=1)
        findings["imbalance_ratio_max_over_min"] = round(float(totals.max() / totals.min()), 2)
        print(totals.to_string())
        print("Largest / smallest class:", findings["imbalance_ratio_max_over_min"])
        '''),
        md("""
        ## 2. Source vs class: the confounding check

        If a class comes from only one dataset, the model can learn *"photos that look like dataset X"*
        instead of the disease. 100% in one column below = that class and that source are inseparable.
        """),
        code('''
        share = (pd.crosstab(data["label"], data["source"], normalize="index") * 100).reindex(TARGET_CLASSES).round(0)
        plt.figure(figsize=(6, 4))
        sns.heatmap(share, annot=True, fmt=".0f", cmap="Oranges", cbar_kws={"label": "% of class"})
        plt.title("Where each class's images come from (%)")
        save_figure("source_by_class")

        single_source = share[(share == 100).any(axis=1)].index.tolist()
        findings["single_source_classes"] = single_source
        print("Classes from a single source:", single_source)
        '''),
        md("## 3. Photos per patient\nPatients with many photos dominate if split carelessly; this is why notebook 05 splits by patient."),
        code('''
        per_patient = data.groupby(["source", "group_id"]).size().rename("photos").reset_index()
        display(per_patient.groupby("source")["photos"].describe().round(1))
        findings["max_photos_one_patient"] = int(per_patient["photos"].max())

        per_patient[per_patient["source"] != "mcsi"].hist(column="photos", by="source", bins=20, figsize=(10, 3), sharey=True)
        save_figure("photos_per_patient")
        '''),
        md("## 4. Image size by source\nDifferent sizes per source are a clue a model could exploit; resizing every image to the same size removes it."),
        code('''
        display(data.groupby("source")[["width", "height"]].describe().round(0))
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=data, x="width", y="height", hue="source", alpha=0.5, s=15)
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Original image size (log scale)")
        save_figure("image_size_by_source")
        '''),
        md("## 5. Colour, brightness and contrast\nDo the sources look systematically different? Compared within the same class where possible."),
        code('''
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, column in zip(axes, ["brightness", "contrast", "mean_r"]):
            sns.boxplot(data=data, x="label", y=column, hue="source", ax=ax, order=TARGET_CLASSES, fliersize=1)
            ax.tick_params(axis="x", rotation=40)
            ax.set_title(column)
        save_figure("colour_by_class_and_source")
        '''),
        md("""
        ## 6. How guessable is the source from simple statistics?

        A plain logistic regression sees **only** size and average colour (no pixels, no lesion). If it can
        name the source accurately, a deep network certainly can, and Healthy/Psoriasis could be "solved"
        through the source. Notebook 11 repeats this test with a real CNN.
        """),
        code('''
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        simple_features = ["width", "height", "mean_r", "mean_g", "mean_b", "brightness", "contrast"]
        X = np.column_stack([np.log(data["width"]), np.log(data["height"]), data[simple_features[2:]]])
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        folds = GroupKFold(n_splits=5)

        source_acc = cross_val_score(model, X, data["source"], groups=data["group_id"], cv=folds).mean()
        label_acc = cross_val_score(model, X, data["label"], groups=data["group_id"], cv=folds).mean()
        majority = data["source"].value_counts(normalize=True).max()
        findings["source_from_simple_stats_accuracy"] = round(float(source_acc), 3)
        findings["label_from_simple_stats_accuracy"] = round(float(label_acc), 3)
        print(f"Source predicted from size+colour only: {source_acc:.1%} (always guessing the biggest source: {majority:.1%})")
        print(f"Label  predicted from size+colour only: {label_acc:.1%}")
        '''),
        md("## 7. Skin tone coverage\nOnly DermaCon-IN has Fitzpatrick (FST) and Monk (MST) labels; MCSI has a light/dark tag."),
        code('''
        derma = data[data["source"] == "dermaconin"]
        fst_table = pd.crosstab(derma["label"], derma["fst"].astype("Int64"), margins=True)
        display(fst_table)
        display(data.loc[data["source"] == "mcsi", "tone_ld"].value_counts().rename("Healthy images").to_frame())
        findings["images_with_fst"] = int(data["fst"].notna().sum())
        findings["fst_counts"] = derma["fst"].value_counts().sort_index().to_dict()

        derma["mst"].plot.hist(bins=range(1, 12), rwidth=0.9, figsize=(6, 3), title="Monk skin tone (DermaCon-IN)")
        save_figure("monk_skin_tone")
        '''),
        md("## 8. Label confidence and demographics"),
        code('''
        display(pd.crosstab(derma["label"], derma["label_confidence"]))
        display(pd.crosstab(data["source"], data["sex"].fillna("unknown")))
        '''),
        md("## 9. Most common body sites per class (DermaCon-IN; one photo can list several)"),
        code('''
        sites = derma.assign(site=derma["body_site"].str.split(",")).explode("site")
        sites["site"] = sites["site"].str.strip()
        display(sites.groupby("label")["site"].agg(lambda s: ", ".join(s.value_counts().head(4).index)).to_frame("top sites"))
        '''),
        md("## 10. Look at the data\nFive random photos per class. Always look before training: label errors and odd images show up here."),
        code('''
        from dermalens.data import load_rgb

        per_row = 5
        fig, axes = plt.subplots(len(TARGET_CLASSES), per_row, figsize=(per_row * 2.4, len(TARGET_CLASSES) * 2.5))
        for row, label in enumerate(TARGET_CLASSES):
            examples = data[data["label"] == label].sample(min(per_row, (data["label"] == label).sum()), random_state=CONFIG["seed"])
            for col in range(per_row):
                ax = axes[row, col]
                ax.axis("off")
                if col < len(examples):
                    item = examples.iloc[col]
                    image = load_rgb(paths.raw / item["rel_path"])
                    image.thumbnail((256, 256))
                    ax.imshow(image)
                    ax.set_title(f"{item['source']}", fontsize=8)
            axes[row, 0].text(-0.1, 0.5, label, transform=axes[row, 0].transAxes, ha="right", va="center", fontsize=10)
        save_figure("sample_grid")
        '''),
        md("""
        ## 10b. Patient metadata: does it carry diagnostic signal? (v3)

        If body region or age differs strongly between classes, giving the model this information should
        raise accuracy. The heat maps show, for each class, the share of photos per body region and per
        age group. A chi-square test checks whether the differences could be chance.
        """),
        code('''
        from scipy.stats import chi2_contingency
        from dermalens.metadata import BODY_REGIONS

        regions = data.assign(region=data["meta_regions"].fillna("").str.split("|")).explode("region")
        regions = regions[regions["region"].isin(list(BODY_REGIONS))]
        region_share = (pd.crosstab(regions["label"], regions["region"], normalize="index") * 100).round(0)
        age_share = (pd.crosstab(data["label"], data["meta_age"], normalize="index") * 100).round(0)
        fig, axes = plt.subplots(1, 2, figsize=(16, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})
        sns.heatmap(region_share, annot=True, fmt=".0f", cmap="Blues", ax=axes[0], cbar=False)
        axes[0].set_title("Body region by class (% of the class's photos)")
        sns.heatmap(age_share, annot=True, fmt=".0f", cmap="Greens", ax=axes[1], cbar=False)
        axes[1].set_title("Age group by class (%)")
        save_figure("metadata_by_class")

        for name, table in {"body region": pd.crosstab(regions["label"], regions["region"]),
                            "age group": pd.crosstab(data["label"], data["meta_age"]),
                            "sex": pd.crosstab(data["label"], data["meta_sex"])}.items():
            chi2, p_value, dof, _ = chi2_contingency(table)
            print(f"{name:<12} chi2 = {chi2:8.1f}  dof = {dof:3d}  p = {p_value:.2g}")
            findings[f"metadata_{name.replace(' ', '_')}_p_value"] = float(p_value)
        '''),
        md("A tiny p-value means the field differs between classes → worth feeding to the model (notebook 08 measures the actual gain)."),
        md("## 11. Findings → decisions"),
        code('''
        save_json(findings, paths.results / "eda_summary.json")
        for key, value in findings.items():
            print(f"- {key}: {value}")
        '''),
        md("""
        | If the numbers above show… | …then we |
        |---|---|
        | an imbalance ratio well above 1 | use inverse-frequency class weights (05) and report macro-F1, not accuracy |
        | single-source classes | report results per source (09) and run the shortcut test (11) |
        | sources guessable from size/colour | resize every image identically (06) and use random crops |
        | many photos for some patients | split by patient group (05) |
        | few images with FST 6 | report FST 3-4 vs 5-6 with confidence intervals (09) |
        """),
    ]


def nb05():
    return [
        header("05", "Splits & Class Weights",
               "Create the train / validation / test split once, leakage-free and as balanced as possible, then freeze it.",
               "`data/processed/manifest_clean.csv`",
               "`data/splits/{train,val,test,manifest_split}.csv`, `data/splits/class_info.json`",
               "~1 min", "06 · Dataset & Transforms"),
        SETUP_MD, SETUP_CODE,
        md("""
        ## 1. Choose the most balanced seed

        `StratifiedGroupKFold` keeps every patient in one fold and balances classes approximately. How well
        it balances depends on the random seed, so we try many seeds and keep the one whose val and test
        class mix is closest to train's. A seed is rejected if any class has too few val/test images.
        """),
        code('''
        import json
        from datetime import date

        import matplotlib.pyplot as plt
        import pandas as pd
        from dermalens import splits as sp
        from dermalens.labels import model_classes
        from dermalens.utils import save_json

        TARGET_CLASSES = model_classes(open_set=CONFIG["data"]["open_set"])   # includes "Other" in v3

        data = pd.read_csv(paths.processed / "manifest_clean.csv", dtype={"phash": str})
        settings = CONFIG["data"]

        best_seed, seed_scores = sp.choose_split_seed(
            data, seeds=range(settings["split_seed_trials"]), n_folds=settings["split_folds"],
            min_per_class=settings["min_images_per_class_in_eval"])
        seed_scores.to_csv(paths.audit / "split_seed_search.csv", index=False)
        print("Chosen seed:", best_seed)
        display(seed_scores.head(8))
        '''),
        md("## 2. Make the split and prove there is no leakage"),
        code('''
        split = sp.make_grouped_split(data, seed=best_seed, n_folds=settings["split_folds"])
        sp.check_no_leakage(split)                    # stops with an error if any patient is shared
        print("No patient or file appears in two splits.")

        counts = sp.split_summary(split)
        display(counts)
        shares = (pd.crosstab(split["label"], split["split"], normalize="columns") * 100).round(1)[["train", "val", "test"]]
        display(shares)
        print(f"Largest class-share drift from train: {sp.split_balance_score(split):.1f} percentage points")
        '''),
        md("## 3. Sources and skin-tone labels inside each split"),
        code('''
        display(pd.crosstab(split["source"], split["split"])[["train", "val", "test"]])
        display(split.groupby("split")["fst"].agg(with_fst="count", total="size").loc[["train", "val", "test"]])
        shares.plot.bar(figsize=(9, 4), title="Class share per split (%)")
        plt.tight_layout()
        plt.savefig(paths.figures / "split_balance.png", dpi=150)
        plt.show()
        '''),
        md("""
        ## 4. Class weights

        `w_c = N / (C · n_c)` from the **training** split only. A class with half the average size gets
        weight ≈ 2, so the loss pays twice as much attention to it.
        """),
        code('''
        train = split[split["split"] == "train"]
        class_weights = sp.compute_class_weights(train, TARGET_CLASSES)
        display(pd.Series(class_weights, name="weight").to_frame())
        '''),
        md("## 5. Save and freeze\nAfter this cell, nobody edits `data/splits/` without a team decision: every result depends on it."),
        code('''
        for name in ("train", "val", "test"):
            split[split["split"] == name].to_csv(paths.splits / f"{name}.csv", index=False)
        split.to_csv(paths.splits / "manifest_split.csv", index=False)

        class_info = {
            "classes": TARGET_CLASSES,
            "class_weights": class_weights,
            "split_seed": best_seed,
            "split_folds": settings["split_folds"],
            "images_per_split": split["split"].value_counts().to_dict(),
            "patients_per_split": split.groupby("split")["group_id"].nunique().to_dict(),
            "created": date.today().isoformat(),
            "open_set": CONFIG["data"]["open_set"],
            "note": "Healthy = MCSI (CC BY 4.0); Psoriasis = DermaCon-IN only; Other = non-target diseases.",
        }
        save_json(class_info, paths.splits / "class_info.json")
        print(json.dumps(class_info, indent=1))
        '''),
    ]


def nb06():
    return [
        header("06", "Dataset & Transforms",
               "Prepare fast image loading (resized cache) and check, by eye, what the model will actually see.",
               "`data/splits/manifest_split.csv`", "`data/cache/*.jpg`, `cache_path` column in the split CSVs, figures",
               "15-30 min the first time (resizes every image once)", "07 · Baseline Training"),
        SETUP_MD, SETUP_CODE,
        md("""
        ## 1. Build the resized-image cache

        Some originals are 4,608 px wide. Decoding them every epoch from Drive would make each epoch take
        many minutes. We shrink every image **once** so its shorter side is 256 px and save it to
        `data/cache/`. Training reads these small files, then crops 224 px.
        """),
        code('''
        import time

        import matplotlib.pyplot as plt
        import pandas as pd
        from dermalens.data import build_transforms, cache_resized_images, denormalize, make_loader, SkinDataset, load_rgb
        from dermalens.utils import load_json

        TARGET_CLASSES = load_json(paths.splits / "class_info.json")["classes"]
        split = pd.read_csv(paths.splits / "manifest_split.csv", dtype={"phash": str})
        split = cache_resized_images(split, paths.raw, paths.cache, short_side=CONFIG["data"]["cache_short_side"])

        for name in ("train", "val", "test"):                  # save cache_path into the frozen splits
            split[split["split"] == name].to_csv(paths.splits / f"{name}.csv", index=False)
        split.to_csv(paths.splits / "manifest_split.csv", index=False)
        print("Cached images:", len(list((paths.cache / f"s{CONFIG['data']['cache_short_side']}").glob("*.jpg"))))
        '''),
        md("""
        ### Optional on Colab: copy the cache to the local disk

        Reading from the Colab machine's own disk is much faster than reading from Drive. The copy lasts
        until the session ends; notebooks 07-11 use it automatically when it exists.
        """),
        code('''
        import shutil
        from pathlib import Path

        LOCAL_CACHE = Path("/content/cache")
        if IN_COLAB and not LOCAL_CACHE.exists():
            shutil.copytree(paths.cache, LOCAL_CACHE)
        IMAGE_ROOT = LOCAL_CACHE if LOCAL_CACHE.exists() else paths.cache
        print("Training will read images from:", IMAGE_ROOT)
        '''),
        md("## 2. Dataset sanity checks"),
        code('''
        image_size = CONFIG["data"]["image_size"]
        train = split[split["split"] == "train"]
        dataset = SkinDataset(train, IMAGE_ROOT, TARGET_CLASSES, build_transforms(image_size, train=True))

        tensor, meta_vector, class_index = dataset[0]
        print("Metadata vector (21 numbers: age one-hot+missing, sex+missing, 11 regions+missing):", meta_vector.tolist())
        print("Images in train:", len(dataset))
        print("One sample:", tuple(tensor.shape), "| values from", round(float(tensor.min()), 2), "to", round(float(tensor.max()), 2))
        print("Its class:", class_index, "=", TARGET_CLASSES[class_index])
        print("Class index map:", {name: i for i, name in enumerate(TARGET_CLASSES)})
        '''),
        md("""
        ## 3. What augmentation does

        Each row: the cached photo, then four random training versions of it. They should still clearly
        show the lesion, with natural colours. If an augmentation hides the lesion or changes skin colour,
        it is too strong.
        """),
        code('''
        augment = build_transforms(image_size, train=True)
        fig, axes = plt.subplots(len(TARGET_CLASSES), 5, figsize=(12, 2.6 * len(TARGET_CLASSES)))
        for row, label in enumerate(TARGET_CLASSES):
            item = train[train["label"] == label].iloc[0]
            original = load_rgb(IMAGE_ROOT / item["cache_path"])
            axes[row, 0].imshow(original)
            axes[row, 0].set_title(label, fontsize=9)
            for col in range(1, 5):
                axes[row, col].imshow(denormalize(augment(original)).permute(1, 2, 0))
            for ax in axes[row]:
                ax.axis("off")
        plt.tight_layout()
        plt.savefig(paths.figures / "augmentation_preview.png", dpi=120)
        plt.show()
        '''),
        md("## 4. How fast can we feed the GPU?"),
        code('''
        loader = make_loader(train, IMAGE_ROOT, TARGET_CLASSES, image_size, train=True,
                             batch_size=CONFIG["train"]["batch_size"], num_workers=CONFIG["train"]["num_workers"])
        started, images_seen = time.time(), 0
        for batch_number, (images, metas, targets) in enumerate(loader):
            images_seen += len(targets)
            if batch_number == 4:
                break
        rate = images_seen / (time.time() - started)
        print(f"{rate:.0f} images/second -> one epoch of {len(train)} images ≈ {len(train) / rate:.0f} s of loading")
        '''),
    ]
