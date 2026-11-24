"""Notebooks 07-14 (v3): training recipe, comparison + metadata ablation, ensemble evaluation,
open-set evaluation, Grad-CAM, shortcut test, grouped taxonomy, export."""

from .common import SETUP_CODE, SETUP_MD, code, header, md

LOAD = '''
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from dermalens.data import make_loader, resolve_images
        from dermalens.utils import get_device, load_json, save_json

        info = load_json(paths.splits / "class_info.json")       # classes + weights frozen in notebook 05
        CLASSES = info["classes"]                                  # 6 targets + "Other"
        WEIGHTS = [info["class_weights"][name] for name in CLASSES]
        train = pd.read_csv(paths.splits / "train.csv")
        val = pd.read_csv(paths.splits / "val.csv")
        IMAGE_ROOT, PATH_COL = resolve_images(train, paths)        # cache if present, else raw files
        DEVICE = get_device()
        IMAGE_SIZE = CONFIG["data"]["image_size"]
        BATCH, WORKERS = CONFIG["train"]["batch_size"], CONFIG["train"]["num_workers"]
        USE_META, COLOR = CONFIG["data"]["use_metadata"], CONFIG["data"]["color_constancy"]


        def loader(rows, train_mode=False, size=IMAGE_SIZE, use_meta=USE_META, classes=None, label_col="label",
                   seed=0):
            """One DataLoader with the project settings (augmentation + metadata dropout only when training)."""
            return make_loader(rows, IMAGE_ROOT, classes or CLASSES, size, train=train_mode, batch_size=BATCH,
                               num_workers=WORKERS, path_col=PATH_COL, label_col=label_col, use_metadata=use_meta,
                               meta_dropout=CONFIG["data"]["meta_dropout"], color_constancy=COLOR, seed=seed)


        print(f"train {len(train)} | val {len(val)} | classes {len(CLASSES)} | device {DEVICE} | "
              f"images from {IMAGE_ROOT} ({PATH_COL})")
'''


def _train_run_code(indent_note: str = "") -> str:
    return '''
        import time
        from dermalens.evaluate import headline_metrics, predict_logits
        from dermalens.model import build_model, count_parameters, zoo_entry
        from dermalens.train import fit, model_from_checkpoint, train_config_from
        from dermalens.utils import set_seed

        TRAIN_CFG = train_config_from(CONFIG["train"])


        def train_run(model_name, seed, use_meta, folder, rows=None, classes=None, weights=None, label_col="label"):
            """Train one model (or reuse it if done.json exists). Returns its validation record."""
            run_dir = paths.models / folder / f"{model_name}_{'meta' if use_meta else 'img'}_seed{seed}"
            if (run_dir / "done.json").exists():
                print("skip (done):", run_dir.name)
                return load_json(run_dir / "done.json")
            rows = rows if rows is not None else (train, val)
            classes, weights = classes or CLASSES, weights or WEIGHTS
            set_seed(seed)
            size = IMAGE_SIZE if zoo_entry(model_name)[2] == "cnn" else min(IMAGE_SIZE, 224)
            model = build_model(model_name, len(classes), CONFIG["model"]["pretrained"], use_meta,
                                CONFIG["model"]["drop_rate"], image_size=size)
            facts = {"model_name": model_name, "classes": classes, "image_size": size, "seed": seed,
                     "use_metadata": use_meta}
            started = time.time()
            history = fit(model, loader(rows[0], True, size, use_meta, classes, label_col, seed),
                          loader(rows[1], False, size, use_meta, classes, label_col), weights, TRAIN_CFG, DEVICE,
                          run_dir, facts, seed=seed)
            history.to_csv(run_dir / "history.csv", index=False)
            best, _ = model_from_checkpoint(run_dir / "best.pt", DEVICE)
            logits, y_true = predict_logits(best, loader(rows[1], False, size, use_meta, classes, label_col), DEVICE)
            record = {"model": model_name, "seed": seed, "use_metadata": use_meta, "image_size": size,
                      "checkpoint": str((run_dir / "best.pt").relative_to(paths.root)),
                      "params_m": round(count_parameters(best)[0] / 1e6, 1),
                      "minutes": round((time.time() - started) / 60, 1),
                      **{k: round(float(v), 4) for k, v in headline_metrics(y_true, logits.argmax(1)).items()}}
            save_json(record, run_dir / "done.json")
            print(record)
            return record
'''


TRAIN_RUN = _train_run_code()


def nb07():
    return [
        header("07", "Training Recipe (one strong model)",
               "Fine-tune one pretrained model with the v3 recipe - metadata fusion, layer-wise learning "
               "rates, mixup, EMA - and read its learning curves.",
               "`data/splits/*`, `data/cache/*` (optional)",
               "`models/baseline/<model>_meta_seed42/`, `reports/results/baseline.json`, figures",
               "25-45 min (ConvNeXt-S, 288 px)", "08 · Model Comparison"),
        SETUP_MD, SETUP_CODE,
        md("""
        ## Why v2 accuracy was low, and what v3 changes

        | Problem seen in v2 | v3 fix | Where |
        |---|---|---|
        | small ImageNet-1k CNNs at 224 px | ConvNeXt-S / EfficientNetV2-S pretrained on **ImageNet-22k**, 288 px | `model.py` |
        | one learning rate for the whole backbone, too low to adapt | **layer-wise LR decay**: last blocks learn fast, early blocks slowly | `train.py` |
        | over-fitting on ~1,900 training images | stronger augmentation, **mixup**, drop-path, weight decay 0.05 | `data.py`, `train.py` |
        | noisy last-epoch weights | **EMA** of weights (smoother, usually +1-2 points) | `train.py` |
        | photo only | **age group, sex and body region** fused with the image features | `model.py` |
        | camera colour differs per dataset | **Shades-of-Gray colour constancy** on every image | `data.py` |
        | any photo forced into 6 classes | **"Other" class** + confidence threshold + skin check | notebooks 03, 10 |
        """),
        md("## 1. Data and the training function"),
        code(LOAD), code(TRAIN_RUN),
        md("""
        ## 2. Train

        If Colab disconnects, run this cell again: training resumes from `last.pt`.
        """),
        code('''
        MODEL_NAME = CONFIG["model"]["name"]
        record = train_run(MODEL_NAME, CONFIG["seed"], USE_META, "baseline")
        save_json(record, paths.results / "baseline.json")
        '''),
        md("""
        ## 3. Learning curves

        *Healthy:* validation macro-F1 rises then flattens. *Over-fitting:* train accuracy near 100% while
        validation falls - early stopping keeps the best epoch.
        """),
        code('''
        from dermalens.evaluate import plot_training_curves

        run_dir = (paths.root / record["checkpoint"]).parent
        history = pd.read_csv(run_dir / "history.csv")
        plot_training_curves(history, paths.figures / f"curves_baseline_{MODEL_NAME}.png", title=f"{MODEL_NAME}")
        plt.show()
        '''),
        md("## 4. Validation confusion matrix\nThe test set stays untouched until notebook 09."),
        code('''
        from dermalens.evaluate import per_class_table, plot_confusion_matrix

        model, facts = model_from_checkpoint(paths.root / record["checkpoint"], DEVICE)
        logits, y_true = predict_logits(model, loader(val, size=facts["image_size"]), DEVICE)
        display(per_class_table(y_true, logits.argmax(1), CLASSES))
        plot_confusion_matrix(y_true, logits.argmax(1), CLASSES, paths.figures / "confusion_val_baseline.png",
                              title=f"{MODEL_NAME} - validation")
        plt.show()
        '''),
    ]


def nb08():
    return [
        header("08", "Model Comparison & Metadata Ablation",
               "Train five architectures with two seeds each on the same split; then measure exactly how much "
               "metadata adds by retraining the best one image-only.",
               "`data/splits/*`, `configs/baseline.yaml` (`compare`)",
               "`models/compare/*`, `reports/results/model_comparison*.csv`, `metadata_ablation.json`, `best_models.json`",
               "4-8 h in total (split across teammates: finished runs are skipped)", "09 · Ensemble & Test Evaluation"),
        SETUP_MD, SETUP_CODE, code(LOAD), code(TRAIN_RUN),
        md("""
        ## 1. Architectures compared

        | Model | Family | Pretraining | Why it is in the comparison |
        |---|---|---|---|
        | EfficientNet-B0 | CNN | ImageNet-1k | v2 baseline, small and fast |
        | EfficientNetV2-S | CNN | ImageNet-21k | modern efficient CNN |
        | ConvNeXt-S | CNN | ImageNet-22k | strongest CNN family on small medical sets |
        | Swin-T | transformer | ImageNet-1k | windowed attention, good at texture |
        | ViT-B/14 DINOv2 | transformer | 142 M images, self-supervised | very general features |

        Each finished run writes `done.json`; teammates can run different models on different Colab accounts.
        """),
        code('''
        results = [train_run(name, seed, USE_META, "compare")
                   for name in CONFIG["compare"]["models"] for seed in CONFIG["compare"]["seeds"]]
        runs = pd.DataFrame(results)
        runs.to_csv(paths.results / "model_comparison_runs.csv", index=False)
        summary = runs.groupby("model").agg(
            macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"), balanced_acc_mean=("balanced_accuracy", "mean"),
            params_m=("params_m", "first"), minutes_per_run=("minutes", "mean"),
        ).round(4).sort_values("macro_f1_mean", ascending=False)
        summary.to_csv(paths.results / "model_comparison.csv")
        display(summary)
        ax = summary["macro_f1_mean"].plot.bar(yerr=summary["macro_f1_std"].fillna(0), capsize=4, figsize=(8, 4),
                                                title="Validation macro-F1 (mean ± std over seeds)")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(paths.figures / "model_comparison.png", dpi=150)
        plt.show()
        '''),
        md("""
        ## 2. Metadata ablation: image only vs image + metadata

        Same model, same seeds, same split; the only difference is the metadata branch. The difference in
        macro-F1 is the contribution of age, sex and body region - a key result for the paper.
        """),
        code('''
        best_name = summary.index[0]
        image_only = pd.DataFrame([train_run(best_name, seed, False, "ablation") for seed in CONFIG["compare"]["seeds"]])
        with_meta = runs[runs["model"] == best_name]
        ablation = {"model": best_name,
                    "image_only_macro_f1": round(float(image_only["macro_f1"].mean()), 4),
                    "image_plus_metadata_macro_f1": round(float(with_meta["macro_f1"].mean()), 4),
                    "image_only_accuracy": round(float(image_only["accuracy"].mean()), 4),
                    "image_plus_metadata_accuracy": round(float(with_meta["accuracy"].mean()), 4)}
        ablation["gain_macro_f1_points"] = round(100 * (ablation["image_plus_metadata_macro_f1"]
                                                        - ablation["image_only_macro_f1"]), 1)
        save_json(ablation, paths.results / "metadata_ablation.json")
        display(pd.Series(ablation).to_frame("value"))
        '''),
        md("""
        ## 3. Choose the ensemble

        The best run (seed) of each of the top-k architectures joins the ensemble. Different architectures
        make different mistakes, so averaging them is usually 1-3 points better than the best single model.
        """),
        code('''
        k = CONFIG["compare"]["ensemble_top_k"]
        chosen = [runs[runs["model"] == name].sort_values("macro_f1", ascending=False).iloc[0].to_dict()
                  for name in summary.index[:k]]
        save_json({"members": chosen}, paths.results / "best_models.json")
        display(pd.DataFrame(chosen)[["model", "seed", "macro_f1", "accuracy", "checkpoint"]])
        '''),
    ]


EVAL_IMPORTS = '''
        from dermalens.evaluate import (bootstrap_ci, ensemble_probabilities, expected_calibration_error,
                                        fit_temperature, grouped_metrics, headline_metrics, metrics_by_group,
                                        per_class_table, plot_confusion_matrix, plot_reliability, predict_logits,
                                        softmax)
        from dermalens.labels import GROUPED_LABELS, OTHER_CLASS
        from dermalens.train import model_from_checkpoint

        test = pd.read_csv(paths.splits / "test.csv")
        members = load_json(paths.results / "best_models.json")["members"]


        def member_logits(member, rows):
            """TTA logits of one ensemble member on a table of rows."""
            model, facts = model_from_checkpoint(paths.root / member["checkpoint"], DEVICE)
            return predict_logits(model, loader(rows, size=facts["image_size"], use_meta=facts["use_metadata"]),
                                  DEVICE, tta=True)
'''


def nb09():
    return [
        header("09", "Ensemble, Calibration & Test Evaluation",
               "Combine the best models, calibrate them, and score them once on the untouched test set: "
               "confidence intervals, per class, per source, per skin type, grouped taxonomy.",
               "`best_models.json`, `data/splits/*`", "`reports/results/test_results.json`, `test_predictions.csv`, figures",
               "~10 min", "10 · Open-Set Evaluation"),
        SETUP_MD, SETUP_CODE,
        md("> **Rule:** the test set is used **once**. Everything (temperatures, weights, thresholds) is chosen on validation."),
        code(LOAD + EVAL_IMPORTS),
        md("""
        ## 1. Validation and test logits for each member, with test-time augmentation

        TTA averages 4 flipped views of every photo; each member gets its own temperature `T` fitted on
        validation so that its probabilities are honest before averaging.
        """),
        code('''
        val_logits, test_logits, temperatures = [], [], []
        for member in members:
            v, val_true = member_logits(member, val)
            t, y_true = member_logits(member, test)
            temperatures.append(fit_temperature(v, val_true))
            val_logits.append(v)
            test_logits.append(t)
            single = headline_metrics(val_true, v.argmax(1))
            print(f"{member['model']:<18} T = {temperatures[-1]:.2f}  val acc (TTA) {single['accuracy']:.3f}  "
                  f"macro-F1 {single['macro_f1']:.3f}")
        val_probs = ensemble_probabilities(val_logits, temperatures)
        probs = ensemble_probabilities(test_logits, temperatures)
        print("Ensemble val:", {k: round(v, 3) for k, v in headline_metrics(val_true, val_probs.argmax(1)).items()})
        '''),
        md("""
        ## 2. Two views of the test result

        * **7-way** - every test photo, including "Other" photos, scored on all 7 outputs.
        * **Known classes (6-way)** - only photos of the 6 target classes, the model's answer restricted to
          those 6. This is the number comparable with other papers and with v2.
        """),
        code('''
        other_index = CLASSES.index(OTHER_CLASS) if OTHER_CLASS in CLASSES else None
        known_classes = [c for c in CLASSES if c != OTHER_CLASS]
        known_rows = np.array([CLASSES[i] != OTHER_CLASS for i in y_true])
        known_probs = probs[known_rows][:, [CLASSES.index(c) for c in known_classes]]
        known_true = np.array([known_classes.index(CLASSES[i]) for i in y_true[known_rows]])
        known_pred = known_probs.argmax(1)
        y_pred = probs.argmax(1)
        groups_known = test.loc[known_rows, "group_id"].to_numpy()

        n_boot = CONFIG["evaluate"]["bootstrap_samples"]
        metric_functions = {
            "accuracy": lambda t, p: (t == p).mean(),
            "macro_f1": lambda t, p: headline_metrics(t, p)["macro_f1"],
            "balanced_accuracy": lambda t, p: headline_metrics(t, p)["balanced_accuracy"],
        }
        headline = {}
        for name, function in metric_functions.items():
            point, low, high = bootstrap_ci(known_true, known_pred, groups_known, function, n_boot, CONFIG["seed"])
            headline[name] = {"value": round(point, 4), "ci95": [round(low, 4), round(high, 4)]}
            print(f"6 known classes  {name:<18} {point:.3f}  (95% CI {low:.3f} - {high:.3f})")
        seven_way = {k: round(float(v), 4) for k, v in headline_metrics(y_true, y_pred).items()}
        print("7-way (with Other):", seven_way)
        '''),
        md("## 3. Per class and confusion matrix"),
        code('''
        class_table = per_class_table(known_true, known_pred, known_classes)
        display(class_table)
        plot_confusion_matrix(known_true, known_pred, known_classes, paths.figures / "confusion_test.png",
                              title="Test set, ensemble (6 known classes)")
        plt.show()
        '''),
        md("""
        ## 4. Grouped taxonomy view

        Contact dermatitis is a sub-type of dermatitis, and dermatologists often cannot tell it from atopic
        eczema in a photo either. Merging the two into **"Eczema / Dermatitis"** (5 classes) shows how much of
        the remaining error is this one clinically-hard pair. Notebook 13 also *trains* on the merged labels.
        """),
        code('''
        grouped = grouped_metrics(known_true, known_pred, known_classes, GROUPED_LABELS)
        grouped = {k: round(float(v), 4) for k, v in grouped.items()}
        print("Grouped (5 classes, same predictions):", grouped)
        '''),
        md("## 5. By source and by skin type (fairness)"),
        code('''
        results = test.assign(true_label=[CLASSES[i] for i in y_true], predicted=[CLASSES[i] for i in y_pred],
                              confidence=probs.max(1), y_true=y_true, y_pred=y_pred)
        results["correct"] = results["y_true"] == results["y_pred"]
        fst = results["fst"] if "fst" in results.columns else pd.Series(np.nan, index=results.index)
        results["fst_group"] = pd.cut(fst, bins=[0, 2, 4, 6], labels=["FST 1-2", "FST 3-4", "FST 5-6"]).astype("string")
        known_results = results[results["true_label"] != OTHER_CLASS]
        by_source = metrics_by_group(known_results, "source", CLASSES)
        display(by_source)
        derma = known_results[known_results["fst_group"].notna()]
        by_fst = metrics_by_group(derma, "fst_group", CLASSES)
        display(by_fst)
        fst_gap = float(by_fst["accuracy"].max() - by_fst["accuracy"].min()) if len(by_fst) >= 2 else float("nan")
        print(f"Accuracy gap between skin-type groups: {fst_gap * 100:.1f} points")
        '''),
        md("## 6. Calibration"),
        code('''
        raw = np.mean([softmax(t) for t in test_logits], axis=0)
        ece_before, ece_after = expected_calibration_error(raw, y_true), expected_calibration_error(probs, y_true)
        print(f"ECE {ece_before:.3f} before -> {ece_after:.3f} after temperature scaling")
        plot_reliability(probs, y_true, paths.figures / "reliability_test.png")
        plt.show()
        '''),
        md("## 7. Save (all report numbers come from these files)"),
        code('''
        np.save(paths.results / "val_ensemble_probs.npy", val_probs)
        np.save(paths.results / "val_true.npy", val_true)
        np.save(paths.results / "test_ensemble_probs.npy", probs)
        columns = ["image_id", "source", "group_id", "true_label", "predicted", "confidence", "correct", "fst",
                   "meta_age", "meta_sex", "meta_regions", PATH_COL]
        results[[c for c in columns if c in results.columns]].to_csv(paths.results / "test_predictions.csv", index=False)
        save_json({"members": members, "temperatures": temperatures, "headline_known_classes": headline,
                   "seven_way": seven_way, "grouped_taxonomy": grouped,
                   "per_class": class_table.reset_index().to_dict(orient="records"),
                   "by_source": by_source.to_dict(orient="records"), "by_fst": by_fst.to_dict(orient="records"),
                   "fst_gap_points": round(fst_gap * 100, 1), "ece_before": ece_before, "ece_after": ece_after,
                   "test_images": int(len(test)), "test_images_known": int(known_rows.sum())},
                  paths.results / "test_results.json")
        print("Saved test_results.json")
        '''),
    ]


def nb10():
    return [
        header("10", "Open-Set Evaluation (\"Not identified\" and \"Not skin\")",
               "Tune and measure how well the system refuses photos it should not name: other skin diseases "
               "and photos that are not skin at all.",
               "`val_ensemble_probs.npy`, `test_ensemble_probs.npy`, `test_predictions.csv`",
               "`reports/results/open_set.json`, figures", "~3 min", "11 · Grad-CAM"),
        SETUP_MD, SETUP_CODE, code(LOAD + '''
        from dermalens import ood
        from dermalens.labels import OTHER_CLASS

        val_probs = np.load(paths.results / "val_ensemble_probs.npy")
        val_true = np.load(paths.results / "val_true.npy")
        test_probs = np.load(paths.results / "test_ensemble_probs.npy")
        predictions = pd.read_csv(paths.results / "test_predictions.csv")
        test_true = np.array([CLASSES.index(c) for c in predictions["true_label"]])
        assert OTHER_CLASS in CLASSES, "Open-set evaluation needs data.open_set: true (notebooks 03 and 05)."
        other = CLASSES.index(OTHER_CLASS)
        '''),
        md("""
        ## 1. How the app decides

        1. **Skin check** - share of pixels in a skin-colour range (YCrCb). Below 15% → *not skin*.
        2. **"Other" wins** → *not identified* (looks like a disease we do not cover).
        3. **Confidence below a threshold** → *not identified* (too unsure).
        4. Otherwise → the condition, or *healthy*.

        The confidence threshold is set on **validation** so that 95% of known-class photos are still
        answered; we then measure on test how many "Other" photos are correctly refused.
        """),
        code('''
        known_columns = [i for i in range(len(CLASSES)) if i != other]


        def known_confidence(probs):
            """Highest probability among the 6 known classes.

            ('Other' winning is handled separately by ood.decide. v3.0 set this score to 0 whenever
            'Other' won; since ~10% of known validation photos lose to 'Other', the 5% quantile became
            0.0 and the confidence check was switched off. Fixed in v3.1.)
            """
            return probs[:, known_columns].max(1)

        val_known = val_true != other
        THRESHOLD = ood.tune_threshold(known_confidence(val_probs)[val_known], CONFIG["open_set"]["target_tpr"])
        print(f"Confidence threshold (keeps {CONFIG['open_set']['target_tpr']:.0%} of known val photos): {THRESHOLD:.3f}")

        test_known = test_true != other
        conf = known_confidence(test_probs)
        auroc = ood.open_set_auroc(conf[test_known], conf[~test_known]) if (~test_known).any() else float("nan")
        decisions = [ood.decide(p, CLASSES, THRESHOLD)[0] for p in test_probs]
        predictions["status"] = decisions
        rejected_other = float(np.mean([d == "not_identified" for d, k in zip(decisions, test_known) if not k]))
        answered_known = float(np.mean([d != "not_identified" for d, k in zip(decisions, test_known) if k]))
        correct_when_answered = float(predictions.loc[test_known & (predictions["status"] != "not_identified"),
                                                      "correct"].mean())
        print(f"Open-set AUROC (known vs other diseases): {auroc:.3f}")
        print(f"Other-disease photos refused ('not identified'): {rejected_other:.1%}")
        print(f"Known photos answered: {answered_known:.1%} | accuracy on answered: {correct_when_answered:.1%}")
        '''),
        md("""
        ## 2. Accuracy-coverage trade-off

        Raising the threshold refuses more photos but makes the answers that remain more accurate. This
        curve is a standard figure in selective-prediction papers: report accuracy at 100%, 90% and 80%
        coverage.
        """),
        code('''
        order = np.argsort(-conf[test_known])
        correct_sorted = predictions.loc[test_known, "correct"].to_numpy()[order]
        coverage = np.arange(1, len(order) + 1) / len(order)
        accuracy_at = np.cumsum(correct_sorted) / np.arange(1, len(order) + 1)
        plt.figure(figsize=(6, 4))
        plt.plot(coverage, accuracy_at)
        plt.xlabel("Coverage (share of known photos answered)")
        plt.ylabel("Accuracy on answered photos")
        plt.title("Selective prediction")
        plt.grid(alpha=0.3)
        plt.savefig(paths.figures / "accuracy_coverage.png", dpi=150, bbox_inches="tight")
        plt.show()
        selective = {f"accuracy_at_{int(c * 100)}pct_coverage": round(float(accuracy_at[max(int(c * len(order)) - 1, 0)]), 4)
                     for c in (1.0, 0.9, 0.8, 0.7)}
        print(selective)
        '''),
        md("""
        ## 3. Skin check on photos that are not skin

        Put any non-skin photos (pets, objects, documents, screenshots) into `data/raw/non_skin/` to test with
        real images. Generated pictures (plain colours, noise) are used in addition so the check always runs.
        """),
        code('''
        from PIL import Image
        from dermalens.data import load_rgb

        rng = np.random.default_rng(0)
        synthetic = [Image.new("RGB", (256, 256), tuple(int(v) for v in rng.integers(0, 255, 3))) for _ in range(30)]
        synthetic += [Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)) for _ in range(10)]
        user_folder = paths.raw / "non_skin"
        real_non_skin = [load_rgb(p) for p in sorted(user_folder.glob("*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png")] \\
            if user_folder.exists() else []
        skin_rows = predictions.sample(min(200, len(predictions)), random_state=0)
        skin_scores = [ood.skin_pixel_ratio(load_rgb(IMAGE_ROOT / p)) for p in skin_rows[PATH_COL]]
        non_skin_scores = [ood.skin_pixel_ratio(im) for im in synthetic + real_non_skin]
        limit = CONFIG["open_set"]["min_skin_ratio"]
        skin_kept = float(np.mean(np.array(skin_scores) >= limit))
        non_skin_refused = float(np.mean(np.array(non_skin_scores) < limit))
        print(f"Real skin photos passing the check: {skin_kept:.1%} (want ~100%)")
        print(f"Non-skin images refused: {non_skin_refused:.1%} ({len(real_non_skin)} real + {len(synthetic)} generated)")
        plt.hist(skin_scores, bins=20, alpha=0.6, label="skin photos")
        plt.hist(non_skin_scores, bins=20, alpha=0.6, label="non-skin")
        plt.axvline(limit, color="k", linestyle="--")
        plt.legend(); plt.xlabel("skin-pixel ratio"); plt.savefig(paths.figures / "skin_check.png", dpi=150); plt.show()
        '''),
        code('''
        save_json({"confidence_threshold": THRESHOLD, "open_set_auroc": auroc, "other_refused": rejected_other,
                   "known_answered": answered_known, "accuracy_when_answered": correct_when_answered,
                   **selective, "min_skin_ratio": limit, "skin_photos_kept": skin_kept,
                   "non_skin_refused": non_skin_refused, "real_non_skin_images": len(real_non_skin)},
                  paths.results / "open_set.json")
        display(predictions["status"].value_counts().to_frame("test photos"))
        '''),
    ]


def nb11():
    return [
        header("11", "Explainability (Grad-CAM)",
               "Show where each ensemble member looks. A trustworthy model looks at the lesion, not the background.",
               "`best_models.json`, `test_predictions.csv`",
               "`reports/figures/gradcam_<class>.png`, `reports/results/gradcam_review.csv`", "~5 min",
               "12 · Source-Shortcut Test"),
        SETUP_MD, SETUP_CODE, code(LOAD + '''
        from PIL import Image
        from dermalens import metadata as meta
        from dermalens.data import build_transforms, center_square, load_rgb
        from dermalens.explain import GradCAM, overlay_heatmap
        from dermalens.model import gradcam_target_layer
        from dermalens.train import model_from_checkpoint

        predictions = pd.read_csv(paths.results / "test_predictions.csv")
        member = load_json(paths.results / "best_models.json")["members"][0]
        model, facts = model_from_checkpoint(paths.root / member["checkpoint"], DEVICE)
        cam = GradCAM(model, gradcam_target_layer(model))
        to_tensor = build_transforms(facts["image_size"], train=False, color_constancy=COLOR)
        print("Explaining", member["model"], "- works for CNNs, ViT (token maps) and Swin")
        '''),
        md("Per class: the 3 most confident correct predictions and up to 3 wrong ones. Red = evidence for the predicted class."),
        code('''
        review_rows = []
        for label in CLASSES:
            in_class = predictions[predictions["true_label"] == label]
            chosen = pd.concat([in_class[in_class["correct"]].nlargest(3, "confidence"),
                                in_class[~in_class["correct"]].nlargest(3, "confidence")])
            if chosen.empty:
                continue
            fig, axes = plt.subplots(2, len(chosen), figsize=(2.8 * len(chosen), 5.6), squeeze=False)
            for column, (_, item) in enumerate(chosen.iterrows()):
                photo = load_rgb(IMAGE_ROOT / item[PATH_COL])
                vector = meta.encode_row(item) if facts["use_metadata"] else None
                heatmap, _, _ = cam(to_tensor(photo), meta_vector=vector)
                shown = center_square(photo).resize((facts["image_size"], facts["image_size"]))
                axes[0, column].imshow(shown)
                axes[1, column].imshow(overlay_heatmap(shown, heatmap))
                verdict = "correct" if item["correct"] else f"WRONG: {item['predicted']}"
                axes[0, column].set_title(f"{verdict}\\n{item['confidence']:.0%}", fontsize=8,
                                          color="green" if item["correct"] else "red")
                for ax in axes[:, column]:
                    ax.axis("off")
                review_rows.append({"image_id": item["image_id"], "true_label": label, "predicted": item["predicted"],
                                    "correct": item["correct"], "attention_on_lesion (yes/no/partly)": "", "notes": ""})
            fig.suptitle(f"True class: {label}")
            plt.tight_layout()
            plt.savefig(paths.figures / f"gradcam_{label.replace(' ', '_').replace('/', '').lower()}.png", dpi=130)
            plt.show()
        cam.remove()
        pd.DataFrame(review_rows).to_csv(paths.results / "gradcam_review.csv", index=False)
        print(f"{len(review_rows)} images to review by a teammate -> gradcam_review.csv")
        '''),
    ]


def nb12():
    return [
        header("12", "Source-Shortcut Test",
               "Measure how easily a model tells which dataset a photo came from, with and without colour "
               "constancy. High = class scores may partly reflect dataset style.",
               "`data/splits/*`", "`reports/results/shortcut_test.json`, figures", "15-30 min",
               "13 · Grouped Taxonomy"),
        SETUP_MD, SETUP_CODE, code(LOAD), code(TRAIN_RUN + '''
        test = pd.read_csv(paths.splits / "test.csv")
        SHARED = ["Contact Dermatitis", "Eczema", "Scabies", "Tinea Corporis"]   # in DermaCon-IN and SkinDisNet
        pick = lambda rows: rows[rows["label"].isin(SHARED) & rows["source"].isin(["dermaconin", "skindisnet"])]
        '''),
        md("""
        Same-disease source test: only the 4 classes present in **both** DermaCon-IN and SkinDisNet; the model
        predicts the **source**. The disease is the same, so balanced accuracy above 0.5 comes from photo
        style alone. We run it twice - with and without Shades-of-Gray colour constancy - to see whether the
        v3 colour normalisation reduces the shortcut.
        """),
        code('''
        from dermalens.evaluate import headline_metrics, predict_logits

        SOURCES = ["dermaconin", "skindisnet"]
        outcome = {}
        for colour in (False, True):
            COLOR = colour                                       # loader() reads this global
            name = "shortcut_cc" if colour else "shortcut_raw"
            record = train_run("resnet50" if CONFIG["model"]["pretrained"] else CONFIG["model"]["name"], CONFIG["seed"],
                               False, name, rows=(pick(train), pick(val)), classes=SOURCES, weights=[1.0, 1.0],
                               label_col="source")
            model, facts = model_from_checkpoint(paths.root / record["checkpoint"], DEVICE)
            logits, y_true = predict_logits(model, loader(pick(test), size=facts["image_size"], use_meta=False,
                                                          classes=SOURCES, label_col="source"), DEVICE)
            outcome[name] = round(float(headline_metrics(y_true, logits.argmax(1))["balanced_accuracy"]), 3)
        COLOR = CONFIG["data"]["color_constancy"]
        verdict = lambda b: "low" if b < 0.65 else "moderate" if b < 0.90 else "high"
        result = {**outcome, "risk_without_cc": verdict(outcome["shortcut_raw"]),
                  "risk_with_cc": verdict(outcome["shortcut_cc"])}
        save_json(result, paths.results / "shortcut_test.json")
        display(pd.Series(result).to_frame("value"))
        '''),
        md("""
        | Balanced accuracy | Meaning |
        |---|---|
        | 0.50-0.65 | sources look alike - low shortcut risk |
        | 0.65-0.90 | style partly recognisable - report per-source results |
        | > 0.90 | trivially separable - Healthy/Psoriasis scores may be inflated |
        """),
    ]


def nb13():
    return [
        header("13", "Grouped Taxonomy Experiment",
               "Retrain the best architecture with Contact Dermatitis and Eczema merged into one "
               "'Eczema / Dermatitis' class, and compare with the 6-class model.",
               "`data/splits/*`, `best_models.json`", "`models/grouped/*`, `reports/results/grouped_taxonomy.json`",
               "30-60 min", "14 · Export"),
        SETUP_MD, SETUP_CODE, code(LOAD), code(TRAIN_RUN + '''
        from dermalens.labels import group_label
        from dermalens.splits import compute_class_weights

        test = pd.read_csv(paths.splits / "test.csv")
        regroup = lambda rows: rows.assign(label=rows["label"].map(group_label))
        g_train, g_val, g_test = regroup(train), regroup(val), regroup(test)
        G_CLASSES = sorted(set(g_train["label"]) - {"Other"}) + (["Other"] if "Other" in CLASSES else [])
        weights = compute_class_weights(g_train, G_CLASSES)
        print(G_CLASSES)
        '''),
        md("""
        **Why:** in a clinical photo, allergic/irritant contact dermatitis and atopic eczema share redness,
        scaling and lichenification; the distinction often needs patch testing or history. Many published
        systems group them. If the grouped model is much better, the report should present both results and
        explain the trade-off (a coarser but more reliable answer).
        """),
        code('''
        best = load_json(paths.results / "best_models.json")["members"][0]
        record = train_run(best["model"], CONFIG["seed"], USE_META, "grouped", rows=(g_train, g_val),
                           classes=G_CLASSES, weights=[weights[c] for c in G_CLASSES])
        model, facts = model_from_checkpoint(paths.root / record["checkpoint"], DEVICE)
        logits, y_true = predict_logits(model, loader(g_test, size=facts["image_size"], classes=G_CLASSES), DEVICE,
                                        tta=True)
        known = np.array([G_CLASSES[i] != "Other" for i in y_true])
        known_idx = [i for i, c in enumerate(G_CLASSES) if c != "Other"]
        pred = np.array(known_idx)[logits[known][:, known_idx].argmax(1)]
        grouped_test = {k: round(float(v), 4) for k, v in headline_metrics(y_true[known], pred).items()}
        six_class = load_json(paths.results / "test_results.json")["headline_known_classes"]
        comparison = {"grouped_5_class_test": grouped_test,
                      "six_class_test": {k: v["value"] for k, v in six_class.items()}}
        save_json(comparison, paths.results / "grouped_taxonomy.json")
        display(pd.DataFrame(comparison))
        '''),
    ]


def nb14():
    return [
        header("14", "Export for the Web App",
               "Package the ensemble, thresholds and metrics into the bundle the FastAPI app loads, and test it end to end.",
               "`best_models.json`, `test_results.json`, `open_set.json`",
               "`models/deploy/{member_*.pt, model_card.json}`", "~3 min", "the web app (docs/WEB_APP_GUIDE.md)"),
        SETUP_MD, SETUP_CODE, code(LOAD + '''
        import torch
        from dermalens.export import load_bundle, measure_latency, save_bundle
        from dermalens.train import model_from_checkpoint

        members = load_json(paths.results / "best_models.json")["members"]
        test_results = load_json(paths.results / "test_results.json")
        open_set = load_json(paths.results / "open_set.json")
        '''),
        code('''
        bundle_members = []
        for member, temperature in zip(members, test_results["temperatures"]):
            model, facts = model_from_checkpoint(paths.root / member["checkpoint"], "cpu")
            bundle_members.append({"model": model, "name": member["model"], "image_size": facts["image_size"],
                                   "temperature": temperature})
        metrics = {k: v["value"] for k, v in test_results["headline_known_classes"].items()}
        metrics.update({"open_set_auroc": round(open_set["open_set_auroc"], 4),
                        "grouped_accuracy": test_results["grouped_taxonomy"]["accuracy"],
                        "test_images": test_results["test_images_known"]})
        bundle_dir = save_bundle(bundle_members, paths.models / "deploy", CLASSES, metrics,
                                 confidence_threshold=open_set["confidence_threshold"],
                                 min_skin_ratio=open_set["min_skin_ratio"], color_constancy=COLOR,
                                 notes=f"Split seed {info['split_seed']}; Healthy = MCSI, Psoriasis = DermaCon-IN only.")
        for file in sorted(bundle_dir.iterdir()):
            print(f"{file.name:<18} {file.stat().st_size / 1e6:.1f} MB")
        print(metrics)
        '''),
        md("## Reload as the app does, check speed and one full prediction"),
        code('''
        reloaded, card = load_bundle(bundle_dir, "cpu")
        torch.set_num_threads(2)
        for model, member in reloaded:
            print(member["model_name"], measure_latency(model, member["image_size"], runs=10))

        sys.path.insert(0, str(CODE_DIR / "app"))
        from inference import Predictor
        from dermalens.data import load_rgb
        predictor = Predictor(bundle_dir)
        sample = val.iloc[0]
        regions = [r for r in str(sample.get("meta_regions") or "").split("|") if r and r != "nan"]
        answer = predictor.predict(load_rgb(IMAGE_ROOT / sample[PATH_COL]),
                                   age_group=sample.get("meta_age") if isinstance(sample.get("meta_age"), str) else None,
                                   sex=sample.get("meta_sex") if isinstance(sample.get("meta_sex"), str) else None,
                                   body_regions=regions)
        print("True label:", sample["label"])
        print({k: v for k, v in answer.items() if k not in ("heatmap_png_base64", "condition")})
        '''),
        md("""
        **Next:** copy `models/deploy/` into the repository's `models/deploy/` and start the site:
        `uvicorn app.api:app --port 8000` (see `docs/WEB_APP_GUIDE.md`).
        """),
    ]
