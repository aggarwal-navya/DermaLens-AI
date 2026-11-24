# Model Card: DermaLens AI v3

Fill the † fields from `reports/results/{test_results,open_set,metadata_ablation,shortcut_test,grouped_taxonomy}.json`
(notebooks 08-14). `models/deploy/model_card.json` holds the same numbers in machine-readable form.

| | |
|---|---|
| Model | ensemble of † (from `best_models.json`), soft voting of temperature-calibrated probabilities, 2-view TTA in the app |
| Inputs | one RGB clinical photo (Shades-of-Gray colour constancy, resize, centre crop to † px) + optional age group, sex, body sites |
| Outputs | 6 classes + "Other"; the app maps them to identified / healthy / not identified / not skin |
| Training | ImageNet-21k/22k or DINOv2 pretraining; layer-wise LR decay, warmup-cosine, mixup, EMA, class-weighted CE with label smoothing, early stopping on validation macro-F1 |
| Code | `src/dermalens/`, notebooks 07-14, `configs/baseline.yaml` |

## Intended use
Decision support and education for five common skin conditions plus healthy skin, on clinical photos of
Indian / South Asian skin. **Not** a diagnosis, not for skin cancer, not for treatment decisions, not a medical device.

## Results (held-out, patient-grouped test set)

| Metric | Value | 95% CI |
|---|---|---|
| Accuracy, 6 known classes | † | † |
| Macro-F1, 6 known classes | † | † |
| Balanced accuracy | † | † |
| Accuracy, grouped (Eczema / Dermatitis merged), same predictions | † | |
| Accuracy, grouped model retrained (notebook 13) | † | |
| Gain from metadata (macro-F1 points, notebook 08) | † | |
| Open-set AUROC (known vs other diseases) | † | |
| Other-disease photos refused | † | |
| Accuracy at 90% / 80% coverage | † / † | |
| Accuracy FST 3-4 vs FST 5-6 | † vs † | |
| ECE after temperature scaling | † | |
| Same-disease source test, without / with colour constancy | † / † | |

## Limitations
* Healthy (MCSI) and Psoriasis (DermaCon-IN) each come from one dataset; their scores may partly reflect style.
* Contact Dermatitis vs Eczema remains the hardest pair; the grouped result is reported alongside.
* "Other" covers only the diseases in the training sources; unseen rare conditions may still be misnamed.
* The skin check is a colour rule: skin-coloured objects can pass, very dark/over-exposed photos can fail.
* Metadata comes from the source datasets' records; age/site coding differs between sources.
* Skin-tone results cover DermaCon-IN only; FST 6 has few test images.

## Ethics
Photos are processed in memory and never stored. Every result shows a disclaimer. Dataset licences are respected;
MCSI images are CC BY 4.0 and credited.
