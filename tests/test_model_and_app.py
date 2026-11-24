"""Dataset, training loop, evaluation helpers, export bundle and the web API work together."""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dermalens import data, evaluate, export, model, ood, splits, train
from dermalens import metadata as meta
from dermalens.labels import OTHER_CLASS, TARGET_CLASSES

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny_run(fake_paths, clean_manifest, tmp_path_factory):
    """Train resnet18 (no pretrained weights) for 2 epochs on the fake data and export a bundle."""
    split = splits.make_grouped_split(clean_manifest, seed=0)
    split = data.cache_resized_images(split, fake_paths.raw, fake_paths.cache, short_side=80)
    loaders = {name: data.make_loader(split[split["split"] == name], fake_paths.cache, TARGET_CLASSES, 64,
                                      train=(name == "train"), batch_size=16, num_workers=0)
               for name in ("train", "val")}
    weights = splits.compute_class_weights(split[split["split"] == "train"], TARGET_CLASSES)
    net = model.build_model("resnet18", len(TARGET_CLASSES), pretrained=False)
    run_dir = tmp_path_factory.mktemp("run")
    history = train.fit(net, loaders["train"], loaders["val"], [weights[c] for c in TARGET_CLASSES],
                        train.TrainConfig(epochs=2, freeze_epochs=1, ema_decay=0.9), "cpu", run_dir,
                        {"model_name": "resnet18", "classes": TARGET_CLASSES})
    bundle = export.save_bundle([{"model": net, "name": "resnet18", "image_size": 64}], run_dir / "deploy",
                                TARGET_CLASSES, {"accuracy": 0.5}, confidence_threshold=0.0)
    return {"history": history, "run_dir": run_dir, "bundle": bundle, "split": split, "model": net,
            "loaders": loaders}


def test_dataset_returns_tensor_and_index(fake_paths, tiny_run):
    dataset = data.SkinDataset(tiny_run["split"], fake_paths.cache, TARGET_CLASSES, data.build_transforms(64, False))
    tensor, vector, index = dataset[0]
    assert tuple(tensor.shape) == (3, 64, 64) and 0 <= index < len(TARGET_CLASSES)
    assert tuple(vector.shape) == (meta.META_DIM,) and float(vector.sum()) == 3.0   # one flag per field


def test_missing_cache_falls_back_to_raw_images(fake_paths, clean_manifest):
    """v2 bug 'KeyError: cache_path' in notebook 07: now the raw files are used instead."""
    root, column = data.resolve_images(clean_manifest, fake_paths)
    assert column == "rel_path" and root == fake_paths.raw


def test_training_writes_checkpoints_and_history(tiny_run):
    assert (tiny_run["run_dir"] / "best.pt").exists() and (tiny_run["run_dir"] / "last.pt").exists()
    assert list(tiny_run["history"]["stage"]) == ["frozen", "finetune"]


def test_bootstrap_interval_contains_point():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 200)
    y_pred = np.where(rng.random(200) < 0.8, y_true, (y_true + 1) % 3)
    point, low, high = evaluate.bootstrap_ci(y_true, y_pred, n_boot=200)
    assert low <= point <= high


def test_temperature_never_changes_the_winning_class(tiny_run):
    logits, targets = evaluate.predict_logits(tiny_run["model"], tiny_run["loaders"]["val"], "cpu")
    temperature = evaluate.fit_temperature(logits, targets)
    assert (evaluate.softmax(logits, temperature).argmax(1) == logits.argmax(1)).all()


def test_bundle_reloads_identically(tiny_run):
    reloaded, card = export.load_bundle(tiny_run["bundle"])
    assert card["classes"] == TARGET_CLASSES
    import torch
    x, m = torch.randn(2, 3, 64, 64), torch.zeros(2, meta.META_DIM)
    with torch.no_grad():
        assert torch.allclose(tiny_run["model"].eval()(x, m), reloaded[0][0](x, m), atol=1e-5)


def test_open_set_decisions():
    classes = TARGET_CLASSES + [OTHER_CLASS]
    sure = np.eye(len(classes))
    assert ood.decide(sure[classes.index("Psoriasis")], classes, 0.5) == ("identified", "Psoriasis")
    assert ood.decide(sure[classes.index("Healthy")], classes, 0.5) == ("healthy", "Healthy")
    assert ood.decide(sure[classes.index(OTHER_CLASS)], classes, 0.5)[0] == "not_identified"
    assert ood.decide(np.full(len(classes), 1 / len(classes)), classes, 0.5)[0] == "not_identified"
    assert ood.decide(sure[0], classes, 0.5, skin_ratio=0.01)[0] == "not_skin"


def test_skin_gate_separates_skin_from_other_colours():
    assert ood.skin_pixel_ratio(Image.new("RGB", (64, 64), (190, 140, 120))) > 0.9    # light-brown skin
    assert ood.skin_pixel_ratio(Image.new("RGB", (64, 64), (90, 56, 40))) > 0.9       # dark skin
    assert ood.skin_pixel_ratio(Image.new("RGB", (64, 64), (40, 90, 200))) < 0.05     # blue
    assert ood.skin_pixel_ratio(Image.new("RGB", (64, 64), (60, 160, 60))) < 0.05     # green


def test_open_set_manifest_has_other_class_and_metadata(fake_paths):
    from dermalens import manifest as mf

    table = mf.build_manifest(fake_paths.raw, open_set=True)
    assert OTHER_CLASS in set(table["label"]) and {"meta_age", "meta_sex", "meta_regions"} <= set(table.columns)
    assert not table.loc[table["label"] == OTHER_CLASS, "source_label"].str.contains("Eczema|Psoria", case=False).any()


def test_api_predict_endpoint(tiny_run, monkeypatch):
    monkeypatch.setenv("DERMALENS_BUNDLE", str(tiny_run["bundle"]))
    sys.modules.pop("app.api", None)
    sys.path.insert(0, str(REPO))
    from fastapi.testclient import TestClient

    from app.api import app

    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        buffer = io.BytesIO()
        Image.new("RGB", (300, 200), (190, 140, 120)).save(buffer, format="JPEG")
        answer = client.post("/api/predict", files={"file": ("skin.jpg", buffer.getvalue(), "image/jpeg")},
                             data={"age_group": "20-40", "sex": "female", "body_regions": "hand,arm"}).json()
        assert answer["status"] in ("identified", "healthy") and answer["label"] in TARGET_CLASSES
        assert answer["used_metadata"] is True and answer["message"]
        info = client.get("/api/info").json()
        assert info["metadata_options"]["body_regions"] and OTHER_CLASS not in info["classes"]
        blue = io.BytesIO()
        Image.new("RGB", (300, 200), (40, 90, 200)).save(blue, format="PNG")
        not_skin = client.post("/api/predict", files={"file": ("b.png", blue.getvalue(), "image/png")}).json()
        assert not_skin["status"] == "not_skin" and not_skin["top3"] == []
        wrong = client.post("/api/predict", files={"file": ("skin.jpg", buffer.getvalue(), "image/jpeg")},
                            data={"sex": "robot"})
        assert wrong.status_code == 400
        assert abs(sum(t["probability"] for t in answer["top3"]) - 1) < 0.5
        assert answer["heatmap_png_base64"]
        bad = client.post("/api/predict", files={"file": ("x.txt", b"hello", "text/plain")})
        assert bad.status_code == 415
