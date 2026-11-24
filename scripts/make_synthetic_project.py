"""Create a small fake copy of the project's data/raw folder for testing.

It mirrors the real Drive layout and file formats exactly (same folder names,
same metadata columns), but with ~400 small generated images. Tests and the
notebook smoke-run use it so the pipeline can be checked without the real data.

It also plants the problems the pipeline must catch:
    - one exact duplicate file           -> must be removed by SHA-256
    - one resized copy of another photo  -> must be grouped by perceptual hash
    - one DermaCon-IN row marked "quality insufficient" -> must be dropped
    - SCIN case IDs damaged into scientific notation    -> must be counted

Usage:  python scripts/make_synthetic_project.py /tmp/fake_project
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

RNG = np.random.default_rng(0)
CLASS_COLOURS = {          # each class gets its own tint so a model can learn something
    "Contact Dermatitis": (200, 90, 80), "Eczema": (180, 120, 90), "Psoriasis": (220, 200, 190),
    "Scabies": (170, 80, 90), "Tinea Corporis": (150, 110, 70), "Vitiligo": (240, 230, 220),
    "Atopic Dermatitis": (190, 100, 100), "Seborrheic Dermatitis": (210, 170, 120), "normal": (190, 150, 120),
    "acne": (200, 100, 120),
}


def fake_photo(label: str, size: tuple[int, int]) -> Image.Image:
    """Skin-coloured noise with a class-coloured blob in the middle."""
    width, height = size
    # Smooth, low-frequency texture (like real skin), not pixel-level noise.
    texture = Image.fromarray(RNG.normal(128, 40, (height // 32, width // 32, 3)).clip(0, 255).astype(np.uint8))
    texture = np.asarray(texture.resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32) - 128
    base = np.array([185, 140, 110]) + texture * 0.4
    yy, xx = np.mgrid[0:height, 0:width]
    blob = ((xx - width / 2) ** 2 / (width / 4) ** 2 + (yy - height / 2) ** 2 / (height / 4) ** 2) < 1
    base[blob] = np.array(CLASS_COLOURS[label]) + texture[blob] * 0.6
    return Image.fromarray(base.clip(0, 255).astype(np.uint8))


def make_dermaconin(raw: Path) -> None:
    folder = raw / "dermaconin" / "DATASET" / "DATASET_0"
    folder.mkdir(parents=True)
    (raw / "dermaconin" / "METADATA").mkdir(parents=True)
    rows, image_number = [], 1
    labels = ["Contact Dermatitis", "Eczema", "Psoriasis", "Scabies", "Tinea Corporis", "Vitiligo"]
    for label in labels:
        for subject in range(14):
            for _ in range(2):                                   # 2 photos per subject
                name = f"IMG_{image_number:04d}.jpg"
                fake_photo(label, (640, 480)).save(folder / name, quality=90)
                rows.append({
                    "Age": "20 - 40", "Body_part": "Back", "Descriptors": "Plaque", "Confidence": 5,
                    "Disease_label": label, "Fitzpatrick": f"FST {3 + subject % 4}", "Gradability": "Yes",
                    "Image_name": name, "Main_class": "Inflammatory Disorders",
                    "Monk_skin_tone": f"MST {5 + subject % 4}",
                    "Quality": "Yes, the quality is sufficient for me or another colleague to make a diagnosis.",
                    "Sex": "Female" if subject % 2 else "Male", "Sub_class": "x",
                    "Subject_ID": f"SUB{labels.index(label):02d}{subject:03d}",
                })
                image_number += 1
    rows[0]["Quality"] = "No, the quality is insufficient for me or another colleague to make a diagnosis"
    pd.DataFrame(rows).to_csv(raw / "dermaconin" / "METADATA" / "Skin_Metadata-1.csv", index=False)

    # Planted exact duplicate: IMG_0004 becomes a byte-for-byte copy of IMG_0003 (same label).
    shutil.copyfile(folder / "IMG_0003.jpg", folder / "IMG_0004.jpg")


def make_skindisnet(raw: Path) -> None:
    codes = {"Contact Dermatitis": "CD", "Eczema": "EC", "Scabies": "SC",
             "Tinea Corporis": "TC", "Atopic Dermatitis": "AD", "Seborrheic Dermatitis": "SD"}
    meta_rows, patient = [], 1
    for label, code in codes.items():
        folder_name = f"{label} ({code})"
        for kind in ("skindisnet1/Preprocessed", "skindisnet2/Preprocessed",
                     "skindisnet1/Augmented", "skindisnet2/Augmented"):
            (raw / "skindisnet" / kind / folder_name).mkdir(parents=True, exist_ok=True)
        for n in range(1, 25):
            if n % 3 == 1:
                patient += 1
            image = fake_photo(label, (512, 512))
            image.save(raw / "skindisnet" / "skindisnet2/Preprocessed" / folder_name / f"{code} ({n}).jpg")
            image.save(raw / "skindisnet" / "skindisnet1/Preprocessed" / folder_name / f"{code}_{n:02d}.jpg")
            image.rotate(90).save(raw / "skindisnet" / "skindisnet2/Augmented" / folder_name / f"aug_{n}.jpg")
            meta_rows.append({"Folder_name": folder_name, "Patient_id": f"P_{patient:03d}",
                              "Image_id": f"{code} ({n})", "Age": "30", "Sex": "Female",
                              "Leision_location": "Hand", "Diagnosis": label})
    pd.DataFrame(meta_rows).to_csv(raw / "skindisnet" / "skindisnet2" / "SkinDisNet_Metadata.csv", index=False)

    # Planted near duplicate: "EC (2)" re-saved smaller as "CD (5)"'s neighbour in another patient.
    source = raw / "skindisnet/skindisnet2/Preprocessed/Eczema (EC)/EC (2).jpg"
    target = raw / "skindisnet/skindisnet2/Preprocessed/Eczema (EC)/EC (24).jpg"
    Image.open(source).resize((400, 400)).save(target, quality=80)


def make_mcsi(raw: Path) -> None:
    folder = raw / "mcsi" / "extracted" / "images"
    folder.mkdir(parents=True)
    rows, tones = [], []
    for n in range(1, 41):
        diagnostic = "normal" if n <= 30 else "acne"
        name = f"{diagnostic}_{n}.jpg"
        fake_photo(diagnostic, (300, 300)).save(folder / name)
        rows.append({"img_id": name, "diagnostic": diagnostic})
        tones.append({"Filename": name, "Type": "dark" if n % 3 == 0 else "light"})
    pd.DataFrame(rows).to_csv(raw / "mcsi" / "extracted" / "metadata.csv", index=False)
    pd.DataFrame(tones).to_csv(raw / "mcsi" / "extracted" / "skin_tone_labels.csv", index=False)


def make_scin(raw: Path) -> None:
    folder = raw / "scin"
    folder.mkdir(parents=True)
    damaged = ["-1.00E+18", "-1.00E+18", "-2.31E+18", "4.55E+17"]
    pd.DataFrame({"case_id": damaged, "image_1_path": [f"dataset/images/{i}.png" for i in range(4)]}
                 ).to_csv(folder / "dataset_scin_cases.csv", index=False)
    pd.DataFrame({"case_id": damaged, "weighted_skin_condition_label": ["{'Eczema': 1.0}"] * 4}
                 ).to_csv(folder / "dataset_scin_labels.csv", index=False)


def make_project(root: Path) -> Path:
    """Build the fake project and return its root folder."""
    if root.exists():
        shutil.rmtree(root)
    raw = root / "data" / "raw"
    for maker in (make_dermaconin, make_skindisnet, make_mcsi, make_scin):
        maker(raw)
    shutil.copytree(Path(__file__).resolve().parents[1] / "configs", root / "configs")
    return root


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dermalens_fake_project")
    print("Fake project created at", make_project(target))
