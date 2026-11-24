"""Turning manifest rows into tensors the model can learn from.

Pieces:
    cache_resized_images  - shrink every image once so training reads small files
    resolve_images        - choose cached files, or fall back to the raw files
    ShadesOfGray          - colour constancy: removes each camera's colour cast,
                            which weakens the "which dataset is this?" shortcut
    build_transforms      - training augmentation / fixed evaluation resize
    SkinDataset           - row -> (image tensor, metadata vector, class index)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from . import metadata as meta

# ImageNet colour statistics: pretrained models expect inputs normalised with these.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_rgb(path: str | Path) -> Image.Image:
    """Open an image upright (phone rotation flag applied) and in RGB."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


# ---------------------------------------------------------------------------
# 1. Resized-image cache
# ---------------------------------------------------------------------------
def _resize_short_side(image: Image.Image, short_side: int) -> Image.Image:
    """Scale the image so its shorter side equals `short_side` (never enlarges)."""
    width, height = image.size
    scale = short_side / min(width, height)
    if scale >= 1:
        return image
    return image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)


def cache_resized_images(manifest: pd.DataFrame, raw_dir: Path, cache_dir: Path,
                         short_side: int = 448, workers: int = 16) -> pd.DataFrame:
    """Save a smaller JPEG copy of every image and add a 'cache_path' column.

    Copies live in cache_dir/s<short_side>/ so caches of different sizes never
    mix. 'cache_path' is relative to cache_dir. Existing files are skipped.
    """
    folder = Path(cache_dir) / f"s{short_side}"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = manifest.copy()
    manifest["cache_path"] = f"s{short_side}/" + manifest["sha256"].str.slice(0, 32) + ".jpg"

    def make_one(rel_path: str, cache_name: str) -> None:
        target = Path(cache_dir) / cache_name
        if target.exists():
            return
        _resize_short_side(load_rgb(raw_dir / rel_path), short_side).save(target, "JPEG", quality=95)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(make_one, manifest["rel_path"], manifest["cache_path"]))
    return manifest


def image_root(paths) -> Path:
    """Cache folder to read from: the fast local copy on Colab if present, else data/cache on Drive."""
    local_copy = Path("/content/cache")
    return local_copy if local_copy.exists() else paths.cache


def resolve_images(table: pd.DataFrame, paths) -> tuple[Path, str]:
    """Return (folder, column) to load images from.

    Uses the resized cache when the table has a 'cache_path' column whose first
    file exists; otherwise falls back to the original files (slower, but it never
    fails with KeyError: 'cache_path' as in v2 when notebook 06 was skipped).
    """
    root = image_root(paths)
    if "cache_path" in table.columns and table["cache_path"].notna().all():
        if (root / table["cache_path"].iloc[0]).exists():
            return root, "cache_path"
    print("Note: resized cache not found - reading original images from data/raw "
          "(run notebook 06 once to make training much faster).")
    return paths.raw, "rel_path"


# ---------------------------------------------------------------------------
# 2. Colour constancy and transforms
# ---------------------------------------------------------------------------
class ShadesOfGray:
    """Shades-of-Gray colour constancy (Finlayson & Trezzi, 2004), p = 6.

    Estimates the light source colour from the image and divides it out, so the
    same lesion photographed with two phones gets similar colours. Widely used
    in dermatology image challenges (ISIC). Deterministic: applied in training
    and at inference alike.
    """

    def __init__(self, power: float = 6.0):
        self.power = power

    def __call__(self, image: Image.Image) -> Image.Image:
        pixels = np.asarray(image, dtype=np.float32)
        illuminant = np.power(np.mean(np.power(pixels, self.power), axis=(0, 1)), 1 / self.power)
        illuminant = illuminant / (np.linalg.norm(illuminant) + 1e-6) * np.sqrt(3)
        balanced = np.clip(pixels / (illuminant + 1e-6), 0, 255)
        return Image.fromarray(balanced.astype(np.uint8))


def build_transforms(image_size: int = 224, train: bool = True, color_constancy: bool = True):
    """Image transforms.

    Training: random crop (50-100% of the photo), flips, rotation, mild
    brightness/contrast, occasional blur. Hue is not changed: redness is a
    symptom and skin tone is what the fairness analysis measures.
    Evaluation: resize, centre crop, no randomness.
    """
    from torchvision import transforms as T

    first = [ShadesOfGray()] if color_constancy else []
    normalise = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train:
        return T.Compose(first + [
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.8, 1.25)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(25),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
        ] + normalise)
    return T.Compose(first + [
        T.Resize(int(round(image_size * 1.14))),
        T.CenterCrop(image_size),
    ] + normalise)


def center_square(image: Image.Image) -> Image.Image:
    """The centre square the evaluation transform crops - use it to line a heatmap up with the photo."""
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def denormalize(tensor):
    """Undo Normalize so a tensor can be shown as a picture (values back in 0-1)."""
    import torch

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)


# ---------------------------------------------------------------------------
# 3. Dataset and DataLoader
# ---------------------------------------------------------------------------
class SkinDataset:
    """PyTorch Dataset over a manifest table.

    dataset[i] returns (image_tensor, metadata_vector, class_index) for row i.
    With use_metadata=False the metadata vector says "everything missing".
    In training, meta_dropout hides each metadata field at random.
    """

    def __init__(self, table: pd.DataFrame, image_root: Path, classes: list[str], transform,
                 path_col: str = "cache_path", label_col: str = "label", use_metadata: bool = True,
                 meta_dropout: float = 0.0, seed: int = 0):
        self.table = table.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.class_index = {name: i for i, name in enumerate(classes)}
        self.transform = transform
        self.path_col = path_col
        self.label_col = label_col
        self.meta_dropout = meta_dropout
        self.rng = np.random.default_rng(seed)
        unknown = set(self.table[label_col]) - set(self.class_index)
        if unknown:
            raise ValueError(f"Labels not in the class list: {unknown}")
        missing_vector = meta.encode(None, None, [])
        has_meta = use_metadata and "meta_age" in self.table.columns
        self.meta_vectors = np.stack([meta.encode_row(row) if has_meta else missing_vector
                                      for _, row in self.table.iterrows()]) if len(self.table) else \
            np.zeros((0, meta.META_DIM), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int):
        import torch

        row = self.table.iloc[index]
        image = self.transform(load_rgb(self.image_root / row[self.path_col]))
        worker = torch.utils.data.get_worker_info()
        if worker is not None and getattr(self, "_worker_id", None) != worker.id:
            self.rng = np.random.default_rng(worker.seed % 2**32)   # each worker: its own dropout pattern
            self._worker_id = worker.id
        vector = meta.drop_fields(self.meta_vectors[index], self.rng, self.meta_dropout)
        return image, torch.from_numpy(vector), self.class_index[row[self.label_col]]


def make_loader(table: pd.DataFrame, image_root: Path, classes: list[str], image_size: int, train: bool,
                batch_size: int = 32, num_workers: int = 2, path_col: str = "cache_path",
                label_col: str = "label", use_metadata: bool = True, meta_dropout: float = 0.0,
                color_constancy: bool = True, seed: int = 0):
    """Wrap a manifest table in a DataLoader (shuffled + augmented for training only)."""
    import torch
    from torch.utils.data import DataLoader

    dataset = SkinDataset(table, image_root, classes, build_transforms(image_size, train, color_constancy),
                          path_col, label_col, use_metadata, meta_dropout if train else 0.0, seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=train and len(dataset) > batch_size,
                      persistent_workers=num_workers > 0)
