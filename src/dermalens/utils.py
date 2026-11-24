"""Small helpers used everywhere: reproducible seeds, device choice, JSON files."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Make Python, NumPy and PyTorch produce the same random numbers every run."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # data notebooks do not need PyTorch
        pass


def get_device() -> str:
    """Return 'cuda' when a GPU is available (Colab T4), otherwise 'cpu'."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def save_json(data: Any, path: str | Path) -> Path:
    """Write a dict/list to a pretty-printed JSON file and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


def load_json(path: str | Path) -> Any:
    """Read a JSON file back into Python objects."""
    return json.loads(Path(path).read_text())
