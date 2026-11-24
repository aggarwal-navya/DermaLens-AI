"""One place that knows where every project folder lives.

Every notebook does:

    paths = ProjectPaths.discover()
    paths.raw / "dermaconin"          # instead of typing a Drive path by hand

so if the Drive folder is renamed, only this file (or one environment
variable) changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Folders we try, in order, when no environment variable is set.
DEFAULT_PROJECT_CANDIDATES = (
    Path("/content/drive/MyDrive/DermaLens_AI"),          # Google Colab + Drive
    Path("/content/drive/MyDrive/skin-disease-detection"),
    Path.cwd().parent,                                     # running from notebooks/
    Path.cwd(),                                            # running from the repo root
)


@dataclass(frozen=True)
class ProjectPaths:
    """All project folders, built from one root folder."""

    root: Path

    @classmethod
    def discover(cls) -> ProjectPaths:
        """Find the project folder.

        1. If the environment variable DERMALENS_PROJECT_DIR is set, use it.
        2. Otherwise use the first candidate folder that contains data/raw.
        """
        env_value = os.environ.get("DERMALENS_PROJECT_DIR")
        if env_value:
            return cls(Path(env_value))
        for candidate in DEFAULT_PROJECT_CANDIDATES:
            if (candidate / "data" / "raw").is_dir():
                return cls(candidate)
        raise FileNotFoundError(
            "Project folder not found. Set DERMALENS_PROJECT_DIR or check that "
            "Google Drive is mounted and DermaLens_AI/data/raw exists."
        )

    # ---- data -------------------------------------------------------------
    @property
    def raw(self) -> Path:            # untouched downloads, never modified
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:      # cleaned tables built from raw data
        return self.root / "data" / "processed"

    @property
    def splits(self) -> Path:         # train/val/test CSVs + class_info.json
        return self.root / "data" / "splits"

    @property
    def cache(self) -> Path:          # resized copies of the images, for fast training
        return self.root / "data" / "cache"

    # ---- results ----------------------------------------------------------
    @property
    def models(self) -> Path:         # checkpoints and the exported web-app bundle
        return self.root / "models"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def audit(self) -> Path:          # data-audit CSVs
        return self.reports / "dataset_audit"

    @property
    def figures(self) -> Path:        # every PNG used in the report or slides
        return self.reports / "figures"

    @property
    def results(self) -> Path:        # metric tables (CSV/JSON) from experiments
        return self.reports / "results"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    def ensure(self) -> ProjectPaths:
        """Create every output folder that does not exist yet (raw is never created)."""
        for folder in (self.processed, self.splits, self.cache, self.models,
                       self.audit, self.figures, self.results):
            folder.mkdir(parents=True, exist_ok=True)
        return self

    def describe(self) -> str:
        """A short multi-line summary, printed at the top of each notebook."""
        rows = [
            ("project", self.root),
            ("raw data", self.raw),
            ("processed", self.processed),
            ("splits", self.splits),
            ("models", self.models),
            ("reports", self.reports),
        ]
        return "\n".join(f"{name:<10} {path}  [{'ok' if path.exists() else 'missing'}]"
                         for name, path in rows)
