"""Shared pieces for building the project notebooks with nbformat."""

from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf


def md(text: str):
    """A markdown cell (text is dedented, so it can be indented in the source)."""
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    """A code cell."""
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def header(number: str, title: str, purpose: str, reads: str, writes: str, runtime: str, next_step: str):
    """The standard first cell: what the notebook does, what it reads and writes."""
    return md(f"""
    # {number} · {title}

    **Purpose.** {purpose}

    | | |
    |---|---|
    | **Reads** | {reads} |
    | **Writes** | {writes} |
    | **Run time (Colab T4)** | {runtime} |
    | **Next notebook** | {next_step} |

    Run the cells from top to bottom. All shared logic lives in `src/dermalens/`; this
    notebook only calls it and explains each decision.
    """)


SETUP_MD = md("""
## 0. Setup

The same cell starts every notebook. It (1) connects Google Drive on Colab,
(2) installs missing libraries on Colab, (3) makes `src/` importable,
(4) finds every project folder and (5) loads `configs/baseline.yaml`.
""")

SETUP_CODE = code('''
import os
import sys
from pathlib import Path

# 1) On Colab, connect Google Drive so the project folder is visible.
try:
    from google.colab import drive                      # this import only works on Colab
    drive.mount("/content/drive")
    os.environ.setdefault("DERMALENS_PROJECT_DIR", "/content/drive/MyDrive/DermaLens_AI")
    IN_COLAB = True
except ImportError:
    IN_COLAB = False                                    # local run: set DERMALENS_PROJECT_DIR yourself

# 2) Where the data lives (PROJECT_DIR) and where the code lives (CODE_DIR, normally the same folder).
PROJECT_DIR = Path(os.environ.get("DERMALENS_PROJECT_DIR", Path.cwd().parent))
CODE_DIR = Path(os.environ.get("DERMALENS_CODE_DIR", PROJECT_DIR))

# 3) Colab starts empty every session: install the few libraries it does not ship with.
if IN_COLAB:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(CODE_DIR / "requirements.txt")], check=True)

# 4) Make `import dermalens` work.
sys.path.insert(0, str(CODE_DIR / "src"))

from dermalens.config import load_config
from dermalens.paths import ProjectPaths
from dermalens.utils import set_seed

# 5) Folders and settings.
paths = ProjectPaths(PROJECT_DIR).ensure()
CONFIG = load_config(os.environ.get("DERMALENS_CONFIG", CODE_DIR / "configs" / "baseline.yaml"))
set_seed(CONFIG["seed"])
print(paths.describe())
''')


def build(path: Path, cells: list) -> None:
    """Write a notebook with Colab-friendly metadata."""
    notebook = nbf.v4.new_notebook()
    notebook.cells = cells
    notebook.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    }
    nbf.write(notebook, path)
