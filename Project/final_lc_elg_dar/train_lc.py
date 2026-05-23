"""Wrapper entrypoint for the ELG-lite training script."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "experiments" / "lc_dar_elg" / "train_lc_elg.py"
    runpy.run_path(str(target), run_name="__main__")
