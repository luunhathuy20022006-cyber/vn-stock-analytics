from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from src.config import EXTRACT_DIR
from src.pipeline.cafef import CafeFDataset


def extract_zip(zip_path: Path, dataset: CafeFDataset) -> Path:
    target = EXTRACT_DIR / dataset.folder_date
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target)

    csv_files = sorted(target.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"Zip không chứa file CSV: {zip_path}")
    return target
