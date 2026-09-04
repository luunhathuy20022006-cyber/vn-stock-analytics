from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import META_PATH, PRICES_CSV, PRICES_PATH, PROCESSED_DIR, SIGNALS_CSV, SIGNALS_PATH
from src.pipeline.cafef import CafeFDataset


def _json_default(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _save_table(df: pd.DataFrame, parquet_path: Path, csv_path: Path) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        df.to_csv(csv_path, index=False, compression="gzip")
        return csv_path


def _load_table(parquet_path: Path, csv_path: Path, date_col: str) -> pd.DataFrame:
    df = pd.DataFrame()
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
        except Exception:
            df = pd.DataFrame()
    if df.empty and csv_path.exists():
        df = pd.read_csv(csv_path, compression="gzip")
    if df.empty:
        return df
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def save_prices(df: pd.DataFrame) -> Path:
    return _save_table(df, PRICES_PATH, PRICES_CSV)


def save_signals(df: pd.DataFrame) -> Path:
    return _save_table(df, SIGNALS_PATH, SIGNALS_CSV)


def save_meta(dataset: CafeFDataset, stats: dict, extra: dict | None = None) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "CafeF",
        "page_url": "https://cafef.vn/du-lieu/du-lieu-download.chn",
        "price_type": "adjusted",
        "data_kind": "upto_3_exchanges",
        "label": "Đã điều chỉnh - Số liệu giao dịch - Upto 3 sàn",
        "source_url": dataset.url,
        "dataset_date": dataset.dataset_date,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
    }
    if extra:
        payload.update(extra)
    META_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return META_PATH


def load_meta() -> dict | None:
    if not META_PATH.exists():
        return None
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def load_prices() -> pd.DataFrame:
    return _load_table(PRICES_PATH, PRICES_CSV, "date")


def load_signals() -> pd.DataFrame:
    return _load_table(SIGNALS_PATH, SIGNALS_CSV, "signal_date")
