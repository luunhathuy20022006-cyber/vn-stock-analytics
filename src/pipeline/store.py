from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from src.config import META_PATH, PRICES_PATH, PROCESSED_DIR, SIGNALS_PATH
from src.pipeline.cafef import CafeFDataset


def _json_default(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def save_prices(df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PRICES_PATH, index=False)
    return PRICES_PATH


def save_signals(df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SIGNALS_PATH, index=False)
    return SIGNALS_PATH


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
    if not PRICES_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(PRICES_PATH)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_signals() -> pd.DataFrame:
    if not SIGNALS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(SIGNALS_PATH)
    if "signal_date" in df.columns:
        df["signal_date"] = pd.to_datetime(df["signal_date"])
    return df
