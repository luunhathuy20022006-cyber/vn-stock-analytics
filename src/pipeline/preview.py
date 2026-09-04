from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import EXTRACT_DIR, RAW_DIR
from src.pipeline.clean import infer_exchange

CLEAN_COLUMNS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "exchange",
    "prev_close",
    "floor",
    "ceiling",
    "limit_pct",
]


def extract_dir_from_meta(meta: dict | None) -> Path | None:
    if not meta:
        return None
    url = str(meta.get("source_url") or "")
    folder = None
    if "/ami_data/" in url:
        folder = url.split("/ami_data/", 1)[1].split("/", 1)[0]
    if not folder:
        date = str(meta.get("dataset_date") or "").replace("-", "")
        folder = date or None
    if not folder:
        return None
    path = EXTRACT_DIR / folder
    return path if path.exists() else None


def list_download_files(meta: dict | None) -> list[dict]:
    extract_dir = extract_dir_from_meta(meta)
    items: list[dict] = []
    if extract_dir:
        for path in sorted(extract_dir.glob("*.csv")):
            items.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "exchange": infer_exchange(path.name),
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                    "kind": "csv",
                }
            )
    if RAW_DIR.exists():
        for path in sorted(RAW_DIR.glob("*.zip")):
            items.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "exchange": "3 sàn",
                    "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                    "kind": "zip",
                }
            )
    order = {"HOSE": 0, "HNX": 1, "UPCOM": 2}
    items.sort(key=lambda item: (0 if item["kind"] == "csv" else 1, order.get(item["exchange"], 9), item["name"]))
    return items


def preview_raw_csv(path: Path, nrows: int = 200, ticker: str | None = None) -> pd.DataFrame:
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return pd.read_csv(path, encoding="utf-8-sig", nrows=nrows)

    ticker_col = None
    hits: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, encoding="utf-8-sig", chunksize=250_000, low_memory=False):
        if ticker_col is None:
            ticker_col = chunk.columns[0]
        matched = chunk[chunk[ticker_col].astype(str).str.strip().str.upper() == ticker]
        if not matched.empty:
            hits.append(matched)
            if sum(len(part) for part in hits) >= nrows:
                break
    if not hits:
        return pd.DataFrame()
    return pd.concat(hits, ignore_index=True).head(nrows)


def preview_clean(prices: pd.DataFrame, nrows: int = 200, ticker: str | None = None, exchange: str | None = None) -> pd.DataFrame:
    cols = [c for c in CLEAN_COLUMNS if c in prices.columns]
    view = prices[cols].copy()
    if exchange and exchange != "Tất cả":
        view = view[view["exchange"] == exchange]
    ticker = (ticker or "").strip().upper()
    if ticker:
        view = view[view["ticker"].astype(str).str.upper() == ticker]
    return view.sort_values(["date", "ticker"], ascending=[False, True]).head(nrows).reset_index(drop=True)


def cleaning_step_table(stats: dict) -> pd.DataFrame:
    raw = int(stats.get("raw_rows") or 0)
    steps = [
        ("1. Ghép 3 sàn theo dòng dựa trên Ticker", 0, raw),
        ("2. Chuẩn hóa thời gian YYYY-MM-DD + lọc missing", int(stats.get("dropped_missing") or 0), 0),
        ("3. Loại numeric không hợp lệ", int(stats.get("dropped_invalid_numeric") or 0), 0),
        ("4. Loại nến sai (high < low)", int(stats.get("dropped_invalid_ohlc") or 0), 0),
        ("5. Lọc sàn/trần HOSE ±7% · HNX ±10% · UPCOM ±15%", int(stats.get("dropped_limit_band") or 0), 0),
        ("6. Xóa trùng (cùng Ticker + cùng ngày)", int(stats.get("dropped_duplicates") or 0), 0),
        ("7. Dữ liệu sau tiền xử lý", 0, int(stats.get("clean_rows") or 0)),
    ]
    remaining = raw
    rows = []
    for label, dropped, explicit_remaining in steps:
        if explicit_remaining:
            remaining = explicit_remaining
        else:
            remaining = max(remaining - dropped, 0)
        rows.append(
            {
                "Bước": label,
                "Số dòng loại": dropped,
                "Số dòng còn lại": remaining,
            }
        )
    return pd.DataFrame(rows)
