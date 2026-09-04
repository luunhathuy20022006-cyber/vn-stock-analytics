from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import (
    EXCHANGE_MAP,
    LIMIT_TICK_TOLERANCE,
    PRICE_LIMITS,
)

CSV_COLUMNS = ["ticker", "date_code", "open", "high", "low", "close", "volume"]
OUTPUT_COLUMNS = [
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


def infer_exchange(filename: str) -> str:
    name = filename.upper()
    if "UPCOM" in name:
        return "UPCOM"
    if "HNX" in name:
        return "HNX"
    if "HSX" in name or "HOSE" in name:
        return "HOSE"
    return "UNKNOWN"


def _read_cafef_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        header=0,
        names=CSV_COLUMNS,
        usecols=range(7),
        dtype={
            "ticker": "string",
            "date_code": "string",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        },
        na_values=["", "NA", "N/A", "null", "None", "--"],
        low_memory=False,
    )
    df["exchange"] = infer_exchange(path.name)
    df["exchange"] = df["exchange"].replace(EXCHANGE_MAP)
    df["source_file"] = path.name
    return df


def concat_three_exchanges(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Ghép 3 bảng HOSE / HNX / UPCOM theo dòng, sắp theo Ticker."""
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=0, ignore_index=True)
    df["ticker"] = df["ticker"].astype("string").str.strip().str.upper()
    return df


def standardize_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Đưa biến thời gian về định dạng chuẩn YYYY-MM-DD."""
    raw = df["date_code"].astype("string").str.strip()
    digits = raw.str.replace(r"[^0-9]", "", regex=True)
    parsed = pd.to_datetime(digits, format="%Y%m%d", errors="coerce")
    still_na = parsed.isna() & raw.notna()
    if still_na.any():
        parsed.loc[still_na] = pd.to_datetime(raw.loc[still_na], errors="coerce", dayfirst=True)
    df = df.copy()
    df["date"] = parsed.dt.normalize()
    return df


def drop_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    missing = (
        df["ticker"].isna()
        | (df["ticker"].str.len() == 0)
        | df["date"].isna()
        | df[["open", "high", "low", "close"]].isna().any(axis=1)
    )
    return df.loc[~missing].copy(), int(missing.sum())


def filter_floor_ceiling(df: pd.DataFrame) -> tuple[pd.DataFrame, int, dict]:
    """
    Lọc quan sát thiếu/lỗi theo biên độ sàn–trần từng sàn:
      HOSE ±7%, HNX ±10%, UPCOM ±15% so với giá đóng cửa phiên trước.
    Ngày giao dịch đầu tiên của mỗi mã được giữ vì chưa có prev_close.
    """
    out = df.copy()
    out["limit_pct"] = out["exchange"].map(PRICE_LIMITS)
    unknown_limit = out["limit_pct"].isna()
    out = out.loc[~unknown_limit]
    out = out.sort_values(["ticker", "date"], kind="mergesort")
    out["prev_close"] = out.groupby("ticker", sort=False)["close"].shift(1)
    out["ceiling"] = out["prev_close"] * (1.0 + out["limit_pct"])
    out["floor"] = out["prev_close"] * (1.0 - out["limit_pct"])

    first_day = out["prev_close"].isna()
    lo = out["floor"] * (1.0 - LIMIT_TICK_TOLERANCE)
    hi = out["ceiling"] * (1.0 + LIMIT_TICK_TOLERANCE)
    within = (
        out["low"].ge(lo)
        & out["high"].le(hi)
        & out["open"].ge(lo)
        & out["open"].le(hi)
        & out["close"].ge(lo)
        & out["close"].le(hi)
    )
    keep = first_day | within
    dropped = int((~keep).sum())
    kept = out.loc[keep].copy()
    by_exchange = (
        out.loc[~keep].groupby("exchange").size().astype(int).to_dict() if dropped else {}
    )
    return kept, dropped, by_exchange


def drop_duplicate_ticker_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Nếu một mã xuất hiện trùng thời gian thì xóa bản ghi trùng, giữ dòng sau cùng."""
    before = len(df)
    out = (
        df.sort_values(["ticker", "date"], kind="mergesort")
        .drop_duplicates(subset=["ticker", "date"], keep="last")
    )
    return out, int(before - len(out))


def _normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats: dict = {"raw_rows": int(len(df)), "price_limits": PRICE_LIMITS}
    df = df.copy()
    df["ticker"] = df["ticker"].astype("string").str.strip().str.upper()

    df = standardize_datetime(df)
    stats["dropped_invalid_date"] = int(df["date"].isna().sum())

    df, dropped_missing = drop_missing(df)
    stats["dropped_missing"] = dropped_missing

    numeric_ok = (
        df["open"].gt(0)
        & df["high"].gt(0)
        & df["low"].gt(0)
        & df["close"].gt(0)
        & df["volume"].fillna(0).ge(0)
    )
    stats["dropped_invalid_numeric"] = int((~numeric_ok).sum())
    df = df.loc[numeric_ok]

    ohlc_ok = df["high"] >= df["low"]
    stats["dropped_invalid_ohlc"] = int((~ohlc_ok).sum())
    df = df.loc[ohlc_ok]

    ticker_ok = df["ticker"].str.fullmatch(r"[A-Z0-9.]{2,20}", na=False)
    stats["dropped_invalid_ticker"] = int((~ticker_ok).sum())
    df = df.loc[ticker_ok]

    df, dropped_limit, dropped_limit_by_ex = filter_floor_ceiling(df)
    stats["dropped_limit_band"] = dropped_limit
    stats["dropped_limit_by_exchange"] = dropped_limit_by_ex

    df, dropped_dup = drop_duplicate_ticker_dates(df)
    stats["dropped_duplicates"] = dropped_dup

    df["volume"] = df["volume"].fillna(0).astype("int64")
    df = df[OUTPUT_COLUMNS]
    df = df.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True)
    stats["clean_rows"] = int(len(df))
    stats["tickers"] = int(df["ticker"].nunique())
    return df, stats


def load_and_clean(extract_dir: Path) -> tuple[pd.DataFrame, dict]:
    files = sorted(extract_dir.glob("*.csv"))
    if not files:
        raise RuntimeError(f"Không tìm thấy CSV trong {extract_dir}")

    frames = [_read_cafef_csv(path) for path in files]
    combined = concat_three_exchanges(frames)
    combined = combined.sort_values(["ticker", "date_code"], kind="mergesort")
    cleaned, stats = _normalize(combined)
    stats["files"] = [path.name for path in files]
    stats["exchanges"] = (
        cleaned.groupby("exchange")["ticker"].nunique().astype(int).to_dict()
    )
    if cleaned.empty:
        raise RuntimeError("Dữ liệu rỗng sau khi làm sạch.")
    return cleaned, stats
