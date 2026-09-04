from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.trendlines import CMF_PERIOD, MA_PERIOD, TL_LENGTH, chaikin_money_flow, trendlines_with_breaks


def _wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return _wilder(tr, period)


def _ta_one(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    close = g["close"]
    high = g["high"]
    low = g["low"]
    volume = g["volume"]

    g["sma_20"] = close.rolling(20, min_periods=20).mean()
    g["sma_50"] = close.rolling(MA_PERIOD, min_periods=MA_PERIOD).mean()
    g["atr_14"] = _atr(high, low, close, TL_LENGTH)
    g["cmf_20"] = chaikin_money_flow(high, low, close, volume, CMF_PERIOD)
    g["volume_ma_20"] = volume.rolling(20, min_periods=20).mean()
    g["pct_change"] = close.pct_change() * 100

    tl = trendlines_with_breaks(high, low, close, g["atr_14"], length=TL_LENGTH)
    for col in tl.columns:
        g[col] = tl[col]
    return g


REQUIRED_TA_COLUMNS = ["sma_50", "cmf_20", "tl_upos", "tl_dnos", "tl_upper", "tl_lower"]
BASE_OHLC_COLUMNS = [
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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Tính MA, CMF và Trendlines with Breaks (LuxAlgo) theo từng mã."""
    if df.empty:
        return df
    ordered = df.sort_values(["ticker", "date"])
    parts = [_ta_one(group) for _, group in ordered.groupby("ticker", sort=False, observed=True)]
    return pd.concat(parts, ignore_index=True)


def add_chart_ta(df: pd.DataFrame) -> pd.DataFrame:
    """Bổ sung EMA, RSI, MACD, Bollinger, ATR cho một mã trên biểu đồ (Task 2 đề bài)."""
    if df.empty:
        return df
    g = df.sort_values("date").copy()
    close = g["close"]
    if "ema_20" not in g.columns:
        g["ema_20"] = close.ewm(span=20, min_periods=20, adjust=False).mean()
    if "ema_50" not in g.columns:
        g["ema_50"] = close.ewm(span=50, min_periods=50, adjust=False).mean()
    if "bb_mid" not in g.columns:
        g["bb_mid"] = close.rolling(20, min_periods=20).mean()
        std = close.rolling(20, min_periods=20).std()
        g["bb_up"] = g["bb_mid"] + 2 * std
        g["bb_dn"] = g["bb_mid"] - 2 * std
    if "rsi_14" not in g.columns:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        g["rsi_14"] = 100 - (100 / (1 + rs))
    if "macd" not in g.columns:
        ema12 = close.ewm(span=12, min_periods=12, adjust=False).mean()
        ema26 = close.ewm(span=26, min_periods=26, adjust=False).mean()
        g["macd"] = ema12 - ema26
        g["macd_signal"] = g["macd"].ewm(span=9, min_periods=9, adjust=False).mean()
        g["macd_hist"] = g["macd"] - g["macd_signal"]
    if "atr_14" not in g.columns:
        g["atr_14"] = _atr(g["high"], g["low"], close, 14)
    return g


def ensure_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Tính lại MA/CMF/Trendline nếu parquet hoặc cache chưa có các cột chiến lược."""
    if df.empty:
        return df
    if all(col in df.columns for col in REQUIRED_TA_COLUMNS):
        return df
    keep = [col for col in BASE_OHLC_COLUMNS if col in df.columns]
    return add_indicators(df[keep])
