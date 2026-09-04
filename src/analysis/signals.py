from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.factors import FACTOR_SPECS  # noqa: F401 — re-export for style.py
from src.analysis.indicators import add_indicators

STRATEGY_KEY = "tl_ma_cmf"
STRATEGIES = {
    STRATEGY_KEY: "Trendline Break (LuxAlgo) + MA + CMF",
}
DEFAULT_STRATEGY = STRATEGY_KEY


_TA_COLUMNS = ["sma_50", "cmf_20", "tl_upos", "tl_dnos", "tl_upper", "tl_lower"]
_OHLC_COLUMNS = [
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


def apply_strategy(df: pd.DataFrame, strategy: str = DEFAULT_STRATEGY) -> pd.DataFrame:
    if df.empty:
        return df
    data = df.sort_values(["ticker", "date"])
    if not all(col in data.columns for col in _TA_COLUMNS):
        keep = [col for col in _OHLC_COLUMNS if col in data.columns]
        data = add_indicators(data[keep])
    return _tl_ma_cmf(data)


def generate_signals(
    df: pd.DataFrame,
    strategy: str = DEFAULT_STRATEGY,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Tín hiệu phiên as_of (mặc định: ngày giao dịch mới nhất trong dữ liệu)."""
    if df.empty:
        return pd.DataFrame()
    data = apply_strategy(df, strategy)
    session = pd.Timestamp(as_of if as_of is not None else data["date"].max()).normalize()
    on_session = pd.to_datetime(data["date"]).dt.normalize().eq(session)
    latest = data.loc[on_session].copy()
    if latest.empty:
        return pd.DataFrame()
    latest = latest.sort_values(["ticker", "date"]).drop_duplicates("ticker", keep="last")
    latest["strategy"] = DEFAULT_STRATEGY
    latest["signal_date"] = session
    latest["can_buy"] = latest["signal"].eq("BUY")
    latest["buy_label"] = latest["signal"].map({"BUY": "MUA", "SELL": "BÁN", "HOLD": "GIỮ"})
    latest["indicator"] = latest.apply(_describe, axis=1)
    latest["break_mark"] = np.where(latest["break_ok"], "✓ Upper break", "✗ chưa break")
    latest["ma_mark"] = [
        f"{'✓' if ok else '✗'} {txt}"
        for ok, txt in zip(latest["ma_ok"], latest["ma_text"])
    ]
    latest["cmf_mark"] = [
        f"{'✓' if ok else '✗'} {txt}"
        for ok, txt in zip(latest["cmf_ok"], latest["cmf_text"])
    ]
    latest = _rank_today(latest)
    columns = [
        "ticker",
        "exchange",
        "close",
        "signal",
        "can_buy",
        "buy_label",
        "rank",
        "priority_score",
        "priority_why",
        "indicator",
        "signal_date",
        "strategy",
        "sma_50",
        "cmf_20",
        "tl_upper",
        "tl_lower",
        "tl_upos",
        "tl_dnos",
        "tl_break_up",
        "tl_break_dn",
        "pct_change",
        "volume",
        "volume_ma_20",
        "break_ok",
        "ma_ok",
        "cmf_ok",
        "break_mark",
        "ma_mark",
        "cmf_mark",
        "ma_text",
        "cmf_text",
    ]
    existing = [col for col in columns if col in latest.columns]
    return latest[existing].reset_index(drop=True)


def _tl_ma_cmf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kết hợp 3 chân:
      1) Trendline with Breaks (LuxAlgo): đã phá kháng cự (tl_upos = 1)
      2) MA: Close > SMA50
      3) CMF: CMF20 > 0
    BUY khi đủ 3 chân. SELL khi phá hỗ trợ + Close < SMA50 + CMF < 0.
    """
    df = df.copy()
    df["break_ok"] = df["tl_upos"].eq(1)
    df["ma_ok"] = df["close"] > df["sma_50"]
    df["cmf_ok"] = df["cmf_20"] > 0
    df["ma_text"] = [
        f"{c:.2f}>{m:.2f}" if pd.notna(c) and pd.notna(m) else "—"
        for c, m in zip(df["close"], df["sma_50"])
    ]
    df["cmf_text"] = df["cmf_20"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")

    buy = df["break_ok"] & df["ma_ok"] & df["cmf_ok"]
    sell = df["tl_dnos"].eq(1) & (df["close"] < df["sma_50"]) & (df["cmf_20"] < 0)
    df["signal"] = "HOLD"
    df.loc[sell, "signal"] = "SELL"
    df.loc[buy, "signal"] = "BUY"
    missing = df[["sma_50", "cmf_20", "tl_upos"]].isna().any(axis=1)
    df.loc[missing, "signal"] = "HOLD"
    df["can_buy"] = df["signal"].eq("BUY")
    return df


def _rank_today(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ưu tiên hành động hôm nay:
      1) Break đúng phiên này (mới phát tín hiệu)
      2) CMF mạnh (dòng tiền cùng chiều)
      3) Volume cao hơn trung bình 20 phiên
      4) Biến động giá hôm nay cùng chiều
    """
    out = df.copy()
    vol_ma = out["volume_ma_20"] if "volume_ma_20" in out.columns else pd.Series(np.nan, index=out.index)
    vol_ratio = (out["volume"] / vol_ma.replace(0, np.nan)).clip(upper=5).fillna(1.0)
    cmf = out["cmf_20"].fillna(0.0)
    chg = out["pct_change"].fillna(0.0) if "pct_change" in out.columns else pd.Series(0.0, index=out.index)
    fresh_up = out["tl_break_up"].fillna(0).astype(bool) if "tl_break_up" in out.columns else False
    fresh_dn = out["tl_break_dn"].fillna(0).astype(bool) if "tl_break_dn" in out.columns else False

    liq_penalty = np.where(vol_ratio < 0.5, 35.0, 0.0)
    buy_score = (
        fresh_up.astype(float) * 100
        + cmf.clip(lower=0) * 80
        + np.log1p(vol_ratio) * 12
        + chg.clip(lower=0, upper=8)
        - liq_penalty
    )
    sell_score = (
        fresh_dn.astype(float) * 100
        + (-cmf.clip(upper=0)) * 80
        + np.log1p(vol_ratio) * 12
        + (-chg.clip(upper=0, lower=-8))
        - liq_penalty
    )
    out["priority_score"] = np.select(
        [out["signal"].eq("BUY"), out["signal"].eq("SELL")],
        [buy_score, sell_score],
        default=0.0,
    )
    out["rank"] = 0
    for sig in ("BUY", "SELL"):
        mask = out["signal"].eq(sig)
        if mask.any():
            out.loc[mask, "rank"] = (
                out.loc[mask, "priority_score"].rank(ascending=False, method="first").astype(int)
            )

    reasons = []
    for row, ratio in zip(out.itertuples(index=False), vol_ratio):
        bits = []
        if row.signal == "BUY":
            bits.append("Break↑ hôm nay" if bool(getattr(row, "tl_break_up", False)) else "Đang trên kháng cự")
        elif row.signal == "SELL":
            bits.append("Break↓ hôm nay" if bool(getattr(row, "tl_break_dn", False)) else "Đang dưới hỗ trợ")
        else:
            bits.append("Chưa đủ 3 chân")
        if pd.notna(getattr(row, "cmf_20", np.nan)):
            bits.append(f"CMF {row.cmf_20:+.3f}")
        if pd.notna(ratio):
            bits.append(f"Vol {ratio:.1f}x")
        if pd.notna(getattr(row, "pct_change", np.nan)):
            bits.append(f"{row.pct_change:+.2f}%")
        reasons.append(" · ".join(bits))
    out["priority_why"] = reasons
    return out


def _describe(row: pd.Series) -> str:
    parts = []
    if bool(row.get("tl_break_up")):
        parts.append("Break↑ hôm nay")
    elif bool(row.get("break_ok")):
        parts.append("Đang trên TL kháng cự")
    elif bool(row.get("tl_break_dn")):
        parts.append("Break↓ hôm nay")
    elif pd.notna(row.get("tl_dnos")) and int(row.get("tl_dnos")) == 1:
        parts.append("Đang dưới TL hỗ trợ")
    else:
        parts.append("Chưa break")
    if pd.notna(row.get("sma_50")):
        parts.append(f"SMA50={row['sma_50']:.2f}")
    if pd.notna(row.get("cmf_20")):
        parts.append(f"CMF={row['cmf_20']:.3f}")
    return " | ".join(parts)
