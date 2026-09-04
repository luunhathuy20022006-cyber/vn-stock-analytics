"""
VN Stock Analytics — đề Duan_1
Một file nộp: pipeline CafeF + chỉ báo + tín hiệu + dashboard Streamlit.

Chạy:
    pip install streamlit pandas numpy plotly pyarrow requests
    streamlit run Duan_1.py

Nguồn: CafeF — Đã điều chỉnh / Số liệu giao dịch / Upto 3 sàn (HOSE, HNX, UPCOM).
Chiến lược: LuxAlgo Trendlines with Breaks + SMA50 + CMF20.
"""
from __future__ import annotations

import gc
import json
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="VN Stock Analytics",
    page_icon="VN",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Cấu hình
# =============================================================================

ROOT = Path(__file__).resolve().parent
CAFEF_PAGE_URL = "https://cafef.vn/du-lieu/du-lieu-download.chn"
CAFEF_CDN_HOST = "https://cafef1.mediacdn.vn"
ADJUSTED_UPTO_NAME = "CafeF.SolieuGD.Upto"

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTRACT_DIR = DATA_DIR / "extracted"
PROCESSED_DIR = DATA_DIR / "processed"
PRICES_PATH = PROCESSED_DIR / "prices.parquet"
SIGNALS_PATH = PROCESSED_DIR / "signals.parquet"
META_PATH = PROCESSED_DIR / "meta.json"

EXCHANGE_MAP = {"HSX": "HOSE", "HOSE": "HOSE", "HNX": "HNX", "UPCOM": "UPCOM"}
PRICE_LIMITS = {"HOSE": 0.07, "HNX": 0.10, "UPCOM": 0.15}
LIMIT_TICK_TOLERANCE = 0.001
PREPROCESS_VERSION = "floor-ceiling-v1"
CLOUD_HISTORY_YEARS = 3
REQUIRED_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume", "exchange"]
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

STRATEGY_KEY = "tl_ma_cmf"
STRATEGIES = {STRATEGY_KEY: "Trendline Break (LuxAlgo) + MA + CMF"}
DEFAULT_STRATEGY = STRATEGY_KEY
FACTOR_SPECS = {
    STRATEGY_KEY: [
        ("break", "⚡", "Break", "Break kháng cự Trendline (LuxAlgo)"),
        ("ma", "▲", "MA50", "Close > SMA50"),
        ("cmf", "▮", "CMF", "CMF20 > 0"),
    ]
}

TL_LENGTH = 14
TL_SLOPE_K = 1.0
MA_PERIOD = 50
CMF_PERIOD = 20

CSV_COLUMNS = ["ticker", "date_code", "open", "high", "low", "close", "volume"]
OUTPUT_COLUMNS = [
    "ticker", "date", "open", "high", "low", "close", "volume",
    "exchange", "prev_close", "floor", "ceiling", "limit_pct",
]
CLEAN_COLUMNS = OUTPUT_COLUMNS
_TA_COLUMNS = ["sma_50", "cmf_20", "tl_upos", "tl_dnos", "tl_upper", "tl_lower"]
_OHLC_COLUMNS = OUTPUT_COLUMNS
REQUIRED_TA_COLUMNS = _TA_COLUMNS
BASE_OHLC_COLUMNS = OUTPUT_COLUMNS

ProgressCb = Callable[[str, float | None], None]


def is_streamlit_cloud() -> bool:
    if os.getenv("STREAMLIT_CLOUD", "").lower() in {"1", "true", "yes"}:
        return True
    if os.getenv("STREAMLIT_SHARING"):
        return True
    if Path("/mount/src").exists():
        return True
    return "streamlit" in os.getenv("HOSTNAME", "").lower()


def history_years() -> int | None:
    raw = os.getenv("STOCK_HISTORY_YEARS", "").strip()
    if raw:
        try:
            years = int(raw)
        except ValueError:
            return None
        return years if years > 0 else None
    if is_streamlit_cloud():
        return CLOUD_HISTORY_YEARS
    return None


# =============================================================================
# Phân tích kỹ thuật — Trendline LuxAlgo + CMF + MA
# =============================================================================

def _pivot_high(high: np.ndarray, length: int) -> np.ndarray:
    n = len(high)
    out = np.full(n, np.nan)
    for i in range(length * 2, n):
        center = i - length
        left = high[center - length : center]
        right = high[center + 1 : center + length + 1]
        if left.size == 0 or right.size == 0:
            continue
        value = high[center]
        if value > left.max() and value > right.max():
            out[i] = value
    return out


def _pivot_low(low: np.ndarray, length: int) -> np.ndarray:
    n = len(low)
    out = np.full(n, np.nan)
    for i in range(length * 2, n):
        center = i - length
        left = low[center - length : center]
        right = low[center + 1 : center + length + 1]
        if left.size == 0 or right.size == 0:
            continue
        value = low[center]
        if value < left.min() and value < right.min():
            out[i] = value
    return out


def trendlines_with_breaks(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    length: int = TL_LENGTH,
    k: float = TL_SLOPE_K,
) -> pd.DataFrame:
    """
    Trendlines with Breaks theo LuxAlgo:
    - Pivot high/low với lookback `length`
    - Độ dốc ATR: slope = ATR(length) / length * k
    - Upper neo pivot high (kháng cự), Lower neo pivot low (hỗ trợ)
    - upos/dnos: đã break, reset khi có pivot mới
    """
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    a = atr.to_numpy(dtype=float)
    n = len(c)
    slope = np.divide(a, length, out=np.zeros(n), where=np.isfinite(a)) * k

    ph = _pivot_high(h, length)
    pl = _pivot_low(l, length)

    upper_raw = np.full(n, np.nan)
    lower_raw = np.full(n, np.nan)
    upper_line = np.full(n, np.nan)
    lower_line = np.full(n, np.nan)
    upos = np.zeros(n, dtype=np.int8)
    dnos = np.zeros(n, dtype=np.int8)

    u = np.nan
    lo = np.nan
    sph = 0.0
    spl = 0.0
    prev_upos = 0
    prev_dnos = 0

    for i in range(n):
        sl = slope[i] if np.isfinite(slope[i]) else 0.0
        if np.isfinite(ph[i]):
            u = ph[i]
            sph = sl
            prev_upos = 0
        elif np.isfinite(u):
            u = u - sph

        if np.isfinite(pl[i]):
            lo = pl[i]
            spl = sl
            prev_dnos = 0
        elif np.isfinite(lo):
            lo = lo + spl

        uline = u - sph * length if np.isfinite(u) else np.nan
        lline = lo + spl * length if np.isfinite(lo) else np.nan
        upper_raw[i] = u
        lower_raw[i] = lo
        upper_line[i] = uline
        lower_line[i] = lline

        if np.isfinite(ph[i]):
            cur_upos = 0
        elif np.isfinite(uline) and c[i] > uline:
            cur_upos = 1
        else:
            cur_upos = prev_upos

        if np.isfinite(pl[i]):
            cur_dnos = 0
        elif np.isfinite(lline) and c[i] < lline:
            cur_dnos = 1
        else:
            cur_dnos = prev_dnos

        upos[i] = cur_upos
        dnos[i] = cur_dnos
        prev_upos = cur_upos
        prev_dnos = cur_dnos

    prev_u = np.roll(upos, 1)
    prev_d = np.roll(dnos, 1)
    prev_u[0] = 0
    prev_d[0] = 0
    return pd.DataFrame(
        {
            "tl_upper": upper_line,
            "tl_lower": lower_line,
            "tl_upos": upos,
            "tl_dnos": dnos,
            "tl_break_up": (upos == 1) & (prev_u == 0),
            "tl_break_dn": (dnos == 1) & (prev_d == 0),
        },
        index=high.index,
    )


def chaikin_money_flow(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = CMF_PERIOD
) -> pd.Series:
    hl = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl
    mfm = mfm.fillna(0.0)
    mfv = mfm * volume
    vol_sum = volume.rolling(period, min_periods=period).sum().replace(0, np.nan)
    return mfv.rolling(period, min_periods=period).sum() / vol_sum


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
    close, high, low, volume = g["close"], g["high"], g["low"], g["volume"]
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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Tính MA, CMF và Trendlines with Breaks theo từng mã."""
    if df.empty:
        return df
    ordered = df.sort_values(["ticker", "date"])
    chunks: list[pd.DataFrame] = []
    buf: list[pd.DataFrame] = []
    for i, (_, group) in enumerate(ordered.groupby("ticker", sort=False, observed=True), start=1):
        buf.append(_ta_one(group))
        if i % 200 == 0:
            chunks.append(pd.concat(buf, ignore_index=True))
            buf.clear()
            gc.collect()
    if buf:
        chunks.append(pd.concat(buf, ignore_index=True))
    out = pd.concat(chunks, ignore_index=True) if chunks else df
    del chunks, buf, ordered
    gc.collect()
    return out


def add_chart_ta(df: pd.DataFrame) -> pd.DataFrame:
    """Bổ sung EMA, RSI, MACD, Bollinger, ATR cho một mã trên biểu đồ."""
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
    if df.empty:
        return df
    if all(col in df.columns for col in REQUIRED_TA_COLUMNS):
        return df
    keep = [col for col in BASE_OHLC_COLUMNS if col in df.columns]
    return add_indicators(df[keep])


def _tl_ma_cmf(df: pd.DataFrame) -> pd.DataFrame:
    """
    BUY khi đủ 3 chân: break kháng cự (tl_upos=1) + Close > SMA50 + CMF > 0.
    SELL khi phá hỗ trợ + Close < SMA50 + CMF < 0.
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


def apply_strategy(df: pd.DataFrame, strategy: str = DEFAULT_STRATEGY) -> pd.DataFrame:
    if df.empty:
        return df
    data = df.sort_values(["ticker", "date"])
    if not all(col in data.columns for col in _TA_COLUMNS):
        keep = [col for col in _OHLC_COLUMNS if col in data.columns]
        data = add_indicators(data[keep])
    return _tl_ma_cmf(data)


def _rank_today(df: pd.DataFrame) -> pd.DataFrame:
    """Ưu tiên: break đúng hôm nay → CMF mạnh → volume/MA20 → % giá cùng chiều."""
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


def generate_signals(
    df: pd.DataFrame,
    strategy: str = DEFAULT_STRATEGY,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
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
    latest["ma_mark"] = [f"{'✓' if ok else '✗'} {txt}" for ok, txt in zip(latest["ma_ok"], latest["ma_text"])]
    latest["cmf_mark"] = [f"{'✓' if ok else '✗'} {txt}" for ok, txt in zip(latest["cmf_ok"], latest["cmf_text"])]
    latest = _rank_today(latest)
    columns = [
        "ticker", "exchange", "close", "signal", "can_buy", "buy_label", "rank",
        "priority_score", "priority_why", "indicator", "signal_date", "strategy",
        "sma_50", "cmf_20", "tl_upper", "tl_lower", "tl_upos", "tl_dnos",
        "tl_break_up", "tl_break_dn", "pct_change", "volume", "volume_ma_20",
        "break_ok", "ma_ok", "cmf_ok", "break_mark", "ma_mark", "cmf_mark",
        "ma_text", "cmf_text",
    ]
    return latest[[c for c in columns if c in latest.columns]].reset_index(drop=True)


# =============================================================================
# Pipeline CafeF — tải, giải nén, làm sạch, lưu
# =============================================================================

ADJUSTED_UPTO_RE = re.compile(
    rf"{re.escape(CAFEF_CDN_HOST)}/data/ami_data/(\d{{8}})/{re.escape(ADJUSTED_UPTO_NAME)}(\d{{8}})\.zip",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CafeFDataset:
    url: str
    folder_date: str
    file_date: str
    dataset_date: str

    @property
    def zip_name(self) -> str:
        return f"{ADJUSTED_UPTO_NAME}{self.file_date}.zip"

    def zip_path(self) -> Path:
        return RAW_DIR / self.zip_name


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Referer": CAFEF_PAGE_URL})
    return session


def _to_iso(yyyymmdd: str) -> str:
    return datetime.strptime(yyyymmdd, "%Y%m%d").strftime("%Y-%m-%d")


def parse_adjusted_upto_links(html: str) -> list[CafeFDataset]:
    found: dict[str, CafeFDataset] = {}
    for folder_date, file_date in ADJUSTED_UPTO_RE.findall(html):
        url = f"{CAFEF_CDN_HOST}/data/ami_data/{folder_date}/{ADJUSTED_UPTO_NAME}{file_date}.zip"
        found[folder_date] = CafeFDataset(
            url=url, folder_date=folder_date, file_date=file_date, dataset_date=_to_iso(folder_date)
        )
    return [found[key] for key in sorted(found, reverse=True)]


def _candidate_dates(days: int = 14) -> list[datetime]:
    today = datetime.now()
    return [today - timedelta(days=i) for i in range(days)]


def _build_dataset(day: datetime) -> CafeFDataset:
    folder_date = day.strftime("%Y%m%d")
    file_date = day.strftime("%d%m%Y")
    url = f"{CAFEF_CDN_HOST}/data/ami_data/{folder_date}/{ADJUSTED_UPTO_NAME}{file_date}.zip"
    return CafeFDataset(url=url, folder_date=folder_date, file_date=file_date, dataset_date=day.strftime("%Y-%m-%d"))


def discover_latest_adjusted_upto(timeout: int = 30) -> CafeFDataset:
    with _session() as session:
        response = session.get(CAFEF_PAGE_URL, timeout=timeout)
        response.raise_for_status()
        datasets = parse_adjusted_upto_links(response.text)
        if datasets:
            return datasets[0]
        for day in _candidate_dates():
            dataset = _build_dataset(day)
            head = session.head(dataset.url, timeout=timeout, allow_redirects=True)
            if head.status_code == 200:
                return dataset
    raise RuntimeError(
        "Không tìm thấy dữ liệu CafeF Đã điều chỉnh / Upto 3 sàn. "
        "Hãy kiểm tra kết nối hoặc trang https://cafef.vn/du-lieu/du-lieu-download.chn"
    )


def download_dataset(dataset: CafeFDataset, progress_cb=None, timeout: int = 180) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = dataset.zip_path()
    if dest.exists() and dest.stat().st_size > 0:
        if progress_cb:
            progress_cb(1.0)
        return dest
    tmp = dest.with_suffix(".zip.part")
    with _session() as session:
        with session.get(dataset.url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=256 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    done += len(chunk)
                    if progress_cb and total:
                        progress_cb(min(done / total, 1.0))
    tmp.replace(dest)
    if progress_cb:
        progress_cb(1.0)
    return dest


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
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, axis=0, ignore_index=True)
    df["ticker"] = df["ticker"].astype("string").str.strip().str.upper()
    return df


def standardize_datetime(df: pd.DataFrame) -> pd.DataFrame:
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
    """Lọc theo biên độ sàn–trần: HOSE ±7%, HNX ±10%, UPCOM ±15%."""
    out = df.copy()
    out["limit_pct"] = out["exchange"].map(PRICE_LIMITS)
    out = out.loc[~out["limit_pct"].isna()]
    out = out.sort_values(["ticker", "date"], kind="mergesort")
    out["prev_close"] = out.groupby("ticker", sort=False)["close"].shift(1)
    out["ceiling"] = out["prev_close"] * (1.0 + out["limit_pct"])
    out["floor"] = out["prev_close"] * (1.0 - out["limit_pct"])
    first_day = out["prev_close"].isna()
    lo = out["floor"] * (1.0 - LIMIT_TICK_TOLERANCE)
    hi = out["ceiling"] * (1.0 + LIMIT_TICK_TOLERANCE)
    within = (
        out["low"].ge(lo) & out["high"].le(hi)
        & out["open"].ge(lo) & out["open"].le(hi)
        & out["close"].ge(lo) & out["close"].le(hi)
    )
    keep = first_day | within
    dropped = int((~keep).sum())
    kept = out.loc[keep].copy()
    by_exchange = out.loc[~keep].groupby("exchange").size().astype(int).to_dict() if dropped else {}
    return kept, dropped, by_exchange


def drop_duplicate_ticker_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    out = df.sort_values(["ticker", "date"], kind="mergesort").drop_duplicates(subset=["ticker", "date"], keep="last")
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
        df["open"].gt(0) & df["high"].gt(0) & df["low"].gt(0) & df["close"].gt(0)
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
    stats["exchanges"] = cleaned.groupby("exchange")["ticker"].nunique().astype(int).to_dict()
    if cleaned.empty:
        raise RuntimeError("Dữ liệu rỗng sau khi làm sạch.")
    return cleaned, stats


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
        "page_url": CAFEF_PAGE_URL,
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
    META_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
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
            items.append({
                "name": path.name, "path": str(path), "exchange": infer_exchange(path.name),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2), "kind": "csv",
            })
    if RAW_DIR.exists():
        for path in sorted(RAW_DIR.glob("*.zip")):
            items.append({
                "name": path.name, "path": str(path), "exchange": "3 sàn",
                "size_mb": round(path.stat().st_size / (1024 * 1024), 2), "kind": "zip",
            })
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


def preview_clean(
    prices: pd.DataFrame, nrows: int = 200, ticker: str | None = None, exchange: str | None = None
) -> pd.DataFrame:
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
        remaining = explicit_remaining if explicit_remaining else max(remaining - dropped, 0)
        rows.append({"Bước": label, "Số dòng loại": dropped, "Số dòng còn lại": remaining})
    return pd.DataFrame(rows)


def run_pipeline(force: bool = False, progress: ProgressCb | None = None) -> dict:
    def emit(message: str, ratio: float | None = None) -> None:
        if progress:
            progress(message, ratio)

    emit("Đang tìm dữ liệu CafeF mới nhất...", 0.02)
    dataset = discover_latest_adjusted_upto()
    emit(f"Tìm thấy {dataset.dataset_date}: Đã điều chỉnh / Upto 3 sàn", 0.08)

    meta = load_meta()
    already_current = (
        not force and meta
        and meta.get("dataset_date") == dataset.dataset_date
        and meta.get("preprocess_version") == PREPROCESS_VERSION
        and PRICES_PATH.exists()
    )
    if already_current:
        prices = load_prices()
        if "cmf_20" not in prices.columns or "tl_upper" not in prices.columns:
            emit("Đang tính lại Trendline + MA + CMF...", 0.72)
            base_cols = [c for c in prices.columns if c in set(OUTPUT_COLUMNS)]
            priced = add_indicators(prices[base_cols])
            save_prices(priced)
            signals = generate_signals(priced, DEFAULT_STRATEGY)
            save_signals(signals)
            extra = {
                "date_min": priced["date"].min().strftime("%Y-%m-%d"),
                "date_max": priced["date"].max().strftime("%Y-%m-%d"),
                "row_count": int(len(priced)),
                "ticker_count": int(priced["ticker"].nunique()),
                "signal_counts": signals["signal"].value_counts().to_dict(),
                "preprocess_version": PREPROCESS_VERSION,
                "strategy": DEFAULT_STRATEGY,
                "price_limits": {"HOSE": "±7%", "HNX": "±10%", "UPCOM": "±15%"},
            }
            save_meta(dataset, meta.get("stats") or {}, extra)
            emit("Đã cập nhật bộ lọc Trendline + MA + CMF.", 1.0)
            return {"status": "updated", "dataset": dataset, "meta": load_meta()}
        emit("Dataset hiện tại đã là bản mới nhất, bỏ qua tải lại.", 1.0)
        return {"status": "up_to_date", "dataset": dataset, "meta": meta}

    emit("Đang tải file Upto 3 sàn đã điều chỉnh...", 0.10)

    def download_progress(ratio: float) -> None:
        emit(f"Đang tải dữ liệu... {ratio:.0%}", 0.10 + 0.35 * ratio)

    zip_path = download_dataset(dataset, progress_cb=download_progress)
    emit("Đang giải nén dữ liệu 3 sàn...", 0.48)
    extract_dir = extract_zip(zip_path, dataset)
    emit("Đang làm sạch và chuẩn hóa dữ liệu...", 0.58)
    prices, stats = load_and_clean(extract_dir)

    years = history_years()
    if years:
        cutoff = prices["date"].max() - pd.DateOffset(years=years)
        before = len(prices)
        prices = prices.loc[prices["date"] >= cutoff].copy()
        stats["history_years"] = years
        stats["rows_before_history_trim"] = int(before)
        emit(f"Cloud: giữ {years} năm gần nhất ({len(prices):,} dòng)...", 0.66)

    if is_streamlit_cloud():
        num_cols = [
            c for c in ("open", "high", "low", "close", "volume", "prev_close", "floor", "ceiling", "limit_pct")
            if c in prices.columns
        ]
        prices[num_cols] = prices[num_cols].apply(pd.to_numeric, errors="coerce").astype("float32")
        shutil.rmtree(extract_dir, ignore_errors=True)
        gc.collect()

    emit("Đang tính chỉ báo kỹ thuật...", 0.72)
    priced = add_indicators(prices)
    save_prices(priced)
    emit("Đang sinh tín hiệu MUA / BÁN / GIỮ...", 0.90)
    signals = generate_signals(priced, DEFAULT_STRATEGY)
    save_signals(signals)
    extra = {
        "date_min": priced["date"].min().strftime("%Y-%m-%d"),
        "date_max": priced["date"].max().strftime("%Y-%m-%d"),
        "row_count": int(len(priced)),
        "ticker_count": int(priced["ticker"].nunique()),
        "signal_counts": signals["signal"].value_counts().to_dict(),
        "preprocess_version": PREPROCESS_VERSION,
        "strategy": DEFAULT_STRATEGY,
        "price_limits": {"HOSE": "±7%", "HNX": "±10%", "UPCOM": "±15%"},
        "history_years": years,
        "runtime": "streamlit-cloud" if is_streamlit_cloud() else "local",
    }
    meta_path = save_meta(dataset, stats, extra)
    emit("Cập nhật hoàn tất.", 1.0)
    return {
        "status": "updated",
        "dataset": dataset,
        "meta": load_meta(),
        "meta_path": str(meta_path),
        "rows": int(len(priced)),
        "tickers": int(priced["ticker"].nunique()),
    }


# =============================================================================
# Giao diện — theme + HTML
# =============================================================================

CSS = """
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap");

html, body, [class*="css"], .stApp, .stMarkdown, .stText, .stSelectbox, button {
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif !important;
}
.stApp { background: #080c16; color: #e6edf7; }
.block-container { padding-top: 1.15rem; padding-bottom: 2.4rem; max-width: 1480px; }

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0c1322 0%, #080c16 100%);
  border-right: 1px solid #1c2740;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

h1, h2, h3 { letter-spacing: -0.03em; font-weight: 700 !important; }
hr { border-color: #1c2740 !important; }

div[data-testid="stMetric"] {
  background: linear-gradient(180deg, #121a2c 0%, #0e1524 100%);
  border: 1px solid #22304a; border-radius: 16px;
  padding: 14px 16px 12px 16px; box-shadow: 0 8px 24px rgba(0,0,0,.18);
}
div[data-testid="stMetric"] label { color: #8b9bb4 !important; font-size: 0.78rem !important; letter-spacing: .04em; text-transform: uppercase; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-family: "IBM Plex Mono", monospace; font-weight: 600; }

.stTabs [data-baseweb="tab-list"] {
  gap: 8px; background: #0e1524; border: 1px solid #1c2740; border-radius: 14px; padding: 6px;
}
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 10px; color: #8b9bb4; font-weight: 600; padding: 10px 16px; }
.stTabs [aria-selected="true"] { background: #1a2744 !important; color: #e6edf7 !important; box-shadow: inset 0 0 0 1px #2d4166; }
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

.stButton > button { border-radius: 12px; font-weight: 650; border: 1px solid #2d4166; }
.stButton > button[kind="primary"] { background: linear-gradient(180deg, #3b82f6, #2563eb); border: 0; }

[data-testid="stDataFrame"] { border: 1px solid #1c2740; border-radius: 16px; overflow: hidden; background: #0e1524; }
[data-testid="stExpander"] { background: #0e1524; border: 1px solid #1c2740; border-radius: 14px; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

.hero { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; margin: 0 0 18px 0; padding: 4px 2px 16px 2px; border-bottom: 1px solid #1c2740; }
.hero h1 { margin: 0; font-size: 1.85rem; color: #f4f7fb; }
.hero p { margin: 6px 0 0 0; color: #8b9bb4; font-size: 0.95rem; }
.badge { display: inline-flex; align-items: center; gap: 8px; background: #102033; color: #93c5fd; border: 1px solid #1e3a5f; border-radius: 999px; padding: 6px 12px; font-size: 0.8rem; font-weight: 600; white-space: nowrap; }
.badge.ok { background: #0c241c; color: #6ee7b7; border-color: #14532d; }

.brand { display: flex; gap: 12px; align-items: center; margin-bottom: 18px; }
.brand-mark { width: 42px; height: 42px; border-radius: 12px; background: linear-gradient(145deg, #3b82f6, #1d4ed8); color: white; display: flex; align-items: center; justify-content: center; font-weight: 800; letter-spacing: .04em; }
.brand-name { font-weight: 700; font-size: 1.05rem; color: #f4f7fb; }
.brand-sub { color: #8b9bb4; font-size: 0.78rem; margin-top: 2px; }

.kpi-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 4px 0 16px 0; }
.kpi { background: linear-gradient(180deg, #121a2c 0%, #0e1524 100%); border: 1px solid #22304a; border-radius: 16px; padding: 16px 18px; }
.kpi .k { color: #8b9bb4; font-size: 0.75rem; letter-spacing: .08em; text-transform: uppercase; font-weight: 600; }
.kpi .v { font-family: "IBM Plex Mono", monospace; font-size: 1.7rem; font-weight: 600; margin-top: 6px; }
.kpi.buy { box-shadow: inset 3px 0 0 #10b981; }
.kpi.buy .v { color: #34d399; }
.kpi.sell { box-shadow: inset 3px 0 0 #f43f5e; }
.kpi.sell .v { color: #fb7185; }
.kpi.neutral .v { color: #93c5fd; }

.signal { display: flex; align-items: center; justify-content: space-between; gap: 16px; border-radius: 18px; padding: 16px 20px; margin: 4px 0 14px 0; border: 1px solid; }
.signal .sym { font-size: 1.45rem; font-weight: 800; letter-spacing: .04em; }
.signal .lbl { font-size: 0.82rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 700; opacity: .9; }
.signal.buy { background: linear-gradient(90deg, #052e1c, #0e1524); border-color: #14532d; color: #6ee7b7; }
.signal.sell { background: linear-gradient(90deg, #3f0d18, #0e1524); border-color: #7f1d1d; color: #fda4af; }
.signal.hold { background: linear-gradient(90deg, #172554, #0e1524); border-color: #1e3a5f; color: #93c5fd; }

.factors { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 12px; }
.factor { border-radius: 14px; padding: 12px 14px; border: 1px solid; min-height: 86px; }
.factor .name { font-weight: 700; font-size: 0.92rem; }
.factor .detail { margin-top: 6px; font-family: "IBM Plex Mono", monospace; font-size: 0.88rem; }
.factor .rule { margin-top: 4px; font-size: 0.75rem; opacity: .75; }
.factor.ok { background: rgba(16,185,129,.08); border-color: #14532d; color: #86efac; }
.factor.bad { background: rgba(244,63,94,.08); border-color: #7f1d1d; color: #fda4af; }

.rank-strip { display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 10px 0; }
.rank-chip { display: flex; align-items: center; gap: 8px; background: #121a2c; border: 1px solid #22304a; border-radius: 999px; padding: 5px 10px 5px 6px; font-size: 0.82rem; }
.rank-chip .n { width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-family: "IBM Plex Mono", monospace; font-size: 0.72rem; font-weight: 700; }
.rank-chip.buy .n { background: #065f46; color: #a7f3d0; }
.rank-chip.sell .n { background: #9f1239; color: #fecdd3; }
.rank-chip .t { font-weight: 700; color: #e6edf7; }
.rank-chip .p { font-family: "IBM Plex Mono", monospace; color: #8b9bb4; }

.stPlotlyChart, div[data-testid="stPlotlyChart"] {
  background: #ffffff !important; border: 1px solid #d0d5dd; border-radius: 8px; overflow: hidden;
}
"""


def inject_theme() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def brand_html() -> str:
    return """
    <div class="brand">
      <div class="brand-mark">VN</div>
      <div>
        <div class="brand-name">Stock Analytics</div>
        <div class="brand-sub">Sàng lọc tự động · 3 sàn</div>
      </div>
    </div>
    """


def hero_html(title: str, subtitle: str, badge: str = "", ok: bool = True) -> str:
    badge_html = f'<span class="badge {"ok" if ok else ""}">{badge}</span>' if badge else ""
    return f"""
    <div class="hero">
      <div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      {badge_html}
    </div>
    """


def kpi_row_html(buy_n: int, sell_n: int, session: str) -> str:
    return f"""
    <div class="kpi-grid">
      <div class="kpi buy"><div class="k">Nên mua hôm nay</div><div class="v">{buy_n}</div></div>
      <div class="kpi sell"><div class="k">Nên bán hôm nay</div><div class="v">{sell_n}</div></div>
      <div class="kpi neutral"><div class="k">Phiên giao dịch</div><div class="v">{session}</div></div>
    </div>
    """


def signal_banner_html(ticker: str, action: str | None, can_buy: bool) -> str:
    if action == "BÁN":
        klass, label = "sell", "Nên bán hôm nay"
    elif action == "MUA" or can_buy:
        klass, label = "buy", "Nên mua hôm nay"
    else:
        klass, label = "hold", "Chưa đủ tín hiệu"
    return f"""
    <div class="signal {klass}">
      <div class="sym">{ticker}</div>
      <div class="lbl">{label}</div>
    </div>
    """


def factor_row_html(break_ok: bool, ma_ok: bool, cmf_ok: bool, ma_text: str, cmf_text: str) -> str:
    def one(ok: bool, name: str, detail: str, rule: str) -> str:
        cls = "ok" if ok else "bad"
        flag = "Đạt" if ok else "Chưa"
        return (
            f'<div class="factor {cls}"><div class="name">{flag} · {name}</div>'
            f'<div class="detail">{detail}</div><div class="rule">{rule}</div></div>'
        )

    return (
        '<div class="factors">'
        + one(break_ok, "Trendline break", "Phá kháng cự LuxAlgo" if break_ok else "Chưa break upper", "upos = 1")
        + one(ma_ok, "Xu hướng MA50", ma_text or "—", "Close > SMA50")
        + one(cmf_ok, "Dòng tiền CMF", cmf_text or "—", "CMF(20) > 0")
        + "</div>"
    )


def rank_strip_html(df: pd.DataFrame, side: str, n: int = 5) -> str:
    if df.empty:
        return ""
    ranked = df.sort_values("rank", kind="mergesort").head(n)
    chips = []
    cls = "buy" if side == "MUA" else "sell"
    for row in ranked.itertuples(index=False):
        chg = getattr(row, "pct_change", None)
        chg_s = f"{chg:+.1f}%" if pd.notna(chg) else ""
        chips.append(
            f'<div class="rank-chip {cls}"><span class="n">{int(getattr(row, "rank"))}</span>'
            f'<span class="t">{row.ticker}</span><span class="p">{chg_s}</span></div>'
        )
    return '<div class="rank-strip">' + "".join(chips) + "</div>"


# =============================================================================
# Biểu đồ nến (TradingView-style)
# =============================================================================

BG, GRID, AXIS, TEXT, MUTED = "#ffffff", "#e0e3eb", "#787b86", "#131722", "#6a6d78"
UP, DOWN, CROSS = "#089981", "#f23645", "#9598a1"
SMA_COLOR, SMA20_COLOR, SUPPORT, RESIST = "#2962ff", "#ff9800", "#26a69a", "#ef5350"

INDICATOR_KEYS = {
    "tl": "Trendline", "sma50": "SMA 50", "sma20": "SMA 20", "ema20": "EMA 20",
    "bb": "Bollinger", "bs": "Tín hiệu B/S", "vol": "Khối lượng",
    "cmf": "CMF 20", "rsi": "RSI 14", "macd": "MACD", "atr": "ATR 14",
}
DEFAULT_INDICATORS = ["tl", "sma50", "bs", "vol", "cmf"]
PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": True, "displayModeBar": False}


def _gap_resets(series: pd.Series) -> pd.Series:
    y = pd.to_numeric(series, errors="coerce")
    jump = y.diff().abs()
    med_y = y.abs().median()
    med_j = jump.median()
    med_y = float(med_y) if pd.notna(med_y) else 0.0
    med_j = float(med_j) if pd.notna(med_j) else 0.0
    cutoff = max(med_y * 0.02, med_j * 6, 1e-6)
    out = y.copy()
    out[jump > cutoff] = np.nan
    return out


def _tv_axes(fig: go.Figure, row: int, *, show_x: bool, x_range: list) -> None:
    fig.update_xaxes(
        showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, showline=False,
        color=AXIS, tickfont=dict(size=11, color=AXIS, family="Trebuchet MS"),
        showticklabels=show_x, rangeslider_visible=False, showspikes=True,
        spikemode="across", spikesnap="cursor", spikecolor=CROSS, spikethickness=1,
        spikedash="solid", type="date", range=x_range, tickformat="%d/%m", nticks=8,
        row=row, col=1,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, showline=False,
        side="right", color=AXIS, tickfont=dict(size=11, color=AXIS, family="Trebuchet MS"),
        showspikes=True, spikemode="across", spikecolor=CROSS, spikethickness=1,
        spikedash="solid", row=row, col=1,
    )


def normalize_overlays(overlays: list[str] | None) -> set[str]:
    if overlays is None:
        return set(DEFAULT_INDICATORS)
    alias = {
        "SMA50": "sma50", "SMA20": "sma20", "Trendline": "tl", "CMF": "cmf",
        "Khối lượng": "vol", "Tín hiệu B/S": "bs",
        **{label: key for key, label in INDICATOR_KEYS.items()},
    }
    return {alias.get(item, item) for item in overlays}


def _last_tag(fig: go.Figure, x, y: float, text: str, color: str, row: int) -> None:
    if y is None or not np.isfinite(y):
        return
    fig.add_annotation(
        x=x, y=y, text=f" {text} ", showarrow=False, xanchor="left", yanchor="middle",
        xshift=6, bgcolor=color, font=dict(color="#ffffff", size=10, family="Trebuchet MS"),
        borderpad=2, row=row, col=1,
    )


def _soft_line(
    fig: go.Figure, x, y, *, name: str, color: str, width: float = 1.8,
    dash: str | None = None, row: int = 1, hover: str | None = None,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines", name=name,
            line=dict(color=color, width=width + 4, dash=dash),
            opacity=0.14, hoverinfo="skip", showlegend=False, connectgaps=False,
        ),
        row=row, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="lines", name=name,
            line=dict(color=color, width=width, dash=dash, shape="linear"),
            hovertemplate=hover or f"{name} %{{y:.2f}}<extra></extra>",
            showlegend=False, connectgaps=False,
        ),
        row=row, col=1, secondary_y=False,
    )


def _fmt(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:,.1f}"
    return f"{v:.2f}"


def price_chart(df: pd.DataFrame, ticker: str, overlays: list[str] | None = None) -> go.Figure:
    on = normalize_overlays(overlays)
    tagged = apply_strategy(df) if "tl_upper" not in df.columns else df
    if "break_ok" not in tagged.columns:
        tagged = apply_strategy(df)
    tagged = tagged.copy()
    tagged["date"] = pd.to_datetime(tagged["date"], errors="coerce")
    tagged = tagged.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    if getattr(tagged["date"].dt, "tz", None) is not None:
        tagged["date"] = tagged["date"].dt.tz_convert(None)
    tagged = tagged.reset_index(drop=True)
    tagged = add_chart_ta(tagged)
    x = tagged["date"]
    x_range = [x.iloc[0] - pd.Timedelta(days=2), x.iloc[-1] + pd.Timedelta(days=4)]

    last = tagged.iloc[-1]
    prev = float(tagged.iloc[-2]["close"]) if len(tagged) > 1 else float(last["close"])
    chg = float(last["close"]) - prev
    chg_pct = (chg / prev * 100) if prev else 0.0
    last_color = UP if chg >= 0 else DOWN
    exchange = str(last["exchange"]) if "exchange" in tagged.columns and pd.notna(last.get("exchange")) else "HOSE"

    panes: list[str] = []
    if "cmf" in on and "cmf_20" in tagged.columns:
        panes.append("cmf")
    if "rsi" in on and "rsi_14" in tagged.columns:
        panes.append("rsi")
    if "macd" in on and "macd" in tagged.columns:
        panes.append("macd")
    if "atr" in on and "atr_14" in tagged.columns:
        panes.append("atr")
    n_rows = 1 + len(panes)
    extra_h = 0.16 * len(panes)
    price_h = max(0.42, 1.0 - extra_h)
    heights = [price_h] + ([extra_h / len(panes)] * len(panes) if panes else [])
    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=heights,
        specs=[[{"secondary_y": True}]] + [[{"secondary_y": False}]] * len(panes),
    )

    fig.add_trace(
        go.Candlestick(
            x=x, open=tagged["open"], high=tagged["high"], low=tagged["low"], close=tagged["close"],
            name="Nến",
            increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
            decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN),
            whiskerwidth=0.9, hoverinfo="skip", showlegend=False,
        ),
        row=1, col=1, secondary_y=False,
    )

    if "tl" in on and "tl_upper" in tagged.columns:
        upper = _gap_resets(tagged["tl_upper"])
        _soft_line(fig, x, upper, name="TL kháng cự", color=RESIST, width=1.7, dash="dot", hover="Kháng cự %{y:.2f}<extra></extra>")
        last_u = pd.to_numeric(upper, errors="coerce").dropna()
        if not last_u.empty:
            _last_tag(fig, x.iloc[last_u.index[-1]], float(last_u.iloc[-1]), f"R {_fmt(float(last_u.iloc[-1]))}", RESIST, 1)
    if "tl" in on and "tl_lower" in tagged.columns:
        lower = _gap_resets(tagged["tl_lower"])
        _soft_line(fig, x, lower, name="TL hỗ trợ", color=SUPPORT, width=1.7, dash="dot", hover="Hỗ trợ %{y:.2f}<extra></extra>")
        last_l = pd.to_numeric(lower, errors="coerce").dropna()
        if not last_l.empty:
            _last_tag(fig, x.iloc[last_l.index[-1]], float(last_l.iloc[-1]), f"S {_fmt(float(last_l.iloc[-1]))}", SUPPORT, 1)

    if "sma50" in on and "sma_50" in tagged.columns:
        sma50 = pd.to_numeric(tagged["sma_50"], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=x, y=sma50, mode="lines", name="SMA 50",
                line=dict(color=SMA_COLOR, width=2.1, shape="spline", smoothing=0.7),
                hovertemplate="SMA50 %{y:.2f}<extra></extra>", showlegend=False, connectgaps=True,
            ),
            row=1, col=1, secondary_y=False,
        )
        if pd.notna(sma50.iloc[-1]):
            _last_tag(fig, x.iloc[-1], float(sma50.iloc[-1]), f"MA50 {_fmt(float(sma50.iloc[-1]))}", SMA_COLOR, 1)
    if "sma20" in on and "sma_20" in tagged.columns:
        sma20 = pd.to_numeric(tagged["sma_20"], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=x, y=sma20, mode="lines", name="SMA 20",
                line=dict(color=SMA20_COLOR, width=1.7, shape="spline", smoothing=0.7),
                hovertemplate="SMA20 %{y:.2f}<extra></extra>", showlegend=False, connectgaps=True,
            ),
            row=1, col=1, secondary_y=False,
        )
        if pd.notna(sma20.iloc[-1]):
            _last_tag(fig, x.iloc[-1], float(sma20.iloc[-1]), f"MA20 {_fmt(float(sma20.iloc[-1]))}", SMA20_COLOR, 1)
    if "ema20" in on and "ema_20" in tagged.columns:
        ema20 = pd.to_numeric(tagged["ema_20"], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=x, y=ema20, mode="lines", name="EMA 20",
                line=dict(color="#7c3aed", width=1.7, shape="spline", smoothing=0.65),
                hovertemplate="EMA20 %{y:.2f}<extra></extra>", showlegend=False, connectgaps=True,
            ),
            row=1, col=1, secondary_y=False,
        )
    if "bb" in on and "bb_up" in tagged.columns:
        fig.add_trace(
            go.Scatter(x=x, y=tagged["bb_up"], mode="lines", name="BB Up", line=dict(color="#94a3b8", width=1, dash="dot"), showlegend=False, hovertemplate="BB Up %{y:.2f}<extra></extra>"),
            row=1, col=1, secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=x, y=tagged["bb_dn"], mode="lines", name="BB Dn", line=dict(color="#94a3b8", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(148,163,184,0.10)", showlegend=False, hovertemplate="BB Dn %{y:.2f}<extra></extra>"),
            row=1, col=1, secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=x, y=tagged["bb_mid"], mode="lines", name="BB Mid", line=dict(color="#64748b", width=1), showlegend=False, hovertemplate="BB Mid %{y:.2f}<extra></extra>"),
            row=1, col=1, secondary_y=False,
        )

    if "bs" in on and "tl_break_up" in tagged.columns:
        buys = tagged[tagged["tl_break_up"].astype(bool)]
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["date"], y=buys["low"] * 0.988, mode="markers+text",
                    text=["B"] * len(buys), textposition="middle center",
                    textfont=dict(color="#ffffff", size=8, family="Arial Black"),
                    marker=dict(symbol="square", size=15, color=UP, line=dict(width=1, color="#ffffff")),
                    name="B", hovertemplate="Mua · Break ↑<extra></extra>", showlegend=False,
                ),
                row=1, col=1, secondary_y=False,
            )
    if "bs" in on and "tl_break_dn" in tagged.columns:
        sells = tagged[tagged["tl_break_dn"].astype(bool)]
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["date"], y=sells["high"] * 1.012, mode="markers+text",
                    text=["S"] * len(sells), textposition="middle center",
                    textfont=dict(color="#ffffff", size=8, family="Arial Black"),
                    marker=dict(symbol="square", size=15, color=DOWN, line=dict(width=1, color="#ffffff")),
                    name="S", hovertemplate="Bán · Break ↓<extra></extra>", showlegend=False,
                ),
                row=1, col=1, secondary_y=False,
            )

    if "vol" in on:
        vol_colors = [UP if c >= o else DOWN for c, o in zip(tagged["close"], tagged["open"])]
        fig.add_trace(
            go.Bar(
                x=x, y=tagged["volume"], name="Vol",
                marker=dict(color=vol_colors, line=dict(width=0)), opacity=0.32,
                hovertemplate="Vol %{y:,.0f}<extra></extra>", showlegend=False,
            ),
            row=1, col=1, secondary_y=True,
        )
        vmax = float(pd.to_numeric(tagged["volume"], errors="coerce").max() or 1)
        fig.update_yaxes(range=[0, vmax * 3.8], showgrid=False, showticklabels=False, showspikes=False, title_text="", row=1, col=1, secondary_y=True)

    for i, pane in enumerate(panes, start=2):
        if pane == "cmf":
            cmf = pd.to_numeric(tagged["cmf_20"], errors="coerce")
            fig.add_trace(
                go.Scatter(
                    x=x, y=cmf, name="CMF", mode="lines",
                    line=dict(color=UP, width=2, shape="spline", smoothing=0.55),
                    fill="tozeroy", fillcolor="rgba(8,153,129,0.10)",
                    hovertemplate="CMF %{y:.3f}<extra></extra>", showlegend=False, connectgaps=True,
                ),
                row=i, col=1,
            )
            fig.add_hline(y=0, line_color="#c7cbd4", line_width=1, row=i, col=1)
            if pd.notna(cmf.iloc[-1]):
                _last_tag(fig, x.iloc[-1], float(cmf.iloc[-1]), f"CMF {float(cmf.iloc[-1]):.2f}", UP if cmf.iloc[-1] >= 0 else DOWN, i)
        elif pane == "rsi":
            rsi = pd.to_numeric(tagged["rsi_14"], errors="coerce")
            fig.add_trace(
                go.Scatter(x=x, y=rsi, name="RSI", mode="lines", line=dict(color="#f59e0b", width=1.7), hovertemplate="RSI %{y:.1f}<extra></extra>", showlegend=False),
                row=i, col=1,
            )
            fig.add_hline(y=70, line_dash="dot", line_color=DOWN, line_width=1, row=i, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color=UP, line_width=1, row=i, col=1)
            fig.update_yaxes(range=[0, 100], row=i, col=1)
            if pd.notna(rsi.iloc[-1]):
                _last_tag(fig, x.iloc[-1], float(rsi.iloc[-1]), f"RSI {float(rsi.iloc[-1]):.1f}", "#f59e0b", i)
        elif pane == "macd":
            hist = tagged["macd_hist"]
            colors = [UP if v >= 0 else DOWN for v in hist.fillna(0)]
            fig.add_trace(go.Bar(x=x, y=hist, name="MACD hist", marker=dict(color=colors, line=dict(width=0)), opacity=0.55, showlegend=False, hovertemplate="Hist %{y:.3f}<extra></extra>"), row=i, col=1)
            fig.add_trace(go.Scatter(x=x, y=tagged["macd"], name="MACD", mode="lines", line=dict(color="#2563eb", width=1.5), showlegend=False, hovertemplate="MACD %{y:.3f}<extra></extra>"), row=i, col=1)
            fig.add_trace(go.Scatter(x=x, y=tagged["macd_signal"], name="Signal", mode="lines", line=dict(color="#f97316", width=1.3), showlegend=False, hovertemplate="Signal %{y:.3f}<extra></extra>"), row=i, col=1)
        elif pane == "atr":
            atr = pd.to_numeric(tagged["atr_14"], errors="coerce")
            fig.add_trace(
                go.Scatter(x=x, y=atr, name="ATR", mode="lines", line=dict(color="#0ea5e9", width=1.6), fill="tozeroy", fillcolor="rgba(14,165,233,0.08)", showlegend=False, hovertemplate="ATR %{y:.2f}<extra></extra>"),
                row=i, col=1,
            )
            if pd.notna(atr.iloc[-1]):
                _last_tag(fig, x.iloc[-1], float(atr.iloc[-1]), f"ATR {_fmt(float(atr.iloc[-1]))}", "#0ea5e9", i)

    last_close = float(last["close"])
    fig.add_hline(y=last_close, line_dash="dot", line_color=last_color, line_width=1, row=1, col=1)
    fig.add_annotation(
        x=x.iloc[-1], y=last_close, text=f" {_fmt(last_close)} ", showarrow=False,
        xanchor="left", yanchor="middle", xshift=6, bgcolor=last_color,
        font=dict(color="#ffffff", size=11, family="Trebuchet MS"), borderpad=3, row=1, col=1,
    )

    chg_txt = f"{chg:+.2f} ({chg_pct:+.2f}%)"
    header = (
        f"<b style='font-size:14px;color:{TEXT}'>{ticker}</b>"
        f"<span style='color:{MUTED}'>  ·  1D  ·  {exchange}</span>&nbsp;&nbsp;"
        f"<span style='color:{MUTED}'>O</span> {_fmt(float(last['open']))}  "
        f"<span style='color:{MUTED}'>H</span> {_fmt(float(last['high']))}  "
        f"<span style='color:{MUTED}'>L</span> {_fmt(float(last['low']))}  "
        f"<span style='color:{MUTED}'>C</span> "
        f"<span style='color:{last_color}'>{_fmt(float(last['close']))}  {chg_txt}</span>"
        f"<br><span style='color:{MUTED};font-size:11px'>"
        + (
            f"LuxAlgo — Trendlines with Breaks&nbsp;&nbsp;{TL_LENGTH}&nbsp;&nbsp;{TL_SLOPE_K:g}&nbsp;&nbsp;Atr"
            if "tl" in on
            else " · ".join(INDICATOR_KEYS[k] for k in DEFAULT_INDICATORS if k in on)
        )
        + "</span>"
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0, y=1.0, xanchor="left", yanchor="top",
        text=header, showarrow=False, align="left", bgcolor="rgba(255,255,255,0.88)",
        borderpad=6, font=dict(family="Trebuchet MS", size=12, color=TEXT),
    )
    labels = {"cmf": f"CMF {CMF_PERIOD}", "rsi": "RSI 14", "macd": "MACD", "atr": "ATR 14"}
    for i, pane in enumerate(panes, start=2):
        fig.update_yaxes(title_text=labels[pane], title_font=dict(size=11, color=MUTED), row=i, col=1)

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Trebuchet MS, Segoe UI, sans-serif", color=TEXT, size=12),
        margin=dict(l=6, r=68, t=10, b=10), showlegend=False, hovermode="x",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e0e3eb", font=dict(family="Trebuchet MS", size=12, color=TEXT)),
        dragmode="pan", bargap=0.22, xaxis_rangeslider_visible=False, height=620 + 120 * len(panes),
    )
    _tv_axes(fig, 1, show_x=not panes, x_range=x_range)
    for i in range(2, n_rows + 1):
        _tv_axes(fig, i, show_x=(i == n_rows), x_range=x_range)
        fig.update_yaxes(tickformat=".2f", row=i, col=1)
    fig.update_yaxes(tickformat=".2f", separatethousands=True, row=1, col=1, secondary_y=False)
    return fig


# =============================================================================
# Tab dữ liệu + dashboard Streamlit
# =============================================================================

CLEANING_CODE = '''# 1) Chuẩn hóa biến thời gian về YYYY-MM-DD
digits = df["date_code"].astype("string").str.replace(r"[^0-9]", "", regex=True)
df["date"] = pd.to_datetime(digits, format="%Y%m%d", errors="coerce").dt.normalize()

# 2) Lọc dữ liệu thiếu
missing = (
    df["ticker"].isna() | (df["ticker"].str.len() == 0)
    | df["date"].isna()
    | df[["open", "high", "low", "close"]].isna().any(axis=1)
)
df = df.loc[~missing]

# 3) Ghép 3 bảng HOSE / HNX / UPCOM theo dòng, dựa trên Ticker
hose, hnx, upcom = ...  # 3 CSV CafeF
df = pd.concat([hose, hnx, upcom], axis=0, ignore_index=True)
df = df.sort_values(["ticker", "date"])

# 4) Sàn / trần theo biến động từng sàn
PRICE_LIMITS = {"HOSE": 0.07, "HNX": 0.10, "UPCOM": 0.15}
df["limit_pct"] = df["exchange"].map(PRICE_LIMITS)
df["prev_close"] = df.groupby("ticker")["close"].shift(1)
df["ceiling"] = df["prev_close"] * (1 + df["limit_pct"])
df["floor"] = df["prev_close"] * (1 - df["limit_pct"])
first_day = df["prev_close"].isna()
within = (
    df["low"].between(df["floor"], df["ceiling"])
    & df["high"].between(df["floor"], df["ceiling"])
    & df["open"].between(df["floor"], df["ceiling"])
    & df["close"].between(df["floor"], df["ceiling"])
)
df = df.loc[first_day | within]

# 5) Xóa trùng: cùng một mã nếu trùng thời gian thì xóa
df = (
    df.sort_values(["ticker", "date"])
      .drop_duplicates(subset=["ticker", "date"], keep="last")
)
'''


@st.cache_data(show_spinner=False)
def cached_raw_preview(file_path: str, nrows: int, ticker: str) -> pd.DataFrame:
    return preview_raw_csv(Path(file_path), nrows=nrows, ticker=ticker or None)


def data_quality_tab(prices: pd.DataFrame, meta: dict) -> None:
    st.markdown("##### Dữ liệu gốc và dữ liệu đã làm sạch")
    st.caption("CafeF đã điều chỉnh · Upto 3 sàn. Trái: CSV giải nén. Phải: dataset sau chuẩn hóa.")

    stats = meta.get("stats") or {}
    dropped = (
        int(stats.get("dropped_missing") or 0)
        + int(stats.get("dropped_invalid_numeric") or 0)
        + int(stats.get("dropped_invalid_ohlc") or 0)
        + int(stats.get("dropped_invalid_ticker") or 0)
        + int(stats.get("dropped_duplicates") or 0)
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dòng tải về", f"{int(stats.get('raw_rows') or 0):,}")
    m2.metric("Dòng sau làm sạch", f"{int(stats.get('clean_rows') or 0):,}")
    m3.metric("Đã loại", f"{dropped:,}")
    m4.metric("Số mã", f"{int(stats.get('tickers') or 0):,}")

    files = [item for item in list_download_files(meta) if item["kind"] == "csv"]
    extract_dir = extract_dir_from_meta(meta)
    if not files or extract_dir is None:
        st.warning("Chưa thấy file CSV đã giải nén. Bấm **Cập nhật dữ liệu mới nhất** để tải lại.")
        files = []

    controls = st.columns([1.4, 1.2, 1.2, 1.2])
    with controls[0]:
        file_labels = [f"{item['exchange']} — {item['name']} ({item['size_mb']} MB)" for item in files]
        selected = st.selectbox("File tải về", file_labels) if file_labels else None
    with controls[1]:
        exchange = st.selectbox(
            "Sàn (bảng sạch)",
            ["Tất cả"] + sorted(prices["exchange"].dropna().unique().tolist()),
            key="clean_exchange",
        )
    with controls[2]:
        ticker = st.text_input("Lọc mã (tùy chọn)", placeholder="VNM").strip().upper()
    with controls[3]:
        nrows = st.slider("Số dòng xem", min_value=50, max_value=1000, value=200, step=50)

    selected_file = files[file_labels.index(selected)] if files and selected else None

    raw_col, clean_col = st.columns(2)
    with raw_col:
        st.markdown("##### Dữ liệu tải về")
        if selected_file:
            st.caption(
                f"{selected_file['name']} · encoding UTF-8 BOM · "
                "cột gốc `<Ticker>, <DTYYYYMMDD>, <Open>, <High>, <Low>, <Close>, <Volume>`"
            )
            with st.spinner("Đang đọc CSV gốc..."):
                raw_df = cached_raw_preview(selected_file["path"], nrows, ticker)
            if raw_df.empty:
                st.info("Không có dòng nào khớp bộ lọc trong file gốc.")
            else:
                st.dataframe(raw_df, width="stretch", hide_index=True, height=420)
                st.download_button(
                    "Tải mẫu CSV gốc",
                    raw_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"raw_{selected_file['exchange']}.csv",
                    mime="text/csv",
                    key="dl_raw",
                )
        else:
            st.info("Không có file CSV để hiển thị.")

    with clean_col:
        st.markdown("##### Dữ liệu đã làm sạch")
        st.caption("ticker, date (YYYY-MM-DD), OHLCV, exchange, prev_close, sàn (floor), trần (ceiling), limit_pct")
        clean_df = preview_clean(prices, nrows=nrows, ticker=ticker, exchange=exchange)
        if clean_df.empty:
            st.info("Không có dòng nào khớp bộ lọc trong dữ liệu đã làm sạch.")
        else:
            show = clean_df.copy()
            show["date"] = pd.to_datetime(show["date"]).dt.strftime("%Y-%m-%d")
            st.dataframe(show, width="stretch", hide_index=True, height=420)
            st.download_button(
                "Tải mẫu dữ liệu sạch",
                show.to_csv(index=False).encode("utf-8-sig"),
                file_name="cleaned_prices_sample.csv",
                mime="text/csv",
                key="dl_clean",
            )

    st.markdown("##### Nhật ký làm sạch")
    st.dataframe(cleaning_step_table(stats), width="stretch", hide_index=True)

    st.markdown("##### Cách làm sạch dữ liệu bằng Python")
    st.write(
        "Pipeline chạy tự động sau khi giải nén zip CafeF. "
        "Tiền xử lý: chuẩn hóa ngày, lọc missing theo sàn/trần 3 sàn, ghép theo Ticker, xóa trùng cùng mã + cùng ngày."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            **1. Chuẩn hóa biến thời gian**  
            CafeF để ngày dạng `20260903`. Python đổi thành `datetime` chuẩn `YYYY-MM-DD`
            (`2026-09-03`). Ngày không parse được bị loại như dữ liệu thiếu.

            **2. Lọc dữ liệu thiếu + sàn/trần 3 sàn**  
            Loại dòng thiếu ticker/ngày/OHLC. Sau đó so với **giá đóng cửa phiên trước**:
            - HOSE: ±**7%**
            - HNX: ±**10%**
            - UPCOM: ±**15%**  
            `trần = prev_close × (1 + biên độ)`, `sàn = prev_close × (1 − biên độ)`.
            Open/High/Low/Close phải nằm trong [sàn, trần]. Ngày đầu của mỗi mã được giữ.

            **3. Ghép 3 bảng theo dòng dựa trên Ticker**  
            `pd.concat([HOSE, HNX, UPCOM], axis=0)` rồi `sort_values(["ticker", "date"])`.
            """
        )
    with c2:
        st.markdown(
            """
            **4. Xóa trùng lặp theo mã + thời gian**  
            Với mọi ticker, nếu hai dòng cùng ngày thì
            `drop_duplicates(subset=["ticker", "date"], keep="last")`.

            **5. Cột kiểm soát sau xử lý**  
            `prev_close`, `floor` (sàn), `ceiling` (trần), `limit_pct` được giữ lại
            để đối chiếu trên bảng dữ liệu sạch.

            **6. Lưu**  
            Volume ép `int64`, sắp theo Ticker / ngày, ghi Parquet cho phân tích kỹ thuật.
            """
        )

    st.code(CLEANING_CODE, language="python")
    st.caption("Đoạn trên là lõi xử lý, chạy mỗi lần bấm cập nhật dữ liệu.")


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def _parquet_signature() -> tuple[int, int]:
    if not PRICES_PATH.exists():
        return (0, 0)
    stat = PRICES_PATH.stat()
    return (int(stat.st_mtime_ns), int(stat.st_size))


def _with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or all(col in df.columns for col in _TA_COLUMNS):
        return df
    keep = [col for col in _OHLC_COLUMNS if col in df.columns]
    return add_indicators(df[keep])


@st.cache_data(show_spinner=False)
def cached_prices(dataset_date: str, file_sig: tuple[int, int], _v: int = 7) -> pd.DataFrame:
    return _with_indicators(load_prices())


@st.cache_data(show_spinner=False)
def cached_signals(dataset_date: str, strategy: str, file_sig: tuple[int, int], _v: int = 11) -> pd.DataFrame:
    prices = cached_prices(dataset_date, file_sig)
    if prices.empty:
        return pd.DataFrame()
    return generate_signals(prices, strategy, as_of=dataset_date)


def render_sidebar(meta: dict | None) -> None:
    st.sidebar.markdown(brand_html(), unsafe_allow_html=True)
    st.sidebar.caption("CafeF · Đã điều chỉnh · Upto HOSE / HNX / UPCOM")

    if st.sidebar.button("Cập nhật dữ liệu mới nhất", type="primary", width="stretch"):
        status = st.sidebar.status("Đang cập nhật từ CafeF...", expanded=True)

        def progress(message: str, ratio: float | None) -> None:
            status.update(label=message, state="running")

        try:
            result = run_pipeline(force=False, progress=progress)
            cached_prices.clear()
            cached_signals.clear()
            cached_raw_preview.clear()
            if result["status"] == "up_to_date":
                status.update(label="Dataset đã là bản mới nhất.", state="complete")
                st.sidebar.success("Dữ liệu hiện tại đã là bản mới nhất trên CafeF.")
            else:
                status.update(label="Cập nhật hoàn tất.", state="complete")
                st.sidebar.success(f"Đã tải dữ liệu {result['dataset'].dataset_date}.")
            st.session_state["just_updated"] = True
            _rerun()
        except Exception as exc:
            status.update(label="Cập nhật thất bại.", state="error")
            st.sidebar.error(str(exc))

    st.sidebar.divider()
    if meta:
        st.sidebar.markdown(
            f"""
            <div class="side-stat">Phiên dữ liệu<br><b>{meta.get("dataset_date", "—")}</b></div>
            <div class="side-stat">Số dòng<br><b>{meta.get("row_count", "—")}</b></div>
            <div class="side-stat">Số mã<br><b>{meta.get("ticker_count", "—")}</b></div>
            <div class="side-stat">Khoảng<br><b>{meta.get("date_min", "—")} → {meta.get("date_max", "—")}</b></div>
            """,
            unsafe_allow_html=True,
        )
        st.sidebar.caption(meta.get("source_url", ""))
    else:
        st.sidebar.info("Chưa có dữ liệu. Bấm cập nhật để tải Upto 3 sàn đã điều chỉnh.")


def empty_state() -> None:
    st.markdown(
        hero_html(
            "Phân tích chứng khoán Việt Nam",
            "Tải dữ liệu CafeF đã điều chỉnh, làm sạch, rồi sinh tín hiệu mua / bán theo Trendline + MA + CMF.",
        ),
        unsafe_allow_html=True,
    )
    st.info(
        "Bấm **Cập nhật dữ liệu mới nhất** ở thanh bên để bắt đầu. "
        "Lần đầu trên Streamlit Cloud mất vài phút (tải CafeF + tính chỉ báo)."
    )


def indicator_controls(key: str = "chart_indicators") -> list[str]:
    labels = list(INDICATOR_KEYS.values())
    default_labels = [INDICATOR_KEYS[k] for k in DEFAULT_INDICATORS]
    if hasattr(st, "pills"):
        picked = st.pills(
            "Chỉ báo trên biểu đồ", labels, selection_mode="multi", default=default_labels, key=key,
        )
    else:
        picked = st.multiselect("Chỉ báo trên biểu đồ", labels, default=default_labels, key=key)
    return list(picked or [])


def _ticker_panel(view: pd.DataFrame, ticker: str, overlays: list[str], action: str | None = None) -> None:
    if view.empty:
        st.warning("Không có dữ liệu trong khoảng thời gian đã chọn.")
        return
    last = view.iloc[-1]
    if "break_ok" not in view.columns:
        tagged = apply_strategy(view)
        last = tagged.iloc[-1]
        view = tagged
    prev_close = view.iloc[-2]["close"] if len(view) > 1 else last["close"]
    change_pct = ((last["close"] - prev_close) / prev_close * 100) if prev_close else 0
    can_buy = bool(last.get("can_buy")) if "can_buy" in last.index else False
    st.markdown(signal_banner_html(ticker, action, can_buy), unsafe_allow_html=True)
    st.markdown(
        factor_row_html(
            bool(last.get("break_ok")), bool(last.get("ma_ok")), bool(last.get("cmf_ok")),
            str(last.get("ma_text", "—")), str(last.get("cmf_text", "—")),
        ),
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Close", f"{last['close']:.2f}", f"{change_pct:+.2f}%")
    c2.metric("Khối lượng", f"{int(last['volume']):,}")
    c3.metric("SMA 50", f"{last['sma_50']:.2f}" if pd.notna(last.get("sma_50")) else "—")
    c4.metric("CMF 20", f"{last['cmf_20']:.3f}" if pd.notna(last.get("cmf_20")) else "—")
    c5.metric("TL kháng cự", f"{last['tl_upper']:.2f}" if pd.notna(last.get("tl_upper")) else "—")
    c6.metric("TL hỗ trợ", f"{last['tl_lower']:.2f}" if pd.notna(last.get("tl_lower")) else "—")
    st.plotly_chart(price_chart(view, ticker, overlays), width="stretch", config=PLOTLY_CONFIG, theme=None)


def dashboard_tab(prices: pd.DataFrame) -> None:
    exchanges = ["Tất cả"] + sorted(prices["exchange"].dropna().unique().tolist())
    c1, c2, c3 = st.columns([1.1, 1.1, 2.2])
    with c1:
        exchange = st.selectbox("Sàn", exchanges)
    with c2:
        only_stocks = st.checkbox("Chỉ mã 3 ký tự", value=True, key="chart_only_stocks")
    filtered = prices if exchange == "Tất cả" else prices[prices["exchange"] == exchange]
    if only_stocks:
        filtered = filtered[filtered["ticker"].astype(str).str.fullmatch(r"[A-Z]{3}", na=False)]
    tickers = sorted(filtered["ticker"].unique().tolist())
    default_idx = tickers.index("VNM") if "VNM" in tickers else 0
    with c3:
        ticker = st.selectbox("Mã", tickers, index=default_idx)
    ticker_df = filtered[filtered["ticker"] == ticker].sort_values("date")
    min_d, max_d = ticker_df["date"].min().date(), ticker_df["date"].max().date()
    default_start = max(min_d, max_d - timedelta(days=365))
    overlays = indicator_controls("chart_indicators")
    picked = st.date_input(
        "Khoảng thời gian", value=(default_start, max_d), min_value=min_d, max_value=max_d, format="DD/MM/YYYY",
    )
    if isinstance(picked, (list, tuple)) and len(picked) == 2:
        start, end = picked
    else:
        start = end = picked
    view = ticker_df[(ticker_df["date"].dt.date >= start) & (ticker_df["date"].dt.date <= end)]
    _ticker_panel(view, ticker, overlays)


def _event_rows(event) -> list[int]:
    if event is None:
        return []
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    if selection is None:
        return []
    rows = getattr(selection, "rows", None)
    if rows is None and isinstance(selection, dict):
        rows = selection.get("rows", [])
    return [int(i) for i in (rows or [])]


def render_screener_chart(prices: pd.DataFrame, ticker: str, action: str | None = None) -> None:
    full = prices[prices["ticker"].astype(str) == str(ticker)].sort_values("date")
    if full.empty:
        st.warning(f"Không có dữ liệu giá cho {ticker}.")
        return
    tagged = apply_strategy(full)
    end = pd.Timestamp(tagged["date"].max())
    view = tagged[pd.to_datetime(tagged["date"]) >= end - timedelta(days=365)]
    if view.empty:
        view = tagged
    overlays = indicator_controls("screener_indicators")
    _ticker_panel(view, ticker, overlays, action)


def _action_table(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.sort_values(["rank", "ticker"], kind="mergesort")
    if "volume_ma_20" in ranked.columns:
        vol_x = pd.to_numeric(ranked["volume"] / ranked["volume_ma_20"].replace(0, pd.NA), errors="coerce")
    else:
        vol_x = pd.Series([None] * len(ranked), index=ranked.index)
    why = ranked["priority_why"] if "priority_why" in ranked.columns else ranked.get("indicator", "")
    why = why.astype(str).str.replace(" · ", "  ·  ", regex=False)
    return pd.DataFrame({
        "#": ranked["rank"].astype(int),
        "Mã": ranked["ticker"],
        "Sàn": ranked["exchange"],
        "Close": ranked["close"],
        "%": ranked["pct_change"] if "pct_change" in ranked.columns else None,
        "CMF": ranked["cmf_20"],
        "Vol": vol_x,
        "Ghi chú": why,
    })


def _table_config() -> dict:
    return {
        "#": st.column_config.NumberColumn("#", width="small"),
        "Mã": st.column_config.TextColumn("Mã", help="Bấm dòng để xem biểu đồ", width="small"),
        "Sàn": st.column_config.TextColumn("Sàn", width="small"),
        "Close": st.column_config.NumberColumn("Close", format="%.2f"),
        "%": st.column_config.NumberColumn("%", format="%+.2f%%"),
        "CMF": st.column_config.NumberColumn("CMF", format="%+.3f"),
        "Vol": st.column_config.NumberColumn("Vol", format="%.1f×"),
        "Ghi chú": st.column_config.TextColumn("Ghi chú", width="large"),
    }


def screener_tab(prices: pd.DataFrame, dataset_date: str) -> None:
    f1, f2 = st.columns([1.2, 1.2])
    with f1:
        exchanges = ["Tất cả"] + sorted(prices["exchange"].dropna().unique().tolist())
        exchange = st.selectbox("Sàn", exchanges, key="screener_exchange")
    with f2:
        only_stocks = st.checkbox("Chỉ mã cổ phiếu 3 ký tự", value=True)

    with st.expander("Cách xếp thứ tự ưu tiên"):
        st.markdown(
            """
            **Mua** khi đủ 3 chân: break kháng cự LuxAlgo + Close > SMA50 + CMF > 0.  
            **Bán** khi phá hỗ trợ + Close < SMA50 + CMF < 0.

            1. Break **đúng hôm nay** đứng trước mã đã break từ trước.  
            2. CMF càng mạnh cùng chiều.  
            3. Khối lượng / trung bình 20 phiên càng cao.  
            4. % giá hôm nay cùng chiều tín hiệu.
            """
        )

    with st.spinner("Đang xếp tín hiệu mua / bán hôm nay..."):
        signals = cached_signals(dataset_date, DEFAULT_STRATEGY, _parquet_signature())

    base = signals.copy()
    if only_stocks:
        base = base[base["ticker"].astype(str).str.fullmatch(r"[A-Z]{3}", na=False)]
    if exchange != "Tất cả":
        base = base[base["exchange"] == exchange]

    buys = base[base["signal"].eq("BUY")].copy()
    sells = base[base["signal"].eq("SELL")].copy()
    if not buys.empty and "priority_score" in buys.columns:
        buys["rank"] = buys["priority_score"].rank(ascending=False, method="first").astype(int)
    if not sells.empty and "priority_score" in sells.columns:
        sells["rank"] = sells["priority_score"].rank(ascending=False, method="first").astype(int)

    st.markdown(kpi_row_html(len(buys), len(sells), dataset_date), unsafe_allow_html=True)

    buy_show = _action_table(buys) if not buys.empty else pd.DataFrame()
    sell_show = _action_table(sells) if not sells.empty else pd.DataFrame()
    cfg = _table_config()

    left, right = st.columns(2)
    with left:
        st.markdown("**Nên mua** · ưu tiên từ trên xuống")
        st.markdown(rank_strip_html(buys, "MUA"), unsafe_allow_html=True)
        if buy_show.empty:
            st.info("Không có mã mua trong phiên này.")
            buy_event = None
        else:
            buy_event = st.dataframe(
                buy_show, width="stretch", hide_index=True, height=400, key="buy_table",
                on_select="rerun", selection_mode="single-row", column_config=cfg,
            )
    with right:
        st.markdown("**Nên bán** · ưu tiên từ trên xuống")
        st.markdown(rank_strip_html(sells, "BÁN"), unsafe_allow_html=True)
        if sell_show.empty:
            st.info("Không có mã bán trong phiên này.")
            sell_event = None
        else:
            sell_event = st.dataframe(
                sell_show, width="stretch", hide_index=True, height=400, key="sell_table",
                on_select="rerun", selection_mode="single-row", column_config=cfg,
            )

    buy_rows = _event_rows(buy_event)
    sell_rows = _event_rows(sell_event)
    prev_buy = st.session_state.get("_prev_buy_rows", [])
    prev_sell = st.session_state.get("_prev_sell_rows", [])
    if buy_rows and buy_rows != prev_buy:
        st.session_state["screener_picked"] = str(buy_show.iloc[buy_rows[0]]["Mã"])
        st.session_state["screener_picked_side"] = "MUA"
    elif sell_rows and sell_rows != prev_sell:
        st.session_state["screener_picked"] = str(sell_show.iloc[sell_rows[0]]["Mã"])
        st.session_state["screener_picked_side"] = "BÁN"
    st.session_state["_prev_buy_rows"] = buy_rows
    st.session_state["_prev_sell_rows"] = sell_rows

    picked = st.session_state.get("screener_picked")
    if picked:
        st.markdown("")
        render_screener_chart(prices, picked, st.session_state.get("screener_picked_side"))
    else:
        st.caption("Bấm một dòng trong bảng để mở biểu đồ nến, trendline và CMF.")

    export = pd.concat(
        [
            _action_table(buys).assign(Lệnh="MUA") if not buys.empty else pd.DataFrame(),
            _action_table(sells).assign(Lệnh="BÁN") if not sells.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    if not export.empty:
        cols = ["Lệnh"] + [c for c in export.columns if c != "Lệnh"]
        export = export[cols]
        st.download_button(
            "Tải CSV mua / bán hôm nay",
            export.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"tin_hieu_hom_nay_{dataset_date}.csv",
            mime="text/csv",
        )


def main() -> None:
    meta = load_meta()
    render_sidebar(meta)
    if not meta:
        empty_state()
        return

    prices = cached_prices(meta["dataset_date"], _parquet_signature())
    if prices.empty:
        empty_state()
        return

    if st.session_state.pop("just_updated", False):
        st.toast("Đã cập nhật dữ liệu CafeF", icon="✅")

    st.markdown(
        hero_html(
            "Bảng điều khiển giao dịch",
            "Chiến lược duy nhất: Trendline Break (LuxAlgo) kết hợp MA50 và CMF.",
            f"CafeF · phiên {meta['dataset_date']}",
        ),
        unsafe_allow_html=True,
    )

    tab_chart, tab_screener, tab_data = st.tabs(["Diễn biến giá", "Mua / Bán hôm nay", "Dữ liệu"])
    with tab_chart:
        dashboard_tab(prices)
    with tab_screener:
        screener_tab(prices, meta["dataset_date"])
    with tab_data:
        data_quality_tab(prices, meta)


inject_theme()
try:
    main()
except Exception as exc:
    st.error("App gặp lỗi khi chạy.")
    st.exception(exc)
