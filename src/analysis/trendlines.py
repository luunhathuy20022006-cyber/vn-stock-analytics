from __future__ import annotations

import numpy as np
import pandas as pd

# LuxAlgo Trendlines with Breaks — tham số mặc định
TL_LENGTH = 14
TL_SLOPE_K = 1.0
MA_PERIOD = 50
CMF_PERIOD = 20


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
    Trendlines with Breaks theo cách xây của LuxAlgo:
    - Pivot high/low với lookback `length`
    - Độ dốc ATR: slope = ATR(length) / length * k
    - Upper: neo tại pivot high, mỗi nến trừ slope (kháng cự hướng xuống)
    - Lower: neo tại pivot low, mỗi nến cộng slope (hỗ trợ hướng lên)
    - Không backpaint: so close với đường chiếu tới nến hiện tại
      upper_line = upper - slope_ph * length
      lower_line = lower + slope_pl * length
    - upos/dnos: trạng thái đã break, reset khi có pivot mới
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


def chaikin_money_flow(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = CMF_PERIOD) -> pd.Series:
    hl = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl
    mfm = mfm.fillna(0.0)
    mfv = mfm * volume
    vol_sum = volume.rolling(period, min_periods=period).sum().replace(0, np.nan)
    return mfv.rolling(period, min_periods=period).sum() / vol_sum
