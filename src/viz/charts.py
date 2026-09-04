from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.analysis.indicators import add_chart_ta
from src.analysis.signals import apply_strategy
from src.analysis.trendlines import CMF_PERIOD, TL_LENGTH, TL_SLOPE_K

# TradingView light (đúng theme ảnh FPT)
BG = "#ffffff"
GRID = "#e0e3eb"
AXIS = "#787b86"
TEXT = "#131722"
MUTED = "#6a6d78"
UP = "#089981"
DOWN = "#f23645"
CROSS = "#9598a1"
SMA_COLOR = "#2962ff"
SMA20_COLOR = "#ff9800"
SUPPORT = "#26a69a"
RESIST = "#ef5350"

INDICATOR_KEYS = {
    "tl": "Trendline",
    "sma50": "SMA 50",
    "sma20": "SMA 20",
    "ema20": "EMA 20",
    "bb": "Bollinger",
    "bs": "Tín hiệu B/S",
    "vol": "Khối lượng",
    "cmf": "CMF 20",
    "rsi": "RSI 14",
    "macd": "MACD",
    "atr": "ATR 14",
}
DEFAULT_INDICATORS = ["tl", "sma50", "bs", "vol", "cmf"]

PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "displayModeBar": False,
}

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
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        zeroline=False,
        showline=False,
        color=AXIS,
        tickfont=dict(size=11, color=AXIS, family="Trebuchet MS"),
        showticklabels=show_x,
        rangeslider_visible=False,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor=CROSS,
        spikethickness=1,
        spikedash="solid",
        type="date",
        range=x_range,
        tickformat="%d/%m",
        nticks=8,
        row=row,
        col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        zeroline=False,
        showline=False,
        side="right",
        color=AXIS,
        tickfont=dict(size=11, color=AXIS, family="Trebuchet MS"),
        showspikes=True,
        spikemode="across",
        spikecolor=CROSS,
        spikethickness=1,
        spikedash="solid",
        row=row,
        col=1,
    )


def normalize_overlays(overlays: list[str] | None) -> set[str]:
    if overlays is None:
        return set(DEFAULT_INDICATORS)
    alias = {
        "SMA50": "sma50",
        "SMA20": "sma20",
        "Trendline": "tl",
        "CMF": "cmf",
        "Khối lượng": "vol",
        "Tín hiệu B/S": "bs",
        **{label: key for key, label in INDICATOR_KEYS.items()},
    }
    return {alias.get(item, item) for item in overlays}


def _last_tag(fig: go.Figure, x, y: float, text: str, color: str, row: int) -> None:
    if y is None or not np.isfinite(y):
        return
    fig.add_annotation(
        x=x,
        y=y,
        text=f" {text} ",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        xshift=6,
        bgcolor=color,
        font=dict(color="#ffffff", size=10, family="Trebuchet MS"),
        borderpad=2,
        row=row,
        col=1,
    )


def _soft_line(
    fig: go.Figure,
    x,
    y,
    *,
    name: str,
    color: str,
    width: float = 1.8,
    dash: str | None = None,
    row: int = 1,
    hover: str | None = None,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
            line=dict(color=color, width=width + 4, dash=dash),
            opacity=0.14,
            hoverinfo="skip",
            showlegend=False,
            connectgaps=False,
        ),
        row=row,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
            line=dict(color=color, width=width, dash=dash, shape="linear"),
            hovertemplate=hover or f"{name} %{{y:.2f}}<extra></extra>",
            showlegend=False,
            connectgaps=False,
        ),
        row=row,
        col=1,
        secondary_y=False,
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
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=heights,
        specs=[[{"secondary_y": True}]] + [[{"secondary_y": False}]] * len(panes),
    )

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=tagged["open"],
            high=tagged["high"],
            low=tagged["low"],
            close=tagged["close"],
            name="Nến",
            increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
            decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN),
            whiskerwidth=0.9,
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
        secondary_y=False,
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
                x=x,
                y=sma50,
                mode="lines",
                name="SMA 50",
                line=dict(color=SMA_COLOR, width=2.1, shape="spline", smoothing=0.7),
                hovertemplate="SMA50 %{y:.2f}<extra></extra>",
                showlegend=False,
                connectgaps=True,
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
        if pd.notna(sma50.iloc[-1]):
            _last_tag(fig, x.iloc[-1], float(sma50.iloc[-1]), f"MA50 {_fmt(float(sma50.iloc[-1]))}", SMA_COLOR, 1)
    if "sma20" in on and "sma_20" in tagged.columns:
        sma20 = pd.to_numeric(tagged["sma_20"], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=sma20,
                mode="lines",
                name="SMA 20",
                line=dict(color=SMA20_COLOR, width=1.7, shape="spline", smoothing=0.7),
                hovertemplate="SMA20 %{y:.2f}<extra></extra>",
                showlegend=False,
                connectgaps=True,
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
        if pd.notna(sma20.iloc[-1]):
            _last_tag(fig, x.iloc[-1], float(sma20.iloc[-1]), f"MA20 {_fmt(float(sma20.iloc[-1]))}", SMA20_COLOR, 1)
    if "ema20" in on and "ema_20" in tagged.columns:
        ema20 = pd.to_numeric(tagged["ema_20"], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=x,
                y=ema20,
                mode="lines",
                name="EMA 20",
                line=dict(color="#7c3aed", width=1.7, shape="spline", smoothing=0.65),
                hovertemplate="EMA20 %{y:.2f}<extra></extra>",
                showlegend=False,
                connectgaps=True,
            ),
            row=1,
            col=1,
            secondary_y=False,
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
                    x=buys["date"],
                    y=buys["low"] * 0.988,
                    mode="markers+text",
                    text=["B"] * len(buys),
                    textposition="middle center",
                    textfont=dict(color="#ffffff", size=8, family="Arial Black"),
                    marker=dict(
                        symbol="square",
                        size=15,
                        color=UP,
                        line=dict(width=1, color="#ffffff"),
                    ),
                    name="B",
                    hovertemplate="Mua · Break ↑<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=1,
                secondary_y=False,
            )
    if "bs" in on and "tl_break_dn" in tagged.columns:
        sells = tagged[tagged["tl_break_dn"].astype(bool)]
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["date"],
                    y=sells["high"] * 1.012,
                    mode="markers+text",
                    text=["S"] * len(sells),
                    textposition="middle center",
                    textfont=dict(color="#ffffff", size=8, family="Arial Black"),
                    marker=dict(
                        symbol="square",
                        size=15,
                        color=DOWN,
                        line=dict(width=1, color="#ffffff"),
                    ),
                    name="S",
                    hovertemplate="Bán · Break ↓<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=1,
                secondary_y=False,
            )

    if "vol" in on:
        vol_colors = [UP if c >= o else DOWN for c, o in zip(tagged["close"], tagged["open"])]
        fig.add_trace(
            go.Bar(
                x=x,
                y=tagged["volume"],
                name="Vol",
                marker=dict(color=vol_colors, line=dict(width=0)),
                opacity=0.32,
                hovertemplate="Vol %{y:,.0f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
            secondary_y=True,
        )
        vmax = float(pd.to_numeric(tagged["volume"], errors="coerce").max() or 1)
        fig.update_yaxes(
            range=[0, vmax * 3.8],
            showgrid=False,
            showticklabels=False,
            showspikes=False,
            title_text="",
            row=1,
            col=1,
            secondary_y=True,
        )

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
        x=x.iloc[-1],
        y=last_close,
        text=f" {_fmt(last_close)} ",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        xshift=6,
        bgcolor=last_color,
        font=dict(color="#ffffff", size=11, family="Trebuchet MS"),
        borderpad=3,
        row=1,
        col=1,
    )

    chg_txt = f"{chg:+.2f} ({chg_pct:+.2f}%)"
    header = (
        f"<b style='font-size:14px;color:{TEXT}'>{ticker}</b>"
        f"<span style='color:{MUTED}'>  ·  1D  ·  {exchange}</span>"
        f"&nbsp;&nbsp;"
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
        xref="paper",
        yref="paper",
        x=0,
        y=1.0,
        xanchor="left",
        yanchor="top",
        text=header,
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.88)",
        borderpad=6,
        font=dict(family="Trebuchet MS", size=12, color=TEXT),
    )
    labels = {"cmf": f"CMF {CMF_PERIOD}", "rsi": "RSI 14", "macd": "MACD", "atr": "ATR 14"}
    for i, pane in enumerate(panes, start=2):
        fig.update_yaxes(title_text=labels[pane], title_font=dict(size=11, color=MUTED), row=i, col=1)

    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family="Trebuchet MS, Segoe UI, sans-serif", color=TEXT, size=12),
        margin=dict(l=6, r=68, t=10, b=10),
        showlegend=False,
        hovermode="x",
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#e0e3eb",
            font=dict(family="Trebuchet MS", size=12, color=TEXT),
        ),
        dragmode="pan",
        bargap=0.22,
        xaxis_rangeslider_visible=False,
        height=620 + 120 * len(panes),
    )
    _tv_axes(fig, 1, show_x=not panes, x_range=x_range)
    for i in range(2, n_rows + 1):
        _tv_axes(fig, i, show_x=(i == n_rows), x_range=x_range)
        fig.update_yaxes(tickformat=".2f", row=i, col=1)
    fig.update_yaxes(tickformat=".2f", separatethousands=True, row=1, col=1, secondary_y=False)
    return fig
