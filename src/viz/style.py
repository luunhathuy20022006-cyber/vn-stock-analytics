from __future__ import annotations

import pandas as pd

from src.analysis.signals import FACTOR_SPECS

GREEN_BG = "#14532d"
GREEN_FG = "#bbf7d0"
RED_BG = "#7f1d1d"
RED_FG = "#fecaca"
GREEN_SOFT = "rgba(22, 163, 74, 0.16)"
RED_SOFT = "rgba(220, 38, 38, 0.14)"

OHLC_ICONS = {
    "ticker": "🏷 Ticker",
    "<Ticker>": "🏷 Ticker",
    "date": "📅 Ngày",
    "<DTYYYYMMDD>": "📅 Ngày",
    "open": "○ Open",
    "<Open>": "○ Open",
    "high": "▲ High",
    "<High>": "▲ High",
    "low": "▼ Low",
    "<Low>": "▼ Low",
    "close": "● Close",
    "<Close>": "● Close",
    "volume": "▮ Volume",
    "<Volume>": "▮ Volume",
    "exchange": "🏛 Sàn",
}


def buy_banner_html(can_buy: bool, ticker: str = "") -> str:
    if can_buy:
        bg, border, fg, text = "#052e16", "#16a34a", "#86efac", "🟢 MUA ĐƯỢC"
    else:
        bg, border, fg, text = "#450a0a", "#dc2626", "#fca5a5", "🔴 KHÔNG MUA"
    label = f"{ticker} · {text}" if ticker else text
    return (
        f'<div style="padding:14px 18px;border-radius:14px;background:{bg};'
        f'border:1px solid {border};color:{fg};font-size:1.35rem;font-weight:800;'
        f'letter-spacing:0.04em;text-align:center;">{label}</div>'
    )


def factor_chip_html(icon: str, name: str, ok: bool, detail: str, rule: str) -> str:
    if ok:
        bg, border, fg = "#052e16", "#16a34a", "#86efac"
        flag = "✓"
    else:
        bg, border, fg = "#450a0a", "#dc2626", "#fca5a5"
        flag = "✗"
    return (
        f'<div style="padding:10px 12px;border-radius:12px;background:{bg};'
        f'border:1px solid {border};color:{fg};min-height:92px;">'
        f'<div style="font-size:1.15rem;font-weight:700;">{icon} {flag} {name}</div>'
        f'<div style="font-size:0.95rem;margin-top:4px;">{detail}</div>'
        f'<div style="font-size:0.75rem;opacity:0.8;margin-top:4px;">{rule}</div>'
        f"</div>"
    )


def factor_legend_html(strategy: str) -> str:
    chips = []
    for _key, icon, short, rule in FACTOR_SPECS[strategy]:
        chips.append(
            f'<span style="margin-right:14px;white-space:nowrap;">'
            f'<b>{icon} {short}</b> <span style="opacity:0.75;">({rule})</span></span>'
        )
    return (
        '<div style="margin:6px 0 12px 0;line-height:1.8;">'
        + " ".join(chips)
        + '<div style="margin-top:6px;"><span style="color:#86efac;">■ Xanh = đạt / mua được</span>'
        '&nbsp;&nbsp;<span style="color:#fca5a5;">■ Đỏ = không đạt / không mua</span></div></div>'
    )


def style_screener(df: pd.DataFrame, strategy: str) -> pd.Styler:
    factor_cols = [f"{icon} {short}" for _k, icon, short, _r in FACTOR_SPECS[strategy]]

    def _row_bg(row: pd.Series) -> list[str]:
        buy = "MUA ĐƯỢC" in str(row.get("Kết luận", ""))
        color = GREEN_SOFT if buy else RED_SOFT
        return [f"background-color: {color}"] * len(row)

    def _verdict(val: object) -> str:
        text = str(val)
        if "MUA ĐƯỢC" in text:
            return f"background-color: {GREEN_BG}; color: {GREEN_FG}; font-weight: 800; text-align: center;"
        return f"background-color: {RED_BG}; color: {RED_FG}; font-weight: 800; text-align: center;"

    def _factor(val: object) -> str:
        text = str(val)
        if "✓" in text:
            return f"background-color: {GREEN_BG}; color: {GREEN_FG}; font-weight: 700;"
        if "✗" in text:
            return f"background-color: {RED_BG}; color: {RED_FG}; font-weight: 700;"
        return ""

    styler = df.style.apply(_row_bg, axis=1)
    if "Kết luận" in df.columns:
        styler = styler.map(_verdict, subset=["Kết luận"])
    present = [col for col in factor_cols if col in df.columns]
    if present:
        styler = styler.map(_factor, subset=present)
    if "Δ %" in df.columns:
        styler = styler.map(_delta, subset=["Δ %"])
    return styler.hide(axis="index")


def _delta(val: object) -> str:
    text = str(val)
    if text.startswith("+"):
        return f"color: {GREEN_FG}; font-weight: 700;"
    if text.startswith("-"):
        return f"color: {RED_FG}; font-weight: 700;"
    return ""


def style_ohlc(df: pd.DataFrame, open_col: str, close_col: str) -> pd.Styler:
    def _row_bg(row: pd.Series) -> list[str]:
        try:
            up = float(row[close_col]) >= float(row[open_col])
        except Exception:
            return [""] * len(row)
        color = GREEN_SOFT if up else RED_SOFT
        return [f"background-color: {color}"] * len(row)

    def _cell(col: pd.Series) -> list[str]:
        styles = []
        name = str(col.name)
        for idx, val in col.items():
            try:
                number = float(val)
            except Exception:
                styles.append("")
                continue
            if name == close_col:
                try:
                    up = number >= float(df.loc[idx, open_col])
                    styles.append(
                        f"color: {GREEN_FG}; font-weight: 800;" if up else f"color: {RED_FG}; font-weight: 800;"
                    )
                except Exception:
                    styles.append("")
            elif "High" in name:
                styles.append("color: #fdba74; font-weight: 700;")
            elif "Low" in name:
                styles.append("color: #93c5fd; font-weight: 700;")
            elif "Volume" in name:
                styles.append("color: #e9d5ff; font-weight: 700;")
            elif "Open" in name:
                styles.append("color: #fef08a; font-weight: 600;")
            else:
                styles.append("")
        return styles

    styler = df.style.apply(_row_bg, axis=1)
    for col in df.columns:
        styler = styler.apply(_cell, subset=[col])
    return styler.hide(axis="index")


def style_cleaning_log(df: pd.DataFrame) -> pd.Styler:
    def _dropped(val: object) -> str:
        try:
            n = int(val)
        except Exception:
            return ""
        if n > 0:
            return f"background-color: {RED_BG}; color: {RED_FG}; font-weight: 700;"
        return f"color: {GREEN_FG};"

    def _remain(col: pd.Series) -> list[str]:
        styles = []
        last = len(col) - 1
        for i, _val in enumerate(col):
            if i == 0:
                styles.append("font-weight: 700;")
            elif i == last:
                styles.append(f"background-color: {GREEN_BG}; color: {GREEN_FG}; font-weight: 800;")
            else:
                styles.append("")
        return styles

    return (
        df.style.map(_dropped, subset=["Số dòng loại"])
        .apply(_remain, subset=["Số dòng còn lại"])
        .hide(axis="index")
    )


def rename_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {col: OHLC_ICONS[col] for col in df.columns if col in OHLC_ICONS}
    return df.rename(columns=mapping)


def screener_display(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    specs = FACTOR_SPECS[strategy]
    out = pd.DataFrame(
        {
            "🏷 Ticker": df["ticker"],
            "🏛 Sàn": df["exchange"],
            "● Close": df["close"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—"),
            "Kết luận": df["buy_label"],
            "Tín hiệu": df["signal"].map({"BUY": "BUY ▲", "SELL": "SELL ▼", "HOLD": "HOLD ●"}).fillna(df["signal"]),
        }
    )
    if "pct_change" in df.columns:
        out["Δ %"] = df["pct_change"].map(lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
    for key, icon, short, _rule in specs:
        col = f"{key}_mark"
        if col in df.columns:
            out[f"{icon} {short}"] = df[col]
    if "signal_date" in df.columns:
        out["📅 Ngày"] = pd.to_datetime(df["signal_date"]).dt.strftime("%Y-%m-%d")
    return out
