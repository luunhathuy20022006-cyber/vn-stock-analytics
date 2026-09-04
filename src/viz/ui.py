from __future__ import annotations

import pandas as pd
import streamlit as st

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
  border: 1px solid #22304a;
  border-radius: 16px;
  padding: 14px 16px 12px 16px;
  box-shadow: 0 8px 24px rgba(0,0,0,.18);
}
div[data-testid="stMetric"] label { color: #8b9bb4 !important; font-size: 0.78rem !important; letter-spacing: .04em; text-transform: uppercase; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-family: "IBM Plex Mono", monospace; font-weight: 600; }

.stTabs [data-baseweb="tab-list"] {
  gap: 8px;
  background: #0e1524;
  border: 1px solid #1c2740;
  border-radius: 14px;
  padding: 6px;
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  border-radius: 10px;
  color: #8b9bb4;
  font-weight: 600;
  padding: 10px 16px;
}
.stTabs [aria-selected="true"] {
  background: #1a2744 !important;
  color: #e6edf7 !important;
  box-shadow: inset 0 0 0 1px #2d4166;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

.stButton > button {
  border-radius: 12px;
  font-weight: 650;
  border: 1px solid #2d4166;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(180deg, #3b82f6, #2563eb);
  border: 0;
}

[data-testid="stDataFrame"] {
  border: 1px solid #1c2740;
  border-radius: 16px;
  overflow: hidden;
  background: #0e1524;
}
[data-testid="stExpander"] {
  background: #0e1524;
  border: 1px solid #1c2740;
  border-radius: 14px;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

.hero {
  display: flex; align-items: flex-end; justify-content: space-between; gap: 16px;
  margin: 0 0 18px 0; padding: 4px 2px 16px 2px;
  border-bottom: 1px solid #1c2740;
}
.hero h1 { margin: 0; font-size: 1.85rem; color: #f4f7fb; }
.hero p { margin: 6px 0 0 0; color: #8b9bb4; font-size: 0.95rem; }
.badge {
  display: inline-flex; align-items: center; gap: 8px;
  background: #102033; color: #93c5fd; border: 1px solid #1e3a5f;
  border-radius: 999px; padding: 6px 12px; font-size: 0.8rem; font-weight: 600;
  white-space: nowrap;
}
.badge.ok { background: #0c241c; color: #6ee7b7; border-color: #14532d; }

.brand { display: flex; gap: 12px; align-items: center; margin-bottom: 18px; }
.brand-mark {
  width: 42px; height: 42px; border-radius: 12px;
  background: linear-gradient(145deg, #3b82f6, #1d4ed8);
  color: white; display: flex; align-items: center; justify-content: center;
  font-weight: 800; letter-spacing: .04em;
}
.brand-name { font-weight: 700; font-size: 1.05rem; color: #f4f7fb; }
.brand-sub { color: #8b9bb4; font-size: 0.78rem; margin-top: 2px; }

.kpi-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 4px 0 16px 0; }
.kpi {
  background: linear-gradient(180deg, #121a2c 0%, #0e1524 100%);
  border: 1px solid #22304a; border-radius: 16px; padding: 16px 18px;
}
.kpi .k { color: #8b9bb4; font-size: 0.75rem; letter-spacing: .08em; text-transform: uppercase; font-weight: 600; }
.kpi .v { font-family: "IBM Plex Mono", monospace; font-size: 1.7rem; font-weight: 600; margin-top: 6px; }
.kpi.buy { box-shadow: inset 3px 0 0 #10b981; }
.kpi.buy .v { color: #34d399; }
.kpi.sell { box-shadow: inset 3px 0 0 #f43f5e; }
.kpi.sell .v { color: #fb7185; }
.kpi.neutral .v { color: #93c5fd; }

.panel {
  background: #0e1524; border: 1px solid #1c2740; border-radius: 18px;
  padding: 16px 16px 8px 16px; margin-bottom: 14px;
}
.panel h3 { margin: 0 0 4px 0; font-size: 1.05rem; }
.panel .hint { color: #8b9bb4; font-size: 0.82rem; margin-bottom: 10px; }

.signal {
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  border-radius: 18px; padding: 16px 20px; margin: 4px 0 14px 0; border: 1px solid;
}
.signal .sym { font-size: 1.45rem; font-weight: 800; letter-spacing: .04em; }
.signal .lbl { font-size: 0.82rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 700; opacity: .9; }
.signal.buy { background: linear-gradient(90deg, #052e1c, #0e1524); border-color: #14532d; color: #6ee7b7; }
.signal.sell { background: linear-gradient(90deg, #3f0d18, #0e1524); border-color: #7f1d1d; color: #fda4af; }
.signal.hold { background: linear-gradient(90deg, #172554, #0e1524); border-color: #1e3a5f; color: #93c5fd; }

.factors { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 12px; }
.factor {
  border-radius: 14px; padding: 12px 14px; border: 1px solid; min-height: 86px;
}
.factor .name { font-weight: 700; font-size: 0.92rem; }
.factor .detail { margin-top: 6px; font-family: "IBM Plex Mono", monospace; font-size: 0.88rem; }
.factor .rule { margin-top: 4px; font-size: 0.75rem; opacity: .75; }
.factor.ok { background: rgba(16,185,129,.08); border-color: #14532d; color: #86efac; }
.factor.bad { background: rgba(244,63,94,.08); border-color: #7f1d1d; color: #fda4af; }

.rank-strip { display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 10px 0; }
.rank-chip {
  display: flex; align-items: center; gap: 8px;
  background: #121a2c; border: 1px solid #22304a; border-radius: 999px;
  padding: 5px 10px 5px 6px; font-size: 0.82rem;
}
.rank-chip .n {
  width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-family: "IBM Plex Mono", monospace; font-size: 0.72rem; font-weight: 700;
}
.rank-chip.buy .n { background: #065f46; color: #a7f3d0; }
.rank-chip.sell .n { background: #9f1239; color: #fecdd3; }
.rank-chip .t { font-weight: 700; color: #e6edf7; }
.rank-chip .p { font-family: "IBM Plex Mono", monospace; color: #8b9bb4; }

.stPlotlyChart, div[data-testid="stPlotlyChart"] {
  background: #ffffff !important;
  border: 1px solid #d0d5dd;
  border-radius: 8px;
  overflow: hidden;
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


def panel_open(title: str, hint: str = "") -> str:
    hint_html = f'<div class="hint">{hint}</div>' if hint else ""
    return f'<div class="panel"><h3>{title}</h3>{hint_html}'


def panel_close() -> str:
    return "</div>"
