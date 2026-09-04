from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.analysis.indicators import add_indicators
from src.analysis.signals import DEFAULT_STRATEGY, apply_strategy, generate_signals
from src.config import PRICES_PATH
from src.pipeline.run import run_pipeline
from src.pipeline.store import load_meta, load_prices
from src.viz.charts import DEFAULT_INDICATORS, INDICATOR_KEYS, PLOTLY_CONFIG, price_chart
from src.viz.data_view import cached_raw_preview, data_quality_tab
from src.viz.ui import (
    brand_html,
    factor_row_html,
    hero_html,
    inject_theme,
    kpi_row_html,
    rank_strip_html,
    signal_banner_html,
)

st.set_page_config(
    page_title="VN Stock Analytics",
    page_icon="VN",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


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
            "Chỉ báo trên biểu đồ",
            labels,
            selection_mode="multi",
            default=default_labels,
            key=key,
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
            bool(last.get("break_ok")),
            bool(last.get("ma_ok")),
            bool(last.get("cmf_ok")),
            str(last.get("ma_text", "—")),
            str(last.get("cmf_text", "—")),
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
    st.plotly_chart(
        price_chart(view, ticker, overlays),
        width="stretch",
        config=PLOTLY_CONFIG,
        theme=None,
    )


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
        "Khoảng thời gian",
        value=(default_start, max_d),
        min_value=min_d,
        max_value=max_d,
        format="DD/MM/YYYY",
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
        vol_x = pd.to_numeric(
            ranked["volume"] / ranked["volume_ma_20"].replace(0, pd.NA),
            errors="coerce",
        )
    else:
        vol_x = pd.Series([None] * len(ranked), index=ranked.index)
    why = ranked["priority_why"] if "priority_why" in ranked.columns else ranked.get("indicator", "")
    why = why.astype(str).str.replace(" · ", "  ·  ", regex=False)
    return pd.DataFrame(
        {
            "#": ranked["rank"].astype(int),
            "Mã": ranked["ticker"],
            "Sàn": ranked["exchange"],
            "Close": ranked["close"],
            "%": ranked["pct_change"] if "pct_change" in ranked.columns else None,
            "CMF": ranked["cmf_20"],
            "Vol": vol_x,
            "Ghi chú": why,
        }
    )


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
                buy_show,
                width="stretch",
                hide_index=True,
                height=400,
                key="buy_table",
                on_select="rerun",
                selection_mode="single-row",
                column_config=cfg,
            )
    with right:
        st.markdown("**Nên bán** · ưu tiên từ trên xuống")
        st.markdown(rank_strip_html(sells, "BÁN"), unsafe_allow_html=True)
        if sell_show.empty:
            st.info("Không có mã bán trong phiên này.")
            sell_event = None
        else:
            sell_event = st.dataframe(
                sell_show,
                width="stretch",
                hide_index=True,
                height=400,
                key="sell_table",
                on_select="rerun",
                selection_mode="single-row",
                column_config=cfg,
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


main()
