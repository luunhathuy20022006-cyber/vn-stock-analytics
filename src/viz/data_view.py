from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.pipeline.preview import (
    cleaning_step_table,
    extract_dir_from_meta,
    list_download_files,
    preview_clean,
    preview_raw_csv,
)

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
df["ceiling"] = df["prev_close"] * (1 + df["limit_pct"])   # trần
df["floor"] = df["prev_close"] * (1 - df["limit_pct"])     # sàn
# Giữ ngày đầu của mỗi mã; các ngày sau phải nằm trong [sàn, trần]
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
        "Module `src/pipeline/clean.py` chạy tự động sau khi giải nén zip CafeF. "
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
    st.caption("Đoạn trên là lõi xử lý trong `src/pipeline/clean.py`, chạy mỗi lần bấm cập nhật dữ liệu.")
