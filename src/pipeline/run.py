from __future__ import annotations

import gc
import shutil
from typing import Callable

import pandas as pd

from src.analysis.indicators import add_indicators
from src.analysis.signals import DEFAULT_STRATEGY, generate_signals
from src.config import PRICES_CSV, PRICES_PATH, PREPROCESS_VERSION, history_years, is_streamlit_cloud
from src.pipeline.cafef import discover_latest_adjusted_upto, download_dataset
from src.pipeline.clean import load_and_clean
from src.pipeline.extract import extract_zip
from src.pipeline.store import load_meta, load_prices, save_meta, save_prices, save_signals


ProgressCb = Callable[[str, float | None], None]


def run_pipeline(
    force: bool = False,
    progress: ProgressCb | None = None,
) -> dict:
    """
    Phát hiện dataset CafeF mới nhất (Đã điều chỉnh - Upto 3 sàn),
    tải, giải nén, làm sạch, tính chỉ báo và sinh tín hiệu.
    """

    def emit(message: str, ratio: float | None = None) -> None:
        if progress:
            progress(message, ratio)

    emit("Đang tìm dữ liệu CafeF mới nhất...", 0.02)
    dataset = discover_latest_adjusted_upto()
    emit(f"Tìm thấy {dataset.dataset_date}: Đã điều chỉnh / Upto 3 sàn", 0.08)

    meta = load_meta()
    already_current = (
        not force
        and meta
        and meta.get("dataset_date") == dataset.dataset_date
        and meta.get("preprocess_version") == PREPROCESS_VERSION
        and (PRICES_PATH.exists() or PRICES_CSV.exists())
    )
    if already_current:
        prices = load_prices()
        if "cmf_20" not in prices.columns or "tl_upper" not in prices.columns:
            emit("Đang tính lại Trendline + MA + CMF...", 0.72)
            base_cols = [c for c in prices.columns if c in {
                "ticker", "date", "open", "high", "low", "close", "volume", "exchange",
                "prev_close", "floor", "ceiling", "limit_pct",
            }]
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
        return {
            "status": "up_to_date",
            "dataset": dataset,
            "meta": meta,
        }

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


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    def _print(message: str, ratio: float | None) -> None:
        suffix = f" ({ratio:.0%})" if ratio is not None else ""
        print(message + suffix)

    result = run_pipeline(progress=_print)
    print(result["status"], result.get("meta", {}).get("dataset_date"))
