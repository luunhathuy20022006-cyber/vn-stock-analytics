import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CAFEF_PAGE_URL = "https://cafef.vn/du-lieu/du-lieu-download.chn"
CAFEF_CDN_HOST = "https://cafef1.mediacdn.vn"

# Đã điều chỉnh + Số liệu giao dịch + Upto 3 sàn
# Không lấy file Raw (chưa điều chỉnh).
ADJUSTED_UPTO_NAME = "CafeF.SolieuGD.Upto"

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTRACT_DIR = DATA_DIR / "extracted"
PROCESSED_DIR = DATA_DIR / "processed"

PRICES_PATH = PROCESSED_DIR / "prices.parquet"
SIGNALS_PATH = PROCESSED_DIR / "signals.parquet"
META_PATH = PROCESSED_DIR / "meta.json"

EXCHANGE_MAP = {
    "HSX": "HOSE",
    "HOSE": "HOSE",
    "HNX": "HNX",
    "UPCOM": "UPCOM",
}

# Biên độ dao động ngày (sàn / trần) theo quy định 3 sàn Việt Nam.
PRICE_LIMITS = {
    "HOSE": 0.07,
    "HNX": 0.10,
    "UPCOM": 0.15,
}

# Dung sai làm tròn tick khi so giá với sàn/trần.
LIMIT_TICK_TOLERANCE = 0.001

PREPROCESS_VERSION = "floor-ceiling-v1"

# Streamlit Community Cloud ~1GB RAM: giữ vài năm gần nhất là đủ SMA50/CMF/TL.
CLOUD_HISTORY_YEARS = 5


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

REQUIRED_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume", "exchange"]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
