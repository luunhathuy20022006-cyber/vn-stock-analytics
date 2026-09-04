from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import requests

from src.config import (
    ADJUSTED_UPTO_NAME,
    CAFEF_CDN_HOST,
    CAFEF_PAGE_URL,
    RAW_DIR,
    USER_AGENT,
)

# Chỉ lấy Đã điều chỉnh / Số liệu giao dịch / Upto 3 sàn.
# File chưa điều chỉnh có dạng CafeF.SolieuGD.Raw.UptoDDMMYYYY.zip nên không khớp.
ADJUSTED_UPTO_RE = re.compile(
    rf"{re.escape(CAFEF_CDN_HOST)}/data/ami_data/(\d{{8}})/{re.escape(ADJUSTED_UPTO_NAME)}(\d{{8}})\.zip",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CafeFDataset:
    url: str
    folder_date: str  # YYYYMMDD trên CDN
    file_date: str  # DDMMYYYY trong tên file
    dataset_date: str  # YYYY-MM-DD

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
        url = (
            f"{CAFEF_CDN_HOST}/data/ami_data/{folder_date}/"
            f"{ADJUSTED_UPTO_NAME}{file_date}.zip"
        )
        found[folder_date] = CafeFDataset(
            url=url,
            folder_date=folder_date,
            file_date=file_date,
            dataset_date=_to_iso(folder_date),
        )
    return [found[key] for key in sorted(found, reverse=True)]


def _candidate_dates(days: int = 14) -> list[datetime]:
    today = datetime.now()
    return [today - timedelta(days=i) for i in range(days)]


def _build_dataset(day: datetime) -> CafeFDataset:
    folder_date = day.strftime("%Y%m%d")
    file_date = day.strftime("%d%m%Y")
    url = f"{CAFEF_CDN_HOST}/data/ami_data/{folder_date}/{ADJUSTED_UPTO_NAME}{file_date}.zip"
    return CafeFDataset(
        url=url,
        folder_date=folder_date,
        file_date=file_date,
        dataset_date=day.strftime("%Y-%m-%d"),
    )


def discover_latest_adjusted_upto(timeout: int = 30) -> CafeFDataset:
    """Tìm file Đã điều chỉnh - Số liệu giao dịch - Upto 3 sàn mới nhất trên CafeF."""
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


def download_dataset(
    dataset: CafeFDataset,
    progress_cb=None,
    timeout: int = 180,
) -> Path:
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
