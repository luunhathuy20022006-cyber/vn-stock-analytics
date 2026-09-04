# VN Stock Analytics

Hệ thống phân tích và sàng lọc cổ phiếu Việt Nam tự động (đề **Duan_1**).

Nguồn: [CafeF – dữ liệu lịch sử 3 sàn](https://cafef.vn/du-lieu/du-lieu-download.chn)  
Chỉ dùng file **Đã điều chỉnh → Số liệu giao dịch → Upto 3 sàn** (HOSE, HNX, UPCOM).

```
CafeF → Tải → Giải nén → Làm sạch → Lưu → Chỉ báo → Tín hiệu → Sàng lọc → Biểu đồ
```

## Đáp ứng đề bài

| Task | Nội dung | Trong app |
| --- | --- | --- |
| 1. Pipeline tự động | Tải CafeF, giải nén, gộp 3 sàn, chuẩn hóa ngày, lọc thiếu/sai, sàn-trần, xóa trùng, lưu Parquet | Nút **Cập nhật dữ liệu mới nhất** |
| 2. Dashboard giá | Chọn sàn, mã, khoảng thời gian; nến OHLC; volume; bật/tắt SMA, EMA, Bollinger, RSI, MACD, ATR, Trendline, CMF | Tab **Diễn biến giá** |
| 3. Screener | BUY / SELL / HOLD theo luật rõ ràng, xếp ưu tiên phiên hôm nay | Tab **Mua / Bán hôm nay** |
| 4. Tự động hóa | Một nút cập nhật toàn bộ pipeline + giao diện thống nhất | Sidebar + 3 tab |

**Chiến lược tín hiệu (duy nhất):** LuxAlgo Trendlines with Breaks + SMA50 + CMF20.

- **MUA** khi break kháng cự (`upos=1`) **và** Close > SMA50 **và** CMF > 0  
- **BÁN** khi phá hỗ trợ **và** Close < SMA50 **và** CMF < 0  
- Thứ tự ưu tiên: break đúng hôm nay → CMF mạnh → volume / MA20 → % giá cùng chiều  

Làm sạch: ngày `YYYY-MM-DD`; loại missing; sàn/trần HOSE ±7%, HNX ±10%, UPCOM ±15%; ghép 3 sàn theo ticker; `drop_duplicates(ticker, date)`.

## Chạy local

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
streamlit run app.py
```

Lần đầu bấm **Cập nhật dữ liệu mới nhất**. Tải ~50MB, tính chỉ báo vài phút.

## Deploy Streamlit Community Cloud

File `prices.parquet` ~270MB, **không** đưa lên GitHub (giới hạn 100MB). Cloud chạy pipeline khi bấm cập nhật. Trên Cloud chỉ giữ **5 năm** gần nhất để vừa RAM ~1GB.

1. Repo GitHub public, file `app.py`, Python 3.12.  
2. [share.streamlit.io](https://share.streamlit.io) → **Create app**.  
3. Deploy. App mở trống → bấm **Cập nhật dữ liệu mới nhất**.

Hoặc chạy server riêng:

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Cấu trúc

```
app.py                 # Streamlit
src/pipeline/          # CafeF download, extract, clean, store
src/analysis/          # Trendline, MA, CMF, tín hiệu
src/viz/               # Chart TradingView-style, theme, tab dữ liệu
data/processed/        # parquet + meta (sinh khi cập nhật)
```
