# Hệ Thống Mock API Streaming & Polling (PDF Papers)

Hệ thống này mô phỏng cơ chế Streaming (Polling) giữa Client và Server để tự động đồng bộ hóa các bài báo khoa học (PDF) và siêu dữ liệu (metadata JSON) vào các thư mục tương ứng.

## 🏗️ Cấu trúc hệ thống

- **`server.py`**: Backend Mock API (FastAPI). Tiếp nhận file từ người dùng, lưu tạm vào thư mục `api_data/` và cung cấp API để Client có thể "hỏi thăm" (poll) dữ liệu mới.
- **`ui.py`**: Giao diện người dùng (Streamlit). Cho phép bạn hoặc người khác trên cùng mạng nhập thông tin và tải lên file PDF.
- **`client.py`**: Trình đồng bộ (Polling). Cứ mỗi 5 giây sẽ gọi lên Server, nếu có file mới sẽ tải về, phân loại vào thư mục `data/` và ghi siêu dữ liệu (metadata) vào các file trong thư mục `json/`.

## 🛠️ Cài đặt môi trường

1. Đảm bảo bạn đã cài đặt Python.
2. Mở Terminal trong thư mục dự án và tạo môi trường ảo (Virtual Environment):
   ```powershell
   python -m venv venv
   ```
3. Kích hoạt môi trường ảo:
   - Trên Windows:
     ```powershell
     .\venv\Scripts\activate
     ```
   - Trên macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
4. Cài đặt các thư viện cần thiết:
   ```powershell
   pip install -r requirements.txt
   ```

## 🚀 Hướng dẫn khởi chạy

Để hệ thống hoạt động hoàn chỉnh, bạn cần mở **3 cửa sổ Terminal** đồng thời. *(Lưu ý: Đảm bảo cả 3 terminal đều đã kích hoạt môi trường ảo `venv`)*.

### 1. Khởi động Máy chủ (Backend)
Ở **Terminal 1**, chạy lệnh sau để khởi động FastAPI Server:
```powershell
python server.py
```
> Server sẽ chạy ở địa chỉ `http://localhost:8000` và tự động tạo thư mục đệm `api_data/` nếu chưa có.

### 2. Khởi động Giao diện Cập nhật (Frontend)
Ở **Terminal 2**, chạy lệnh sau để mở giao diện web:
```powershell
streamlit run ui.py
```
> Trình duyệt sẽ tự động mở tab mới tại `http://localhost:8501`. Bạn có thể dùng trang web này để tải file PDF lên.

### 3. Khởi động Trình đồng bộ (Client)
Ở **Terminal 3**, chạy lệnh sau để lắng nghe và đồng bộ dữ liệu:
```powershell
python client.py
```
> Script sẽ liên tục in ra "[Client] Bắt đầu polling từ server..." mỗi 5 giây để kiểm tra file mới.

## 🧪 Cách kiểm thử hệ thống

1. Sau khi cả 3 command trên đang chạy, mở tab trình duyệt của Streamlit (`http://localhost:8501`).
2. Nhập "Tên Bài báo", chọn "Nguồn" (ACL, arXiv, hoặc CVF) và tải lên một file PDF bất kỳ.
3. Bấm **Tải lên (Upload)**.
4. Mở lại VS Code và quan sát **Terminal 3**. Bạn sẽ thấy các bản log thông báo đã phát hiện file mới, tải file thành công và lưu metadata.
5. Kiểm tra thư mục `data/` (đã có file PDF mới) và thư mục `json/` (file JSON đã được thêm 1 dòng mới).
