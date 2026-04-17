# ☁️ Hướng dẫn chạy Backend trên Google Colab (Miễn phí GPU)

## Tổng quan

Chạy backend AI Designer trên **Google Colab** (GPU T4 miễn phí) kết nối với frontend local.

## Cách hoạt động

```
[Máy bạn - Frontend]
        │
        │  https://xxxx.ngrok-free.app/api
        │
        ▼
[Google Colab - Backend + GPU T4]
        │
        │  stabilityai/sdxl-turbo (6GB)
        │  LoRA adapters (nếu có)
        ▼
[HuggingFace Hub - Download models]
```

---

## BƯỚC 1: Chuẩn bị project

### Option A: Clone từ GitHub (khuyến nghị)
```
# Clone repo lên Google Drive (THƯ MỤC: AI_GEN_IMAGE)
!git clone https://github.com/YOUR_USERNAME/AI_GEN.git /content/drive/MyDrive/AI_GEN_IMAGE
cd /content/drive/MyDrive/AI_GEN_IMAGE
```

### Option B: Upload thủ công
1. Nén thư mục `app/` và `requirements.txt` thành ZIP
2. Upload lên Google Drive
3. Trong Colab: Mount Drive → giải nén vào `/content/drive/MyDrive/AI_GEN_IMAGE`

---

## BƯỚC 2: Mở notebook Colab

1. Mở file `colab_backend.ipynb` trong repo (phiên bản **v3** — dùng `nohup` không bị kill)
2. Upload lên Google Colab
3. Chọn **Runtime → Change runtime type → GPU T4**

---

## BƯỚC 3: Thiết lập HuggingFace Token

1. Đăng ký tài khoản **huggingface.co** (miễn phí)
2. Lấy token tại: https://huggingface.co/settings/tokens
3. Nhập token vào cell khi được yêu cầu

**Token free** có thể download SDXL Turbo không giới hạn.

---

## BƯỚC 4: Thiết lập ngrok (tunnel public URL)

1. Đăng ký **ngrok.com** (miễn phí)
2. Lấy Authtoken tại: https://dashboard.ngrok.com/get-started/your-authtoken
3. Nhập Authtoken vào cell khi được yêu cầu

**ngrok free** cho phép tunnel ~2 giờ, đủ cho hầu hết use case.

---

## BƯỚC 5: Chạy tất cả cells

1. **Runtime → Run all** (Ctrl+F9)
2. Đợi ~3-5 phút cho lần đầu (download SDXL Turbo ~6GB)
3. Sau khi xong, copy **public URL** (dạng `https://xxxx.ngrok-free.app`)

---

## BƯỚC 6: Cập nhật frontend

### Cách 1: Qua giao diện (khuyến nghị)
1. Mở frontend → Settings (⚙️)
2. Tìm **Backend URL**
3. Dán URL Colab: `https://xxxx.ngrok-free.app/api`
4. Nhấn **Kết nối**

### Cách 2: Qua file .env
```env
VITE_API_BASE=https://xxxx.ngrok-free.app
```
Sau đó build lại:
```bash
cd frontend
npm run build
```

---

## BƯỚC 7: Mở frontend

```bash
cd frontend
npm run dev
```

Truy cập `http://localhost:5173` — frontend sẽ tự động kết nối Colab backend!

---

## Lưu ý quan trọng

| Vấn đề | Giải pháp |
|---------|-----------|
| ngrok hết hạn sau ~2h | Chạy lại cell BƯỚC 7 trong Colab |
| Colab GPU hết 90 phút | Runtime → Disconnect and delete → Run all lại |
| Cần giữ alive lâu | Dùng **Colab Pro** (1000 min/tháng) |
| Model load chậm lần đầu | Bình thường — download 6GB từ HuggingFace |
| Lỗi 500/502 | Reload trang, kiểm tra Colab vẫn đang chạy |
| Không có GPU T4 | Đổi Runtime type → GPU T4 trong Colab |

---

## Nếu muốn dùng LoRA đã train

Upload thư mục `outputs/` lên Google Drive:
```python
# Trong Colab
!cp -r /content/drive/MyDrive/AI_GEN/outputs /content/AI_GEN/outputs
```

Backend sẽ tự động discover LoRA adapters.

---

## Khắc phục lỗi thường gặp

### ❌ Colab bị disconnect liên tục
- **Nguyên nhân:** Không tương tác với Colab quá lâu
- **Giải pháp:** Dùng **Colab Advanced** (tab Chrome mới) hoặc Colab Pro

### ❌ Backend không chạy (trang trắng ngrok)
- **Nguyên nhân:** Server bị crash trước khi ngrok tunnel mở, HOẶC port 8000 bị block
- **Giải pháp:**
  1. Kill tất cả process cũ: `!pkill -f uvicorn`
  2. Restart cell 7 (backend)
  3. Xem log: `!tail -50 /content/drive/MyDrive/AI_GEN/backend.log`

### ❌ Lỗi CORS (dù đã fix)
- **Nguyên nhân:** File `main.py` trên Colab chưa được cập nhật CORS fix
- **Giải pháp:** Chạy **Cell 6 (Fix CORS)** → restart backend

### ❌ Model load MemoryError
- **Nguyên nhân:** GPU T4 Colab hết RAM
- **Giải pháp:** Thử lại, thường là tạm thời do RAM bị chiếm bởi process khác

### ❌ `git clone` bị lỗi
- **Nguyên nhân:** URL sai hoặc chưa thay YOUR_USERNAME
- **Giải pháp:** Đảm bảo dùng URL đúng: `https://github.com/YOUR_USERNAME/AI_GEN.git`

---

## So sánh Local vs Colab

| | Local (máy bạn) | Colab (GPU T4) |
|---|---|---|
| GPU | Tùy máy | **Miễn phí T4** |
| RAM | Hạn chế | ~12-13 GB |
| Model | CPU (chậm) | **GPU (nhanh)** |
| LoRA | Không support | **Hỗ trợ đầy đủ** |
| Runtime | Vô hạn | ~90 phút (free) |
| Cần internet | Không | Có |

**Colab là giải pháp tốt nhất cho máy yếu vì:**
- GPU T4 miễn phí ~10-15 it/s (vs CPU ~0.5 it/s)
- Hỗ trợ đầy đủ LoRA adapters
- SDXL Turbo chạy mượt mà
