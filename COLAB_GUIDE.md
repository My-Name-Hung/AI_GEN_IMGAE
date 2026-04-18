# Hướng dẫn chạy Backend trên Cloud (Colab / Kaggle)

## Tổng quan

Chạy backend AI Designer trên **Google Colab** hoặc **Kaggle** (GPU miễn phí) kết nối với frontend local.

## Cách hoạt động

```
[Máy bạn - Frontend]
        │
        │  https://xxxx.ngrok-free.app/api
        ▼
[Google Colab / Kaggle - Backend + GPU]
        │
        │  stabilityai/sdxl-turbo (6GB)
        │  LoRA adapters (nếu có)
        ▼
[HuggingFace Hub - Download models]
```

---

## So sánh Colab vs Kaggle

| | Google Colab | Kaggle |
|---|---|---|
| GPU | T4 miễn phí | T4/P100 miễn phí |
| VRAM | ~15 GB | ~16 GB |
| Session timeout | ~90 phút (free) | ~9 giờ |
| Dung lượng | Google Drive 15 GB | 20 GB /workspace |
| Dễ setup | ⭐⭐⭐ | ⭐⭐ |
| Package có sẵn | Ít | Nhiều (PyTorch, TF...) |

**Kaggle tốt hơn** vì: session lâu hơn, VRAM nhiều hơn, pre-installed packages nhiều hơn.

---

## BƯỚC 1: Chuẩn bị project

### Upload lên Kaggle (khuyến nghị)

1. Nén thư mục `AI_GEN/` thành file ZIP
2. Vào **kaggle.com → New Dataset → Upload** file ZIP
3. Tạo Notebook mới → **Add Data → Your Datasets → Dataset vừa upload**

### Clone từ GitHub

```bash
# Colab
!git clone https://github.com/YOUR_USERNAME/AI_GEN.git /content/drive/MyDrive/AI_GEN_IMAGE

# Kaggle
!git clone https://github.com/YOUR_USERNAME/AI_GEN.git /kaggle/working/AI_GEN_IMAGE
```

---

## BƯỚC 2: Thiết lập GPU

### Colab
- **Runtime → Change runtime type → GPU T4**

### Kaggle
- **Notebook settings (bên phải) → Accelerator → GPU T4**

---

## BƯỚC 3: Thiết lập HuggingFace Token

1. Đăng ký **huggingface.co** (miễn phí)
2. Lấy token: https://huggingface.co/settings/tokens
3. Nhập token vào cell khi được yêu cầu

Token free cho phép download SDXL Turbo không giới hạn.

---

## BƯỚC 4: Thiết lập ngrok

1. Đăng ký **ngrok.com** (miễn phí)
2. Lấy Authtoken: https://dashboard.ngrok.com/get-started/your-authtoken
3. Nhập Authtoken vào cell

**ngrok free** tunnel ~2 giờ. Kaggle session dài hơn nên cần chú ý thời gian.

---

## BƯỚC 5: Chạy Notebook

### Colab — Dùng `colab_backend.ipynb`
1. Upload notebook lên Google Colab
2. Chạy lần lượt: Cell 1 → 2 → 3 → 4 → 5 → 6
3. Cell 6 giữ chạy — copy public URL

### Kaggle — Dùng `kaggle_backend.ipynb`
1. Upload notebook lên Kaggle (hoặc tạo notebook mới, copy cells)
2. Add Dataset đã upload ở BƯỚC 1
3. Chạy lần lượt: Cell 1 → 2 → 3 → 4 → 5 → 6
4. Cell 6 giữ chạy — copy public URL

**Lần đầu chạy:** Đợi ~5-10 phút để download SDXL Turbo (~6GB).

---

## BƯỚC 6: Cập nhật frontend

### Cách 1: Qua giao diện Settings (khuyến nghị)
1. Mở frontend → Settings (⚙️)
2. Tìm **Backend URL**
3. Dán URL backend: `https://xxxx.ngrok-free.app/api`
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

Truy cập `http://localhost:5173` — frontend sẽ tự kết nối backend!

---

## Lưu ý quan trọng

| Vấn đề | Giải pháp |
|---------|-----------|
| ngrok hết hạn (~2h) | Chạy lại cell backend |
| Colab disconnect | Runtime → Disconnect → Run all |
| Kaggle session hết (~9h) | Chạy lại notebook |
| Model load chậm lần đầu | Bình thường — download 6GB |
| Lỗi 500/502 | Reload trang, kiểm tra backend còn chạy |
| IndentationError layout.py | Pull code mới nhất từ repo |
| Không có GPU | Kiểm tra Runtime settings đã bật GPU chưa |

---

## Khắc phục lỗi thường gặp

### Backend không chạy (502 Bad Gateway)
```bash
# Colab
!pkill -f uvicorn
!cat /content/drive/MyDrive/AI_GEN_IMAGE/backend.log

# Kaggle
!pkill -f uvicorn
!cat /kaggle/working/AI_GEN_IMAGE/backend.log
```
Xem log để tìm lỗi, fix rồi chạy lại cell backend.

### Lỗi CORS
Chạy cell Fix CORS → restart backend cell.

### Lỗi `IndentationError` ở layout.py
Pull code mới nhất từ repo:
```bash
# Colab
!cd /content/drive/MyDrive/AI_GEN_IMAGE && git pull

# Kaggle
!cd /kaggle/working/AI_GEN_IMAGE && git pull
```

### MemoryError trên Colab
Thử chạy lại. Nếu vẫn lỗi, dùng Kaggle thay Colab.

### Kaggle: ModuleNotFoundError
Cài thêm package:
```python
!pip install -q tên-package
```
