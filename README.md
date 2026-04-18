# Design AI / AI Designer System

Hệ thống tạo **logo/poster bằng AI** gồm:
- **Backend FastAPI** (generate, train LoRA, vectorize)
- **Frontend React** (UI kiểu chat)
- **Pipeline dữ liệu + training LoRA**

---

## 1) Project này làm được gì?

### Tính năng chính
1. Nhập prompt -> tạo ảnh bằng diffusion model (`/api/generate`)
2. Có thể gắn **LoRA đã train** để ra đúng style của dataset
3. Có thể train LoRA từ dữ liệu ảnh + caption (`/api/train`, `/api/train/auto`)
4. Có endpoint vectorize PNG -> SVG (`/api/vectorize`)
5. Frontend chat để thao tác nhanh

### Kết quả đầu ra
- Ảnh trả về dạng base64 qua API
- Có thể bật `save_outputs=true` để lưu PNG vào thư mục `outputs/generated/...`

---

## 2) Cấu trúc thư mục (quan trọng)

```text
AI_GEN/
├── app/
│   ├── main.py                  # FastAPI entry
│   ├── routers/
│   │   ├── generate.py          # /api/generate
│   │   ├── train.py             # /api/train, /api/train/auto
│   │   └── vectorize.py         # /api/vectorize
│   ├── services/
│   │   ├── diffusion.py         # load model, load LoRA, generate
│   │   ├── trainer.py           # train LoRA service
│   │   ├── clip.py              # CLIP score (optional)
│   │   ├── layout.py            # layout analysis
│   │   └── vectorizer.py        # PNG -> SVG
│   └── models/schemas.py        # request/response schema
├── frontend/                    # React + Vite UI
├── training/
│   ├── data_pipeline/           # import/prepare dataset
│   └── lora_trainer/            # script train LoRA
├── dataset/                     # dữ liệu thô (ảnh + txt)
├── dataset_processed/           # dữ liệu đã chuẩn hóa
├── outputs/                     # LoRA + ảnh generate lưu ra
├── kaggle_training.ipynb        # notebook train trên Kaggle
├── requirements.txt
└── README.md
```

---

## 3) Luồng hoạt động tổng quát

## 3.1 Luồng generate (dùng hàng ngày)
1. Frontend gửi prompt đến `POST /api/generate`
2. Backend `diffusion.py`:
   - xác định device (GPU/CPU)
   - load base model
   - nếu có `lora_path` hoặc `DEFAULT_LORA_PATH` thì thử load LoRA
3. Sinh ảnh
4. Trả ảnh base64 + metadata
5. Nếu `save_outputs=true`, backend lưu PNG vào `outputs/generated/...`

## 3.2 Luồng training LoRA
1. Chuẩn bị dữ liệu `dataset/` (ảnh + caption txt)
2. Import thành dataset chuẩn `dataset_processed/`
3. Train LoRA
4. Nhận output tại `outputs/.../final/unet_lora/`
5. Dùng path này trong generate để áp style

---

## 4) Cài đặt môi trường

## 4.1 Yêu cầu
- Python 3.10+
- Node.js 18+
- (Khuyến nghị mạnh) GPU NVIDIA + CUDA cho SDXL + LoRA

## 4.2 Cài backend

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4.3 Cài frontend

```bash
cd frontend
npm install
```

---

## 5) Chạy chuẩn cho người mới

## 5.1 Chạy backend

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
ngrok http 8000
```

Kiểm tra:
- `http://localhost:8000/health`
- `http://localhost:8000/api/model/status`

## 5.2 Chạy frontend

```bash
cd frontend
npm run dev
```

Mở: `http://localhost:5173`

---

## 6) Profile chạy model (rất quan trọng)

Project có profile để cân bằng tốc độ/chất lượng.

### 6.1 GPU chất lượng thật + LoRA SDXL (khuyên dùng)

```bash
$env:LOCAL_INFER_PROFILE="balanced"
$env:DEFAULT_LORA_PATH="C:\Users\.Freelancer\AI_GEN\outputs\lora_poster\final\unet_lora"
uvicorn app.main:app --reload --port 8000
```

Sau đó kiểm tra `/api/model/status` phải thấy:
- `device = cuda`
- model SDXL đang load

### 6.2 CPU test nhanh (chỉ test flow, không phải chất lượng thật)

```bash
$env:LOCAL_INFER_PROFILE="fast"
$env:SKIP_PRIMARY_CPU_MODEL="true"
$env:CPU_DIFFUSION_MODEL="segmind/tiny-sd"
$env:CPU_FALLBACK_MODEL="segmind/tiny-sd"
uvicorn app.main:app --reload --port 8000
```

Lưu ý: CPU fast mode không phù hợp để đánh giá chất lượng ảnh.

---

## 7) Dùng API generate

## 7.1 Request mẫu (GPU + SDXL + LoRA)

```json
{
  "prompt": "poster_style, modern movie poster, cinematic lighting",
  "mode": "quality",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 20,
  "guidance_scale": 7.5,
  "num_images": 1,
  "use_default_lora": true,
  "lora_scale": 1.0,
  "save_outputs": true,
  "output_subdir": "frontend_runs"
}
```

## 7.2 Curl mẫu

```bash
curl -X POST "http://localhost:8000/api/generate" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"poster_style, modern movie poster\",\"mode\":\"quality\",\"width\":1024,\"height\":1024,\"num_inference_steps\":20,\"guidance_scale\":7.5,\"num_images\":1,\"use_default_lora\":true,\"save_outputs\":true,\"output_subdir\":\"manual_test\"}"
```

## 7.3 Ảnh được lưu ở đâu?
- `outputs/generated/<output_subdir>/...png`
- nếu không truyền `output_subdir`, backend tự tạo thư mục timestamp

---

## 8) Train LoRA từ đầu

## 8.1 Chuẩn bị dữ liệu
- `dataset/POSTER (n).png` (hoặc jpg)
- `dataset/POSTER (n).txt` (caption)

## 8.2 Import dữ liệu

```bash
python -m training.data_pipeline.paired_poster_import ^
  --input dataset ^
  --output dataset_processed ^
  --target_size 512 ^
  --merge_short_title
```

## 8.3 Train LoRA

```bash
python -m training.lora_trainer.train_lora ^
  --model_name stabilityai/sdxl-turbo ^
  --dataset dataset_processed ^
  --output outputs/lora_poster ^
  --rank 8 --alpha 16 --lr 1e-4 ^
  --epochs 10 --batch_size 1 --grad_accum 8 ^
  --resolution 512 --save_steps 100 ^
  --validation_prompt "poster_style, modern movie poster, cinematic lighting"
```

Output quan trọng:
- `outputs/lora_poster/final/unet_lora/`

Dùng folder này làm `lora_path` hoặc set `DEFAULT_LORA_PATH`.

---

## 9) Frontend hiện tại

Frontend đã có:
- giao diện chat tên **Design AI**
- theme sáng/tối, icon react-icons
- quick settings cho mode, size, steps, guidance, LoRA scale
- nhập `lora_path` custom
- tải ảnh về máy (download)
- gửi `save_outputs=true` để backend lưu ảnh

---

## 10) FAQ nhanh

### Q1: Vì sao Kaggle nhanh hơn local?
- Kaggle thường chạy GPU + cache model tốt hơn.
- Local CPU dễ thiếu RAM/pagefile khi load model lớn.

### Q2: Vì sao LoRA không áp dụng?
- LoRA SDXL cần base model SDXL tương thích.
- Nếu đang fallback tiny/CPU, LoRA SDXL sẽ bị skip để tránh lỗi.

### Q3: Tôi mới, nên chạy cấu hình nào?
- Nếu có GPU: chạy profile `balanced` + SDXL LoRA.
- Nếu không có GPU: chạy `fast` chỉ để test luồng API/UI.

### Q4: Check hệ thống nhanh thế nào?
1. `/health?warmup=true`
2. `/api/model/status`
3. gửi 1 request generate đơn giản

---

## 11) Gợi ý lộ trình cho người mới

1. Chạy backend + frontend thành công
2. Generate 1 ảnh và thấy ảnh lưu trong `outputs/generated`
3. Test `DEFAULT_LORA_PATH`
4. Hiểu khác biệt giữa CPU fast vs GPU quality
5. Sau đó mới thử train LoRA dataset riêng

---

Nếu bạn muốn, bước tiếp theo mình có thể viết thêm một tài liệu ngắn kiểu **"Checklist debug 5 phút"** (model không load, API lỗi, frontend không kết nối) để bạn tự xử lý rất nhanh khi gặp lỗi.