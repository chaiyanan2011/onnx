# 🤖 Universal ONNX AI API

Backend API สำหรับรันโมเดล ONNX ทุกประเภท บน Render.com (Free Tier)

## 🚀 Deploy บน Render.com

### วิธีที่ 1: Deploy ด้วย Blueprint (แนะนำ)
1. Fork โปรเจคนี้ไป GitHub ของคุณ
2. เข้า [Render Dashboard](https://dashboard.render.com/)
3. Click **New +** → **Blueprint**
4. เลือก repository ของคุณ
5. Render จะอ่าน `render.yaml` และ deploy อัตโนมัติ

### วิธีที่ 2: Deploy แบบ Manual
1. เข้า [Render Dashboard](https://dashboard.render.com/)
2. Click **New +** → **Web Service**
3. เชื่อม GitHub repository
4. ตั้งค่า:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Click **Create Web Service**

## 📁 โครงสร้างโปรเจค

```
onnx_api_project/
├── app.py              # FastAPI Backend
├── requirements.txt    # Python dependencies
├── render.yaml         # Render Blueprint config
├── Procfile            # Process file
├── static/
│   └── index.html      # Frontend UI
└── models/             # โฟลเดอร์เก็บโมเดล ONNX
```

## 📡 API Endpoints

| Endpoint | Method | รายละเอียด |
|----------|--------|-----------|
| `/` | GET | หน้าเว็บหลัก |
| `/api/health` | GET | เช็คสถานะ API |
| `/api/model/upload` | POST | อัปโหลดโมเดล ONNX |
| `/api/models` | GET | แสดงรายการโมเดล |
| `/api/model/{id}/info` | GET | ข้อมูลโมเดล |
| `/api/inference/image` | POST | รัน inference ด้วยรูปภาพ |
| `/api/inference/video` | POST | รัน inference ด้วยวิดีโอ |
| `/api/inference/audio` | POST | รัน inference ด้วยเสียง |
| `/api/inference/text` | POST | รัน inference ด้วยข้อความ |
| `/api/inference/numeric` | POST | รัน inference ด้วยตัวเลข |
| `/api/inference/random` | POST | รัน inference ด้วย random data |

## 📝 ตัวอย่างการใช้งาน API

### อัปโหลดโมเดล
```bash
curl -X POST "https://your-api.onrender.com/api/model/upload" \
  -F "file=@yolo11n.onnx"
```

### รัน Inference ด้วยรูปภาพ
```bash
curl -X POST "https://your-api.onrender.com/api/inference/image" \
  -F "model_id=yolo11n" \
  -F "file=@image.jpg" \
  -F "target_size=640,640" \
  -F "normalize=01"
```

### รัน Inference ด้วยข้อความ
```bash
curl -X POST "https://your-api.onrender.com/api/inference/text" \
  -F "model_id=bert_model" \
  -F "text=Hello World" \
  -F "max_length=512"
```

## 🛠️ รัน Local

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python app.py

# หรือ
uvicorn app:app --reload --port 10000
```

เข้า `http://localhost:10000` บนเบราว์เซอร์

## ⚠️ ข้อจำกัด Free Tier (Render)
- RAM: 512 MB
- CPU: Shared
- Sleep after 15 min inactivity (spin up ~30 sec)
- Disk: Ephemeral (ไฟล์จะหายเมื่อ redeploy)

**แนะนำ**: สำหรับโมเดลใหญ่ (>100MB) ควรใช้ Paid Plan หรือรัน local
