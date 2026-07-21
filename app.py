from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import json
import os
import base64
from typing import List, Optional
import tempfile
import cv2

app = FastAPI(title="Universal ONNX API", version="1.0.0")

# CORS ให้หน้าเว็บยิงมาได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# เก็บโมเดลที่โหลดไว้
loaded_models = {}

# ==================== MOUNT STATIC FILES ====================
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==================== HELPER FUNCTIONS ====================

def get_model_info(model_path):
    """ดึงข้อมูลโมเดล ONNX"""
    try:
        session = ort.InferenceSession(model_path)
        inputs = []
        outputs = []

        for inp in session.get_inputs():
            inputs.append({
                "name": inp.name,
                "shape": list(inp.shape),
                "type": str(inp.type)
            })

        for out in session.get_outputs():
            outputs.append({
                "name": out.name,
                "shape": list(out.shape),
                "type": str(out.type)
            })

        return {
            "inputs": inputs,
            "outputs": outputs,
            "providers": session.get_providers()
        }
    except Exception as e:
        return {"error": str(e)}

def preprocess_image(image_bytes, target_size=(640, 640), normalize="01", channel_order="rgb"):
    """Preprocess รูปภาพสำหรับ ONNX"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype(np.float32)

    # Normalize
    if normalize == "imagenet":
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array / 255.0 - mean) / std
    elif normalize == "01":
        img_array = img_array / 255.0
    elif normalize == "11":
        img_array = img_array / 127.5 - 1.0

    # Channel order
    if channel_order == "bgr":
        img_array = img_array[:, :, ::-1]

    # HWC -> CHW
    img_array = np.transpose(img_array, (2, 0, 1))
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def preprocess_video(video_bytes, target_size=(224, 224), max_frames=10):
    """Preprocess วิดีโอสำหรับ ONNX"""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    frames = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // max_frames)

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = np.transpose(frame, (2, 0, 1))
            frames.append(frame)
        frame_count += 1

    cap.release()
    os.remove(tmp_path)

    if len(frames) == 0:
        raise ValueError("Could not extract frames from video")

    # Pad if needed
    while len(frames) < max_frames:
        frames.append(frames[-1])

    return np.array(frames).astype(np.float32)

def preprocess_audio(audio_bytes, sample_rate=16000, duration=10):
    """Preprocess เสียงสำหรับ ONNX (simplified)"""
    # สำหรับตัวอย่างนี้ จะสร้าง dummy audio data
    # ใน production ควรใช้ librosa หรือ soundfile
    target_samples = sample_rate * duration
    return np.random.randn(1, target_samples).astype(np.float32)

def preprocess_text(text, max_length=512):
    """Preprocess ข้อความสำหรับ ONNX (simplified)"""
    # Simple tokenization - ใน production ควรใช้ tokenizer จริง
    tokens = text.lower().split()[:max_length]
    indices = [hash(t) % 30000 for t in tokens]
    while len(indices) < max_length:
        indices.append(0)
    return np.array([indices]).astype(np.int64)

# ==================== API ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """เสิร์ฟหน้าเว็บหลัก"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/health")
async def health_check():
    """เช็คสถานะ API"""
    return {"status": "ok", "onnx_runtime": ort.get_device()}

@app.post("/api/model/upload")
async def upload_model(file: UploadFile = File(...)):
    """อัปโหลดโมเดล ONNX"""
    if not file.filename.endswith(".onnx"):
        raise HTTPException(400, "Only .onnx files allowed")

    model_id = file.filename.replace(".onnx", "")
    model_path = f"models/{file.filename}"

    with open(model_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # โหลดโมเดลและเก็บไว้
    try:
        session = ort.InferenceSession(model_path)
        loaded_models[model_id] = {
            "session": session,
            "path": model_path,
            "info": get_model_info(model_path)
        }
        return {
            "success": True,
            "model_id": model_id,
            "info": loaded_models[model_id]["info"]
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {str(e)}")

@app.get("/api/models")
async def list_models():
    """แสดงรายการโมเดลที่โหลดไว้"""
    return {
        "models": [
            {
                "id": k,
                "info": v["info"]
            } for k, v in loaded_models.items()
        ]
    }

@app.get("/api/model/{model_id}/info")
async def get_model_info_endpoint(model_id: str):
    """ดูข้อมูลโมเดล"""
    if model_id not in loaded_models:
        raise HTTPException(404, "Model not found")
    return loaded_models[model_id]["info"]

@app.post("/api/inference/image")
async def inference_image(
    model_id: str = Form(...),
    file: UploadFile = File(...),
    target_size: str = Form("640,640"),
    normalize: str = Form("01"),
    channel_order: str = Form("rgb")
):
    """รัน inference ด้วยรูปภาพ"""
    if model_id not in loaded_models:
        raise HTTPException(404, "Model not found")

    session = loaded_models[model_id]["session"]
    image_bytes = await file.read()

    w, h = map(int, target_size.split(","))
    input_data = preprocess_image(image_bytes, (w, h), normalize, channel_order)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})

    return {
        "success": True,
        "outputs": [
            {
                "name": session.get_outputs()[i].name,
                "shape": list(out.shape),
                "preview": out.flatten()[:20].tolist()
            }
            for i, out in enumerate(outputs)
        ]
    }

@app.post("/api/inference/video")
async def inference_video(
    model_id: str = Form(...),
    file: UploadFile = File(...),
    target_size: str = Form("224,224"),
    max_frames: int = Form(10)
):
    """รัน inference ด้วยวิดีโอ"""
    if model_id not in loaded_models:
        raise HTTPException(404, "Model not found")

    session = loaded_models[model_id]["session"]
    video_bytes = await file.read()

    w, h = map(int, target_size.split(","))
    input_data = preprocess_video(video_bytes, (w, h), max_frames)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})

    return {
        "success": True,
        "outputs": [
            {
                "name": session.get_outputs()[i].name,
                "shape": list(out.shape),
                "preview": out.flatten()[:20].tolist()
            }
            for i, out in enumerate(outputs)
        ]
    }

@app.post("/api/inference/audio")
async def inference_audio(
    model_id: str = Form(...),
    file: UploadFile = File(...),
    sample_rate: int = Form(16000),
    duration: int = Form(10)
):
    """รัน inference ด้วยเสียง"""
    if model_id not in loaded_models:
        raise HTTPException(404, "Model not found")

    session = loaded_models[model_id]["session"]
    audio_bytes = await file.read()

    input_data = preprocess_audio(audio_bytes, sample_rate, duration)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})

    return {
        "success": True,
        "outputs": [
            {
                "name": session.get_outputs()[i].name,
                "shape": list(out.shape),
                "preview": out.flatten()[:20].tolist()
            }
            for i, out in enumerate(outputs)
        ]
    }

@app.post("/api/inference/text")
async def inference_text(
    model_id: str = Form(...),
    text: str = Form(...),
    max_length: int = Form(512)
):
    """รัน inference ด้วยข้อความ"""
    if model_id not in loaded_models:
        raise HTTPException(404, "Model not found")

    session = loaded_models[model_id]["session"]
    input_data = preprocess_text(text, max_length)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})

    return {
        "success": True,
        "outputs": [
            {
                "name": session.get_outputs()[i].name,
                "shape": list(out.shape),
                "preview": out.flatten()[:20].tolist()
            }
            for i, out in enumerate(outputs)
        ]
    }

@app.post("/api/inference/numeric")
async def inference_numeric(
    model_id: str = Form(...),
    data: str = Form(...),  # JSON string of array
    shape: Optional[str] = Form(None),
    dtype: str = Form("float32")
):
    """รัน inference ด้วยข้อมูลตัวเลข"""
    if model_id not in loaded_models:
        raise HTTPException(404, "Model not found")

    session = loaded_models[model_id]["session"]
    values = json.loads(data)

    if dtype == "float32":
        arr = np.array(values, dtype=np.float32)
    elif dtype == "int32":
        arr = np.array(values, dtype=np.int32)
    elif dtype == "int64":
        arr = np.array(values, dtype=np.int64)
    else:
        arr = np.array(values, dtype=np.float32)

    if shape:
        arr = arr.reshape(list(map(int, shape.split(","))))

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: arr})

    return {
        "success": True,
        "outputs": [
            {
                "name": session.get_outputs()[i].name,
                "shape": list(out.shape),
                "preview": out.flatten()[:20].tolist()
            }
            for i, out in enumerate(outputs)
        ]
    }

@app.post("/api/inference/random")
async def inference_random(
    model_id: str = Form(...)
):
    """รัน inference ด้วย random data"""
    if model_id not in loaded_models:
        raise HTTPException(404, "Model not found")

    session = loaded_models[model_id]["session"]
    input_info = session.get_inputs()[0]

    shape = [1 if s == "None" or s is None or (isinstance(s, str) and "N" in str(s)) else int(s) for s in input_info.shape]

    if "int" in str(input_info.type).lower():
        input_data = np.random.randint(0, 100, size=shape).astype(np.int64)
    else:
        input_data = np.random.randn(*shape).astype(np.float32)

    outputs = session.run(None, {input_info.name: input_data})

    return {
        "success": True,
        "outputs": [
            {
                "name": session.get_outputs()[i].name,
                "shape": list(out.shape),
                "preview": out.flatten()[:20].tolist()
            }
            for i, out in enumerate(outputs)
        ]
    }

# Load existing models on startup
@app.on_event("startup")
async def load_existing_models():
    """โหลดโมเดลที่มีอยู่ในโฟลเดอร์ models"""
    os.makedirs("models", exist_ok=True)
    for filename in os.listdir("models"):
        if filename.endswith(".onnx"):
            model_id = filename.replace(".onnx", "")
            model_path = f"models/{filename}"
            try:
                session = ort.InferenceSession(model_path)
                loaded_models[model_id] = {
                    "session": session,
                    "path": model_path,
                    "info": get_model_info(model_path)
                }
                print(f"Loaded model: {model_id}")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
