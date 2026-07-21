from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import json
import os
import time
from typing import Optional
import tempfile
import cv2

app = FastAPI(title="Universal ONNX API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

loaded_models = {}
UPLOAD_TIMEOUT = 300  # 5 นาที

# Static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Models directory
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

def get_model_info(model_path):
    try:
        session = ort.InferenceSession(model_path)
        inputs = []
        outputs = []
        for inp in session.get_inputs():
            shape = [str(s) if s is not None else "?" for s in inp.shape]
            inputs.append({"name": inp.name, "shape": shape, "type": str(inp.type)})
        for out in session.get_outputs():
            shape = [str(s) if s is not None else "?" for s in out.shape]
            outputs.append({"name": out.name, "shape": shape, "type": str(out.type)})
        return {"inputs": inputs, "outputs": outputs, "providers": session.get_providers()}
    except Exception as e:
        return {"error": str(e)}

def preprocess_image(image_bytes, target_size=(640, 640), normalize="01", channel_order="rgb"):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype(np.float32)

    if normalize == "imagenet":
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array / 255.0 - mean) / std
    elif normalize == "01":
        img_array = img_array / 255.0
    elif normalize == "11":
        img_array = img_array / 127.5 - 1.0

    if channel_order == "bgr":
        img_array = img_array[:, :, ::-1]

    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==================== API ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>API Running</h1><a href='/static/index.html'>Go to App</a>"

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "onnx_runtime": ort.get_device(),
        "models_loaded": len(loaded_models),
        "models_dir": models_dir,
        "disk_free_gb": round(os.statvfs(models_dir).f_frsize * os.statvfs(models_dir).f_bavail / (1024**3), 2) if hasattr(os, 'statvfs') else "N/A"
    }

@app.post("/api/model/upload")
async def upload_model(file: UploadFile = File(...)):
    """อัปโหลดโมเดล ONNX พร้อมตรวจสอบ"""
    start_time = time.time()

    if not file.filename.endswith(".onnx"):
        raise HTTPException(400, "Only .onnx files allowed")

    model_id = file.filename.replace(".onnx", "")
    model_path = os.path.join(models_dir, file.filename)

    try:
        # อ่านไฟล์เป็นชิ้นๆ (chunked)
        chunk_size = 1024 * 1024  # 1MB chunks
        with open(model_path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)

        # ตรวจสอบขนาดไฟล์
        file_size = os.path.getsize(model_path)
        if file_size == 0:
            os.remove(model_path)
            raise HTTPException(400, "Uploaded file is empty")

        # โหลดโมเดล
        load_start = time.time()
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        load_time = round(time.time() - load_start, 2)

        loaded_models[model_id] = {
            "session": session,
            "path": model_path,
            "info": get_model_info(model_path),
            "size_mb": round(file_size / 1024 / 1024, 2)
        }

        total_time = round(time.time() - start_time, 2)

        return {
            "success": True,
            "model_id": model_id,
            "file_size_mb": round(file_size / 1024 / 1024, 2),
            "load_time_sec": load_time,
            "total_time_sec": total_time,
            "info": loaded_models[model_id]["info"]
        }

    except Exception as e:
        # ลบไฟล์ถ้าเกิด error
        if os.path.exists(model_path):
            os.remove(model_path)
        raise HTTPException(500, f"Failed: {str(e)}")

@app.get("/api/models")
async def list_models():
    return {
        "models": [{"id": k, "size_mb": v.get("size_mb", 0), "info": v["info"]} for k, v in loaded_models.items()]
    }

@app.delete("/api/model/{model_id}")
async def delete_model(model_id: str):
    """ลบโมเดลเพื่อเคลียร์พื้นที่"""
    if model_id in loaded_models:
        model_path = loaded_models[model_id]["path"]
        if os.path.exists(model_path):
            os.remove(model_path)
        del loaded_models[model_id]
        return {"success": True, "message": f"Model {model_id} deleted"}
    raise HTTPException(404, "Model not found")

# ==================== INFERENCE ENDPOINTS ====================

@app.post("/api/inference/image")
async def inference_image(
    model_id: str = Form(...),
    file: UploadFile = File(...),
    target_size: str = Form("640,640"),
    normalize: str = Form("01"),
    channel_order: str = Form("rgb")
):
    if model_id not in loaded_models:
        raise HTTPException(404, "Model not found. Please upload first.")

    try:
        session = loaded_models[model_id]["session"]
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            raise HTTPException(400, "Empty image file")

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
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {str(e)}")

@app.post("/api/inference/random")
async def inference_random(model_id: str = Form(...)):
    """รัน inference ด้วย random data (ทดสอบเร็ว)"""
    if model_id not in loaded_models:
        raise HTTPException(404, "Model not found")

    try:
        session = loaded_models[model_id]["session"]
        input_info = session.get_inputs()[0]

        shape = []
        for s in input_info.shape:
            if s is None or (isinstance(s, str) and "N" in s):
                shape.append(1)
            else:
                try:
                    shape.append(int(s))
                except:
                    shape.append(1)

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
    except Exception as e:
        raise HTTPException(500, f"Random inference failed: {str(e)}")

# Load existing models on startup
@app.on_event("startup")
async def load_existing_models():
    print(f"Loading models from: {models_dir}")
    count = 0
    for filename in os.listdir(models_dir):
        if filename.endswith(".onnx"):
            model_id = filename.replace(".onnx", "")
            model_path = os.path.join(models_dir, filename)
            try:
                file_size = os.path.getsize(model_path)
                if file_size == 0:
                    os.remove(model_path)
                    continue

                session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                loaded_models[model_id] = {
                    "session": session,
                    "path": model_path,
                    "info": get_model_info(model_path),
                    "size_mb": round(file_size / 1024 / 1024, 2)
                }
                count += 1
                print(f"✅ Loaded: {model_id} ({round(file_size/1024/1024, 2)} MB)")
            except Exception as e:
                print(f"❌ Failed {filename}: {e}")
    print(f"Total models loaded: {count}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=120)
