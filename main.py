# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import io
import numpy as np
import tensorflow as tf
import soundfile as sf
import httpx
import re
import os
import requests
import tempfile


# ==============================================================
# 🔧 CONFIGURATION
# ==============================================================

# Google Drive file IDs (replace with yours if needed)
# Google Drive file IDs
YAMNET_ID = "1bQ9NK5TIJsO9bgPJ7uEBUmtIbiZsqD0J"  # yamnet_with_embeddings.tflite
BLOW_ID = "1Jt1OmsprGF8ciKTKY3YeJMfrxlUbWaRj"    # blow_classifier_compact_ffnine16.tflite

MODEL_DIR = "models"
YAMNET_PATH = os.path.join(MODEL_DIR, "yamnet_with_embeddings.tflite")
BLOW_PATH = os.path.join(MODEL_DIR, "blow_classifier_compact_ffnine16.tflite")
# ==============================================================
# 🚀 FASTAPI APP
# ==============================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://blowith-frontend.onrender.com", "https://blowithback.onrender.com"

    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ==============================================================
# 📦 UTIL: DOWNLOAD MODEL FROM GOOGLE DRIVE
# ==============================================================

async def download_from_gdrive(file_id: str, destination: str):
    """Download a file from Google Drive to the destination path if not already present."""
    if os.path.exists(destination):
        print(f"✅ Model already cached: {destination}")
        return

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    print(f"🚀 Starting model setup for {destination}...")
    
    try:
        # Use requests instead of httpx for better cookie handling
        import requests
        
        URL = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        print(f"⬇️ Downloading model from {URL} ...")
        
        session = requests.Session()
        
        # Initial request to get the confirmation token
        response = session.get(URL, stream=True)
        
        if response.status_code == 200:
            # Check if we got a confirmation page (for large files)
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    # We need to confirm the download
                    URL = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
                    response = session.get(URL, stream=True)
                    break
            
            # Now download the file
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                with open(destination, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            downloaded_size += len(chunk)
                            if total_size > 0:
                                percent = (downloaded_size / total_size) * 100
                                print(f"📥 Download progress: {percent:.1f}%", end='\r')
                
                print(f"\n✅ Model saved: {destination} ({downloaded_size} bytes)")
                return
            
        raise Exception(f"Failed to download model (status {response.status_code})")
        
    except Exception as e:
        print(f"❌ Download error: {e}")
        raise RuntimeError(f"Error downloading model from Google Drive: {e}")
# ==============================================================
# 🧠 MODEL LOADING
# ==============================================================

@app.on_event("startup")
async def load_models():
    """Load both TFLite models on app startup."""
    print("🚀 Starting model setup...")
    await download_from_gdrive(YAMNET_ID, YAMNET_PATH)
    await download_from_gdrive(BLOW_ID, BLOW_PATH)

    global yamnet, blow
    yamnet = tf.lite.Interpreter(model_path=YAMNET_PATH)
    yamnet.allocate_tensors()

    blow = tf.lite.Interpreter(model_path=BLOW_PATH)
    blow.allocate_tensors()

    print("✅ All models loaded and ready!")



# ==============================================================
# 🎤 CLASSIFY ENDPOINT
# ==============================================================
@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    try:
        # ✅ Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # ✅ Preprocess the audio (convert to mono, float32)
        data, sr = sf.read(tmp_path, dtype="float32")
        if len(data.shape) > 1:  # Convert stereo → mono
            data = np.mean(data, axis=1)

        # ✅ Normalize audio to [-1, 1]
        data = np.clip(data / np.max(np.abs(data)), -1.0, 1.0)

        # ✅ Load models
        yamnet = tf.lite.Interpreter(model_path="yamnet.tflite")
        yamnet.allocate_tensors()
        blow = tf.lite.Interpreter(model_path="blow_classifier.tflite")
        blow.allocate_tensors()

        # ---------------------------------------------------------------
        # 🔥 YamNet forward pass (robust to 1D vs [1, N] inputs)
        # ---------------------------------------------------------------
        input_details = yamnet.get_input_details()
        output_details = yamnet.get_output_details()
        expected_shape = list(input_details[0]["shape"])

        # Handle dynamic shape (-1)
        if any(d == -1 for d in expected_shape):
            try:
                yamnet.resize_tensor_input(input_details[0]["index"], [int(len(data))])
                yamnet.allocate_tensors()
                input_details = yamnet.get_input_details()
                expected_shape = list(input_details[0]["shape"])
            except Exception as e:
                print("⚠️ yamnet.resize failed:", e)

        # ✅ Set tensor correctly depending on model shape
        if len(expected_shape) == 1:
            yamnet.set_tensor(input_details[0]["index"], data.astype(np.float32))
        elif len(expected_shape) == 2:
            yamnet.set_tensor(input_details[0]["index"], data.astype(np.float32).reshape(1, -1))
        else:
            raise HTTPException(status_code=500, detail=f"Unsupported yamnet input shape: {expected_shape}")

        yamnet.invoke()

        # ✅ Get embedding
        yamnet_outputs = yamnet.get_tensor(output_details[0]["index"])
        embedding = yamnet_outputs[0] if yamnet_outputs.ndim > 1 else yamnet_outputs
        embedding = np.asarray(embedding, dtype=np.float32).flatten()

        # ---------------------------------------------------------------
        # 🔥 Blow classifier forward pass
        # ---------------------------------------------------------------
        blow_input = blow.get_input_details()
        blow_output = blow.get_output_details()
        expected_blow = list(blow_input[0]["shape"])

        inp = np.zeros(tuple(expected_blow), dtype=np.float32)
        flat = inp.ravel()
        src = embedding.flatten()
        n = min(flat.size, src.size)
        flat[:n] = src[:n]
        inp = flat.reshape(expected_blow).astype(np.float32)

        blow.set_tensor(blow_input[0]["index"], inp)
        blow.invoke()
        out = blow.get_tensor(blow_output[0]["index"])
        blow_prob = float(np.array(out).flatten()[0])

        # ✅ Cleanup
        os.remove(tmp_path)

        # ✅ Return result
        return {"prediction": "blow" if blow_prob > 0.5 else "no_blow", "probability": blow_prob}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")


# ==============================================================
# 💌 DEDICATION ENDPOINT
# ==============================================================

@app.get("/dedication/{doc_id}")
async def dedications_proxy(doc_id: str):
    url = f"https://blowithback.onrender.com/view/{doc_id}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {e}")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Upstream returned {r.status_code}")

    text = r.text

    def first(group):
        return (group.group(1).strip() if group else None)

    sender = first(re.search(r"From:\s*(.+)", text, re.I))
    receiver = first(re.search(r"To:\s*(.+)", text, re.I)) or first(re.search(r"Receiver:\s*(.+)", text, re.I))
    message = None
    m = re.search(r"Message:\s*([\s\S]*?)(?:\n\n|$)", text, re.I)
    if m:
        message = m.group(1).strip()
    video_name = first(re.search(r"Video:\s*(.+)", text, re.I))

    return {
        "id": doc_id,
        "senderName": sender,
        "receiverName": receiver,
        "message": message,
        "videoId": video_name,
        "raw": text[:500]
    }


# ==============================================================
# 🏠 ROOT
# ==============================================================

@app.get("/")
def home():
    return {"message": "✅ FastAPI ML Service running with Google Drive models!"}


@app.get("/test")
async def test_endpoint():
    """Simple test to verify the service is working"""
    return {
        "status": "ML service is running",
        "models_loaded": True,
        "endpoints": ["/classify", "/dedication/{id}", "/test"]
    }