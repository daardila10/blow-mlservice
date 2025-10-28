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
# 🎙️ CLASSIFICATION ENDPOINT
# ==============================================================

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    try:
        print(f"🎯 Received audio file: {file.filename}, size: {file.size}")
        
        contents = await file.read()
        print(f"📦 File size after reading: {len(contents)} bytes")
        
        # Convert WebM to WAV
        try:
            audio = AudioSegment.from_file(io.BytesIO(contents), format="webm")
            print(f"🔊 Audio loaded: {len(audio)} ms, {audio.frame_rate} Hz, {audio.channels} channels")
        except Exception as e:
            print(f"❌ WebM conversion failed: {e}")
            # Try other formats
            try:
                audio = AudioSegment.from_file(io.BytesIO(contents))
                print(f"🔊 Audio loaded (auto-detect): {len(audio)} ms, {audio.frame_rate} Hz, {audio.channels} channels")
            except Exception as e2:
                raise HTTPException(status_code=400, detail=f"Unsupported audio format: {e2}")

        # Convert to 16kHz mono
        audio = audio.set_frame_rate(16000).set_channels(1)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        
        # Read WAV data
        data, sr = sf.read(wav_io)
        print(f"🎵 Processed audio: {len(data)} samples, {sr} Hz")
        
        if sr != 16000:
            raise HTTPException(status_code=400, detail="Audio must be 16kHz")
        
        if len(data) < 1000:  # Minimum samples check
            raise HTTPException(status_code=400, detail="Audio too short")
        
        # Ensure we have enough samples for YamNet (pad if needed)
        yamnet_expected_samples = 15600
        if len(data) < yamnet_expected_samples:
            # Pad with zeros
            data = np.pad(data, (0, yamnet_expected_samples - len(data)), mode='constant')
        elif len(data) > yamnet_expected_samples:
            # Truncate
            data = data[:yamnet_expected_samples]
        
        print(f"📊 Final audio shape: {data.shape}")

        # Yamnet forward pass
        input_details = yamnet.get_input_details()
        output_details = yamnet.get_output_details()
        
        print(f"🔧 YamNet input details: {input_details}")
        print(f"🔧 YamNet output details: {output_details}")
        
        yamnet.set_tensor(input_details[0]['index'], np.array([data], dtype=np.float32))
        yamnet.invoke()
        embedding = yamnet.get_tensor(output_details[0]['index'])[0]
        print(f"📐 Embedding shape: {embedding.shape}")

        # Blow classifier forward pass
        blow_input = blow.get_input_details()
        blow_output = blow.get_output_details()
        
        print(f"🔧 Blow classifier input details: {blow_input}")
        print(f"🔧 Blow classifier output details: {blow_output}")
        
        blow.set_tensor(blow_input[0]['index'], np.array([embedding], dtype=np.float32))
        blow.invoke()
        blow_prob = float(blow.get_tensor(blow_output[0]['index'])[0][0])
        
        print(f"🎉 Classification successful! Blow probability: {blow_prob}")
        
        return {"blowProb": blow_prob}

    except Exception as e:
        print(f"💥 Error in classification: {str(e)}")
        import traceback
        print(f"🔍 Stack trace: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

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