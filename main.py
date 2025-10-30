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
        "https://blowith-frontend.onrender.com", 
        "https://blowithback.onrender.com"
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
# 🧠 MODEL LOADING & DEBUG INFO
# ==============================================================

@app.on_event("startup")
async def load_models():
    """Load both TFLite models on app startup."""
    print("🚀 Starting model setup...")
    await download_from_gdrive(YAMNET_ID, YAMNET_PATH)
    await download_from_gdrive(BLOW_ID, BLOW_PATH)

    global yamnet, blow
    
    # Load YamNet model
    yamnet = tf.lite.Interpreter(model_path=YAMNET_PATH)
    yamnet.allocate_tensors()

    # Load Blow classifier model
    blow = tf.lite.Interpreter(model_path=BLOW_PATH)
    blow.allocate_tensors()

    # Debug code to check model details
    print("\n" + "="*50)
    print("🔍 MODEL DEBUG INFORMATION")
    print("="*50)
    
    # YamNet model details
    yamnet_input_details = yamnet.get_input_details()
    yamnet_output_details = yamnet.get_output_details()
    
    print("=== YAMNET MODEL DETAILS ===")
    print("Input details:", yamnet_input_details)
    print("Output details:", yamnet_output_details)
    
    # Blow classifier model details
    blow_input_details = blow.get_input_details()
    blow_output_details = blow.get_output_details()
    
    print("\n=== BLOW CLASSIFIER MODEL DETAILS ===")
    print("Input details:", blow_input_details)
    print("Output details:", blow_output_details)
    
    print("="*50)
    print("✅ All models loaded and ready!")

# ==============================================================
# 🎤 CLASSIFY ENDPOINT
# ==============================================================

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    try:
        global yamnet, blow

        # ✅ Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # ✅ Preprocess audio (convert to mono, float32)
        data, sr = sf.read(tmp_path, dtype="float32")
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # ✅ Normalize to [-1, 1]
        if np.max(np.abs(data)) > 0:
            data = np.clip(data / np.max(np.abs(data)), -1.0, 1.0)

        print(f"🎧 Original audio shape: {data.shape}, sample rate: {sr}")

        # ---------------------------------------------------------------
        # 🔥 YamNet forward pass
        # ---------------------------------------------------------------
        input_details = yamnet.get_input_details()
        output_details = yamnet.get_output_details()
        expected_shape = input_details[0]["shape"]
        
        print(f"📐 YamNet expected input shape: {expected_shape}")

        # Standard YamNet expects 0.975 seconds of audio at 16kHz = 15600 samples
        target_length = 15600
        
        # Resample to 16kHz if needed
        if sr != 16000:
            from scipy import signal
            data = signal.resample(data, int(len(data) * 16000 / sr))
            sr = 16000
            print(f"🔄 Resampled to 16kHz, new shape: {data.shape}")

        # Handle audio length
        if len(data) > target_length:
            # Use middle segment for consistency
            start = (len(data) - target_length) // 2
            data = data[start:start + target_length]
        elif len(data) < target_length:
            # Pad if too short
            data = np.pad(data, (0, target_length - len(data)), mode='constant')
        
        # Ensure correct shape for YamNet: [batch_size, audio_samples]
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=0)  # Add batch dimension
        
        print(f"✅ Final YamNet input shape: {data.shape}, dtype: {data.dtype}")

        # ✅ Feed into YamNet
        yamnet.set_tensor(input_details[0]["index"], data.astype(np.float32))
        yamnet.invoke()

        # ✅ Extract embedding
        yamnet_outputs = yamnet.get_tensor(output_details[0]["index"])
        embedding = yamnet_outputs.flatten().astype(np.float32)

        print(f"📊 YamNet embedding shape: {embedding.shape}")

        # ---------------------------------------------------------------
        # 🔥 Blow classifier forward pass
        # ---------------------------------------------------------------
        blow_input_details = blow.get_input_details()
        blow_output_details = blow.get_output_details()
        expected_blow_shape = blow_input_details[0]["shape"]

        print(f"📐 Blow classifier expected input shape: {expected_blow_shape}")

        # Prepare input for blow classifier
        inp = np.zeros(tuple(expected_blow_shape), dtype=np.float32)
        flat = inp.ravel()
        src = embedding.flatten()
        n = min(flat.size, src.size)
        flat[:n] = src[:n]
        inp = flat.reshape(expected_blow_shape).astype(np.float32)

        print(f"✅ Blow classifier input shape: {inp.shape}")

        blow.set_tensor(blow_input_details[0]["index"], inp)
        blow.invoke()
        out = blow.get_tensor(blow_output_details[0]["index"])
        blow_prob = float(np.array(out).flatten()[0])

        os.remove(tmp_path)

        return {
            "prediction": "blow" if blow_prob > 0.5 else "no_blow", 
            "probability": blow_prob,
            "embedding_shape": embedding.shape
        }

    except Exception as e:
        # Clean up temp file in case of error
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"❌ Classification error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")

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
        "endpoints": ["/classify", "/test"]
    }

@app.get("/model-info")
async def model_info():
    """Endpoint to get model information"""
    try:
        yamnet_input_details = yamnet.get_input_details()
        yamnet_output_details = yamnet.get_output_details()
        blow_input_details = blow.get_input_details()
        blow_output_details = blow.get_output_details()
        
        return {
            "yamnet": {
                "input": yamnet_input_details,
                "output": yamnet_output_details
            },
            "blow_classifier": {
                "input": blow_input_details,
                "output": blow_output_details
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {e}")