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

        # Validate YamNet input shape - it should be reasonable for audio
        if len(expected_shape) == 1 and expected_shape[0] == 1:
            print("⚠️ Warning: YamNet input shape seems incorrect [1]. This might be a model issue.")
            # Try to infer correct shape based on standard YamNet
            # Standard YamNet usually expects [batch_size, audio_samples] where audio_samples > 10000
            expected_shape = [1, 15600]  # Common YamNet input length
            print(f"🔄 Using inferred YamNet input shape: {expected_shape}")

        # Handle dynamic shape or incorrect shape
        if -1 in expected_shape or expected_shape[0] == 1:
            try:
                # Use standard YamNet input length (15600 samples = ~0.975s at 16kHz)
                target_length = 15600
                if len(data) > target_length:
                    # Use middle segment for consistency
                    start = (len(data) - target_length) // 2
                    data = data[start:start + target_length]
                else:
                    # Pad if too short
                    data = np.pad(data, (0, target_length - len(data)), mode='constant')
                
                target_shape = [1, target_length]
                print(f"🔄 Resizing YamNet input to: {target_shape}")
                yamnet.resize_tensor_input(input_details[0]["index"], target_shape)
                yamnet.allocate_tensors()
                # Update input details after resizing
                input_details = yamnet.get_input_details()
                expected_shape = input_details[0]["shape"]
                print(f"✅ New YamNet input shape: {expected_shape}")
            except Exception as e:
                print(f"❌ YamNet resize failed: {e}")
                os.remove(tmp_path)
                raise HTTPException(status_code=500, detail=f"Model input resize failed: {e}")

        # ✅ Ensure correct shape for YamNet input
        data = data.astype(np.float32)
        
        # Reshape data to match expected shape
        if list(data.shape) != list(expected_shape):
            print(f"🔄 Reshaping data from {data.shape} to {expected_shape}")
            try:
                if len(expected_shape) == 1:
                    # If model expects 1D but we have the right length
                    if data.shape[0] == expected_shape[0]:
                        data = data.flatten()
                    else:
                        # Truncate or pad to expected length
                        if data.shape[0] > expected_shape[0]:
                            data = data[:expected_shape[0]]
                        else:
                            data = np.pad(data, (0, expected_shape[0] - data.shape[0]), mode='constant')
                elif len(expected_shape) == 2:
                    # Add batch dimension if needed
                    if len(data.shape) == 1:
                        data = np.expand_dims(data, axis=0)
                    # Handle length mismatch
                    if data.shape[1] > expected_shape[1]:
                        data = data[:, :expected_shape[1]]
                    elif data.shape[1] < expected_shape[1]:
                        padding = expected_shape[1] - data.shape[1]
                        data = np.pad(data, ((0, 0), (0, padding)), mode='constant')
            except Exception as reshape_error:
                print(f"❌ Reshaping failed: {reshape_error}")
                os.remove(tmp_path)
                raise HTTPException(status_code=500, detail=f"Cannot reshape audio from {data.shape} to {expected_shape}")

        print(f"✅ Final YamNet input shape: {data.shape}")

        # ✅ Feed into YamNet
        yamnet.set_tensor(input_details[0]["index"], data)
        yamnet.invoke()

        # ✅ Extract embedding
        yamnet_outputs = yamnet.get_tensor(output_details[0]["index"])
        embedding = yamnet_outputs[0] if yamnet_outputs.ndim > 1 else yamnet_outputs
        embedding = np.asarray(embedding, dtype=np.float32).flatten()

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
        "endpoints": ["/classify", "/dedication/{id}", "/test"]
    }