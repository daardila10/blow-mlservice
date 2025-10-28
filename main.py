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

# ==============================================================
# 🔧 CONFIGURATION
# ==============================================================

# Google Drive file IDs (replace with yours if needed)
YAMNET_ID = "1UwuJetN0CksqYHkdAoHgbhNIafCM2kTn"  # example
BLOW_ID = "1abcYourOtherModelIDHere"  # replace with your second model ID

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
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    print(f"⬇️ Downloading model from {url} ...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code != 200:
                raise Exception(f"Failed to download model (status {response.status_code})")
            with open(destination, "wb") as f:
                f.write(response.content)
        print(f"✅ Model saved: {destination}")
    except Exception as e:
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
        contents = await file.read()
        # Convert WebM to WAV
        audio = AudioSegment.from_file(io.BytesIO(contents), format="webm")
        wav_io = io.BytesIO()
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        data, sr = sf.read(wav_io)
        if sr != 16000:
            raise HTTPException(status_code=400, detail="Audio must be 16kHz")

        # Yamnet forward pass
        input_details = yamnet.get_input_details()
        output_details = yamnet.get_output_details()
        yamnet.set_tensor(input_details[0]['index'], np.array([data], dtype=np.float32))
        yamnet.invoke()
        embedding = yamnet.get_tensor(output_details[0]['index'])[0]

        # Blow classifier forward pass
        blow_input = blow.get_input_details()
        blow_output = blow.get_output_details()
        blow.set_tensor(blow_input[0]['index'], np.array([embedding], dtype=np.float32))
        blow.invoke()
        blow_prob = float(blow.get_tensor(blow_output[0]['index'])[0][0])

        return {"blowProb": blow_prob}

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
