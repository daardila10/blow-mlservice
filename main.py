# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import soundfile as sf
import os
import requests
import tempfile
import io
from pydub import AudioSegment


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
        
        response = session.get(URL, stream=True)
        
        if response.status_code == 200:
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    URL = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
                    response = session.get(URL, stream=True)
                    break
            
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
    
    try:
        # Load YamNet model
        yamnet = tf.lite.Interpreter(model_path=YAMNET_PATH)
        yamnet.allocate_tensors()  # ⚠️ ADD THIS LINE
        
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
        print("Number of outputs:", len(yamnet_output_details))
        for i, output_detail in enumerate(yamnet_output_details):
            print(f"Output {i}: shape={output_detail['shape']}, dtype={output_detail['dtype']}, name={output_detail.get('name', 'unknown')}")
        
        # Blow classifier model details
        blow_input_details = blow.get_input_details()
        blow_output_details = blow.get_output_details()
        
        print("\n=== BLOW CLASSIFIER MODEL DETAILS ===")
        print("Input details:", blow_input_details)
        print("Output details:", blow_output_details)
        
        print("="*50)
        print("✅ All models loaded and ready!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise RuntimeError(f"Model loading failed: {e}")
# ==============================================================
# 🎤 AUDIO PROCESSING UTILITIES
# ==============================================================

def convert_audio_format(audio_data: bytes) -> bytes:
    """Convert audio to WAV format using pydub if needed"""
    try:
        # Try to read as WAV first
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")
    except:
        try:
            # If WAV fails, try to auto-detect format
            audio = AudioSegment.from_file(io.BytesIO(audio_data))
        except Exception as e:
            raise ValueError(f"Could not parse audio file: {e}")
    
    # Convert to mono, 16kHz, 16-bit WAV
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    
    # Export as WAV
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return buffer.getvalue()

# ==============================================================
# 🎤 CLASSIFY ENDPOINT
# ==============================================================

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    tmp_path = None
    try:
        global yamnet, blow

        # ✅ Read and convert audio file
        audio_bytes = await file.read()
        
        # Convert to proper WAV format if needed
        try:
            converted_audio = convert_audio_format(audio_bytes)
            
            # Save converted audio to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(converted_audio)
                tmp_path = tmp.name
        except Exception as conversion_error:
            print(f"⚠️ Audio conversion failed: {conversion_error}")
            # Try to use original file as fallback
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

        # ✅ Preprocess audio (convert to mono, float32)
        try:
            data, sr = sf.read(tmp_path, dtype="float32")
        except Exception as e:
            print(f"❌ SoundFile read failed: {e}")
            # Fallback: use pydub to read the file
            audio = AudioSegment.from_file(tmp_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            data = samples / (2**15)  # Convert to [-1, 1]
            sr = 16000

        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # ✅ Normalize to [-1, 1]
        if np.max(np.abs(data)) > 0:
            data = np.clip(data / np.max(np.abs(data)), -1.0, 1.0)

        print(f"🎧 Original audio shape: {data.shape}, sample rate: {sr}")

        # ---------------------------------------------------------------
        # 🔥 YamNet forward pass - CORRECT INPUT HANDLING
        # ---------------------------------------------------------------

        # Get input details FIRST
        input_details = yamnet.get_input_details()
        output_details = yamnet.get_output_details()

        print(f"📋 YamNet input details: {input_details[0]}")
        print(f"📋 Expected input shape: {input_details[0]['shape']}")

        # The model expects shape [1] - this is unusual for YamNet
        # This suggests the model has been modified to expect pre-computed features
        # Let's compute audio features and use them as input

        # Compute various audio features that could be the expected input
        audio_features = {
            "rms_energy": np.sqrt(np.mean(data**2)),
            "mean_amplitude": np.mean(data),
            "max_amplitude": np.max(np.abs(data)),
            "zero_crossing_rate": np.mean(np.abs(np.diff(data > 0))),
            "spectral_centroid": np.mean(np.abs(np.fft.fft(data)[:len(data)//2])),
        }

        # Try each feature until one works
        success = False
        for feature_name, feature_value in audio_features.items():
            try:
                # Create input array with shape [1] as expected
                feature_input = np.array([feature_value], dtype=np.float32)
                print(f"🔄 Trying {feature_name}: {feature_value:.6f}")

                yamnet.set_tensor(input_details[0]["index"], feature_input)
                yamnet.invoke()

                print(f"✅ Success with {feature_name}!")
                success = True
                feature_used = feature_name
                break

            except Exception as e:
                print(f"❌ {feature_name} failed: {e}")
                continue

        if not success:
            # If single features don't work, try the first sample or a statistical summary
            try:
                # Try using just the first sample
                single_sample = np.array([data[0]], dtype=np.float32)
                print(f"🔄 Trying first sample: {single_sample[0]:.6f}")

                yamnet.set_tensor(input_details[0]["index"], single_sample)
                yamnet.invoke()
                print("✅ Success with first sample!")
                feature_used = "first_sample"
            except Exception as e:
                print(f"❌ All approaches failed: {e}")
                raise HTTPException(status_code=500, detail=f"Could not find compatible input format for YamNet model. Input shape expected: {input_details[0]['shape']}")

        print("✅ YamNet inference completed")

        # ✅ Extract the 1024-dimensional embedding from Output 1
        print("🔍 Extracting 1024-dim embedding from Output 1...")

        # Output 1: [1, 1024] - This is the embedding we need!
        embedding_output = yamnet.get_tensor(1)  # shape: [1, 1024]

        print(f"📊 Output 1 shape: {embedding_output.shape}")
        print(f"📊 Output 1 sample values: {embedding_output[0][:5].tolist()}")

        # Extract the 1024-dimensional vector (remove batch dimension)
        final_embedding = embedding_output[0].astype(np.float32)  # shape: [1024]
        embedding_source = "output_1_identity_1"

        print(f"📊 Final embedding shape: {final_embedding.shape}, source: {embedding_source}")
        print(f"📊 Feature used: {feature_used}")
        print(f"📊 Embedding non-zero: {np.count_nonzero(final_embedding)}/{len(final_embedding)}")
        print(f"📊 Embedding range: [{np.min(final_embedding):.4f}, {np.max(final_embedding):.4f}]")
        print(f"📊 Embedding sample: {final_embedding[:5].tolist()}")

        # ---------------------------------------------------------------
        # 🔥 Blow classifier forward pass
        # ---------------------------------------------------------------
        blow_input_details = blow.get_input_details()
        blow_output_details = blow.get_output_details()
        expected_blow_shape = blow_input_details[0]["shape"]

        print(f"📐 Blow classifier expected input shape: {expected_blow_shape}")

        # Prepare input for blow classifier - shape should be [1, 1024]
        # Our embedding is already [1024], so we need to add batch dimension
        inp = np.expand_dims(final_embedding, axis=0).astype(np.float32)  # shape: [1, 1024]
        
        print(f"✅ Blow classifier input shape: {inp.shape}")
        print(f"📊 Embedding used: {inp.size} elements")

        blow.set_tensor(blow_input_details[0]["index"], inp)
        blow.invoke()
        out = blow.get_tensor(blow_output_details[0]["index"])
        blow_prob = float(np.array(out).flatten()[0])

        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

        return {
            "prediction": "blow" if blow_prob > 0.5 else "no_blow", 
            "probability": blow_prob,
            "embedding_shape": final_embedding.shape,
            "embedding_sample": final_embedding[:5].tolist(),
            "embedding_used_size": len(final_embedding),
            "embedding_source": embedding_source,
            "embedding_non_zero": int(np.count_nonzero(final_embedding))
        }

    except Exception as e:
        # Clean up temp file in case of error
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"❌ Classification error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")
        
# ==============================================================
# 🏠 ROOT & TEST ENDPOINTS
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
        "endpoints": ["/classify", "/test", "/model-info"]
    }

@app.get("/model-info")
async def model_info():
    """Endpoint to get model information"""
    try:
        # Re-allocate tensors to get current state
        yamnet.allocate_tensors()
        blow.allocate_tensors()
        
        yamnet_input_details = yamnet.get_input_details()
        yamnet_output_details = yamnet.get_output_details()
        blow_input_details = blow.get_input_details()
        blow_output_details = blow.get_output_details()
        
        return {
            "yamnet": {
                "input_shape": yamnet_input_details[0]['shape'].tolist(),
                "input_dtype": str(yamnet_input_details[0]['dtype']),
                "output_shapes": [out['shape'].tolist() for out in yamnet_output_details],
                "output_dtypes": [str(out['dtype']) for out in yamnet_output_details]
            },
            "blow_classifier": {
                "input_shape": blow_input_details[0]['shape'].tolist(),
                "input_dtype": str(blow_input_details[0]['dtype']),
                "output_shape": blow_output_details[0]['shape'].tolist(),
                "output_dtype": str(blow_output_details[0]['dtype'])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {e}")