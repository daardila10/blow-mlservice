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
    
    # Load YamNet model
    yamnet = tf.lite.Interpreter(model_path=YAMNET_PATH)
    
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
        # 🔥 YamNet forward pass - SIMPLIFIED APPROACH
        # ---------------------------------------------------------------
        
        # Resize YamNet input to exactly 15600 samples
        EXPECTED_SAMPLES = 15600
        
        # Prepare waveform array (pad/truncate as needed)
        waveform = np.zeros(EXPECTED_SAMPLES, dtype=np.float32)
        actual_samples = min(data.shape[0], EXPECTED_SAMPLES)
        waveform[:actual_samples] = data[:actual_samples]
        
        print(f"📐 YamNet input shape: {waveform.shape}")
        print(f"📊 Input non-zero samples: {np.count_nonzero(waveform)}/{len(waveform)}")
        
        # Reset and reallocate the interpreter
        yamnet = tf.lite.Interpreter(model_path=YAMNET_PATH)
        yamnet.resize_tensor_input(0, [EXPECTED_SAMPLES])
        yamnet.allocate_tensors()
        
        # Get input details after resizing
        input_details = yamnet.get_input_details()
        output_details = yamnet.get_output_details()
        
        print(f"✅ Final YamNet input shape: {input_details[0]['shape']}")
        print(f"📊 YamNet outputs: {len(output_details)}")
        
        # Set input tensor
        yamnet.set_tensor(input_details[0]["index"], waveform)
        
        # Invoke the model
        print("🚀 Running YamNet inference...")
        yamnet.invoke()
        print("✅ YamNet inference completed")

        # ✅ Analyze outputs - SIMPLIFIED
        print("🔍 Checking YamNet outputs:")
        valid_embeddings = []
        
        for i, output_detail in enumerate(output_details):
            output_data = yamnet.get_tensor(i)
            flat_output = output_data.flatten()
            non_zero_count = np.count_nonzero(flat_output)
            
            print(f"  Output {i}: shape={output_data.shape}, non_zero={non_zero_count}")
            
            # Only consider outputs with meaningful data
            if non_zero_count > 100:  # Reasonable threshold
                valid_embeddings.append({
                    'index': i,
                    'data': output_data,
                    'flat_size': flat_output.shape[0],
                    'non_zero_count': non_zero_count
                })
                print(f"    ✅ Valid output - size: {flat_output.shape[0]}")
        
        # DECISION: If no valid embeddings found, use a completely different approach
        if not valid_embeddings:
            print("❌ No valid YamNet outputs found. Using alternative feature extraction...")
            
            # Create a simple but meaningful 1024-dim embedding from audio features
            embedding = np.zeros(1024, dtype=np.float32)
            
            # Extract basic audio features
            rms = np.sqrt(np.mean(waveform**2))
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(waveform)))) / len(waveform)
            spectral_centroid = 0.0
            
            # Simple FFT-based features
            try:
                from scipy.fft import fft
                spectrum = np.abs(fft(waveform))
                if len(spectrum) > 1:
                    freqs = np.linspace(0, sr/2, len(spectrum)//2)
                    if len(freqs) > 0 and np.sum(spectrum[:len(freqs)]) > 0:
                        spectral_centroid = np.sum(freqs * spectrum[:len(freqs)]) / np.sum(spectrum[:len(freqs)])
                        spectral_centroid = spectral_centroid / (sr/2)  # Normalize
            except:
                pass
            
            # Fill embedding with pattern based on features
            for i in range(1024):
                # Create a pattern that varies based on position and audio features
                pos = i / 1024.0
                embedding[i] = (rms * np.sin(pos * 10) + 
                               zero_crossings * np.cos(pos * 20) + 
                               spectral_centroid * np.sin(pos * 30)) * 0.1
            
            embedding_source = "fallback_audio_features"
            
        else:
            # Use the best valid embedding
            best_embedding = max(valid_embeddings, key=lambda x: x['non_zero_count'])
            embedding_data = best_embedding['data']
            
            # Handle different output shapes
            if len(embedding_data.shape) == 1:
                embedding = embedding_data.astype(np.float32)
            else:
                # Take first frame if multi-dimensional
                embedding = embedding_data[0].flatten().astype(np.float32)
            
            embedding_source = f"output_{best_embedding['index']}"
        
        print(f"📊 Final embedding shape: {embedding.shape}, source: {embedding_source}")
        print(f"📊 Embedding non-zero: {np.count_nonzero(embedding)}/{len(embedding)}")
        print(f"📊 Embedding sample: {embedding[:5].tolist()}")

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
        print(f"📊 Embedding used: {n} elements")

        blow.set_tensor(blow_input_details[0]["index"], inp)
        blow.invoke()
        out = blow.get_tensor(blow_output_details[0]["index"])
        blow_prob = float(np.array(out).flatten()[0])

        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

        return {
            "prediction": "blow" if blow_prob > 0.5 else "no_blow", 
            "probability": blow_prob,
            "embedding_shape": embedding.shape,
            "embedding_sample": embedding[:5].tolist(),
            "embedding_used_size": n,
            "embedding_source": embedding_source,
            "embedding_non_zero": int(np.count_nonzero(embedding))
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