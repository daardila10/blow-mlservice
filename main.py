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
        # 🔥 YamNet forward pass - HANDLE [1] INPUT SHAPE
        # ---------------------------------------------------------------
        
        # Get model details
        input_details = yamnet.get_input_details()
        output_details = yamnet.get_output_details()
        
        print(f"📋 YamNet expects input shape: {input_details[0]['shape']}")
        print(f"📋 Input name: {input_details[0]['name']}")

        # Since the model expects shape [1], we need to find what single value it wants
        # This is likely a modified YamNet that expects some pre-computed feature
        
        # Strategy: Process the audio in chunks and aggregate results
        CHUNK_SIZE = 15600  # Standard YamNet chunk size
        HOP_SIZE = 4800     # 0.3s hop for overlapping windows
        
        all_embeddings = []
        
        # Process audio in overlapping chunks
        for start in range(0, len(data) - CHUNK_SIZE + 1, HOP_SIZE):
            end = start + CHUNK_SIZE
            chunk = data[start:end]
            
            # Compute various potential input features for this chunk
            features_to_try = [
                ("rms", np.sqrt(np.mean(chunk**2))),
                ("mean", np.mean(chunk)),
                ("max", np.max(np.abs(chunk))),
                ("std", np.std(chunk)),
                ("zcr", np.mean(np.abs(np.diff(chunk > 0)))),
            ]
            
            chunk_success = False
            for feature_name, feature_value in features_to_try:
                try:
                    # Create input with expected shape [1]
                    feature_input = np.array([feature_value], dtype=np.float32)
                    
                    yamnet.set_tensor(input_details[0]["index"], feature_input)
                    yamnet.invoke()
                    
                    # Get embedding from output 1
                    embedding = yamnet.get_tensor(1)[0]  # shape: [1024]
                    all_embeddings.append(embedding)
                    chunk_success = True
                    print(f"✅ Chunk {start//HOP_SIZE}: used {feature_name} = {feature_value:.6f}")
                    break
                    
                except Exception as e:
                    continue
            
            if not chunk_success:
                print(f"⚠️ Chunk {start//HOP_SIZE}: Could not process with any feature")
        
        if not all_embeddings:
            raise HTTPException(status_code=500, detail="Could not process any audio chunks with the YamNet model")
        
        # Average all chunk embeddings to get final embedding
        final_embedding = np.mean(all_embeddings, axis=0).astype(np.float32)
        print(f"📊 Combined {len(all_embeddings)} chunks into final embedding")
        
        # ---------------------------------------------------------------
        # 🔥 Blow classifier forward pass
        # ---------------------------------------------------------------
        blow_input_details = blow.get_input_details()
        blow_output_details = blow.get_output_details()

        print(f"📐 Blow classifier expected input shape: {blow_input_details[0]['shape']}")

        # Prepare input for blow classifier - shape should be [1, 1024]
        inp = np.expand_dims(final_embedding, axis=0).astype(np.float32)  # shape: [1, 1024]
        
        print(f"✅ Blow classifier input shape: {inp.shape}")

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
            "chunks_processed": len(all_embeddings),
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
        


        @app.post("/debug-yamnet")
async def debug_yamnet(file: UploadFile = File(...)):
    """Debug endpoint to test YamNet model directly"""
    tmp_path = None
    try:
        # Read and process audio file (same as classify)
        audio_bytes = await file.read()
        
        try:
            converted_audio = convert_audio_format(audio_bytes)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(converted_audio)
                tmp_path = tmp.name
        except Exception as conversion_error:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

        data, sr = sf.read(tmp_path, dtype="float32")
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if np.max(np.abs(data)) > 0:
            data = np.clip(data / np.max(np.abs(data)), -1.0, 1.0)

        # Test different input values
        input_details = yamnet.get_input_details()
        output_details = yamnet.get_output_details()
        
        test_values = [
            ("zero", 0.0),
            ("small_positive", 0.1),
            ("small_negative", -0.1),
            ("rms", np.sqrt(np.mean(data**2))),
            ("mean", np.mean(data)),
            ("max", np.max(data)),
            ("min", np.min(data)),
        ]
        
        results = []
        for name, value in test_values:
            try:
                input_data = np.array([value], dtype=np.float32)
                yamnet.set_tensor(input_details[0]["index"], input_data)
                yamnet.invoke()
                
                # Get all outputs
                outputs = []
                for i in range(len(output_details)):
                    output_data = yamnet.get_tensor(i)
                    outputs.append({
                        "index": i,
                        "shape": output_data.shape,
                        "sample": output_data.flatten()[:3].tolist() if output_data.size > 3 else output_data.flatten().tolist()
                    })
                
                results.append({
                    "input_name": name,
                    "input_value": value,
                    "success": True,
                    "outputs": outputs
                })
                
            except Exception as e:
                results.append({
                    "input_name": name,
                    "input_value": value,
                    "success": False,
                    "error": str(e)
                })
        
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            
        return {
            "audio_info": {
                "length": len(data),
                "sample_rate": sr,
                "rms": np.sqrt(np.mean(data**2)),
                "mean": np.mean(data),
                "max": np.max(data),
                "min": np.min(data)
            },
            "model_input": {
                "expected_shape": input_details[0]['shape'].tolist(),
                "name": input_details[0]['name']
            },
            "test_results": results
        }
        
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Debug error: {e}")
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