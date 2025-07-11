# Import necessary libraries
import base64  # For encoding/decoding audio data to/from base64
import io      # For in-memory byte streams
import modal   # For Modal serverless app management
import numpy as np  # For numerical computations
import requests  # For sending HTTP requests
import torch.nn as nn  # For neural network building
import torchaudio.transforms as T  # Audio preprocessing transforms
import torch  # PyTorch core library
from pydantic import BaseModel  # For request data validation
import soundfile as sf  # For reading/writing sound files
import librosa  # For audio processing (e.g., resampling)

# Import the custom CNN model
from model import AudioCNN

# Initialize the Modal application
app = modal.App("audio-cnn-inference")

# Create the Docker image for deployment, install dependencies, and add source files
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")  # Install Python deps
    .apt_install(["libsndfile1"])  # Install system dependency for sound file handling
    .add_local_python_source("model")  # Include local model file
)

# Load a persistent volume that stores the trained model weights
model_volume = modal.Volume.from_name("esc-model")

# Define a class for preprocessing audio
class AudioProcessor:
    def __init__(self):
        # Create a transformation pipeline: Convert waveform to Mel Spectrogram and then to Decibels
        self.transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=22050,
                n_fft=1024,
                hop_length=512,
                n_mels=128,
                f_min=0,
                f_max=11025
            ),
            T.AmplitudeToDB()
        )

    def process_audio_chunk(self, audio_data):
        # Convert numpy array to PyTorch tensor and add batch dimension
        waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
        # Generate spectrogram
        spectrogram = self.transform(waveform)
        return spectrogram.unsqueeze(0)  # Add another batch/channel dim


# Define the expected format of the incoming inference request
class InferenceRequest(BaseModel):
    audio_data: str  # Base64 encoded WAV audio file


# Modal server class for audio classification
@app.cls(image=image, gpu="A10G", volumes={"/models": model_volume}, scaledown_window=15)

class AudioClassifier:
    # Load model and preprocessing tools when container starts
    @modal.enter()
    def load_model(self):
        print("Loading models on enter")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the saved model checkpoint from volume
        checkpoint = torch.load('/models/best_model.pth', map_location=self.device)
        self.classes = checkpoint['classes']

        # Initialize and load model weights
        self.model = AudioCNN(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Initialize audio processor
        self.audio_processor = AudioProcessor()
        print("Model loaded on enter")

    # Define an HTTP POST endpoint for inference
    @modal.fastapi_endpoint(method="POST")
    def inference(self, request: InferenceRequest):
        # production: frontend -> upload file to s3 -> inference endpoint -> download from s3 bucket
        # frontend -> send file directly -> inference endpoint
        
        # Decode base64 audio data to raw bytes
        audio_bytes = base64.b64decode(request.audio_data)

        # Read audio waveform and sample rate
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        # Convert stereo to mono if needed
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample to 44.1 kHz if needed
        if sample_rate != 44100:
            audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=44100)

        # Preprocess the audio into spectrogram
        spectrogram = self.audio_processor.process_audio_chunk(audio_data)
        spectrogram = spectrogram.to(self.device)

        # Perform model inference
        with torch.no_grad():
            output, feature_maps = self.model(spectrogram, return_feature_maps=True)
            output = torch.nan_to_num(output)

            # Get top-3 predictions
            probabilities = torch.softmax(output, dim=1)
            top3_probs, top3_indicies = torch.topk(probabilities[0], 3)
            predictions = [
                {"class": self.classes[idx.item()], "confidence": prob.item()}
                for prob, idx in zip(top3_probs, top3_indicies)
            ]

            # Prepare feature map visualization data
            viz_data = {}
            for name, tensor in feature_maps.items():
                if tensor.dim() == 4:  # Only 4D feature maps
                    aggregated_tensor = torch.mean(tensor, dim=1)  # Reduce channels
                    squeezed_tensor = aggregated_tensor.squeeze(0)
                    numpy_array = squeezed_tensor.cpu().numpy()
                    clean_array = np.nan_to_num(numpy_array)
                    viz_data[name] = {
                        "shape": list(clean_array.shape),
                        "values": clean_array.tolist()
                    }

            # Prepare spectrogram for visualization
            spectrogram_np = spectrogram.squeeze(0).squeeze(0).cpu().numpy()
            clean_spectrogram = np.nan_to_num(spectrogram_np)

            # Downsample waveform for efficient visualization
            max_samples = 8000
            waveform_sample_rate = 44100
            if len(audio_data) > max_samples:
                step = len(audio_data) // max_samples
                waveform_data = audio_data[::step]
            else:
                waveform_data = audio_data

        # Construct the response object
        response = {
            "predictions": predictions,
            "visualization": viz_data,
            "input_spectrogram": {
                "shape": list(clean_spectrogram.shape),
                "values": clean_spectrogram.tolist()
            },
            "waveform": {
                "values": waveform_data.tolist(),
                "sample_rate": waveform_sample_rate,
                "duration": len(audio_data) / waveform_sample_rate
            }
        }

        return response


# Local testing entry point
@app.local_entrypoint()
def main():
    # Read the local WAV file
    audio_data, sample_rate = sf.read("1-13572-A-46.wav")

    # Encode the audio as base64
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    payload = {"audio_data": audio_b64}

    # Start local server and make a test POST request
    server = AudioClassifier()
    url = server.inference.get_web_url()
    response = requests.post(url, json=payload)
    response.raise_for_status()

    # Parse the result
    result = response.json()

    # Print waveform info
    waveform_info = result.get("waveform", {})
    if waveform_info:
        values = waveform_info.get("values", {})
        print(f"First 10 values: {[round(v, 4) for v in values[:10]]}...")
        print(f"Duration: {waveform_info.get('duration', 0)}")

    # Print top-3 predictions
    print("Top predictions:")
    for pred in result.get("predictions", []):
        print(f"  -{pred['class']} {pred['confidence']:0.2%}")
