import tensorflow_hub as hub
import librosa
import numpy as np

# Load YAMNet
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

# Load one audio file
audio_path = "audio_data/crying/babycry-6473.wav"
audio, sr = librosa.load(audio_path, sr=16000, mono=True)

# Send audio to YAMNet
scores, embeddings, spectrogram = yamnet(audio)

# Take average embedding
final_embedding = np.mean(embeddings, axis=0)

print("Embedding shape:", final_embedding.shape)
