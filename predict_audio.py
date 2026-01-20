import numpy as np
import librosa
import tensorflow_hub as hub
import tensorflow as tf
import joblib

# Load YAMNet
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Load trained classifier
model = joblib.load("cry_shout_model.pkl")

# Class labels
LABELS = {0: "Crying", 1: "Shouting"}

def extract_embedding(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    scores, embeddings, spectrogram = yamnet_model(audio)
    embedding = tf.reduce_mean(embeddings, axis=0)
    return embedding.numpy()

# ---- TEST AUDIO FILE ----
audio_file = "testing_data/a-man-sobbing-type-1-265495.wav"   # 👈 put your file name here

embedding = extract_embedding(audio_file)
embedding = embedding.reshape(1, -1)

prediction = model.predict(embedding)[0]
confidence = model.predict_proba(embedding).max()

print("🎧 Audio File:", audio_file)
print("🔮 Prediction:", LABELS[prediction])
print("📊 Confidence:", round(confidence * 100, 2), "%")
