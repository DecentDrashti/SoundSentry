import os
import numpy as np
import librosa
import tensorflow_hub as hub
import tensorflow as tf
import joblib

# Load models
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
model = joblib.load("cry_shout_model.pkl")

LABELS = ["Crying", "Shouting", "Other"]
TEST_DIR = "real_world_test"

def extract_embedding(file_path):
    audio, _ = librosa.load(file_path, sr=16000)
    scores, embeddings, _ = yamnet(audio)
    return tf.reduce_mean(embeddings, axis=0).numpy()

print("\n🔍 REAL-WORLD MODEL VALIDATION\n")

for label in os.listdir(TEST_DIR):
    class_dir = os.path.join(TEST_DIR, label)

    for file in os.listdir(class_dir):
        path = os.path.join(class_dir, file)

        emb = extract_embedding(path).reshape(1, -1)
        pred = model.predict(emb)[0]
        conf = model.predict_proba(emb).max() * 100

        print(f"🎧 {file}")
        print(f"   True: {label}")
        print(f"   Pred: {LABELS[pred]} ({conf:.2f}%)\n")
