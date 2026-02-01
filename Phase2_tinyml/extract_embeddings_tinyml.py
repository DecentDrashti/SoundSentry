import os
import numpy as np
import librosa
import tensorflow_hub as hub
import tensorflow as tf

# ------------------
# CONFIG
# ------------------
SAMPLE_RATE = 16000
EMBEDDING_SIZE = 1024

DATASET_PATH = "C:\\Users\\dr_dr\\OneDrive\\Desktop\\B.Tech CSE\\voice_ml\\audio_data"  # adjust if needed
# OUTPUT_PATH = ""

LABEL_MAP = {
    "crying": 0,
    "shouting": 1,
    "other": 2
}

# ------------------
# LOAD YAMNET
# ------------------
print("🔄 Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print("✅ YAMNet loaded")

X = []
y = []

# ------------------
# PROCESS DATA
# ------------------
for label_name, label_id in LABEL_MAP.items():
    folder = os.path.join(DATASET_PATH, label_name)

    if not os.path.exists(folder):
        print(f"⚠️ Skipping missing folder: {folder}")
        continue

    for file in os.listdir(folder):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(folder, file)

        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        scores, embeddings, spectrogram = yamnet_model(audio)

        embedding = tf.reduce_mean(embeddings, axis=0).numpy()

        if embedding.shape[0] != EMBEDDING_SIZE:
            print(f"❌ Wrong embedding size: {file}")
            continue

        X.append(embedding)
        y.append(label_id)

# ------------------
# SAVE OUTPUT
# ------------------
X = np.array(X)
y = np.array(y)

np.save("X_embeddings.npy", X)
np.save("y_labels.npy", y)

print("✅ Embeddings saved")
print("X shape:", X.shape)
print("y shape:", y.shape)
