import os
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub

# Load YAMNet model
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

DATASET_PATH = "audio_data"
CLASSES = {
    "crying": 0,
    "shouting": 1
}

X = []
y = []

def extract_embedding(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    scores, embeddings, spectrogram = yamnet_model(audio)
    return tf.reduce_mean(embeddings, axis=0).numpy()

for class_name, label in CLASSES.items():
    class_folder = os.path.join(DATASET_PATH, class_name)

    for file in os.listdir(class_folder):
        if file.lower().endswith((".wav", ".mp3", ".ogg")):
            file_path = os.path.join(class_folder, file)
            try:
                embedding = extract_embedding(file_path)
                X.append(embedding)
                y.append(label)
                print(f"Processed: {file_path}")

            except Exception as e:
                print(f"Failed: {file_path} | Reason: {e}")
            # embedding = extract_embedding(file_path)

            # X.append(embedding)
            # y.append(label)

            # print(f"Processed: {file_path}")

X = np.array(X)
y = np.array(y)

np.save("X_embeddings.npy", X)
np.save("y_labels.npy", y)

print("✅ Embeddings saved successfully!")
print("X shape:", X.shape)
print("y shape:", y.shape)



#or endswith((".wav", ".mp3", ".ogg"))
# for class_name, label in CLASSES.items():
#     class_folder = os.path.join(DATASET_PATH, class_name)

#     for file in os.listdir(class_folder):
#         if file.lower().endswith((".wav", ".mp3", ".ogg")):
#             file_path = os.path.join(class_folder, file)

#             try:
#                 embedding = extract_embedding(file_path)
#                 X.append(embedding)
#                 y.append(label)
#                 print(f"Processed: {file_path}")

#             except Exception as e:
#                 print(f"❌ Failed: {file_path} | Reason: {e}")





# What This Step Will Do (No Code Yet)

# For each file:

# Load audio

# Pass through YAMNet

# Get a (1024,) embedding

# Assign label:

# crying → 0

# shouting → 1

# Save everything into arrays

# At the end:

# X → features (numbers)

# y → labels