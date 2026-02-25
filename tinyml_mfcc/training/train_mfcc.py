import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ======================
# CONFIG
# ======================
DATASET_PATH = "../../audio_data"  # Update with your dataset path
SAMPLE_RATE = 16000
CLIP_DURATION = 1  # seconds
N_MFCC = 20
N_FFT = 512
HOP_LENGTH = 320

# ======================
# LOAD + PROCESS AUDIO
# ======================
def extract_features():
    X = []
    y = []

    for label in os.listdir(DATASET_PATH):
        label_path = os.path.join(DATASET_PATH, label)

        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)

            try:
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

                total_samples = len(signal)
                samples_per_clip = SAMPLE_RATE * CLIP_DURATION

                for start in range(0, total_samples - samples_per_clip, samples_per_clip):
                    clip = signal[start:start + samples_per_clip]

                    mfcc = librosa.feature.mfcc(
                        y=clip,
                        sr=SAMPLE_RATE,
                        n_mfcc=N_MFCC,
                        n_fft=N_FFT,
                        hop_length=HOP_LENGTH
                    )

                    mfcc = mfcc.T  # shape → (time, mfcc)
                    X.append(mfcc)
                    y.append(label)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return np.array(X), np.array(y)

# ======================
# MAIN
# ======================
print("Extracting MFCC features...")
X, y = extract_features()

print("Encoding labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X = X[..., np.newaxis]  # Add channel dimension

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ======================
# MODEL (Tiny CNN)
# ======================
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=X_train.shape[1:]),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("Training...")
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

print("Evaluating...")
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

model.save("mfcc_model.h5")