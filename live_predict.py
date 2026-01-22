import sounddevice as sd
import numpy as np
import tensorflow_hub as hub
import joblib
import time

# Load models
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
classifier = joblib.load("cry_shout_model.pkl")

SAMPLE_RATE = 16000
DURATION = 1  # seconds

def countdown():
    print("\n Get your voice ready...")
    time.sleep(1)
    for i in [3, 2, 1]:
        print(f"{i}...")
        time.sleep(1)
    print("🎙️ Go! Listening now...")
    print(" Press Ctrl + C anytime to stop\n")

def predict_live_audio():
    countdown()

    while True:
        # Record audio
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()
       
        audio = audio.flatten()

        # YAMNet embedding
        scores, embeddings, spectrogram = yamnet_model(audio)
        embedding = np.mean(embeddings.numpy(), axis=0).reshape(1, -1)

        # Prediction
        prediction = classifier.predict(embedding)[0]
        confidence = np.max(classifier.predict_proba(embedding)) * 100

        label = "Crying " if prediction == 0 else "Shouting "

        print(f" Prediction: {label} | Confidence: {confidence:.2f}%")
        countdown()

# Graceful exit
try:
    predict_live_audio()
except KeyboardInterrupt:
    print("\n\n Listening stopped by user.")
    print("👋 Thank you for using VoxAlert / SoundSentry!")
