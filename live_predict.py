import sounddevice as sd
import numpy as np
import tensorflow_hub as hub
import joblib
import time
from collections import deque

# Load models
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
classifier = joblib.load("cry_shout_model.pkl")

SAMPLE_RATE = 16000
DURATION = 3 # seconds
prediction_buffer = deque(maxlen=5)  # last 5 seconds

def countdown():
    print("\n Get your voice ready...")
    time.sleep(1)
    for i in [3, 2, 1]:
        print(f"{i}...")
        time.sleep(1)
    print("🎙️ Go! Listening now...")
    print(" Press Ctrl + C anytime to stop\n")

def check_alert(buffer):
    crying = sum(1 for l, c in buffer if l == "Crying" and c > 60)
    shouting = sum(1 for l, c in buffer if l == "Shouting" and c > 60)

    if crying >= 3:
        return "Crying"
    if shouting >= 3:
        return "Shouting"
    return None



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

        labels = ["Crying", "Shouting", "Other"]
        label = labels[prediction]
        # Confidence-based handling (OPTIONAL but safe)
        # THRESHOLD = 60  # you can tune this

        # if confidence < THRESHOLD:
        #     label = "Uncertain (Background / Overlap)"
        # else:
        #     label = labels[prediction]

        prediction_buffer.append((label, confidence))

        print(f" Prediction: {label} | Confidence: {confidence:.2f}%")

        alert = check_alert(prediction_buffer)

        if alert:
            print(f"\n🚨 ALERT: Sustained {alert} detected!")
            print("🛑 Monitoring paused\n")
            break



        #countdown()

# Graceful exit
try:
    predict_live_audio()
except KeyboardInterrupt:
    print("\n\n Listening stopped by user.")
    print("👋 Thank you for using VoxAlert / SoundSentry!")
