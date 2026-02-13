import sounddevice as sd
import numpy as np
import tensorflow_hub as hub
import joblib
import serial
import time

# --------------------
# Serial setup
# --------------------
SERIAL_PORT = "COM3"   # ⚠️ change this
BAUD_RATE = 115200

ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(5)  # allow ESP32 reset


# --------------------
# ML setup
# --------------------
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
classifier = joblib.load("../../../voice_ml/cry_shout_model.pkl")

SAMPLE_RATE = 16000
DURATION = 5

LABELS = ["CRYING", "SHOUTING", "OTHER"]

def countdown():
    print("\n🎤 Get ready...")
    for i in [3, 2, 1]:
        print(i)
        time.sleep(1)
    print("Listening...")

while True:
    countdown()

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    audio = audio.flatten()

    _, embeddings, _ = yamnet_model(audio)
    embedding = np.mean(embeddings.numpy(), axis=0).reshape(1, -1)

    prediction = classifier.predict(embedding)[0]
    label = LABELS[prediction]

    print("Prediction:", label)

    # 🔥 Send to ESP32
    ser.write((label + "\n").encode())

    time.sleep(1)

# 1️⃣ What it sends

# It opens the COM port where the ESP32 is connected (COM3, COM4, etc. on Windows).

# It waits for a short delay (so ESP32 has time to reset after opening the Serial).

# It sends prediction information. Usually, you send one of these:

# 'C'  # Crying
# 'S'  # Shouting
# 'O'  # Other


# ESP32 reads this single character and decides what to do:

# C → blink LED once

# S → blink LED multiple times or trigger buzzer

# O → maybe do nothing