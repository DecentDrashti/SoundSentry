import tensorflow as tf
import numpy as np
import librosa
import glob
import os

# -----------------------
# CONFIG (MUST match training)
# -----------------------
MODEL_PATH = "mfcc_model.h5"
DATASET_PATH = "../../audio_data"
SAMPLE_RATE = 16000
N_MFCC = 20
MAX_LEN = 51   # (51,20,1) input shape

# -----------------------
# MFCC Extraction
# -----------------------
def extract_mfcc(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # Pad or truncate
    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)))
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc.T  # shape -> (51,20)

# -----------------------
# Representative Dataset
# -----------------------
def representative_dataset():
    files = glob.glob(DATASET_PATH + "/**/*.*", recursive=True)
    count = 0

    for file in files:
        try:
            mfcc = extract_mfcc(file)
            mfcc = mfcc.reshape(1, 51, 20, 1).astype(np.float32)
            yield [mfcc]

            count += 1
            if count >= 100:  # 100 samples is enough
                break
        except:
            continue

# -----------------------
# Load Model
# -----------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------
# TFLite INT8 Conversion
# -----------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Force full INT8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# -----------------------
# Save Model
# -----------------------
with open("mfcc_model_int8.tflite", "wb") as f:
    f.write(tflite_model)

size_kb = os.path.getsize("mfcc_model_int8.tflite") / 1024
print("✅ INT8 model saved as mfcc_model_int8.tflite")
print(f"📦 Model size: {size_kb:.2f} KB")