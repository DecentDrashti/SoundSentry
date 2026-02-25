import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("mfcc_model.h5")

# Convert to TFLite (float model)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open("mfcc_model_float.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Float TFLite model saved as mfcc_model_float.tflite")

# Print model size
import os
size_kb = os.path.getsize("mfcc_model_float.tflite") / 1024
print(f"Model size: {size_kb:.2f} KB")