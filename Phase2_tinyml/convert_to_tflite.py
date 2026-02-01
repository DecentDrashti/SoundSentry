import tensorflow as tf

# Load trained Tiny NN model
model = tf.keras.models.load_model("tiny_audio_model.h5")

# Create TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimizations (VERY IMPORTANT for ESP32)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert model
tflite_model = converter.convert()

# Save the converted model
with open("tiny_audio_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion complete!")
print("📦 Saved as tiny_audio_model.tflite")
