import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# -----------------------------
# Load embeddings
# -----------------------------
X = np.load("X_embeddings.npy")
y = np.load("y_labels.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)

# One-hot encode labels
NUM_CLASSES = 3
y_cat = to_categorical(y, NUM_CLASSES)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

# -----------------------------
# Tiny Neural Network
# -----------------------------
model = Sequential([
    Dense(32, activation="relu", input_shape=(1024,)),
    Dense(16, activation="relu"),
    Dense(NUM_CLASSES, activation="softmax")
])

model.summary()

# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=8
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc:.3f}")

# Save model
model.save("tiny_audio_model.h5")
print("✅ Tiny NN model saved as tiny_audio_model.h5")
