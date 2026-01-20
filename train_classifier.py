import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load frozen dataset
X = np.load("X_embeddings.npy")
y = np.load("y_labels.npy")

print("Loaded data:")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Crying", "Shouting"]))

import joblib

joblib.dump(model, "cry_shout_model.pkl")
print("✅ Model saved as cry_shout_model.pkl")
