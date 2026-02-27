# 🎧 SoundSentry  
### Real-Time Crying & Shouting Detection using YAMNet + TinyML + ESP32

---

## 📌 Project Overview

**SoundSentry** is a real-time audio classification system that detects:

- 👶 Crying  
- 📢 Shouting  
- 🔇 Other (background / unknown sounds)

The system uses transfer learning with a pretrained deep learning model (YAMNet) to extract audio embeddings and trains a lightweight classifier on top.

Final goal:  
➡ Deploy an optimized INT8 TinyML model on ESP32

---

## 🧠 Core Idea

Instead of training a deep network from scratch:

Audio → YAMNet → 1024-D Embeddings → Lightweight Classifier → Cry / Shout / Other

We use:

- YAMNet as a frozen feature extractor  
- Logistic Regression / Tiny Neural Network as task-specific classifier  

This dramatically improves performance with small datasets.

---

## 🔍 Why YAMNet?

YAMNet is a pretrained deep learning model trained on Google’s AudioSet (~2 million audio clips).

It:

- Analyzes waveform
- Learns acoustic patterns
- Compresses sound into 1024 learned features (embeddings)

Think of YAMNet as a professional ear 👂

It mathematically analyzes:

- Pitch  
- Loudness  
- Frequency patterns  
- Emotional intensity  
- Temporal patterns  

And converts them into a fixed 1024-dimensional numerical fingerprint.

---

## 📊 Understanding YAMNet Outputs

```python
scores, embeddings, spectrogram = yamnet_model(audio)
---
```

## 📊 Understanding YAMNet Outputs

| Output        | Used? | Purpose |
|--------------|-------|----------|
| scores        | ❌ No | Generic pretrained sound probabilities |
| embeddings    | ✅ YES | Learned numerical representation |
| spectrogram   | ❌ No | Visual frequency map |

**Embedding shape:**
1024


YAMNet always explains every sound in exactly **1024 numerical values**.

---

## 🧮 Why Logistic Regression?

Logistic Regression:

- Predicts probability of belonging to a class  
- Learns decision boundaries  
- Lightweight  
- Fast  
- TinyML-friendly  

Later upgraded to a Tiny Neural Network:
1024 → 32 → 16 → 3


Why this works:

- Balanced capacity  
- Avoids overfitting  
- Efficient for TinyML deployment  

---

# 🗂 Project File Structure & Purpose

---

## 1️⃣ `test_yamnet.py`

**Purpose:** Sanity check  

- Loads YAMNet  
- Processes one audio file  
- Confirms embeddings are generated  
- Embedding shape: (1024,)  

If this works → foundation is correct.

---

## 2️⃣ `extract_embeddings.py`

**Purpose:** Feature Extraction  

- Converts raw audio into embeddings  
- Assigns labels:
  Crying → 0
  Shouting → 1
  Other → 2


Saves:

- `X_embeddings.npy`
- `y_labels.npy`

This transforms unstructured audio into structured numeric data.

---

## 3️⃣ `train_classifier.py`

**Purpose:** Learn decision boundaries  

- Loads X and y  
- Splits into train/test  
- Trains Logistic Regression or Tiny NN  
- Saves model  

Output:
cry_shout_model.pkl


**Important:**  
YAMNet is NOT trained.  
Only the classifier is trained.

---

## 4️⃣ `evaluate_model.py`

**Purpose:** Model diagnosis  

Reports:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion matrix  

Example:
Accuracy: 94–95%


Why high?

Because YAMNet embeddings are already semantically powerful.  
This is **transfer learning**.

---

## 5️⃣ `predict.py`

**Purpose:** Predict new unseen audio  

Pipeline:
Audio → YAMNet → Embedding → Classifier → Prediction

---

# 📂 Project Folder Structure (Functional Grouping)

This repository is organized into structured development phases — from training to embedded deployment.

---

## 🧠 `tinyml_mfcc/`

This folder contains the **TinyML feature extraction and training pipeline using MFCC features**.

It is focused on preparing lightweight models suitable for microcontroller deployment.

### Subfolders:

### 📁 `training/`

Contains Python scripts responsible for:

- MFCC feature extraction
- Model training (Tiny Neural Network / Logistic Regression)
- Dataset preprocessing
- Saving trained models

This is where your TinyML-compatible classifier is trained.
---

Each subfolder represents a purpose-specific dataset split to ensure clean training, testing, and validation workflow.

---

# 🚀 `Phase2_tinyml/`

This folder represents the **model optimization and TinyML preparation stage**.

Purpose:

Convert the trained model into a microcontroller-compatible format.

Typical contents include:

- TFLite conversion scripts  
- Representative dataset functions (for INT8 quantization)  
- Post-training quantization scripts  
- TFLite model evaluation scripts  

Key tasks performed here:

- Float32 → INT8 conversion  
- Size optimization  
- Latency testing  
- Embedded readiness validation  

This phase bridges machine learning development and embedded deployment.

---

# 🔌 `Phase3_ESP32/`

This folder contains everything related to **ESP32 embedded deployment**.

It includes firmware, model headers, and microcontroller integration code.

---

### 📄 `model_data.h`

- Contains the INT8 quantized TFLite model
- Converted into a C byte array
- Generated using tools like `xxd`

Used directly by TensorFlow Lite Micro inside ESP32 firmware.

---

### 📄 `VoiceAI.ino`

- Arduino sketch for ESP32
- Loads TensorFlow Lite Micro model
- Allocates tensor arena
- Performs inference
- Handles prediction outputs

This is the main embedded application file.

---

### 📄 Additional MCU Code

May include:

- I2S microphone configuration
- Audio buffer management
- MFCC extraction on-device
- Decision smoothing logic
- LED / buzzer / alert handling
- Serial debugging output

---

### 📄 Hardware Documentation (Optional)

May include:

- Wiring diagrams
- Pin configurations
- Microphone specifications
- Deployment notes

---

This folder contains everything required to:

- Build
- Flash
- Run
- Test

Your TinyML model on ESP32 hardware.

---

# 🧩 Development Flow Overview

## Phase 1 – Model Training (Laptop)

Audio  
↓  
MFCC Feature Extraction  
↓  
Tiny Neural Network  
↓  
Model Saved (.h5 / .keras)

---

## Phase 2 – TinyML Optimization

Trained Model  
↓  
TFLite Conversion  
↓  
INT8 Quantization  
↓  
model_int8.tflite  

---

## Phase 3 – ESP32 Deployment

model_int8.tflite  
↓  
Convert to C Array (`model_data.h`)  
↓  
Load in Arduino (VoiceAI.ino)  
↓  
On-device Inference  

---

This structured folder organization ensures:

- Clean separation of concerns  
- Reproducible training pipeline  
- Proper deployment workflow  
- Production-ready TinyML architecture  

---

---

# 🎤 Live Audio Handling

We use:

- `sounddevice` (not PyAudio)

Reason:

- Lightweight  
- Modern  
- Cleaner API  
- Better for real-time testing  

---

# 🧠 Adding “Other” Class (Production Thinking)

To reduce false alarms:

We introduced:

- An "Other" class  
- Temporal aggregation  
- Decision smoothing  

**Industry-ready explanation:**

> To make the system robust for real-world deployment, we introduced an ‘Other’ class to handle background noise and unknown sounds. Instead of triggering alerts on a single prediction, we aggregate multiple short-window predictions before making a decision, significantly reducing false alarms.

---

# 🧪 Model Performance

After dataset expansion:

**Accuracy: 94–95%**

Why?

- YAMNet embeddings are pretrained on millions of clips  
- Tiny NN only learns separation  
- This is transfer learning  

**Interview explanation:**

> The accuracy is high because we use pretrained embeddings from YAMNet. Our model learns task-specific separation rather than raw audio representation.

---

# ⚠️ Critical TinyML Deployment Challenge

On Laptop:
Raw Audio → YAMNet → 1024 Embeddings → Tiny NN
On ESP32:
Raw Audio → ??? → Tiny NN


ESP32 does NOT have YAMNet.

YAMNet is:

- Too large  
- Too RAM-heavy  
- Not feasible for microcontrollers  

---

# 🚨 Deployment Solution

Replace YAMNet with MFCC.

| Feature Type | Tool | ESP32 Compatible |
|--------------|------|------------------|
| YAMNet | TensorFlow Hub | ❌ |
| MFCC | Librosa / ESP-DSP | ✅ |

MFCC is:

- Lightweight  
- Industry standard  
- TinyML compatible  

---

# 🔌 Why ESP32 (Not ESP8266)

| Feature | ESP8266 | ESP32 |
|----------|----------|----------|
| TFLite Micro | ❌ | ✅ |
| Audio ML | ❌ | ✅ |
| Mic streaming | Weak | Stable |
| Future scaling | No | Yes |

ESP32 chosen for:

- More RAM  
- DSP capability  
- TinyML support  

---

# 📚 Techniques Used

- YAMNet (Transfer Learning)  
- Logistic Regression  
- Tiny Neural Networks  
- MFCC (planned for deployment)  
- TensorFlow / Keras  
- INT8 Quantization  
- TinyML  
- Real-time audio processing  

---

# 🎯 Practical Applications

- Baby monitoring systems  
- Safety alert wearables  
- Emotion-aware IoT systems  
- Smart surveillance  
- Assistive technology  

---

# 🚀 Future Improvements

- Replace YAMNet embeddings with MFCC fully  
- Convert Tiny NN to INT8 TFLite  
- Deploy fully on ESP32 with I2S mic  
- Add temporal smoothing logic on-device  
- Expand dataset with real-world noise  

---

# 📌 Final Architecture

## 🖥 Development Phase
    Audio
    ↓
    YAMNet (Feature Extractor)
    ↓
    1024-D Embeddings
    ↓
    Tiny Neural Network
    ↓
    Cry / Shout / Other

## 🔌 Deployment Phase
      Audio
    ↓
    MFCC (On ESP32)
    ↓
    Tiny INT8 TFLite Model
    ↓
    Decision Logic

