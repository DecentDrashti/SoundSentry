#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "model.h"   // 👈 your converted model

// ======================
// Hardware
// ======================
#define LED_PIN 2   // ESP32 onboard LED

// ======================
// Tensor Arena
// ======================
constexpr int kTensorArenaSize = 60 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

// ======================
// Globals
// ======================
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// ======================
// Setup
// ======================
void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);

  Serial.println("🔌 ESP32 TinyML Booting...");

  // Load model
  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("❌ Model schema mismatch!");
    while (1);
  }

  // Resolver
  static tflite::AllOpsResolver resolver;

  // Interpreter
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize
  );
  interpreter = &static_interpreter;

  // Allocate memory
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("❌ Tensor allocation failed!");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("✅ Model loaded successfully");
}

// ======================
// Loop
// ======================
void loop() {
  // ---- Fake embedding input ----
  for (int i = 0; i < 1024; i++) {
    input->data.f[i] = 0.5;   // test value
  }

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("❌ Inference failed!");
    return;
  }

  // Output probabilities
  float crying   = output->data.f[0];
  float shouting = output->data.f[1];
  float other    = output->data.f[2];

  Serial.print("Crying: "); Serial.print(crying);
  Serial.print(" | Shouting: "); Serial.print(shouting);
  Serial.print(" | Other: "); Serial.println(other);

  // Alert logic
  if (crying > 0.6 || shouting > 0.6) {
    digitalWrite(LED_PIN, HIGH);
    Serial.println("🚨 ALERT!");
  } else {
    digitalWrite(LED_PIN, LOW);
  }

  delay(2000);
}
