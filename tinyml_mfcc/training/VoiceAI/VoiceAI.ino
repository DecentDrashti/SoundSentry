#include <TensorFlowLite.h>
#include "model_data.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
// #include "tensorflow/lite/version.h"

const tflite::Model* model = tflite::GetModel(model_int8_tflite);
tflite::AllOpsResolver resolver;

constexpr int tensor_arena_size = 20 * 1024;
uint8_t tensor_arena[tensor_arena_size];

tflite::MicroInterpreter interpreter(
  model, resolver, tensor_arena, tensor_arena_size);

void setup() {
  Serial.begin(115200);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    return;
  }

  interpreter.AllocateTensors();
  Serial.println("Model loaded successfully!");
}

void loop() {
}