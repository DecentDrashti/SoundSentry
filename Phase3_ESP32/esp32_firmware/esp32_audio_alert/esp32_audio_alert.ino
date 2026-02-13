#define LED_PIN 2       // Built-in LED (most ESP32 boards)
// #define BUZZER_PIN 15   // Change if needed

void setup() {
  Serial.begin(115200);

  pinMode(LED_PIN, OUTPUT);
  // pinMode(BUZZER_PIN, OUTPUT);

  digitalWrite(LED_PIN, LOW);
  // digitalWrite(BUZZER_PIN, LOW);

  Serial.println("ESP32 Ready to receive predictions...");
}

void loop() {
  if (Serial.available()) {
    String label = Serial.readStringUntil('\n');
    label.trim();

    Serial.print("Received: ");
    Serial.println(label);

    // Reset outputs
    digitalWrite(LED_PIN, LOW);
    // digitalWrite(BUZZER_PIN, LOW);

    if (label == "CRYING") {
      digitalWrite(LED_PIN, HIGH);
    }
    // else if (label == "SHOUTING") {
    //   digitalWrite(BUZZER_PIN, HIGH);
    // }
  }
}
