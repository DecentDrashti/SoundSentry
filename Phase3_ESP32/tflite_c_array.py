import sys

input_file = "tiny_audio_model.tflite"
output_file = "./esp32_firmware/model.h"

with open(input_file, "rb") as f:
    data = f.read()

with open(output_file, "w") as f:
    f.write("const unsigned char g_model[] = {\n")
    for i, byte in enumerate(data):
        f.write(f"0x{byte:02x}, ")
        if (i + 1) % 12 == 0:
            f.write("\n")
    f.write("\n};\n")
    f.write(f"const unsigned int g_model_len = {len(data)};")

print("✅ model.h generated for ESP32")
