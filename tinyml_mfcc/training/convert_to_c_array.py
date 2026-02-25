with open("mfcc_model_int8.tflite", "rb") as f:
    data = f.read()

with open("model_data.h", "w") as f:
    f.write("unsigned char model_int8_tflite[] = {\n")
    for i, byte in enumerate(data):
        f.write(f"0x{byte:02x}, ")
        if (i + 1) % 12 == 0:
            f.write("\n")
    f.write("\n};\n")
    f.write(f"unsigned int model_int8_tflite_len = {len(data)};\n")

print("Conversion completed!")