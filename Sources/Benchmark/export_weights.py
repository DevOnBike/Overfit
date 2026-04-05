import onnx
import numpy as np
import struct

model = onnx.load("benchmark_model.onnx")
weights_tensor = None
bias_tensor = None

for init in model.graph.initializer:
    if "weight" in init.name:
        weights_tensor = onnx.numpy_helper.to_array(init)
    elif "bias" in init.name:
        bias_tensor = onnx.numpy_helper.to_array(init)

if weights_tensor is None or bias_tensor is None:
    print("Błąd: Nie znaleziono wag lub biasu w pliku ONNX.")
    exit()

print(f"Znaleziono wagi o kształcie: {weights_tensor.shape}")
print(f"Znaleziono bias o kształcie: {bias_tensor.shape}")

weights_flat = weights_tensor.flatten()
bias_flat = bias_tensor.flatten()

with open("benchmark_model.bin", "wb") as f:
    # Zapisz wagi
    for w in weights_flat:
        f.write(struct.pack('f', w))
    # Zapisz bias
    for b in bias_flat:
        f.write(struct.pack('f', b))

print("Zapisano 'benchmark_model.bin' z powodzeniem!")