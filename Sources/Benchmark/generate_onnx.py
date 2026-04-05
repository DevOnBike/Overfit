import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(784, 10))
dummy_input = torch.randn(1, 784)

torch.onnx.export(
    model, 
    dummy_input, 
    "benchmark_model.onnx", 
    input_names=['input'], 
    output_names=['output']
)
print("Plik benchmark_model.onnx wygenerowany!")