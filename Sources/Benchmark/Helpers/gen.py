# generate_test_fixture.py
import torch
import torch.nn as nn
import numpy as np

class MnistCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=0)
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 13 * 13, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = MnistCnn()
model.eval()

# Generate deterministic dummy input
torch.manual_seed(42)
dummy_input = torch.randn(1, 1, 28, 28)

# Export ONNX
torch.onnx.export(
    model, dummy_input, "mnist_cnn.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=17,
    dynamic_axes=None  # static shapes for simplicity
)

# Save reference input/output for testing
np.save("mnist_input.npy", dummy_input.numpy())
with torch.no_grad():
    output = model(dummy_input)
np.save("mnist_output.npy", output.numpy())

print("Generated: mnist_cnn.onnx, mnist_input.npy, mnist_output.npy")