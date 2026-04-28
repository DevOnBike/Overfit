"""
Fixture generator for Overfit ONNX importer testing.

Generates a small MNIST CNN, exports to ONNX, and saves reference
input/output as raw float32 binary files (no numpy dependency on read side).

Usage:
    pip install torch numpy
    python generate_fixture.py

Output:
    test_fixtures/mnist_cnn.onnx       - the model
    test_fixtures/mnist_input.bin      - reference input (raw float32, 1*1*28*28 = 784 floats)
    test_fixtures/mnist_output.bin     - reference output (raw float32, 1*10 = 10 floats)
    test_fixtures/mnist_shapes.txt     - shape metadata for verification
"""

import os
import struct
import torch
import torch.nn as nn

OUTPUT_DIR = "test_fixtures"


class MnistCnn(nn.Module):
    """Minimal MNIST CNN exercising Conv + BN + ReLU + Pool + Linear path."""

    def __init__(self):
        super().__init__()
        # Conv: 1 -> 8 channels, 3x3 kernel, no padding -> 28x28 -> 26x26
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=0)
        self.bn = nn.BatchNorm2d(8)
        # MaxPool 2x2 -> 26x26 -> 13x13
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 13 * 13, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten -> Linear input
        x = self.fc(x)
        return x


def save_floats_raw(tensor: torch.Tensor, path: str):
    """Write a tensor as raw little-endian float32 bytes."""
    arr = tensor.detach().cpu().contiguous().numpy().astype("float32").ravel()
    with open(path, "wb") as f:
        for v in arr:
            f.write(struct.pack("<f", float(v)))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    torch.manual_seed(42)

    model = MnistCnn()
    model.eval()

    # Deterministic dummy input
    dummy_input = torch.randn(1, 1, 28, 28)

    # Export ONNX
    onnx_path = os.path.join(OUTPUT_DIR, "mnist_cnn.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
        dynamic_axes=None,
    )
    print(f"Wrote: {onnx_path}")

    # Save input
    input_path = os.path.join(OUTPUT_DIR, "mnist_input.bin")
    save_floats_raw(dummy_input, input_path)
    print(f"Wrote: {input_path}  (shape: {tuple(dummy_input.shape)}, {dummy_input.numel()} floats)")

    # Save reference output
    with torch.no_grad():
        output = model(dummy_input)

    output_path = os.path.join(OUTPUT_DIR, "mnist_output.bin")
    save_floats_raw(output, output_path)
    print(f"Wrote: {output_path} (shape: {tuple(output.shape)}, {output.numel()} floats)")

    # Save shape metadata for verification on the C# side
    shapes_path = os.path.join(OUTPUT_DIR, "mnist_shapes.txt")
    with open(shapes_path, "w") as f:
        f.write(f"input_shape: {list(dummy_input.shape)}\n")
        f.write(f"output_shape: {list(output.shape)}\n")
        f.write(f"output_first_5: {output[0, :5].tolist()}\n")
    print(f"Wrote: {shapes_path}")

    print("\nFixtures generated successfully.")


if __name__ == "__main__":
    main()
