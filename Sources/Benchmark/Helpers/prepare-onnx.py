"""
Generates ONNX models + .bin files for the DevOnBike.Overfit benchmarks.

Usage:
    pip install torch onnx onnxruntime
    python prepare-onnx.py
"""

import struct
import os
import numpy as np
import torch
import torch.nn as nn
import onnx


def export(model, input_size, name):
    dummy = torch.randn(1, input_size)

    # ONNX — export + force inline weights (no .onnx.data)
    onnx_path = f"{name}.onnx"
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    onnx_model = onnx.load(onnx_path)
    onnx.save_model(onnx_model, onnx_path, save_as_external_data=False)

    # Remove the junk .onnx.data file if PyTorch generated one
    data_path = f"{onnx_path}.data"
    if os.path.exists(data_path):
        os.remove(data_path)

    # .bin — Overfit's LinearLayer.Save format:
    #   For each Linear layer in order:
    #     [int32 weight_count][float32 × weight_count]
    #     [int32 bias_count][float32 × bias_count]
    # Weights are stored in [in_size, out_size] layout, which is the transpose of
    # PyTorch's nn.Linear.weight (which uses [out_size, in_size]). Hence the .T below.
    bin_path = f"{name}.bin"
    with open(bin_path, "wb") as f:
        for m in model:
            if isinstance(m, nn.Linear):
                weight_t = m.weight.data.T.contiguous().flatten()
                bias = m.bias.data.flatten()

                # Length prefix for weights (int32, little-endian).
                f.write(struct.pack("i", weight_t.numel()))
                for val in weight_t:
                    f.write(struct.pack("f", val.item()))

                # Length prefix for bias (int32, little-endian).
                f.write(struct.pack("i", bias.numel()))
                for val in bias:
                    f.write(struct.pack("f", val.item()))

    print(f"  {name}: .onnx={os.path.getsize(onnx_path)}B  .bin={os.path.getsize(bin_path)}B")


def main():
    torch.manual_seed(42)

    models = [
        ("benchmark_model",  784,  nn.Sequential(nn.Linear(784, 10))),
        ("benchmark_mlp3",   784,  nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))),
        # Sizes are read by ScalingBenchmark.cs, which expects 64 / 784 / 4096 inputs.
        # Keep these in sync with that file or scaling tests will mismatch shapes.
        ("benchmark_small",  64,   nn.Sequential(nn.Linear(64, 10))),
        ("benchmark_medium", 784,  nn.Sequential(nn.Linear(784, 10))),
        ("benchmark_large",  4096, nn.Sequential(nn.Linear(4096, 10))),
    ]

    for name, input_size, model in models:
        export(model, input_size, name)

    # Verification
    try:
        import onnxruntime as ort
        print("\nVerification:")
        for name, input_size, model in models:
            x = np.random.randn(1, input_size).astype(np.float32)
            onnx_out = ort.InferenceSession(f"{name}.onnx").run(None, {"input": x})[0]
            with torch.no_grad():
                torch_out = model(torch.from_numpy(x)).numpy()
            diff = np.max(np.abs(onnx_out - torch_out))
            print(f"  {name}: diff={diff:.8f} {'OK' if diff < 1e-5 else 'FAIL'}")
    except ImportError:
        print("\nSkipping verification (pip install onnxruntime)")

    print("\nDone!")


if __name__ == "__main__":
    main()