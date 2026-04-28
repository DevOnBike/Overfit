#!/usr/bin/env python3
"""
PyTorch CPU benchmark for the same MNIST CNN ONNX fixture used by Overfit.

This is intentionally a standalone Python benchmark rather than a BenchmarkDotNet
method. It compares the same architecture and weights against Overfit/ONNX Runtime
results without mixing cross-process Python startup cost into BDN.

Requirements:
    pip install torch onnx numpy

Usage:
    python benchmark_pytorch_mnist_cnn.py --fixture-dir test_fixtures --threads 1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import torch
import torch.nn as nn


class MnistCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(8 * 13 * 13, 10, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def load_initializers(model_path: Path) -> dict[str, np.ndarray]:
    model = onnx.load(str(model_path), load_external_data=True)
    result: dict[str, np.ndarray] = {}

    for initializer in model.graph.initializer:
        result[initializer.name] = numpy_helper.to_array(initializer)

    return result


def require_initializer(initializers: dict[str, np.ndarray], *names: str) -> np.ndarray:
    for name in names:
        if name in initializers:
            return initializers[name]

    suffix_matches = [
        value
        for key, value in initializers.items()
        if any(key.endswith(name) for name in names)
    ]

    if len(suffix_matches) == 1:
        return suffix_matches[0]

    raise KeyError(f"Could not find initializer. Tried names={names}. Available={sorted(initializers)}")


def load_model_from_onnx(model_path: Path) -> MnistCnn:
    initializers = load_initializers(model_path)
    model = MnistCnn().eval()

    with torch.no_grad():
        model.conv.weight.copy_(torch.from_numpy(require_initializer(initializers, "conv.weight")).float())
        model.conv.bias.copy_(torch.from_numpy(require_initializer(initializers, "conv.bias")).float())
        model.fc.weight.copy_(torch.from_numpy(require_initializer(initializers, "fc.weight")).float())
        model.fc.bias.copy_(torch.from_numpy(require_initializer(initializers, "fc.bias")).float())

    return model


def load_float_bin(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype="<f4")
    return data.astype(np.float32, copy=False)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan

    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", default="Tests/test_fixtures")
    parser.add_argument("--model", default="mnist_cnn.onnx")
    parser.add_argument("--input", default="mnist_input.bin")
    parser.add_argument("--expected", default="mnist_output.bin")
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--warmup", type=int, default=10_000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    fixture_dir = Path(args.fixture_dir)
    model_path = fixture_dir / args.model
    input_path = fixture_dir / args.input
    expected_path = fixture_dir / args.expected

    model = load_model_from_onnx(model_path)
    input_np = load_float_bin(input_path).reshape(1, 1, 28, 28)
    input_tensor = torch.from_numpy(input_np)

    expected = load_float_bin(expected_path)

    with torch.inference_mode():
        output = model(input_tensor).detach().cpu().numpy().reshape(-1)

    max_abs_diff = float(np.max(np.abs(output - expected)))
    argmax_match = int(np.argmax(output)) == int(np.argmax(expected))

    if max_abs_diff > 1e-4 or not argmax_match:
        raise RuntimeError(
            f"Output mismatch: max_abs_diff={max_abs_diff}, "
            f"argmax_torch={int(np.argmax(output))}, argmax_expected={int(np.argmax(expected))}"
        )

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(input_tensor)

        samples_us: list[float] = []
        checksum = 0.0

        for _ in range(args.repeats):
            start = time.perf_counter()

            for _ in range(args.iterations):
                y = model(input_tensor)
                checksum += float(y[0, 0])

            elapsed = time.perf_counter() - start
            samples_us.append(elapsed * 1_000_000.0 / args.iterations)

    result = {
        "runtime": "PyTorch eager CPU",
        "threads": args.threads,
        "iterations_per_repeat": args.iterations,
        "repeats": args.repeats,
        "mean_us": float(np.mean(samples_us)),
        "median_us": float(np.median(samples_us)),
        "p90_us": percentile(samples_us, 0.90),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
        "max_abs_diff_vs_fixture": max_abs_diff,
        "argmax_match": argmax_match,
        "checksum": checksum,
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("PyTorch MNIST CNN benchmark")
        print(f"  model:      {model_path}")
        print(f"  threads:    {args.threads}")
        print(f"  iterations: {args.iterations} x {args.repeats}")
        print(f"  mean:       {result['mean_us']:.3f} us/op")
        print(f"  median:     {result['median_us']:.3f} us/op")
        print(f"  p90:        {result['p90_us']:.3f} us/op")
        print(f"  min/max:    {result['min_us']:.3f} / {result['max_us']:.3f} us/op")
        print(f"  max diff:   {result['max_abs_diff_vs_fixture']:.8f}")
        print(f"  checksum:   {result['checksum']:.4f}")


if __name__ == "__main__":
    main()
