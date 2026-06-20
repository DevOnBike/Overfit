# Copyright (c) 2026 DevOnBike.
# This file is part of DevonBike Overfit.
# DevonBike Overfit is licensed under the GNU AGPLv3.
# For commercial licensing options, contact: devonbike@gmail.com
"""
Exports a torchvision CNN to ONNX for the Overfit-vs-ONNX-Runtime CPU benchmark.

Random weights on purpose — inference *latency* depends only on the graph shape/FLOPs, not on weight
values, so there is nothing to download (no pretrained checkpoint, no internet). Eval mode + opset 13
produce the clean op set Overfit's ONNX graph importer supports.

Architectures:
  vgg16    — ~15.5 GFLOPs, compute-dominated; uses only 2x2 stride-2 (non-overlapping) MaxPool, which
             Overfit imports. This is the default — the honest "real kernels vs MLAS" test.
  resnet50 — ~4 GFLOPs, but its first MaxPool is 3x3 stride-2 (OVERLAPPING), which Overfit's importer does
             not yet support (MaxPool2DLayer is non-overlapping only). Kept here for when that gap is fixed.

Usage (needs a local Python with torch + torchvision):
    python Scripts/export_cnn_onnx.py --arch vgg16 --out C:\\onnxmodels\\cnn.onnx
    dotnet run -c Release --project Sources/Benchmark -- --filter "*LargeCnnComparison*"
Override the path the benchmark reads with the OVERFIT_CNN_ONNX environment variable.
"""
import argparse
import os

import torch
import torchvision
from torch import nn


def _deepcnn() -> nn.Module:
    # VGG-16's convolutional configuration (~15 GFLOPs, the compute-heavy part) but with a
    # GlobalAveragePool head instead of VGG's 25088x4096 fully-connected layers. That avoids the two
    # importer limits a real torchvision export trips: (1) the giant ~411 MB FC weight tensor that
    # desyncs the hand-rolled protobuf parser, and (2) nothing here uses overlapping MaxPool. Pure
    # Conv 3x3 (pad 1) + ReLU + 2x2 stride-2 (non-overlapping) MaxPool + GlobalAveragePool + a tiny
    # 512->1000 Linear — every op is one Overfit's ONNX graph importer supports.
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
    layers: list[nn.Module] = []
    in_c = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_c, v, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_c = v
    return nn.Sequential(
        *layers,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, 1000),
    )


def build(arch: str):
    if arch == "deepcnn":
        return _deepcnn()
    if arch == "vgg16":
        return torchvision.models.vgg16(weights=None)
    if arch == "resnet50":
        return torchvision.models.resnet50(weights=None)
    raise SystemExit(f"unknown --arch '{arch}' (use deepcnn, vgg16 or resnet50)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a random-weight CNN to ONNX.")
    parser.add_argument("--arch", default="deepcnn", choices=["deepcnn", "vgg16", "resnet50"])
    parser.add_argument("--out", default=r"C:\onnxmodels\cnn.onnx", help="output .onnx path")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset (13 = clean, stable op set)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    model = build(args.arch)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)

    export_kwargs = dict(
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
    )
    try:
        # Force the legacy TorchScript exporter — it emits the canonical op patterns the importer expects.
        torch.onnx.export(model, dummy, args.out, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, args.out, **export_kwargs)

    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"exported {args.arch} -> {args.out}  ({size_mb:.1f} MB, opset {args.opset}, input 1x3x224x224, output 1000)")


if __name__ == "__main__":
    main()
