"""
fixture_resnet.py — generuje mini-ResNet z skip connection do testów OnnxGraphImporter.

Uruchomienie:
    python fixture_resnet.py

Wymagania:
    pip install torch onnx numpy

Generowane pliki (w tests/test_fixtures/):
    tiny_resnet.onnx          — model z skip connection
    tiny_resnet_input.bin     — reference input (8 float32 LE)
    tiny_resnet_output.bin    — reference output (4 float32 LE)

    resnet_block.onnx         — większy model: Conv→BN→ReLU + skip
    resnet_block_input.bin    — reference input (1×4×8×8 = 256 float32)
    resnet_block_output.bin   — reference output (1×4×8×8 = 256 float32)

Model TinyResNet (Linear residual):
    fc1 = Linear(8, 8)
    fc2 = Linear(8, 4)
    forward(x): h = relu(fc1(x)) + x   ← skip connection (Add)
                return fc2(h)

Model ResNetBlock (Conv residual — sprawdza Conv+BN+ReLU+Add):
    conv1 = Conv2d(4, 4, 3, padding=1)
    bn1   = BatchNorm2d(4)  → eksportuje jako BN node (train mode) lub folded (eval)
    forward(x): return relu(bn1(conv1(x))) + x  ← skip connection
"""

import os
import struct
import numpy as np
import torch
import torch.nn as nn

OUT_DIR = os.path.join("tests", "test_fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

def save_bin(arr: np.ndarray, path: str):
    """Saves float32 numpy array as little-endian binary."""
    arr.astype("<f4").tofile(path)
    print(f"  saved: {path}  ({arr.size} float32, {arr.nbytes} bytes)")

def export_onnx(model, dummy, name, **kwargs):
    path = os.path.join(OUT_DIR, name)
    torch.onnx.export(
        model, dummy, path,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        **kwargs,
    )
    print(f"  exported: {path}")
    return path

# ─────────────────────────────────────────────────────────────────────────────
# 1. TinyResNet — Linear(8,8) + skip + Linear(8,4)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] TinyResNet (Linear + skip connection)")

class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        h = torch.relu(self.fc1(x)) + x   # skip: Add node in ONNX graph
        return self.fc2(h)

torch.manual_seed(42)
model = TinyResNet().eval()

dummy = torch.randn(1, 8)
with torch.no_grad():
    ref_out = model(dummy)

save_bin(dummy.numpy().flatten(), os.path.join(OUT_DIR, "tiny_resnet_input.bin"))
save_bin(ref_out.numpy().flatten(), os.path.join(OUT_DIR, "tiny_resnet_output.bin"))
export_onnx(model, dummy, "tiny_resnet.onnx")

# ─────────────────────────────────────────────────────────────────────────────
# 2. ResNetBlock — Conv2d + skip (no BatchNorm — eval() folds it)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] ResNetBlock (Conv2d padding=1 + skip connection, eval mode)")

class ResNetBlock(nn.Module):
    """Standard residual block: x → Conv→BN→ReLU + x"""
    def __init__(self, channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn1(self.conv1(x))) + x   # BN folded in eval()

torch.manual_seed(42)
resnet_block = ResNetBlock(channels=4).eval()

# Warm-up running stats (BN needs a forward pass in train mode first)
resnet_block.train()
dummy_train = torch.randn(8, 4, 8, 8)
with torch.no_grad():
    resnet_block(dummy_train)
resnet_block.eval()

dummy_block = torch.randn(1, 4, 8, 8)
with torch.no_grad():
    ref_block_out = resnet_block(dummy_block)

save_bin(dummy_block.numpy().flatten(),
         os.path.join(OUT_DIR, "resnet_block_input.bin"))
save_bin(ref_block_out.numpy().flatten(),
         os.path.join(OUT_DIR, "resnet_block_output.bin"))
export_onnx(resnet_block, dummy_block, "resnet_block.onnx")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Verify ONNX graphs contain Add nodes (skip connections)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Verification")

try:
    import onnx
    for fname in ["tiny_resnet.onnx", "resnet_block.onnx"]:
        m = onnx.load(os.path.join(OUT_DIR, fname))
        ops = [n.op_type for n in m.graph.node]
        add_count = ops.count("Add")
        print(f"  {fname}: ops = {ops}, Add nodes = {add_count}")
        assert add_count >= 1, f"Expected at least 1 Add node in {fname}"
    print("  ✓ Both models contain Add (skip connection) nodes")
except ImportError:
    print("  (onnx package not installed — skip ONNX graph verification)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Print what to load in C# tests
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
Done. Files saved to: {OUT_DIR}

C# usage — TinyResNet:
    var model = OnnxGraphImporter.Load("test_fixtures/tiny_resnet.onnx", 8, 4);

C# usage — ResNetBlock (Conv + skip):
    // Input shape: [1, 4, 8, 8] → flat size = 256
    // Output shape: [1, 4, 8, 8] → flat size = 256
    var model = OnnxGraphImporter.Load("test_fixtures/resnet_block.onnx", 256, 256);

Tolerance: 1e-4f (float32 drift between PyTorch and Overfit TensorPrimitives).
""")