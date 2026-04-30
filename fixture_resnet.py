"""
fixture_resnet.py — generuje modele ONNX do testów OnnxGraphImporter i AveragePool.

Uruchomienie:
    python fixture_resnet.py

Wymagania:
    pip install torch onnx numpy

Generowane pliki (w tests/test_fixtures/):
    tiny_resnet.onnx / _input.bin / _output.bin
        Linear(8,8) + skip + Linear(8,4)

    resnet_block.onnx / _input.bin / _output.bin
        Conv2d(4,4,3,padding=1) + BN(folded) + ReLU + skip [1,4,8,8]

    avgpool_model.onnx / _input.bin / _output.bin
        Conv2d(1,4,3,padding=1) + AveragePool(3,stride=1,padding=1) + GAP [1,1,8,8]
"""

import os, struct, numpy as np, torch, torch.nn as nn

OUT_DIR = os.path.join("tests", "test_fixtures")
os.makedirs(OUT_DIR, exist_ok=True)

def save_bin(arr, name):
    path = os.path.join(OUT_DIR, name)
    arr.astype("<f4").flatten().tofile(path)
    print(f"  saved: {path}  ({arr.size} float32)")
    return path

def export(model, dummy, name):
    path = os.path.join(OUT_DIR, name)
    torch.onnx.export(model, dummy, path, export_params=True,
                      input_names=["input"], output_names=["output"])
    print(f"  exported: {path}")
    return path


# ─── 1. TinyResNet ────────────────────────────────────────────────────────────
print("\n[1] TinyResNet (Linear + skip connection)")

class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 4)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)) + x)

torch.manual_seed(42)
m = TinyResNet().eval()
dummy = torch.randn(1, 8)
with torch.no_grad(): out = m(dummy)
save_bin(dummy.numpy(), "tiny_resnet_input.bin")
save_bin(out.numpy(), "tiny_resnet_output.bin")
export(m, dummy, "tiny_resnet.onnx")


# ─── 2. ResNetBlock ───────────────────────────────────────────────────────────
print("\n[2] ResNetBlock (Conv2d padding=1 + BN folded + skip)")

class ResNetBlock(nn.Module):
    def __init__(self, c=4):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(c)
        self.relu  = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn1(self.conv1(x))) + x

torch.manual_seed(42)
blk = ResNetBlock(4)
blk.train()
with torch.no_grad(): blk(torch.randn(8, 4, 8, 8))  # warm BN running stats
blk.eval()

dummy_blk = torch.randn(1, 4, 8, 8)
with torch.no_grad(): out_blk = blk(dummy_blk)
save_bin(dummy_blk.numpy(), "resnet_block_input.bin")
save_bin(out_blk.numpy(), "resnet_block_output.bin")
export(blk, dummy_blk, "resnet_block.onnx")


# ─── 3. AveragePool model ─────────────────────────────────────────────────────
print("\n[3] AveragePool model (Conv2d + AveragePool(3,p=1) + GAP)")

class AvgPoolModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)  # same-size pooling
        self.gap  = nn.AdaptiveAvgPool2d(1)               # GAP → [B, 4, 1, 1]
        self.fc   = nn.Linear(4, 2)
    def forward(self, x):
        x = torch.relu(self.conv(x))  # [1,4,8,8]
        x = self.pool(x)               # [1,4,8,8]
        x = self.gap(x)                # [1,4,1,1]
        x = x.flatten(1)               # [1,4]
        return self.fc(x)              # [1,2]

torch.manual_seed(42)
am = AvgPoolModel().eval()
dummy_ap = torch.randn(1, 1, 8, 8)
with torch.no_grad(): out_ap = am(dummy_ap)
save_bin(dummy_ap.numpy(), "avgpool_model_input.bin")
save_bin(out_ap.numpy(), "avgpool_model_output.bin")
export(am, dummy_ap, "avgpool_model.onnx")


# ─── 4. Verify Add nodes ──────────────────────────────────────────────────────
print("\n[4] Verification")
try:
    import onnx
    for fname, expected_add in [
        ("tiny_resnet.onnx", 1),
        ("resnet_block.onnx", 1),
        ("avgpool_model.onnx", 0),
    ]:
        m2 = onnx.load(os.path.join(OUT_DIR, fname))
        ops = [n.op_type for n in m2.graph.node]
        add_count = ops.count("Add")
        avg_count = ops.count("AveragePool")
        print(f"  {fname}: ops={ops}, Add={add_count}, AveragePool={avg_count}")
        assert add_count >= expected_add, f"Expected {expected_add} Add in {fname}"
    print("  ✓ All models verified")
except ImportError:
    print("  (onnx not installed — skip verification)")

print(f"""
Done. Sizes:
  tiny_resnet:    input=8, output=4
  resnet_block:   input=256 (1x4x8x8), output=256
  avgpool_model:  input=64 (1x1x8x8), output=2
""")