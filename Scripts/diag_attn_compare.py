"""Diagnostic (task #95): compare HF vs Overfit attention distributions (last query) per layer/head,
same 3B GGUF. Finds which heads attend differently — points at the broken attention op."""
import numpy as np

hf = np.load(r"D:\Overfit\hf_attn.npy")  # [layers, heads, kv]
L, H, KV = hf.shape
print(f"HF attn {hf.shape}")

ov = []
with open(r"D:\Overfit\overfit_attn.txt") as f:
    first = f.readline()  # header
    for line in f:
        line = line.strip()
        if not line:
            continue
        _, vals = line.split(":", 1)
        ov.append(np.array([float(x) for x in vals.split()]))
print(f"Overfit distributions: {len(ov)} (expected {L*H}={L}x{H})")

# Assume Overfit order = layer*H + head for the first L*H entries.
print("layer | mean |HF-OV| per head (max over heads) | worst head | HF top-key/OV top-key on worst head")
for layer in range(min(L, 4)):
    diffs = []
    for head in range(H):
        idx = layer * H + head
        if idx >= len(ov):
            break
        o = ov[idx]
        h = hf[layer, head]
        n = min(len(o), len(h))
        diffs.append(np.abs(o[:n] - h[:n]).max())
    diffs = np.array(diffs)
    worst = int(diffs.argmax())
    o = ov[layer * H + worst]
    h = hf[layer, worst]
    print(f"  {layer} | maxdiff={diffs.max():.4f} worst_head={worst} | "
          f"HF argmax key={int(h.argmax())}({h.max():.3f})  OV argmax key={int(o.argmax())}({o.max():.3f})")
