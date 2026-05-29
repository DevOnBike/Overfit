"""Diagnostic (task #95): per-layer cosine between HF and Overfit last-position hidden states
(both the SAME 3B Q4_K_M GGUF). Finds the layer where Overfit's forward first diverges in direction."""
import numpy as np

hf = np.load(r"D:\Overfit\hf_hidden.npy").astype(np.float64)   # [36, 2048] block outputs (hidden_states[1..36])
ov = []
with open(r"D:\Overfit\overfit_hidden.txt") as f:
    for line in f:
        line = line.strip()
        if line:
            ov.append(np.array([float(x) for x in line.split()], dtype=np.float64))
ov = np.stack(ov)   # [36, 2048] block 0..35 outputs

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

print(f"hf {hf.shape}  overfit {ov.shape}")
print("layer | cosine | HF_L2 | OV_L2 | argmax_dim HF/OV | maxabs HF/OV")
for k in range(min(len(hf), len(ov))):
    h, o = hf[k], ov[k]
    print(f"  {k:2d} | {cos(h, o):+.5f} | {np.linalg.norm(h):8.2f} | {np.linalg.norm(o):8.2f} | "
          f"{int(np.argmax(np.abs(h))):4d}/{int(np.argmax(np.abs(o))):4d} | "
          f"{np.abs(h).max():8.1f}/{np.abs(o).max():8.1f}")
