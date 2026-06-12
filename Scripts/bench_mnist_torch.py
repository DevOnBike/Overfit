# Honest A/B reference: the EXACT Overfit MNIST beast architecture trained in PyTorch CPU on the
# same box / same data / same batch size, to quantify the training-throughput gap (see
# docs/mnist-cnn-training-audit.md). Arch: Conv(1->8,3x3,VALID) -> ReLU -> MaxPool2 -> FC(1352->64)
# -> ReLU -> FC(64->10), softmax CE, Adam(AdamW). Usage:  python bench_mnist_torch.py [threads]
import sys, time, struct
import numpy as np
import torch
import torch.nn as nn

DIR = r"d:\ml"

def read_idx_images(path):
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, 1, rows, cols).astype(np.float32) / 255.0

def read_idx_labels(path):
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

threads = int(sys.argv[1]) if len(sys.argv) > 1 else 16
torch.set_num_threads(threads)
print(f"torch {torch.__version__} | threads={threads}")

x = torch.from_numpy(read_idx_images(DIR + r"\train-images.idx3-ubyte"))
y = torch.from_numpy(read_idx_labels(DIR + r"\train-labels.idx1-ubyte"))
print(f"data: {x.shape}")

model = nn.Sequential(
    nn.Conv2d(1, 8, 3),            # VALID padding, jak Overfit
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(1352, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

BATCH = 128
EPOCHS = 5
steps = len(x) // BATCH

# warmup (JIT-like: oneDNN primitive caches)
for s in range(3):
    xb = x[s*BATCH:(s+1)*BATCH]; yb = y[s*BATCH:(s+1)*BATCH]
    opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()

t_run = time.perf_counter()
for epoch in range(EPOCHS):
    t0 = time.perf_counter()
    total = 0.0
    for s in range(steps):
        xb = x[s*BATCH:(s+1)*BATCH]
        yb = y[s*BATCH:(s+1)*BATCH]
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total/steps:.4f} | Time: {(time.perf_counter()-t0)*1000:.1f}ms")
print(f"TOTAL: {(time.perf_counter()-t_run)*1000:.0f} ms")
