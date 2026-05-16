#!/usr/bin/env python3
"""
PyTorch CPU training-step benchmark — apples-to-apples counterpart of Overfit's
`GPT1CpuUtilizationProbeTests`.

It builds the SAME 4-layer GPT-1 architecture Overfit's probe trains
(dModel=128, dFF=512, 4 heads, vocab=68, seqLen=128, untied LM head, pre-LN,
tanh-approx GELU) and times a full training step — forward + cross-entropy
loss + backward + grad-norm clip + AdamW step — using the identical
methodology: a few warmup steps, then N timed steps, reported as wall/step.

This is intentionally a standalone Python script rather than a BenchmarkDotNet
method (mirrors `benchmark_pytorch_mnist_cnn.py`): it keeps Python startup and
the torch import out of the .NET process and out of BDN's measurement.

Requirements:
    pip install torch

Usage:
    python benchmark_pytorch_gpt1_training.py
    python benchmark_pytorch_gpt1_training.py --threads 32 --steps 20

Notes on fairness:
    - Overfit's `OverfitParallelFor` spawns Environment.ProcessorCount workers
      (32 logical on the dev box). PyTorch defaults to physical-core count
      (16). Pass --threads to match a specific count; the script prints
      whatever it used so the comparison is explicit.
    - Both sides measure the SAME thing: one optimizer step over one batch,
      post-warmup, wall-clock.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Config — must match GPT1CpuUtilizationProbeTests ────────────────────────
VOCAB_SIZE = 68
CONTEXT_LENGTH = 128
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
D_FF = 512
SEQ_LEN = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0


class TransformerBlock(nn.Module):
    """Pre-LN transformer block: x + Attn(LN(x)), x + FFN(LN(x))."""

    def __init__(self) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(
            D_MODEL, N_HEADS, batch_first=True, bias=True)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff1 = nn.Linear(D_MODEL, D_FF)
        self.ff2 = nn.Linear(D_FF, D_MODEL)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        h = self.ln2(x)
        h = self.ff2(F.gelu(self.ff1(h), approximate="tanh"))
        return x + h


class Gpt1(nn.Module):
    """Decoder-only transformer matching Overfit's GPT1Model (untied head)."""

    def __init__(self) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(CONTEXT_LENGTH, D_MODEL)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        causal = torch.triu(
            torch.full((t, t), float("-inf"), device=idx.device), diagonal=1)
        for block in self.blocks:
            x = block(x, causal)
        return self.head(self.ln_f(x))


def run_for_batch(batch_size: int, steps: int, warmup: int) -> None:
    torch.manual_seed(42)

    model = Gpt1()
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    rng = torch.Generator().manual_seed(123)
    corpus = torch.randint(
        0, VOCAB_SIZE, (1_115_394,), generator=rng, dtype=torch.long)

    def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(
            0, corpus.numel() - SEQ_LEN - 1, (batch_size,), generator=rng)
        inp = torch.stack([corpus[s:s + SEQ_LEN] for s in starts])
        tgt = torch.stack([corpus[s + 1:s + 1 + SEQ_LEN] for s in starts])
        return inp, tgt

    def one_step() -> None:
        inp, tgt = sample_batch()
        optimizer.zero_grad(set_to_none=True)
        logits = model(inp)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

    for _ in range(warmup):
        one_step()

    start = time.perf_counter()
    for _ in range(steps):
        one_step()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print(f"--- BatchSize={batch_size}, SeqLen={SEQ_LEN}, Steps={steps} ---")
    print(f"  wall time total:  {elapsed_ms:.1f} ms "
          f"({elapsed_ms / steps:.2f} ms/step)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0,
                        help="0 = leave PyTorch default")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    print(f"PyTorch {torch.__version__}, CPU threads: {torch.get_num_threads()}")
    print(f"Model: GPT-1 {N_LAYERS}L, dModel={D_MODEL}, dFF={D_FF}, "
          f"heads={N_HEADS}, vocab={VOCAB_SIZE}")
    print()

    for batch_size in (8, 16, 32):
        run_for_batch(batch_size, args.steps, args.warmup)


if __name__ == "__main__":
    main()
