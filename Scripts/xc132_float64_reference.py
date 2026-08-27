"""XC-132: a high-precision reference for position 0, to decide WHICH `quantize` arm is right.

THE QUESTION THIS ANSWERS. At token position 0 the two `quantize` arms disagree at cosine 0.906-0.979,
and llama.cpp cannot arbitrate: it shares the Q8 arithmetic with our `quantize:true` arm and sits on top
of it (cos 0.999979-0.999997). So a third arm is needed that shares arithmetic with NEITHER. This is it —
the same Q8_0 weights, dequantized once, and the whole forward pass done in float64 with numpy.

WHY POSITION 0 IS TRACTABLE AND NO OTHER POSITION IS. The model is causal, so at position 0 the attention
softmax runs over exactly one key and equals 1.0. The attention output is therefore exactly V, and Q, K,
the QK-RMSNorms and RoPE cannot affect the result at all. What remains is an embedding lookup, RMSNorm,
the V and O projections, SwiGLU and the final norm. No KV cache, no rotation, no mixing.

THE ARITHMETIC IS QUOTED FROM THE ENGINE, NOT ASSUMED:
  RMSNorm   CachedGptStack.FinalNorm      y[i] = x[i] / sqrt(mean(x^2) + eps) * gamma[i]
  SwiGLU    CachedFeedForwardBlock        down( silu(gate(h)) * up(h) ),  silu(x) = x * sigmoid(x)
  eps       qwen3.attention.layer_norm_rms_epsilon = 1e-6, read from the file
  GQA       16 query heads over 8 KV heads, so query head i reads KV head i // 2

THE SELF-CHECK IS THE PART THAT MAKES THE ANSWER USABLE. The same code runs at float32 first. Our
`quantize:false` arm dequantizes to F32 and computes in F32, so the float32 run must land close to it —
if it does not, this file is wrong and its float64 output describes nothing. Only then is the float64
number read. A transposed matrix or a missing embedding scale fails that check loudly.

    python Scripts/xc132_float64_reference.py
"""

import argparse
import json
import math
import pathlib
import sys

import gguf
import numpy as np
from gguf import GGUFReader

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = ROOT / "Tests" / "bin" / "Release" / "net10.0" / "xc132-per-position.json"
DEFAULT_GGUF = r"C:\qwen3-embed\Qwen3-Embedding-0.6B-Q8_0.gguf"


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class Model:
    def __init__(self, path):
        self.reader = GGUFReader(path)
        self.by_name = {t.name: t for t in self.reader.tensors}
        self.meta = {}
        for key, field in self.reader.fields.items():
            if key.startswith("qwen3."):
                self.meta[key] = field.parts[field.data[0]].tolist()[0]

        self.layers = int(self.meta["qwen3.block_count"])
        self.d_model = int(self.meta["qwen3.embedding_length"])
        self.heads = int(self.meta["qwen3.attention.head_count"])
        self.kv_heads = int(self.meta["qwen3.attention.head_count_kv"])
        self.head_dim = int(self.meta["qwen3.attention.value_length"])
        self.eps = float(self.meta["qwen3.attention.layer_norm_rms_epsilon"])

    def tensor(self, name, dtype):
        t = self.by_name[name]
        if t.tensor_type.name == "F32":
            return np.asarray(t.data, dtype=dtype)
        return gguf.quants.dequantize(t.data, t.tensor_type).astype(dtype)

    def embedding_row(self, token, dtype):
        """One row only. Dequantizing all 151669 rows costs 622 MB for 1024 numbers."""
        t = self.by_name["token_embd.weight"]
        raw = t.data[token:token + 1]
        return gguf.quants.dequantize(raw, t.tensor_type).astype(dtype).reshape(-1)


def rms_norm(x, gamma, eps):
    return x / np.sqrt(np.mean(x * x) + eps) * gamma


def silu(x):
    """x * sigmoid(x), computed branch-wise so exp() never overflows.

    The naive `x / (1 + exp(-x))` is mathematically identical and produces the right limit, but it raises
    an overflow warning on strongly negative inputs. A reference implementation that warns invites doubt
    about every number under it, which is the opposite of its job.
    """
    positive = x >= 0
    z = np.empty_like(x)
    z[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    e = np.exp(x[~positive])
    z[~positive] = e / (1.0 + e)
    return x * z


def forward_position_zero(model, token, dtype, trace=None):
    x = model.embedding_row(token, dtype)
    repeats = model.heads // model.kv_heads

    for layer in range(model.layers):
        p = "blk.%d." % layer

        h = rms_norm(x, model.tensor(p + "attn_norm.weight", dtype), model.eps)

        # Position 0: softmax over a single key is 1.0, so the attention output IS V.
        # Q, K, the QK-norms and RoPE are unreachable from here by construction.
        v = model.tensor(p + "attn_v.weight", dtype) @ h
        v_heads = v.reshape(model.kv_heads, model.head_dim)
        attended = np.repeat(v_heads, repeats, axis=0).reshape(-1)
        x = x + model.tensor(p + "attn_output.weight", dtype) @ attended

        h2 = rms_norm(x, model.tensor(p + "ffn_norm.weight", dtype), model.eps)
        gate = model.tensor(p + "ffn_gate.weight", dtype) @ h2
        up = model.tensor(p + "ffn_up.weight", dtype) @ h2
        x = x + model.tensor(p + "ffn_down.weight", dtype) @ (silu(gate) * up)

        if trace is not None:
            trace.append((layer, float(np.linalg.norm(x)), float(np.max(np.abs(x))),
                          int(np.argmax(np.abs(x)))))

    return rms_norm(x, model.tensor("output_norm.weight", dtype), model.eps)


def rope_tables(head_dim, theta, positions, dtype):
    """cos/sin for each requested position. freq_i = 1 / theta^(2i/head_dim), angle = pos * freq.

    Quoted from RopeTable: `freq = 1f / MathF.Pow(Theta, 2f * i / HeadDimension)`, `angle = pos * freq`.
    """
    half = head_dim // 2
    i = np.arange(half, dtype=np.float64)
    freq = 1.0 / np.power(np.float64(theta), 2.0 * i / head_dim)
    angle = np.outer(np.asarray(positions, dtype=np.float64), freq)
    return np.cos(angle).astype(dtype), np.sin(angle).astype(dtype)


def apply_rope_split_half(vectors, cos, sin):
    """Qwen3 pairing is SPLIT-HALF, not adjacent — `GgufLlamaLoader:137` puts qwen3 in the split-half set.

    Quoted from RopeKernel: x[i] = x0*cos[i] - x1*sin[i]; x[i+half] = x0*sin[i] + x1*cos[i],
    where x1 = x[i+half]. `vectors` is (positions, heads, head_dim); cos/sin are (positions, half).
    """
    half = vectors.shape[-1] // 2
    x0 = vectors[..., :half]
    x1 = vectors[..., half:]
    c = cos[:, None, :]
    s = sin[:, None, :]
    return np.concatenate([x0 * c - x1 * s, x0 * s + x1 * c], axis=-1)


def rms_norm_rows(x, gamma, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * gamma


def forward_sequence(model, tokens, dtype):
    """Full causal forward for a token sequence. Returns the post-final-norm state per position.

    Unlike forward_position_zero this runs the REAL attention path — QK-RMSNorm, split-half RoPE, the
    causal softmax and the GQA head mapping — which is precisely the machinery `XC-134` suspects. That
    makes it a third independent implementation of the thing under test, so read its self-checks first.
    """
    n = len(tokens)
    heads, kv_heads, hd = model.heads, model.kv_heads, model.head_dim
    repeats = heads // kv_heads
    theta = float(model.meta["qwen3.rope.freq_base"])
    cos, sin = rope_tables(hd, theta, range(n), dtype)
    scale = dtype(1.0 / math.sqrt(hd))

    # Causal mask: position p attends to 0..p.
    mask = np.triu(np.full((n, n), -np.inf, dtype=np.float64), 1).astype(dtype)

    x = np.stack([model.embedding_row(t, dtype) for t in tokens])

    for layer in range(model.layers):
        p = "blk.%d." % layer

        h = rms_norm_rows(x, model.tensor(p + "attn_norm.weight", dtype), model.eps)

        q = (h @ model.tensor(p + "attn_q.weight", dtype).T).reshape(n, heads, hd)
        k = (h @ model.tensor(p + "attn_k.weight", dtype).T).reshape(n, kv_heads, hd)
        v = (h @ model.tensor(p + "attn_v.weight", dtype).T).reshape(n, kv_heads, hd)

        # QK-RMSNorm FIRST, then RoPE — the order in CachedSingleHeadAttention.LoadQueryAndRope.
        # Its eps is a private const 1e-6, not the model's layer_norm_rms_epsilon; they coincide here.
        q = rms_norm_rows(q, model.tensor(p + "attn_q_norm.weight", dtype), dtype(1e-6))
        k = rms_norm_rows(k, model.tensor(p + "attn_k_norm.weight", dtype), dtype(1e-6))

        q = apply_rope_split_half(q, cos, sin)
        k = apply_rope_split_half(k, cos, sin)

        # GQA: query head j reads KV head j // repeats (`group = h / ctx.GroupSize`).
        k = np.repeat(k, repeats, axis=1)
        v = np.repeat(v, repeats, axis=1)

        scores = np.einsum("qhd,khd->hqk", q, k) * scale + mask
        scores = scores - scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        attended = np.einsum("hqk,khd->qhd", weights, v).reshape(n, heads * hd)

        x = x + attended @ model.tensor(p + "attn_output.weight", dtype).T

        h2 = rms_norm_rows(x, model.tensor(p + "ffn_norm.weight", dtype), model.eps)
        gate = h2 @ model.tensor(p + "ffn_gate.weight", dtype).T
        up = h2 @ model.tensor(p + "ffn_up.weight", dtype).T
        x = x + (silu(gate) * up) @ model.tensor(p + "ffn_down.weight", dtype).T

    return rms_norm_rows(x, model.tensor("output_norm.weight", dtype), model.eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", default=str(DEFAULT_CAPTURE))
    parser.add_argument("--gguf", default=DEFAULT_GGUF)
    parser.add_argument("--sequence", action="store_true",
                        help="run the full causal path and compare positions 0 and 1 (XC-134)")
    parser.add_argument("--text-index", type=int, default=0)
    parser.add_argument("--positions", type=int, default=2)
    arguments = parser.parse_args()

    if arguments.sequence:
        return sequence_mode(arguments)

    capture_path = pathlib.Path(arguments.capture)
    if not capture_path.exists():
        raise SystemExit("no capture at %s — run the per-position capture test first" % capture_path)
    capture = json.loads(capture_path.read_text(encoding="utf-8-sig"))

    model = Model(arguments.gguf)
    print("layers=%d d_model=%d heads=%d kv_heads=%d head_dim=%d eps=%g"
          % (model.layers, model.d_model, model.heads, model.kv_heads, model.head_dim, model.eps))
    print()

    texts = capture["texts"]
    ids = capture["token_ids"]
    arm_true = capture["quantize_true"]
    arm_false = capture["quantize_false"]

    print("%-5s %-8s %-12s %-12s %-12s %-12s %s"
          % ("text", "token", "f32 vs F", "f32 vs T", "f64 vs T", "f64 vs F", "verdict"))

    rows = []
    for t, text in enumerate(texts):
        token = ids[t][0]
        ours_t = np.asarray(arm_true[t][0], dtype=np.float64)
        ours_f = np.asarray(arm_false[t][0], dtype=np.float64)

        ref32 = forward_position_zero(model, token, np.float32)
        ref64 = forward_position_zero(model, token, np.float64)

        c32f = cosine(ref32, ours_f)
        c32t = cosine(ref32, ours_t)
        c64t = cosine(ref64, ours_t)
        c64f = cosine(ref64, ours_f)

        verdict = "TRUE closer" if c64t > c64f else "FALSE closer"
        print("%-5d %-8d %-12.6f %-12.6f %-12.6f %-12.6f %s"
              % (t, token, c32f, c32t, c64t, c64f, verdict))
        rows.append((t, token, c32f, c32t, c64t, c64f))

    print()
    print("SELF-CHECK — the float32 run must sit close to the `quantize:false` arm, which is also F32.")
    worst = min(r[2] for r in rows)
    print("  worst float32-vs-false cosine: %.6f" % worst)
    if worst < 0.99:
        print("  !! FAILED. This implementation does not reproduce the F32 arm, so its float64 output")
        print("     describes nothing. Do not read the verdict column above.")
        raise SystemExit(1)
    print("  passed — the float64 column is meaningful.")

    print()
    print("float64 reference norms and the two arms, for scale:")
    for t in range(len(texts)):
        token = ids[t][0]
        ref64 = forward_position_zero(model, token, np.float64)
        print("  text %d token %-8d |ref64|=%8.2f  |T|=%8.2f  |F|=%8.2f"
              % (t, token, np.linalg.norm(ref64),
                 np.linalg.norm(arm_true[t][0]), np.linalg.norm(arm_false[t][0])))


def sequence_mode(arguments):
    """XC-134: does the full causal path in float64 side with our engine or with llama.cpp?"""
    capture = json.loads(pathlib.Path(arguments.capture).read_text(encoding="utf-8-sig"))
    model = Model(arguments.gguf)

    t = arguments.text_index
    tokens = capture["token_ids"][t][:arguments.positions]
    ours_t = capture["quantize_true"][t]
    ours_f = capture["quantize_false"][t]

    print("text %d %r" % (t, capture["texts"][t][:60]))
    print("tokens used: %s" % tokens)
    print()

    print("SELF-CHECK 1 — RoPE at position 0 must be the identity rotation.")
    cos, sin = rope_tables(model.head_dim, float(model.meta["qwen3.rope.freq_base"]), [0], np.float64)
    probe = np.arange(model.head_dim, dtype=np.float64).reshape(1, 1, model.head_dim)
    rotated = apply_rope_split_half(probe, cos, sin)
    drift = float(np.max(np.abs(rotated - probe)))
    print("  max |rotate(x, pos=0) - x| = %.3e" % drift)
    if drift > 1e-12:
        raise SystemExit("  !! FAILED — RoPE is not the identity at position 0; the table is wrong.")
    print("  passed.")
    print()

    print("SELF-CHECK 2 — the full causal path must reproduce the ALREADY-VALIDATED position-0 result.")
    print("  That validates the embedding, both norms, V/O, SwiGLU and the degenerate one-key softmax.")
    seq64 = forward_sequence(model, tokens, np.float64)
    only0 = forward_position_zero(model, tokens[0], np.float64)
    c = cosine(seq64[0], only0)
    print("  cos(sequence[0], position-0-only) = %.8f" % c)
    if c < 0.999999:
        raise SystemExit("  !! FAILED — the two float64 paths disagree at position 0. Fix this first.")
    print("  passed. Only RoPE at position >= 1 and the multi-key softmax remain unvalidated.")
    print()

    seq32 = forward_sequence(model, tokens, np.float32)

    print("%-5s %-14s %-14s %-14s %-14s" % ("pos", "f64 vs ours-F", "f64 vs ours-T", "f32 vs ours-F", "f64 vs f32"))
    for k in range(len(tokens)):
        print("%-5d %-14.6f %-14.6f %-14.6f %-14.6f"
              % (k, cosine(seq64[k], ours_f[k]), cosine(seq64[k], ours_t[k]),
                 cosine(seq32[k], ours_f[k]), cosine(seq64[k], seq32[k])))

    print()
    print("READ IT THIS WAY. At position 0 the float64 arm already sided with ours-false (XC-132).")
    print("If position 1 does the same, our attention path is right and llama.cpp differs. If it")
    print("sides with NEITHER of our arms, the difference is ours and this is where it lives.")
    return 0


if __name__ == "__main__":
    main()
