#!/usr/bin/env python3
"""
forward_from_bin.py — Forward pass using PRE-DEQUANTIZED weights from qwen.bin.
No GGUF dequantization involved — reads float32 weights C# uses directly.

This is the RELIABLE reference — if C# and this disagree, the C# inference code is wrong.
If they agree, the binary weights have a bug.

Usage:
  python3 Scripts/forward_from_bin.py --bin test_fixtures/qwen.bin --token 151643
"""

import argparse, struct, time
import numpy as np

def read_f32(f, count):
    return np.frombuffer(f.read(count * 4), dtype=np.float32).copy()

def rms_norm(x, gamma, eps=1e-6):
    rms = np.sqrt(np.mean(x * x) + eps)
    return (x / rms) * gamma

def silu(x):
    return x / (1.0 + np.exp(-x.astype(np.float64))).astype(np.float32)

def rope_rotate(q, pos, theta=1e6, head_dim=64):
    """Split-half RoPE (LLaMA/Qwen style)."""
    if pos == 0:
        return q  # cos(0)=1, sin(0)=0 → identity
    half = head_dim // 2
    angles = np.array([pos / (theta ** (2*i/head_dim)) for i in range(half)], dtype=np.float32)
    cos_a = np.cos(angles); sin_a = np.sin(angles)
    q_rot = q.copy()
    q_rot[:half] = q[:half]*cos_a - q[half:]*sin_a
    q_rot[half:] = q[:half]*sin_a + q[half:]*cos_a
    return q_rot

def load_model(path):
    """Load all weights from qwen.bin into memory."""
    with open(path, "rb") as f:
        magic   = struct.unpack("<I", f.read(4))[0]
        version = struct.unpack("<I", f.read(4))[0]
        n_layers   = struct.unpack("<i", f.read(4))[0]
        d_model    = struct.unpack("<i", f.read(4))[0]
        n_heads    = struct.unpack("<i", f.read(4))[0]
        n_kv_heads = struct.unpack("<i", f.read(4))[0]
        vocab_size = struct.unpack("<i", f.read(4))[0]
        ctx        = struct.unpack("<i", f.read(4))[0]
        d_ff       = struct.unpack("<i", f.read(4))[0]
        use_rope   = struct.unpack("<i", f.read(4))[0]
        rope_theta = struct.unpack("<f", f.read(4))[0]
        ffn_act    = struct.unpack("<i", f.read(4))[0]
        tie_emb    = struct.unpack("<i", f.read(4))[0]
    
        head_dim = d_model // n_heads
        print(f"Config: {n_layers}L d={d_model} h={n_heads}/{n_kv_heads} hd={head_dim} dff={d_ff} θ={rope_theta:.0f}")

        # Embedding [vocab_size, d_model]
        emb = read_f32(f, vocab_size * d_model).reshape(vocab_size, d_model)
        print(f"Embed loaded: {emb.shape}")

        layers = []
        for l in range(n_layers):
            layer = {}

            # Attention norms
            layer['ln1_gamma'] = read_f32(f, d_model)   # gamma
            layer['ln1_beta']  = read_f32(f, d_model)   # beta (zeros)

            # Q heads: H × ([d_model*head_dim] + [head_dim])
            # Written as Wq.T = [d_model, head_dim] per head (kernel layout)
            # To get Wq[head_dim, d_model]: reshape and transpose
            wq_heads = []
            bq_heads = []
            for h in range(n_heads):
                wq_h = read_f32(f, d_model * head_dim).reshape(d_model, head_dim)  # [d_model, hd]
                bq_h = read_f32(f, head_dim)
                wq_heads.append(wq_h)
                bq_heads.append(bq_h)

            # KV heads: nKV × ([d_model*head_dim] + [head_dim]) × 2
            wk_heads = []; bk_heads = []
            wv_heads = []; bv_heads = []
            for kv in range(n_kv_heads):
                wk_h = read_f32(f, d_model * head_dim).reshape(d_model, head_dim)
                bk_h = read_f32(f, head_dim)
                wv_h = read_f32(f, d_model * head_dim).reshape(d_model, head_dim)
                bv_h = read_f32(f, head_dim)
                wk_heads.append(wk_h); bk_heads.append(bk_h)
                wv_heads.append(wv_h); bv_heads.append(bv_h)

            # O heads: H × ([head_dim*d_model] + [d_model])
            wo_heads = []; bo_heads = []
            for h in range(n_heads):
                wo_h = read_f32(f, head_dim * d_model).reshape(head_dim, d_model)  # [hd, d_model]
                bo_h = read_f32(f, d_model)
                wo_heads.append(wo_h); bo_heads.append(bo_h)

            # FFN norm
            layer['ln2_gamma'] = read_f32(f, d_model)
            layer['ln2_beta']  = read_f32(f, d_model)

            # FFN SwiGLU: gate [d_model, d_ff], up [d_model, d_ff], down [d_ff, d_model]
            layer['ffn_gate'] = read_f32(f, d_model * d_ff).reshape(d_model, d_ff)   # [d_model, dff]
            layer['ffn_up']   = read_f32(f, d_model * d_ff).reshape(d_model, d_ff)
            layer['ffn_down'] = read_f32(f, d_ff * d_model).reshape(d_ff, d_model)    # [dff, d_model]

            layer['wq'] = wq_heads; layer['bq'] = bq_heads
            layer['wk'] = wk_heads; layer['bk'] = bk_heads
            layer['wv'] = wv_heads; layer['bv'] = bv_heads
            layer['wo'] = wo_heads; layer['bo'] = bo_heads
            layers.append(layer)
            print(f"  Layer {l+1}/{n_layers} loaded", end="\r", flush=True)

        print(f"\nAll layers loaded.")

        # Final norm + LM head
        final_norm_gamma = read_f32(f, d_model)
        final_norm_beta  = read_f32(f, d_model)

        # LM head: stored as [d_model, vocab_size] (already transposed by C# loader)
        # File has LM head as [vocab_size, d_model] (same as embedding, no transposition written)
        # C# does TransposeLmHead at load time → [d_model, vocab_size] in memory
        # Python: read as [vocab_size, d_model], compute logits = lm_head @ x (no .T needed)
        lm_head_raw = read_f32(f, vocab_size * d_model).reshape(vocab_size, d_model)
        print(f"LM head loaded: {lm_head_raw.shape} (raw from file)")

    cfg = dict(n_layers=n_layers, d_model=d_model, n_heads=n_heads,
               n_kv_heads=n_kv_heads, head_dim=head_dim, d_ff=d_ff,
               rope_theta=rope_theta)
    return cfg, emb, layers, final_norm_gamma, lm_head_raw

def forward(cfg, emb, layers, final_norm_gamma, lm_head_raw, token_id, position=0):
    D = cfg['d_model']; H = cfg['n_heads']; KV = cfg['n_kv_heads']
    HD = cfg['head_dim']; theta = cfg['rope_theta']
    eps = 1e-6

    # Embedding lookup
    x = emb[token_id].copy()
    print(f"Embed[{token_id}][:5] = {x[:5]}")

    for l, layer in enumerate(layers):
        # Pre-attn RMSNorm
        x_norm = rms_norm(x, layer['ln1_gamma'], eps)

        if l == 0:
            print(f"Layer0 ln1_gamma[:3]  = {layer['ln1_gamma'][:3]}")
            print(f"Layer0 x_norm[:5]     = {x_norm[:5]}")
        if l in (1, 12, 23):
            print(f"Layer {l:2d} LN1 x_norm[:5] = {x_norm[:5]}")

        # Attention (position 0 → trivial single-token attention)
        attn_out = np.zeros(D, dtype=np.float32)
        for h in range(H):
            kvh = h % KV
            # Q, K, V projections
            # Kernel layout: weights [d_model, head_dim] → output = weights^T @ input
            q_h = layer['wq'][h].T @ x_norm + layer['bq'][h]   # [hd, d_model]^T @ [d_model] = [hd]
            k_h = layer['wk'][kvh].T @ x_norm + layer['bk'][kvh]
            v_h = layer['wv'][kvh].T @ x_norm + layer['bv'][kvh]

            if l == 0 and h == 0:
                print(f"Layer0 head0 V[:5]    = {v_h[:5]}")

            # RoPE (identity at pos=0)
            q_h = rope_rotate(q_h, position, theta, HD)
            k_h = rope_rotate(k_h, position, theta, HD)

            # Single-token: context = v_h
            context = v_h

            # O projection: kernel [hd, d_model] → output = weights^T @ context
            attn_out += layer['wo'][h].T @ context   # [hd, d_model]^T @ [hd] = [d_model]

        if l == 0:
            print(f"Layer0 attn_out[:5]   = {attn_out[:5]}")
        # Residual
        x = x + attn_out
        if l == 0:
            print(f"Layer 0 after attn_residual x[:5] = {x[:5]}")

        # Pre-FFN RMSNorm
        x_norm2 = rms_norm(x, layer['ln2_gamma'], eps)
        if l == 0:
            print(f"Layer0 x_norm2[:5]    = {x_norm2[:5]}")

        # SwiGLU FFN
        # gate/up: kernel [d_model, d_ff] → output = weights^T @ input
        gate = layer['ffn_gate'].T @ x_norm2   # [d_model, dff]^T @ [d_model] = [dff]
        up   = layer['ffn_up'].T   @ x_norm2
        intermediate = silu(gate) * up

        # down: kernel [d_ff, d_model] → output = weights^T @ input
        ffn_out = layer['ffn_down'].T @ intermediate   # [dff, d_model]^T @ [dff] = [d_model]

        x = x + ffn_out
        if l == 0:
            print(f"Layer 0 after ffn_residual  x[:5] = {x[:5]}")
        if l in (0, 1, 11, 22, 23):
            print(f"Layer {l:2d} output x[:5] = {x[:5]}")

    # Final RMSNorm
    x = rms_norm(x, final_norm_gamma, eps)

    # LM head: [d_model, vocab_size], kernel: output = weights^T @ input
    logits = lm_head_raw @ x   # [vocab, d_model] @ [d_model] = [vocab] @ [d_model] = [vocab]  (same as kernel: embedding[v,:] @ hidden)

    return logits

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", required=True)
    p.add_argument("--token", type=int, default=151643)
    args = p.parse_args()

    t0 = time.time()
    print(f"Loading {args.bin}...")
    cfg, emb, layers, final_norm_gamma, lm_head_raw = load_model(args.bin)

    print(f"\nForward pass for token {args.token}...")
    logits = forward(cfg, emb, layers, final_norm_gamma, lm_head_raw, args.token)
    print(f"Done in {time.time()-t0:.1f}s")

    top20 = np.argsort(logits)[-20:][::-1]
    print(f"\nLogit range: [{logits.min():.3f}, {logits.max():.3f}]  std={logits.std():.3f}")
    print("TOP-20 (from qwen.bin weights — same as C# uses):")
    for r, tid in enumerate(top20):
        print(f"  #{r+1:2d}  [{tid:7d}]  {logits[tid]:8.3f}")
    print(f"\nlogits[151644] = {logits[151644]:.4f}")
    print(f"logits[198]    = {logits[198]:.4f}")
    print(f"logits[3352]   = {logits[3352]:.4f}  (Python GGUF ref top-1)")
    print(f"logits[50560]  = {logits[50560]:.4f}  (C# top-1)")

if __name__ == "__main__":
    main()