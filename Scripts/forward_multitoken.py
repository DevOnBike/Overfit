#!/usr/bin/env python3
"""
Multi-token forward pass with KV cache — Overfit binary.
Tests: [BOS], [BOS, im_start], and the full 26-token chat prompt.

Usage: python3 Scripts/forward_multitoken.py --bin test_fixtures/qwen.bin
"""
import argparse, struct, time
import numpy as np

def rms_norm(x, gamma, eps=1e-6):
    rms = np.sqrt(np.mean(x.astype(np.float64)**2) + eps)
    return (x.astype(np.float64) / rms * gamma.astype(np.float64)).astype(np.float32)

def silu(x):
    return (x.astype(np.float64) / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)

def rope_rotate(v, pos, theta, half):
    """Adjacent-pair rotation (GPT-NeoX / llama.cpp NEOX style).
    Pairs (v[2i], v[2i+1]) share frequency i.
    Matches GGUF/llama.cpp weight ordering for Qwen2.
    """
    if pos == 0: return v.copy()
    v = v.copy()
    # frequencies for i = 0..half-1
    angles = np.float64(pos) / (theta ** (2.0 * np.arange(half, dtype=np.float64) / (2*half)))
    cos_a = np.cos(angles).astype(np.float32)
    sin_a = np.sin(angles).astype(np.float32)
    x_even = v[0::2].copy()  # v[0], v[2], v[4], ...
    x_odd  = v[1::2].copy()  # v[1], v[3], v[5], ...
    v[0::2] = x_even * cos_a - x_odd * sin_a
    v[1::2] = x_even * sin_a + x_odd * cos_a
    return v

def forward_sequence(emb, layers, fg2, lm_head, tokens, head_dim, n_heads, n_kv_heads, rope_theta, eps=1e-6):
    """
    Causal forward pass, KV-cache, one token at a time.
    Returns (final_hidden_before_norm, logits) at the LAST token.
    """
    half = head_dim // 2
    kv_cache = [[{'K': [], 'V': []} for _ in range(n_kv_heads)] for _ in range(len(layers))]

    x = None
    for pos, tid in enumerate(tokens):
        x = emb[tid].copy()
        for l, (ln1g, wq, bq, wk, bk, wv, bv, wo, bo, ln2g, fg, fu, fd) in enumerate(layers):
            x_norm = rms_norm(x, ln1g, eps)
            attn_out = np.zeros(x.shape, dtype=np.float32)
            for h in range(n_heads):
                kvh = h % n_kv_heads
                q_h = (wq[h].T @ x_norm + bq[h]).astype(np.float32)
                k_h = (wk[kvh].T @ x_norm + bk[kvh]).astype(np.float32)
                v_h = (wv[kvh].T @ x_norm + bv[kvh]).astype(np.float32)
                q_h = rope_rotate(q_h, pos, rope_theta, half)
                k_h = rope_rotate(k_h, pos, rope_theta, half)
                # Write to KV cache once per KV head (h == kvh is the canonical writer)
                if h == kvh:
                    kv_cache[l][kvh]['K'].append(k_h)
                    kv_cache[l][kvh]['V'].append(v_h)
                Ks = np.stack(kv_cache[l][kvh]['K'])   # [seqLen, headDim]
                Vs = np.stack(kv_cache[l][kvh]['V'])
                scores = (Ks @ q_h).astype(np.float64) / np.sqrt(head_dim)
                scores -= scores.max()
                w = np.exp(scores); w /= w.sum()
                ctx = (w @ Vs).astype(np.float32)
                attn_out += (wo[h].T @ ctx).astype(np.float32)
            x = x + attn_out
            x_norm2 = rms_norm(x, ln2g, eps)
            gate = silu(fg.T @ x_norm2)
            up   = (fu.T @ x_norm2).astype(np.float32)
            x    = x + (fd.T @ (gate * up)).astype(np.float32)

    hidden_before_norm = x.copy()
    x_final = rms_norm(x, fg2, eps)
    logits = (lm_head @ x_final).astype(np.float32)
    return hidden_before_norm, logits

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", required=True)
    args = p.parse_args()

    print("Loading weights...")
    t0 = time.time()
    with open(args.bin, "rb") as f:
        magic      = struct.unpack("<I", f.read(4))[0]
        _          = struct.unpack("<I", f.read(4))[0]
        n_layers   = struct.unpack("<i", f.read(4))[0]
        d_model    = struct.unpack("<i", f.read(4))[0]
        n_heads    = struct.unpack("<i", f.read(4))[0]
        n_kv_heads = struct.unpack("<i", f.read(4))[0]
        vocab_size = struct.unpack("<i", f.read(4))[0]
        _ctx       = struct.unpack("<i", f.read(4))[0]
        d_ff       = struct.unpack("<i", f.read(4))[0]
        _          = struct.unpack("<i", f.read(4))[0]
        rope_theta = struct.unpack("<f", f.read(4))[0]
        _          = struct.unpack("<i", f.read(4))[0]
        _          = struct.unpack("<i", f.read(4))[0]
        head_dim   = d_model // n_heads
        print(f"  {n_layers}L d={d_model} h={n_heads}/{n_kv_heads} ff={d_ff} vocab={vocab_size} head_dim={head_dim}")

        def rf(n): return np.frombuffer(f.read(n*4), dtype=np.float32).copy()
        emb = rf(vocab_size * d_model).reshape(vocab_size, d_model)
        layers = []
        for _ in range(n_layers):
            ln1g = rf(d_model); ln1b = rf(d_model)
            wq, bq = [], []
            for h in range(n_heads):
                wq.append(rf(d_model * head_dim).reshape(d_model, head_dim))
                bq.append(rf(head_dim))
            wk, bk, wv, bv = [], [], [], []
            for kv in range(n_kv_heads):
                wk.append(rf(d_model * head_dim).reshape(d_model, head_dim))
                bk.append(rf(head_dim))
                wv.append(rf(d_model * head_dim).reshape(d_model, head_dim))
                bv.append(rf(head_dim))
            wo, bo = [], []
            for h in range(n_heads):
                wo.append(rf(head_dim * d_model).reshape(head_dim, d_model))
                bo.append(rf(d_model))
            ln2g = rf(d_model); _ = rf(d_model)
            fg = rf(d_model * d_ff).reshape(d_model, d_ff)
            fu = rf(d_model * d_ff).reshape(d_model, d_ff)
            fd = rf(d_ff * d_model).reshape(d_ff, d_model)
            layers.append((ln1g, wq, bq, wk, bk, wv, bv, wo, bo, ln2g, fg, fu, fd))
        fg2 = rf(d_model); _ = rf(d_model)
        lm_head = rf(vocab_size * d_model).reshape(vocab_size, d_model)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ─── TEST 1: [BOS] single token ───────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 1: [BOS=151643] single token")
    h1, lg1 = forward_sequence(emb, layers, fg2, lm_head, [151643], head_dim, n_heads, n_kv_heads, rope_theta)
    top5 = np.argsort(lg1)[-5:][::-1]
    print(f"  Top-5: {[(int(t), float(lg1[t])) for t in top5]}")
    print(f"  logit[62406] = {lg1[62406]:.4f}  (C#: 11.9223 ← MUST MATCH)")
    print(f"  hidden[:4]   = {h1[:4]}")

    # ─── TEST 2: [BOS, im_start] ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST 2: [BOS=151643, im_start=151644] — 2 tokens")
    h2, lg2 = forward_sequence(emb, layers, fg2, lm_head, [151643, 151644], head_dim, n_heads, n_kv_heads, rope_theta)
    top5 = np.argsort(lg2)[-5:][::-1]
    print(f"  Top-5: {[(int(t), float(lg2[t])) for t in top5]}")
    print(f"  logit[6622] = {lg2[6622]:.4f}  (C#: 13.457)")
    print(f"  logit[8948]='system' = {lg2[8948]:.4f}  (C#: -7.9325)")
    print(f"  logit[198]='\\n'     = {lg2[198]:.4f}  (C#:  5.0972)")
    print(f"  hidden[:4]   = {h2[:4]}")
    print()
    print(f"  Compare hidden[:4] with C# session.LastHiddenState[:4]:")
    print(f"  Python: {h2[:4]}")
    print(f"  C# should print in L0_TwoToken_HiddenStateVsPython test above")

    # ─── TEST 3: Full 26-token chat prompt ────────────────────────────────────
    print("\n" + "="*60)
    chat_tokens = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13,
                   151645, 198, 151644, 872, 198, 3838, 374, 220, 17, 10,
                   17, 30, 151645, 198, 151644, 77091, 198]
    print(f"TEST 3: Full chat prompt ({len(chat_tokens)} tokens): 'What is 2+2?'")
    t3 = time.time()
    h3, lg3 = forward_sequence(emb, layers, fg2, lm_head, chat_tokens, head_dim, n_heads, n_kv_heads, rope_theta)
    top10 = np.argsort(lg3)[-10:][::-1]
    print(f"  Forward pass: {time.time()-t3:.1f}s")
    print(f"  Top-10 at position 25 (after assistant token):")
    for t in top10:
        print(f"    [{t:7d}] {lg3[t]:8.4f}")
    print(f"  logit[19]=' 4'     = {lg3[19]:.4f}  (working model should have this NEAR TOP)")
    print(f"  logit[19]=' 4' rank = {int(np.sum(lg3 > lg3[19]))+1}")
    print(f"  C# top-1: [72559] ':].' 11.778 ← should be completely different from Python")


    # ─── TEST 4: Correct Qwen2.5 system message ──────────────────────────────
    print("\n" + "="*60)
    print("TEST 4: Correct Qwen2.5 default system message")
    # Tokenize: You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
    # Need tokenizer - using hardcoded tokens from known encoding
    # "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n" (NO system)
    no_system_tokens = [151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198]
    print(f"  No-system format: {len(no_system_tokens)} tokens")
    h4, lg4 = forward_sequence(emb, layers, fg2, lm_head, no_system_tokens, head_dim, n_heads, n_kv_heads, rope_theta)
    top10_4 = np.argsort(lg4)[-10:][::-1]
    print(f"  Top-10 at last position (after assistant token):")
    for t in top10_4:
        print(f"    [{t:7d}] {lg4[t]:8.4f}")
    print(f"  logit[19]=' 4'   = {lg4[19]:.4f}")
    print(f"  logit[19]=' 4' rank = {int(np.sum(lg4 > lg4[19]))+1}")
    print(f"  logit[220]=' '  = {lg4[220]:.4f}")
    print(f"  (Ollama output: 'The result of adding two numbers 2 and 2 is four')")

    print()
    print("SUMMARY:")
    print(f"  With system msg [26 tok]:    top-1=[{np.argmax(lg3)}] {np.max(lg3):.3f}  \'4\' rank={int(np.sum(lg3>lg3[19]))+1}")
    print(f"  Without system msg [15 tok]: top-1=[{np.argmax(lg4)}] {np.max(lg4):.3f}  \'4\' rank={int(np.sum(lg4>lg4[19]))+1}")

if __name__ == "__main__":
    main()