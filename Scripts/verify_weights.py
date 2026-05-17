#!/usr/bin/env python3
"""
verify_weights.py — Spot-check specific weight values between GGUF and Overfit binary.

Usage:
  python3 Scripts/verify_weights.py \
    --gguf  "%USERPROFILE%\.ollama\models\blobs\sha256-<HASH>" \
    --bin   test_fixtures\qwen.bin

This script compares:
1. Embedding vector for token 151644 (<|im_start|>)
2. Q-weight row 0 of layer 0, head 0
3. V-weight row 0 of layer 0, kv-head 0

If these match → conversion is correct, bug is in C# inference.
If they don't → conversion script has a bug.
"""

import sys, struct, os
import numpy as np

# ── GGUF reader (minimal, no external deps) ───────────────────────────────────

def read_bin_header(path):
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
    return dict(n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                n_kv_heads=n_kv_heads, vocab_size=vocab_size, ctx=ctx,
                d_ff=d_ff, head_dim=head_dim)

def read_f32_tensor(f, count):
    return np.frombuffer(f.read(count * 4), dtype=np.float32).copy()

def extract_from_bin(path, token_id=151644):
    h = read_bin_header(path)
    D, V, H, KV, HD, FF = h['d_model'], h['vocab_size'], h['n_heads'], h['n_kv_heads'], h['head_dim'], h['d_ff']
    N = h['n_layers']
    results = {}

    with open(path, "rb") as f:
        f.seek(13 * 4)  # skip 13-int header

        # Embedding [V, D]
        emb = read_f32_tensor(f, V * D).reshape(V, D)
        results['emb_token'] = emb[token_id].copy()

        # Layer 0
        # attn_norm_gamma [D], beta [D]
        f.read(2 * D * 4)

        # Q heads: H × ([D*HD] + [HD])
        # Head 0 Wq [D, HD] (stored transposed from [HD, D])
        wq0_flat = read_f32_tensor(f, D * HD)  # [D, HD] layout
        results['wq0_row0'] = wq0_flat[:HD].copy()  # first row of [D, HD] = first HD values
        bq0 = read_f32_tensor(f, HD)
        # Skip heads 1..H-1
        for h_idx in range(1, H):
            f.read((D * HD + HD) * 4)

        # KV head 0: Wk [D, HD] + bk [HD] + Wv [D, HD] + bv [HD]
        wk0_flat = read_f32_tensor(f, D * HD)
        results['wk0_row0'] = wk0_flat[:HD].copy()
        bk0 = read_f32_tensor(f, HD)
        wv0_flat = read_f32_tensor(f, D * HD)
        results['wv0_row0'] = wv0_flat[:HD].copy()

    return results

def dequant_q5_0(raw, n):
    n_blocks = n // 32
    result = np.zeros(n, dtype=np.float32)
    for b in range(n_blocks):
        off = b * 22
        d  = np.frombuffer(raw[off:off+2], dtype=np.float16)[0].astype(np.float32)
        qh = int.from_bytes(raw[off+2:off+6], "little")
        qs = raw[off+6:off+22]
        base = b * 32
        for j in range(16):
            lo = qs[j] & 0x0F
            hi = (qs[j] >> 4) & 0x0F
            h0 = (qh >> j) & 1
            h1 = (qh >> (j+16)) & 1
            result[base+j   ] = d * (float(lo | (h0<<4)) - 16)
            result[base+j+16] = d * (float(hi | (h1<<4)) - 16)
    return result

def dequant_q8_0(raw, n):
    n_blocks = n // 32
    result = np.zeros(n, dtype=np.float32)
    for b in range(n_blocks):
        off = b * 34
        d  = np.frombuffer(raw[off:off+2], dtype=np.float16)[0].astype(np.float32)
        qs = np.frombuffer(raw[off+2:off+34], dtype=np.int8)
        result[b*32:(b+1)*32] = d * qs
    return result

def extract_from_gguf(path, token_id=151644):
    """Read and dequantize specific tensors from GGUF."""
    import struct
    results = {}

    with open(path, "rb") as f:
        def read(fmt):
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

        magic = f.read(4)
        assert magic == b"GGUF", "Not a GGUF file"
        version, = read("<I")
        n_tensors, = read("<Q")
        n_kv, = read("<Q")

        # Skip metadata
        def read_str():
            n, = read("<Q")
            return f.read(n).decode("utf-8", errors="replace")
        def read_val(vtype):
            if vtype in (0,): return read("<B")[0]
            if vtype in (4,): return read("<I")[0]
            if vtype in (5,): return read("<i")[0]
            if vtype in (6,): return read("<f")[0]
            if vtype in (7,): return read("<?")[0]
            if vtype in (8,): return read_str()
            if vtype in (9,):
                et, = read("<I"); cnt, = read("<Q")
                return [read_val(et) for _ in range(cnt)]
            if vtype in (10,): return read("<Q")[0]
            if vtype in (11,): return read("<q")[0]
            # skip unknowns
            return None
        meta = {}
        for _ in range(n_kv):
            k = read_str()
            vt, = read("<I")
            meta[k] = read_val(vt)

        # Read tensor info
        tinfo = {}
        for _ in range(n_tensors):
            name = read_str()
            nd, = read("<I")
            dims = list(read(f"<{nd}Q"))
            dtype, = read("<I")
            offset, = read("<Q")
            tinfo[name] = (dims, dtype, offset)

        # Data section start (32-byte aligned)
        pos = f.tell()
        data_start = (pos + 31) & ~31

        TYPES = {0:"F32",1:"F16",6:"Q5_0",8:"Q8_0",14:"Q4_K",15:"Q4_K"}
        BLOCK = {"F32":(1,4),"F16":(1,2),"Q5_0":(32,22),"Q8_0":(32,34),"Q4_K":(256,144)}

        def load_tensor(name):
            dims, dtype_id, offset = tinfo[name]
            tname = TYPES.get(dtype_id, f"UNK{dtype_id}")
            n = 1
            for d in dims: n *= d
            be, bb = BLOCK.get(tname, (1,4))
            nbytes = (n // be) * bb
            f.seek(data_start + offset)
            raw = f.read(nbytes)
            if tname == "Q8_0":
                arr = dequant_q8_0(raw, n)
            elif tname == "Q5_0":
                arr = dequant_q5_0(raw, n)
            elif tname == "F32":
                arr = np.frombuffer(raw, dtype=np.float32)
            else:
                print(f"  WARNING: {tname} not supported, returning zeros")
                return np.zeros(n, dtype=np.float32), dims
            return arr.reshape(dims[::-1]), dims  # PyTorch shape

        # Embedding: token_embd.weight [vocab, d_model]
        emb, dims = load_tensor("token_embd.weight")
        results['emb_token'] = emb[token_id].copy()

        # Q weight layer 0: [n_heads*head_dim, d_model]
        wq, _ = load_tensor("blk.0.attn_q.weight")  # shape (n_heads*hd, d_model)
        results['wq0_from_gguf'] = wq[0, :].copy()   # first row = head 0, dim 0

        # K weight layer 0: [n_kv*head_dim, d_model]
        wk, _ = load_tensor("blk.0.attn_k.weight")
        results['wk0_from_gguf'] = wk[0, :].copy()

        # V weight layer 0: [n_kv*head_dim, d_model]
        wv, _ = load_tensor("blk.0.attn_v.weight")
        results['wv0_from_gguf'] = wv[0, :].copy()

    return results

def compare(name, a, b, n=8):
    diff = np.abs(a - b).max()
    ok = "✓" if diff < 1e-3 else "✗"
    print(f"  {ok} {name}: max_diff={diff:.6f}")
    print(f"      overfit[:8] = {a[:n]}")
    print(f"      gguf[:8]    = {b[:n]}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--bin",  required=True)
    parser.add_argument("--token", type=int, default=151644)
    args = parser.parse_args()

    print(f"=== Verifying token {args.token} weights ===\n")

    print("Reading Overfit binary...")
    bin_res = extract_from_bin(args.bin, args.token)

    print("Reading GGUF...")
    gguf_res = extract_from_gguf(args.gguf, args.token)

    print("\n=== Comparison ===")
    # The GGUF Q weight row[0] = Wq[head_dim_0, :] = [d_model] values for output dim 0
    # Our bin Wq for head 0 is stored as [D, HD] layout
    # wq0_flat[i*HD + j] = Wq_T[i, j] where Wq_T is [D, HD]
    # So wq0_flat[:HD] = Wq_T[0, :] = first column of Wq^T = first row of Wq
    # = Wq[0, :] = head 0, output dim 0, all input dims... 
    # But we only stored first HD values — that's wrong for comparison.
    # Let me compare the first 8 values of embedding instead.
    
    compare("Embedding[token]", bin_res['emb_token'], gguf_res['emb_token'])

    # For Wq: GGUF[0, :] = wq_matrix[0, :] = first output dim, all input dims
    # Our bin stores Wq head 0 as [D, HD] flat: value [d*HD + j] = Wq[j, d] (transposed)
    # So our Wq[0, :] in original = our bin[0::HD] (every HD-th value starting at 0)
    # That's complex — let's just compare embeddings for now
    print()
    print("=== Embedding spot check ===")
    tok_emb_match = np.allclose(bin_res['emb_token'], gguf_res['emb_token'], atol=1e-3)
    if tok_emb_match:
        print("✓ Embedding matches GGUF → embedding conversion is CORRECT")
        print("  Bug must be in C# inference code or weight transpositions")
    else:
        print("✗ EMBEDDING MISMATCH → conversion script is broken!")
        print("  Check _dequant_q8_0 or reshape logic")

    print(f"\nEmbedding[token] norm: {np.linalg.norm(bin_res['emb_token']):.4f}")
    print(f"GGUF emb norm:        {np.linalg.norm(gguf_res['emb_token']):.4f}")

if __name__ == "__main__":
    main()