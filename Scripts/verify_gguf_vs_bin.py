#!/usr/bin/env python3
"""
Compare GGUF FP16 weights against the Overfit binary.
Verifies: embedding, layer0 (Q, K, V, O, FFN gate/up/down).

Usage:
  python3 Scripts/verify_gguf_vs_bin.py \
      --gguf "%USERPROFILE%/.ollama/models/blobs/<sha256-hash>" \
      --bin  test_fixtures/qwen.bin
"""
import argparse, struct, sys
import numpy as np

# ─── GGUF parser ──────────────────────────────────────────────────────────────
GGUF_MAGIC = 0x46554747
GGUF_TYPES = {0:'u8',1:'i8',2:'u16',3:'i16',4:'u32',5:'i32',6:'f32',7:'bool',8:'str',9:'arr',10:'u64',11:'i64',12:'f64'}

def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', errors='replace')

def read_value(f, t):
    if t == 0: return struct.unpack('<B', f.read(1))[0]
    if t == 1: return struct.unpack('<b', f.read(1))[0]
    if t == 2: return struct.unpack('<H', f.read(2))[0]
    if t == 3: return struct.unpack('<h', f.read(2))[0]
    if t == 4: return struct.unpack('<I', f.read(4))[0]
    if t == 5: return struct.unpack('<i', f.read(4))[0]
    if t == 6: return struct.unpack('<f', f.read(4))[0]
    if t == 7: return bool(struct.unpack('<B', f.read(1))[0])
    if t == 8: return read_str(f)
    if t == 10: return struct.unpack('<Q', f.read(8))[0]
    if t == 11: return struct.unpack('<q', f.read(8))[0]
    if t == 12: return struct.unpack('<d', f.read(8))[0]
    if t == 9:
        et = struct.unpack('<I', f.read(4))[0]
        n  = struct.unpack('<Q', f.read(8))[0]
        return [read_value(f, et) for _ in range(n)]
    raise ValueError(f"Unknown type {t}")

def load_gguf(path):
    """Returns (metadata dict, tensor_info dict {name: (dtype_str, dims, offset)})"""
    with open(path, 'rb') as f:
        magic   = struct.unpack('<I', f.read(4))[0]
        assert magic == GGUF_MAGIC, f"Not a GGUF file (magic={magic:#x})"
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv      = struct.unpack('<Q', f.read(8))[0]

        meta = {}
        for _ in range(n_kv):
            key = read_str(f)
            t   = struct.unpack('<I', f.read(4))[0]
            meta[key] = read_value(f, t)

        DTYPES = {0:'f32', 1:'f16', 2:'q4_0', 6:'q8_0', 12:'q4_k', 13:'q5_k', 14:'q6_k', 15:'q8_k'}
        tensor_info = {}
        for _ in range(n_tensors):
            name  = read_str(f)
            ndim  = struct.unpack('<I', f.read(4))[0]
            dims  = [struct.unpack('<Q', f.read(8))[0] for _ in range(ndim)]
            dtype = struct.unpack('<I', f.read(4))[0]
            offset= struct.unpack('<Q', f.read(8))[0]
            tensor_info[name] = (DTYPES.get(dtype, f'unk{dtype}'), dims, offset)

        data_offset = f.tell()
        # align to 32 bytes
        data_offset = (data_offset + 31) & ~31
        return meta, tensor_info, data_offset

def read_gguf_tensor(path, tensor_info, data_offset, name):
    dtype_str, dims, off = tensor_info[name]
    n_elem = 1
    for d in dims: n_elem *= d
    with open(path, 'rb') as f:
        f.seek(data_offset + off)
        if dtype_str == 'f16':
            raw = np.frombuffer(f.read(n_elem * 2), dtype=np.float16).astype(np.float32)
        elif dtype_str == 'f32':
            raw = np.frombuffer(f.read(n_elem * 4), dtype=np.float32).copy()
        else:
            print(f"  [SKIP] {name}: dtype={dtype_str} (not F16/F32 — need dequant)")
            return None
    return raw.reshape(dims[::-1])  # reverse dims → numpy C-order

# ─── Binary parser ─────────────────────────────────────────────────────────────
def load_binary_layer0(bin_path):
    with open(bin_path, 'rb') as f:
        _magic   = struct.unpack('<I', f.read(4))[0]
        _ver     = struct.unpack('<I', f.read(4))[0]
        n_layers = struct.unpack('<i', f.read(4))[0]
        d_model  = struct.unpack('<i', f.read(4))[0]
        n_heads  = struct.unpack('<i', f.read(4))[0]
        n_kv_heads = struct.unpack('<i', f.read(4))[0]
        vocab    = struct.unpack('<i', f.read(4))[0]
        _ctx     = struct.unpack('<i', f.read(4))[0]
        d_ff     = struct.unpack('<i', f.read(4))[0]
        _rope_en = struct.unpack('<i', f.read(4))[0]
        _rope_th = struct.unpack('<f', f.read(4))[0]
        _ffn_act = struct.unpack('<i', f.read(4))[0]
        _tie_emb = struct.unpack('<i', f.read(4))[0]
        head_dim = d_model // n_heads

        def rf(n): return np.frombuffer(f.read(n*4), dtype=np.float32).copy()

        emb = rf(vocab * d_model).reshape(vocab, d_model)

        # Layer 0
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
        ln2g = rf(d_model); ln2b = rf(d_model)
        ffn_gate = rf(d_model * d_ff).reshape(d_model, d_ff)
        ffn_up   = rf(d_model * d_ff).reshape(d_model, d_ff)
        ffn_down = rf(d_ff * d_model).reshape(d_ff, d_model)

    return dict(
        emb=emb, ln1g=ln1g, head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
        wq=wq, bq=bq, wk=wk, bk=bk, wv=wv, bv=bv, wo=wo, bo=bo,
        ln2g=ln2g, ffn_gate=ffn_gate, ffn_up=ffn_up, ffn_down=ffn_down,
        d_model=d_model, d_ff=d_ff, vocab=vocab
    )

def compare(name, gguf_raw, bin_arr, atol=1e-3):
    """Compare GGUF tensor (after reshape) vs binary tensor. Returns True if OK."""
    if gguf_raw is None:
        print(f"  SKIP {name}")
        return None
    g = gguf_raw.astype(np.float32).flatten()
    b = bin_arr.astype(np.float32).flatten()
    if g.shape != b.shape:
        print(f"  SHAPE MISMATCH {name}: gguf={g.shape} bin={b.shape} ✗")
        return False
    max_diff = float(np.max(np.abs(g - b)))
    mean_diff = float(np.mean(np.abs(g - b)))
    ok = max_diff < atol
    mark = "✓" if ok else "✗ BUG FOUND"
    print(f"  {mark} {name:50s}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
    if not ok:
        # Show first few mismatches
        diff_idx = np.where(np.abs(g - b) > atol)[0][:5]
        for idx in diff_idx:
            print(f"     elem[{idx}]: gguf={g[idx]:.6f}  bin={b[idx]:.6f}  diff={g[idx]-b[idx]:.6f}")
        print(f"  GGUF[:5] = {g[:5]}")
        print(f"  bin[:5]  = {b[:5]}")
    return ok

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True, help="Path to GGUF FP16 blob")
    p.add_argument("--bin",  required=True, help="Path to qwen.bin")
    args = p.parse_args()

    print(f"Loading GGUF: {args.gguf}")
    meta, tinfo, data_off = load_gguf(args.gguf)

    d_model    = meta.get('qwen2.embedding_length', 896)
    n_heads    = meta.get('qwen2.attention.head_count', 14)
    n_kv_heads = meta.get('qwen2.attention.head_count_kv', 2)
    head_dim   = d_model // n_heads
    print(f"  Config: d={d_model} h={n_heads}/{n_kv_heads} head_dim={head_dim}")
    print(f"  Tensors: {len(tinfo)}")

    def gt(name):
        return read_gguf_tensor(args.gguf, tinfo, data_off, name)

    print(f"\nLoading binary: {args.bin}")
    B = load_binary_layer0(args.bin)

    print("\n─── EMBEDDING ───────────────────────────────────────────────────────")
    g_emb = gt("token_embd.weight")
    if g_emb is not None:
        compare("emb[BOS=151643]", g_emb[151643:151644], B['emb'][151643:151644])
        compare("emb[im_start=151644]", g_emb[151644:151645], B['emb'][151644:151645])

    print("\n─── LAYER 0 ATTENTION Q ─────────────────────────────────────────────")
    g_wq = gt("blk.0.attn_q.weight")   # GGUF: [d_model, n_heads*head_dim] after reversal
    if g_wq is not None:
        # Binary stores per head: wq[h] = [d_model, head_dim], same as Wq.T per head
        # GGUF after reversal: shape [n_heads*head_dim, d_model] (n_heads*head_dim rows)
        print(f"  GGUF wq shape after reversal: {g_wq.shape}")
        for h in [0, 1, 13]:
            # Per-head slice from GGUF: rows h*64..(h+1)*64 → [head_dim, d_model]
            # Binary: wq[h] = [d_model, head_dim], so compare g_wq_h.T vs B['wq'][h]
            g_h = g_wq[h*head_dim:(h+1)*head_dim, :]  # [64, 896]
            b_h = B['wq'][h]                            # [896, 64]
            compare(f"wq head{h} (GGUF vs bin.T)", g_h, b_h.T)

    g_bq = gt("blk.0.attn_q.bias")
    if g_bq is not None:
        print(f"  GGUF bq shape: {g_bq.shape}")
        for h in [0, 1]:
            g_h = g_bq[h*head_dim:(h+1)*head_dim]
            compare(f"bq head{h}", g_h, B['bq'][h])

    print("\n─── LAYER 0 ATTENTION K ─────────────────────────────────────────────")
    g_wk = gt("blk.0.attn_k.weight")   # [n_kv_heads*head_dim, d_model] after reversal
    if g_wk is not None:
        print(f"  GGUF wk shape after reversal: {g_wk.shape}")
        for kv in [0, 1]:
            g_kv = g_wk[kv*head_dim:(kv+1)*head_dim, :]  # [64, 896]
            compare(f"wk kvhead{kv} (GGUF vs bin.T)", g_kv, B['wk'][kv].T)

    g_bk = gt("blk.0.attn_k.bias")
    if g_bk is not None:
        for kv in [0, 1]:
            compare(f"bk kvhead{kv}", g_bk[kv*head_dim:(kv+1)*head_dim], B['bk'][kv])

    print("\n─── LAYER 0 ATTENTION V ─────────────────────────────────────────────")
    g_wv = gt("blk.0.attn_v.weight")
    if g_wv is not None:
        print(f"  GGUF wv shape after reversal: {g_wv.shape}")
        for kv in [0, 1]:
            g_kv = g_wv[kv*head_dim:(kv+1)*head_dim, :]
            compare(f"wv kvhead{kv} (GGUF vs bin.T)", g_kv, B['wv'][kv].T)

    g_bv = gt("blk.0.attn_v.bias")
    if g_bv is not None:
        for kv in [0, 1]:
            compare(f"bv kvhead{kv}", g_bv[kv*head_dim:(kv+1)*head_dim], B['bv'][kv])

    print("\n─── LAYER 0 ATTENTION O ─────────────────────────────────────────────")
    g_wo = gt("blk.0.attn_output.weight")   # [d_model, n_heads*head_dim] after reversal
    if g_wo is not None:
        print(f"  GGUF wo shape after reversal: {g_wo.shape}")
        for h in [0, 1, 13]:
            # Per-head slice from GGUF: cols h*64..(h+1)*head_dim
            # Binary: wo[h] = [head_dim, d_model]
            g_h = g_wo[:, h*head_dim:(h+1)*head_dim]  # [d_model, head_dim]
            compare(f"wo head{h} (GGUF vs bin.T)", g_h, B['wo'][h].T)

    print("\n─── LAYER 0 ATTN NORM (RMSNorm gamma) ───────────────────────────────")
    g_ln1g = gt("blk.0.attn_norm.weight")
    if g_ln1g is not None:
        compare("attn_norm_gamma", g_ln1g.flatten(), B['ln1g'])

    print("\n─── LAYER 0 FFN ─────────────────────────────────────────────────────")
    g_gate = gt("blk.0.ffn_gate.weight")  # [d_ff, d_model] after reversal
    if g_gate is not None:
        print(f"  GGUF gate shape: {g_gate.shape}")
        # Binary: ffn_gate = [d_model, d_ff]
        compare("ffn_gate (GGUF vs bin.T)", g_gate, B['ffn_gate'].T)

    g_up = gt("blk.0.ffn_up.weight")
    if g_up is not None:
        compare("ffn_up (GGUF vs bin.T)", g_up, B['ffn_up'].T)

    g_down = gt("blk.0.ffn_down.weight")  # [d_model, d_ff] after reversal
    if g_down is not None:
        print(f"  GGUF down shape: {g_down.shape}")
        compare("ffn_down (GGUF vs bin.T)", g_down, B['ffn_down'].T)

    print("\nDone.")

if __name__ == "__main__":
    main()