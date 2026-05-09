#!/usr/bin/env python3
"""
forward_pass_reference.py — Full numpy forward pass from GGUF for a single token.
Produces reference logits to compare with C# Overfit output.

Usage:
  python3 Scripts/forward_pass_reference.py \
    --gguf  d:/qwen.bin         (original GGUF)
    --token 151643
"""

import argparse, struct, sys, time
import numpy as np

# ── Dequantize ────────────────────────────────────────────────────────────────

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
            lo = qs[j] & 0x0F; hi = (qs[j] >> 4) & 0x0F
            h0 = (qh >> j) & 1; h1 = (qh >> (j+16)) & 1
            result[base+j   ] = d * (float(lo|(h0<<4)) - 16)
            result[base+j+16] = d * (float(hi|(h1<<4)) - 16)
    return result

def dequant_q8_0(raw, n):
    n_blocks = n // 32
    result = np.zeros(n, dtype=np.float32)
    for b in range(n_blocks):
        off = b * 34
        d  = np.frombuffer(raw[off:off+2], dtype=np.float16)[0].astype(np.float32)
        qs = np.frombuffer(raw[off+2:off+34], dtype=np.int8)
        result[b*32:(b+1)*32] = d * qs.astype(np.float32)
    return result

def dequant_q3_k(raw, n):
    """Q3_K: 110 bytes per 256 elements."""
    QK_K = 256
    n_blocks = n // QK_K
    result = np.zeros(n, dtype=np.float32)
    for b in range(n_blocks):
        off_b = b * 110
        d = np.frombuffer(raw[off_b+108:off_b+110], dtype=np.float16)[0].astype(np.float32)
        if not np.isfinite(d): d = 0.
        hmask  = np.frombuffer(raw[off_b:off_b+32],    dtype=np.uint8)  # [32]
        qs     = np.frombuffer(raw[off_b+32:off_b+96], dtype=np.uint8)  # [64]
        sc_raw = raw[off_b+96:off_b+108]                                 # [12]

        # Unpack scales: 8×6-bit values packed in 12 bytes (from llama.cpp)
        scales = np.zeros(8, dtype=np.float32)
        for i in range(4):
            scales[i]   = ((sc_raw[i] & 0xF) | ((sc_raw[i+8] & 0x03) << 4)) - 32
            scales[i+4] = ((sc_raw[i] >> 4)  | ((sc_raw[i+8] >> 2)   << 4)) - 32

        base = b * QK_K
        # Process 8 sub-blocks of 32 values each
        # qs has 2 bits per value: 64 bytes × 4 values/byte = 256 values (2 bits each)
        # hmask has 1 bit per value: 32 bytes × 8 bits = 256 high bits
        for s in range(8):
            sc = d * scales[s] / 4.0  # d/4 is common factor
            for i in range(32):
                # global element index
                gi = s * 32 + i
                # 2 low bits from qs (packed as 4 per byte)
                qs_byte = gi // 4
                qs_bit  = (gi % 4) * 2
                lo2 = (qs[qs_byte] >> qs_bit) & 0x03
                # 1 high bit from hmask
                hm_byte = gi // 8
                hm_bit  = gi % 8
                hi1 = (hmask[hm_byte] >> hm_bit) & 1
                # 3-bit value: hi1:lo2, signed offset by 4 → range [-4, 3]
                val = (lo2 | (hi1 << 2)) - 4
                result[base + gi] = sc * val
    return result

def dequant_q4_k(raw, n):
    QK_K = 256
    n_blocks = n // QK_K
    result = np.zeros(n, dtype=np.float32)
    for b in range(n_blocks):
        off = b * 144
        d    = np.frombuffer(raw[off:off+2], dtype=np.float16)[0].astype(np.float32)
        dmin = np.frombuffer(raw[off+2:off+4], dtype=np.float16)[0].astype(np.float32)
        if not np.isfinite(d): d = 0.
        if not np.isfinite(dmin): dmin = 0.
        sc = raw[off+4:off+16]
        qs = np.frombuffer(raw[off+16:off+144], dtype=np.uint8)
        scales = np.zeros(8, np.float32); mins = np.zeros(8, np.float32)
        for j in range(8):
            if j < 4:
                scales[j] = sc[j] & 0x3F; mins[j] = sc[j+4] & 0x3F
            else:
                scales[j] = (sc[j+4]&0xF) | ((sc[j-4]>>6)<<4)
                mins[j]   = (sc[j+4]>>4)  | ((sc[j+0]>>6)<<4)
        base = b * QK_K
        for s in range(4):
            qb = qs[s*32:(s+1)*32]
            lo = (qb & 0x0F).astype(np.float32)
            hi = ((qb >> 4) & 0x0F).astype(np.float32)
            result[base+s*64    :base+s*64+32] = d*scales[s*2  ]*lo - dmin*mins[s*2  ]
            result[base+s*64+32 :base+s*64+64] = d*scales[s*2+1]*hi - dmin*mins[s*2+1]
    return result

TYPES = {0:"F32",1:"F16",6:"Q5_0",8:"Q8_0",11:"Q3_K",12:"Q3_K",13:"Q3_K",14:"Q4_K",15:"Q4_K",16:"Q5_K",17:"Q5_K"}
BLOCK = {"F32":(1,4),"F16":(1,2),"Q5_0":(32,22),"Q8_0":(32,34),"Q3_K":(256,110),"Q4_K":(256,144),"Q5_K":(256,176)}

# ── GGUF loader ───────────────────────────────────────────────────────────────

class GGUF:
    def __init__(self, path):
        self._f = open(path,"rb")
        self.meta = {}; self._tinfo = {}; self._data_start = 0
        self._load_header()

    def _r(self, fmt): return struct.unpack(fmt, self._f.read(struct.calcsize(fmt)))
    def _str(self):
        n, = self._r("<Q"); return self._f.read(n).decode("utf-8","replace")
    def _val(self, t):
        if t==0: return self._r("<B")[0]
        if t in(4,): return self._r("<I")[0]
        if t in(5,): return self._r("<i")[0]
        if t in(6,): return self._r("<f")[0]
        if t in(7,): return self._r("<?")[0]
        if t in(8,): return self._str()
        if t in(9,):
            et,=self._r("<I"); cnt,=self._r("<Q")
            return [self._val(et) for _ in range(cnt)]
        if t in(10,): return self._r("<Q")[0]
        if t in(11,): return self._r("<q")[0]
        return None

    def _load_header(self):
        assert self._f.read(4)==b"GGUF"
        ver,=self._r("<I"); nt,=self._r("<Q"); nk,=self._r("<Q")
        print(f"GGUF v{ver}: {nt} tensors, {nk} metadata keys")
        for _ in range(nk):
            k=self._str(); vt,=self._r("<I"); self.meta[k]=self._val(vt)
        for _ in range(nt):
            name=self._str(); nd,=self._r("<I")
            dims=list(self._r(f"<{nd}Q")); dtype,=self._r("<I"); off,=self._r("<Q")
            self._tinfo[name]=(dims,dtype,off)
        pos=self._f.tell(); self._data_start=(pos+31)&~31

    def load(self, name):
        dims, dtype_id, offset = self._tinfo[name]
        tname = TYPES.get(dtype_id, f"UNK{dtype_id}")
        n = 1
        for d in dims: n *= d
        be, bb = BLOCK.get(tname, (1,4))
        self._f.seek(self._data_start + offset)
        raw = self._f.read((n//be)*bb)
        if tname == "F32":   arr = np.frombuffer(raw, dtype=np.float32).copy()
        elif tname == "Q8_0": arr = dequant_q8_0(raw, n)
        elif tname == "Q5_0": arr = dequant_q5_0(raw, n)
        elif tname == "Q3_K": arr = dequant_q3_k(raw, n)
        elif tname == "Q4_K": arr = dequant_q4_k(raw, n)
        elif tname in("Q5_K",): arr = dequant_q4_k(raw, n)  # approx
        else: raise NotImplementedError(f"Unsupported type {tname}")
        return arr.reshape(dims[::-1])  # PyTorch shape [out, in] for 2D

# ── Forward pass ──────────────────────────────────────────────────────────────

def rms_norm(x, gamma, eps=1e-6):
    rms = np.sqrt(np.mean(x*x) + eps)
    return x / rms * gamma

def silu(x): return x / (1 + np.exp(-x))

def rope_rotate(q, pos, theta=1e6, head_dim=64):
    half = head_dim // 2
    angles = np.array([pos / (theta ** (2*i/head_dim)) for i in range(half)], dtype=np.float32)
    cos = np.cos(angles); sin = np.sin(angles)
    q_rot = q.copy()
    q_rot[:half] = q[:half]*cos - q[half:]*sin
    q_rot[half:] = q[:half]*sin + q[half:]*cos
    return q_rot

def forward_one_token(g, token_id):
    arch = g.meta.get("general.architecture", "qwen2")
    n_layers   = g.meta.get(f"{arch}.block_count", 24)
    d_model    = g.meta.get(f"{arch}.embedding_length", 896)
    n_heads    = g.meta.get(f"{arch}.attention.head_count", 14)
    n_kv_heads = g.meta.get(f"{arch}.attention.head_count_kv", 2)
    d_ff       = g.meta.get(f"{arch}.feed_forward_length", 4864)
    rope_theta = float(g.meta.get(f"{arch}.rope.freq_base", 1e6))
    head_dim   = d_model // n_heads
    eps        = 1e-6

    print(f"Config: {n_layers}L d={d_model} h={n_heads}/{n_kv_heads} "
          f"hd={head_dim} dff={d_ff} θ={rope_theta:.0f}")

    # Embedding lookup
    emb = g.load("token_embd.weight")  # [vocab, d_model]
    x = emb[token_id].copy()
    print(f"x[0] after embed: {x[:5]}")

    position = 0
    for layer in range(n_layers):
        p = f"blk.{layer}"
        print(f"  Layer {layer}...", end="\r", flush=True)

        # Pre-attn RMSNorm
        ln1_gamma = g.load(f"{p}.attn_norm.weight")  # [d_model]
        x_norm = rms_norm(x, ln1_gamma, eps)

        # Q projection [n_heads*hd, d_model] × [d_model] → [n_heads*hd]
        Wq = g.load(f"{p}.attn_q.weight")  # [n_heads*hd, d_model]
        bq = g.meta  # will try to load bias
        try: bq_arr = g.load(f"{p}.attn_q.bias")
        except: bq_arr = np.zeros(n_heads*head_dim)
        q_all = Wq @ x_norm + bq_arr  # [n_heads*hd]

        # K projection [n_kv*hd, d_model]
        Wk = g.load(f"{p}.attn_k.weight")
        try: bk_arr = g.load(f"{p}.attn_k.bias")
        except: bk_arr = np.zeros(n_kv_heads*head_dim)
        k_all = Wk @ x_norm + bk_arr  # [n_kv*hd]

        # V projection
        Wv = g.load(f"{p}.attn_v.weight")
        try: bv_arr = g.load(f"{p}.attn_v.bias")
        except: bv_arr = np.zeros(n_kv_heads*head_dim)
        v_all = Wv @ x_norm + bv_arr

        # O projection [d_model, n_heads*hd]
        Wo = g.load(f"{p}.attn_output.weight")  # [d_model, n_heads*hd]

        # Single-position attention (trivial: score=1, out=V)
        attn_out = np.zeros(d_model)
        for h in range(n_heads):
            kvh = h % n_kv_heads
            q_h = q_all[h*head_dim:(h+1)*head_dim].copy()
            k_h = k_all[kvh*head_dim:(kvh+1)*head_dim].copy()
            v_h = v_all[kvh*head_dim:(kvh+1)*head_dim].copy()

            # RoPE
            q_h = rope_rotate(q_h, position, rope_theta, head_dim)
            k_h = rope_rotate(k_h, position, rope_theta, head_dim)

            # Single-token: attention weight = 1.0
            scale = 1.0 / np.sqrt(head_dim)
            score = np.dot(q_h, k_h) * scale
            # softmax([score]) = [1.0]
            context = v_h  # weighted sum = 1.0 * V

            # O projection for this head
            Wo_h = Wo[:, h*head_dim:(h+1)*head_dim]  # [d_model, hd]
            attn_out += Wo_h @ context

        # Residual
        x = x + attn_out

        if layer == 0:
            print(f"Layer 0 after attn_residual x[:5]  = {x[:5]}")

        # Pre-FFN RMSNorm
        ln2_gamma = g.load(f"{p}.ffn_norm.weight")
        x_norm2 = rms_norm(x, ln2_gamma, eps)

        # SwiGLU FFN
        Wgate = g.load(f"{p}.ffn_gate.weight")  # [dff, d_model]
        Wup   = g.load(f"{p}.ffn_up.weight")
        Wdown = g.load(f"{p}.ffn_down.weight")  # [d_model, dff]

        gate = silu(Wgate @ x_norm2)
        up   = Wup @ x_norm2
        ffn_out = Wdown @ (gate * up)

        x = x + ffn_out

        if layer == 0:
            print(f"Layer 0 after ffn_residual  x[:5]  = {x[:5]}")

    # Final RMSNorm
    final_gamma = g.load("output_norm.weight")
    x = rms_norm(x, final_gamma, eps)

    # LM head (tied embeddings)
    try: lm = g.load("output.weight")
    except: lm = emb
    logits = lm @ x  # [vocab_size]

    return logits

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf",  required=True)
    p.add_argument("--token", type=int, default=151643)
    args = p.parse_args()

    print(f"Loading {args.gguf}...")
    t0 = time.time()
    g = GGUF(args.gguf)

    print(f"\nRunning forward pass for token {args.token}...")
    logits = forward_one_token(g, args.token)
    print(f"\nDone in {time.time()-t0:.1f}s")

    top20 = np.argsort(logits)[-20:][::-1]
    print(f"\nLogit range: [{logits.min():.3f}, {logits.max():.3f}]  std={logits.std():.3f}")
    print("TOP-20 LOGITS (Python/GGUF reference):")
    for rank, tid in enumerate(top20):
        print(f"  #{rank+1:2d}  [{tid:7d}]  {logits[tid]:8.3f}")
    print(f"\nlogits[151644] = {logits[151644]:.4f}")
    print(f"logits[198]    = {logits[198]:.4f}")

if __name__ == "__main__":
    main()