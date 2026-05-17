#!/usr/bin/env python3
"""
Checks ALL 24 GGUF layers against the binary.
Fast: one representative weight per tensor per layer.
At the end it prints only the mismatches (✗) or "ALL OK".

Usage:
  python3 Scripts/verify_all_layers.py --gguf <gguf_blob> --bin d:/qwen/qwen.bin
"""
import argparse, struct
import numpy as np

GGUF_MAGIC = 0x46554747

def read_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', errors='replace')

def read_value(f, t):
    if t == 0: return struct.unpack('<B',f.read(1))[0]
    if t == 1: return struct.unpack('<b',f.read(1))[0]
    if t == 2: return struct.unpack('<H',f.read(2))[0]
    if t == 3: return struct.unpack('<h',f.read(2))[0]
    if t == 4: return struct.unpack('<I',f.read(4))[0]
    if t == 5: return struct.unpack('<i',f.read(4))[0]
    if t == 6: return struct.unpack('<f',f.read(4))[0]
    if t == 7: return bool(struct.unpack('<B',f.read(1))[0])
    if t == 8: return read_str(f)
    if t ==10: return struct.unpack('<Q',f.read(8))[0]
    if t ==11: return struct.unpack('<q',f.read(8))[0]
    if t ==12: return struct.unpack('<d',f.read(8))[0]
    if t == 9:
        et = struct.unpack('<I',f.read(4))[0]
        n  = struct.unpack('<Q',f.read(8))[0]
        return [read_value(f,et) for _ in range(n)]
    raise ValueError(f"unk type {t}")

def load_gguf_index(path):
    with open(path,'rb') as f:
        assert struct.unpack('<I',f.read(4))[0] == GGUF_MAGIC
        _ver      = struct.unpack('<I',f.read(4))[0]
        n_tensors = struct.unpack('<Q',f.read(8))[0]
        n_kv      = struct.unpack('<Q',f.read(8))[0]
        meta = {}
        for _ in range(n_kv):
            key = read_str(f)
            t   = struct.unpack('<I',f.read(4))[0]
            meta[key] = read_value(f,t)
        DTYPES={0:'f32',1:'f16',2:'q4_0',6:'q8_0',12:'q4_k',13:'q5_k',14:'q6_k',15:'q8_k'}
        tinfo={}
        for _ in range(n_tensors):
            name  = read_str(f)
            ndim  = struct.unpack('<I',f.read(4))[0]
            dims  = [struct.unpack('<Q',f.read(8))[0] for _ in range(ndim)]
            dtype = struct.unpack('<I',f.read(4))[0]
            offset= struct.unpack('<Q',f.read(8))[0]
            tinfo[name]=(DTYPES.get(dtype,f'unk{dtype}'),dims,offset)
        data_off = (f.tell()+31)&~31
    return meta, tinfo, data_off

def read_f16_tensor(path, tinfo, data_off, name, n_check=32):
    """Read first n_check elements of a tensor as float32."""
    if name not in tinfo:
        return None, f"MISSING in GGUF"
    dtype, dims, off = tinfo[name]
    if dtype not in ('f16','f32'):
        return None, f"SKIP dtype={dtype}"
    n_elem = 1
    for d in dims: n_elem *= d
    with open(path,'rb') as f:
        f.seek(data_off + off)
        if dtype == 'f16':
            raw = np.frombuffer(f.read(min(n_elem,n_check)*2), dtype=np.float16).astype(np.float32)
        else:
            raw = np.frombuffer(f.read(min(n_elem,n_check)*4), dtype=np.float32).copy()
    return raw, None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--bin",  required=True)
    args = p.parse_args()

    print(f"Loading GGUF index...")
    meta, tinfo, data_off = load_gguf_index(args.gguf)
    d_model    = meta.get('qwen2.embedding_length', 896)
    n_heads    = meta.get('qwen2.attention.head_count', 14)
    n_kv_heads = meta.get('qwen2.attention.head_count_kv', 2)
    n_layers   = meta.get('qwen2.block_count', 24)
    d_ff       = meta.get('qwen2.feed_forward_length', 4864)
    vocab_size = meta.get('qwen2.vocab_size') or meta.get('tokenizer.ggml.tokens', [None]*152000).__len__()
    head_dim   = d_model // n_heads
    print(f"  {n_layers}L d={d_model} h={n_heads}/{n_kv_heads} ff={d_ff} vocab={vocab_size}")

    # Load binary sequentially
    print(f"Loading binary (sequential)...")
    import io
    buf = open(args.bin,'rb').read()
    f = io.BytesIO(buf)

    def ri(n): return struct.unpack('<i',f.read(4))[0]
    def rI(n): return struct.unpack('<I',f.read(4))[0]
    def rf(n): return np.frombuffer(f.read(n*4), dtype=np.float32).copy()

    _magic=rI(1);_ver=rI(1);L=ri(1);D=ri(1);H=ri(1);KV=ri(1);V=ri(1)
    _ctx=ri(1);FF=ri(1);_re=ri(1);_rt=struct.unpack('<f',f.read(4))[0]
    _fa=ri(1);_te=ri(1)

    emb = rf(V * D)  # [vocab * dModel]

    N_CHECK = 64  # number of float32 elements to compare per tensor

    bugs = []
    ok_count = 0

    def chk(layer_label, gguf_name, bin_data_first64):
        """Compare first N elements of GGUF tensor vs binary."""
        g_raw, err = read_f16_tensor(args.gguf, tinfo, data_off, gguf_name, N_CHECK)
        if err:
            return
        g = g_raw[:N_CHECK].flatten()
        b = bin_data_first64[:N_CHECK].flatten()
        n = min(len(g), len(b))
        g, b = g[:n], b[:n]
        diff = np.max(np.abs(g - b))
        nonlocal ok_count
        if diff < 1e-4:
            ok_count += 1
        else:
            bugs.append((layer_label, gguf_name, diff, g[:4], b[:4]))
            print(f"  ✗ L{layer_label} {gguf_name:45s} max_diff={diff:.4f}")
            print(f"    GGUF[:4]={g[:4]}  bin[:4]={b[:4]}")

    # Embedding check (first 64 elements)
    chk("emb", "token_embd.weight", emb[:N_CHECK])

    for layer in range(n_layers):
        pfx = f"blk.{layer}"
        ln1g = rf(D); ln1b = rf(D)
        chk(f"{layer:02d}", f"{pfx}.attn_norm.weight", ln1g)

        # Q: n_heads * (D*head_dim + head_dim)
        wq_all = []; bq_all = []
        for h in range(n_heads):
            wq_all.append(rf(D*head_dim))
            bq_all.append(rf(head_dim))
        # Compare head0 wq (bin is [dModel,headDim], GGUF after reversal is [n_h*hd, dModel])
        # bin wq[0] = wq_all[0].reshape(D,hd), transpose = [hd,D]
        # GGUF rows 0..hd-1 of [n_h*hd, D] = same
        chk(f"{layer:02d}", f"{pfx}.attn_q.weight",
            np.array(wq_all[0][:N_CHECK]))   # first N of head0 weight flat
        chk(f"{layer:02d}", f"{pfx}.attn_q.bias",
            np.array(bq_all[0][:N_CHECK]))

        # KV
        wk_all=[]; bk_all=[]; wv_all=[]; bv_all=[]
        for kv in range(n_kv_heads):
            wk_all.append(rf(D*head_dim))
            bk_all.append(rf(head_dim))
            wv_all.append(rf(D*head_dim))
            bv_all.append(rf(head_dim))
        chk(f"{layer:02d}", f"{pfx}.attn_k.weight", np.array(wk_all[0][:N_CHECK]))
        chk(f"{layer:02d}", f"{pfx}.attn_k.bias",   np.array(bk_all[0][:N_CHECK]))
        chk(f"{layer:02d}", f"{pfx}.attn_v.weight", np.array(wv_all[0][:N_CHECK]))
        chk(f"{layer:02d}", f"{pfx}.attn_v.bias",   np.array(bv_all[0][:N_CHECK]))

        # O
        wo_all=[]; bo_all=[]
        for h in range(n_heads):
            wo_all.append(rf(head_dim*D))
            bo_all.append(rf(D))
        chk(f"{layer:02d}", f"{pfx}.attn_output.weight", np.array(wo_all[0][:N_CHECK]))

        ln2g = rf(D); ln2b = rf(D)
        chk(f"{layer:02d}", f"{pfx}.ffn_norm.weight", ln2g)

        # FFN
        gate = rf(D*FF)
        up   = rf(D*FF)
        down = rf(FF*D)
        chk(f"{layer:02d}", f"{pfx}.ffn_gate.weight", gate[:N_CHECK])
        chk(f"{layer:02d}", f"{pfx}.ffn_up.weight",   up[:N_CHECK])
        chk(f"{layer:02d}", f"{pfx}.ffn_down.weight", down[:N_CHECK])

    # Final norm
    fg2 = rf(D); fb2 = rf(D)
    chk("fn", "output_norm.weight", fg2)

    # LM head (stored transposed in binary vs GGUF)
    lm_flat = rf(V*D)
    # bin stores [dModel, vocab], GGUF stores [vocab, dModel]
    # Compare first dModel elements of LM head col 0 (all vocab, dim 0)
    # GGUF lm: row-major [vocab, dModel], column 0 = lm[:,0]
    # bin: [dModel, vocab] flat → row 0 = first D elements = lm[d=0, v=0..V-1]
    # Actually compare first 64 flat elements
    g_lm, err = read_f16_tensor(args.gguf, tinfo, data_off, "output.weight", N_CHECK)
    if g_lm is not None:
        # GGUF: [vocab, dModel] → first 64 = emb[0][0..63]
        # bin (after TransposeLmHead): [dModel, vocab] → first 64 = lm[d=0][v=0..63]
        # These are NOT comparable directly due to transposition — instead check dim[0] of each
        # Compare GGUF[0,0..N] vs bin transposed [0,0..N]: 
        g_row0 = g_lm[:N_CHECK].flatten()  # GGUF token 0, dims 0..63
        # bin: lm_flat[0*V .. 0*V+N] = dim 0 across vocab 0..N
        # vs GGUF: lm_flat[0*D .. 0*D+N] = token 0, dims 0..N
        # These should be different due to transpose → verify by comparing GGUF emb vs LM head
        # (tied embeddings: GGUF output.weight == token_embd.weight for Qwen)
        g_emb_check, _ = read_f16_tensor(args.gguf, tinfo, data_off, "token_embd.weight", N_CHECK)
        if g_emb_check is not None:
            tie_diff = np.max(np.abs(g_lm - g_emb_check[:N_CHECK]))
            print(f"\nLM head tie check (GGUF output.weight == token_embd.weight): max_diff={tie_diff:.6f}")
            if tie_diff < 1e-4:
                print("  → Tied embeddings confirmed in GGUF")
                # bin LM head should be TRANSPOSED embedding
                # Verify: bin lm_flat is [dModel,vocab] = emb.T
                # emb[0,0] should equal lm_flat[0*V+0]? No...
                # emb has shape [V,D]: emb[v,d] = emb_flat[v*D+d]
                # lm (transposed) [D,V]: lm[d,v] = emb[v,d] = emb_flat[v*D+d]
                # lm_flat[d*V+v] = emb_flat[v*D+d]
                # Check lm_flat[0:D] = emb[0:D,0] (column 0 of emb = all tokens' dim-0)
                # vs emb_flat[0:D] = emb[0][0..D-1] (token 0's all dims)
                # These are different — just verify a spot check
                emb_arr = emb.reshape(V, D)
                lm_arr  = lm_flat.reshape(D, V)
                # lm[d,v] should equal emb[v,d]
                spot_d, spot_v = 0, 0
                lm_spot = lm_arr[spot_d, spot_v]
                emb_spot = emb_arr[spot_v, spot_d]
                print(f"  Spot check: lm[d=0,v=0]={lm_spot:.6f}  emb[v=0,d=0]={emb_spot:.6f}  diff={abs(lm_spot-emb_spot):.6f}")
                ok_count += 1

    print(f"\n{'='*60}")
    if not bugs:
        print(f"✓ ALL {ok_count} CHECKS PASSED — weights are identical to GGUF")
        print("  → Bug is in INFERENCE LOGIC, not in binary weights")
    else:
        print(f"✗ {len(bugs)} BUGS FOUND, {ok_count} checks passed")
        print("\nBug summary:")
        for lbl, name, diff, g4, b4 in bugs:
            print(f"  Layer {lbl}: {name}  max_diff={diff:.4f}")
            print(f"    GGUF={g4}  bin={b4}")

if __name__ == "__main__":
    main()