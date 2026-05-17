#!/usr/bin/env python3
"""
debug_q4k.py — Compares our Q4_K dequantization against numpy reference.
Reads blk.0.ffn_down.weight from GGUF and dequantizes block 0 manually,
printing d, dmin, scales, mins, and first 32 values.

Usage:
  python3 Scripts/debug_q4k.py --gguf d:/qwen.bin
"""
import argparse, struct
import numpy as np

def read_gguf_tensor(path, tensor_name):
    """Read raw bytes of a specific tensor from GGUF."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF"
        version, = struct.unpack("<I", f.read(4))
        n_tensors, = struct.unpack("<Q", f.read(8))
        n_kv,      = struct.unpack("<Q", f.read(8))

        def rstr():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", errors="replace")
        def rval(t):
            if t in (0,): return struct.unpack("<B", f.read(1))[0]
            if t in (4,): return struct.unpack("<I", f.read(4))[0]
            if t in (5,): return struct.unpack("<i", f.read(4))[0]
            if t in (6,): return struct.unpack("<f", f.read(4))[0]
            if t in (7,): return struct.unpack("<?", f.read(1))[0]
            if t in (8,): return rstr()
            if t in (9,):
                et, = struct.unpack("<I", f.read(4))
                cnt, = struct.unpack("<Q", f.read(8))
                return [rval(et) for _ in range(cnt)]
            if t in (10,): return struct.unpack("<Q", f.read(8))[0]
            if t in (11,): return struct.unpack("<q", f.read(8))[0]
            return None

        for _ in range(n_kv):
            k = rstr(); vt, = struct.unpack("<I", f.read(4)); rval(vt)

        tinfo = {}
        for _ in range(n_tensors):
            name = rstr()
            nd, = struct.unpack("<I", f.read(4))
            dims = list(struct.unpack(f"<{nd}Q", f.read(8*nd)))
            dtype, = struct.unpack("<I", f.read(4))
            offset, = struct.unpack("<Q", f.read(8))
            tinfo[name] = (dims, dtype, offset)

        pos = f.tell()
        data_start = (pos + 31) & ~31

        if tensor_name not in tinfo:
            raise KeyError(f"{tensor_name} not in GGUF. Keys: {list(tinfo.keys())[:5]}")

        dims, dtype, offset = tinfo[tensor_name]
        n = 1
        for d in dims: n *= d
        BLOCK = {14: (256, 144), 15: (256, 144)}
        be, bb = BLOCK.get(dtype, (1, 4))
        n_blocks = n // be

        f.seek(data_start + offset)
        raw = f.read(n_blocks * bb)

        print(f"Tensor: {tensor_name}, dtype={dtype}, dims={dims}, n_elements={n}, n_blocks={n_blocks}")
        return raw, dims, dtype, n_blocks

def debug_q4k_block(raw, block_idx):
    """Print detailed info about one Q4_K block."""
    off = block_idx * 144
    d    = np.frombuffer(raw[off:off+2],   dtype=np.float16)[0].astype(np.float64)
    dmin = np.frombuffer(raw[off+2:off+4], dtype=np.float16)[0].astype(np.float64)
    sc   = raw[off+4:off+16]    # 12 scale bytes
    qs   = np.frombuffer(raw[off+16:off+144], dtype=np.uint8)

    print(f"\n=== Block {block_idx} ===")
    print(f"  d     = {d:.8f}  (raw: {raw[off:off+2].hex()})")
    print(f"  dmin  = {dmin:.8f}  (raw: {raw[off+2:off+4].hex()})")
    print(f"  scales raw: {sc.hex()}")

    # Unpack 8 scales + 8 mins (6-bit each from 12 bytes)
    scales = np.zeros(8); mins = np.zeros(8)
    for j in range(8):
        if j < 4:
            scales[j] = sc[j] & 0x3F
            mins[j]   = sc[j+4] & 0x3F
        else:
            scales[j] = (sc[j+4] & 0xF) | ((sc[j-4] >> 6) << 4)
            mins[j]   = (sc[j+4] >> 4)  | ((sc[j]   >> 6) << 4)

    print(f"  scales: {scales}")
    print(f"  mins:   {mins}")
    print(f"  max abs weight = d*scale*15 - dmin*0 = {d * scales.max() * 15:.4f}")
    print(f"  min weight     = d*0 - dmin*mins     = {-dmin * mins.max():.4f}")

    # Dequantize first 32 elements (sub-block 0, lo nibbles)
    result = np.zeros(256)
    for s in range(4):
        qb = qs[s*32:(s+1)*32]
        lo = (qb & 0x0F).astype(np.float64)
        hi = ((qb >> 4) & 0x0F).astype(np.float64)
        result[s*64:s*64+32] = d * scales[s*2]   * lo - dmin * mins[s*2]
        result[s*64+32:s*64+64] = d * scales[s*2+1] * hi - dmin * mins[s*2+1]

    print(f"\n  First 16 dequantized values: {result[:16]}")
    print(f"  Max abs value: {np.abs(result).max():.6f}")
    print(f"  Mean value:    {result.mean():.6f}")
    print(f"  Std value:     {result.std():.6f}")

    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    args = p.parse_args()

    raw, dims, dtype, n_blocks = read_gguf_tensor(args.gguf, "blk.0.ffn_down.weight")

    print(f"\nAnalyzing {n_blocks} Q4_K blocks...")

    # Print first few blocks
    for b in range(min(3, n_blocks)):
        debug_q4k_block(raw, b)

    # Statistics across all blocks
    all_d    = np.array([np.frombuffer(raw[b*144:b*144+2],   dtype=np.float16)[0] for b in range(n_blocks)], dtype=np.float64)
    all_dmin = np.array([np.frombuffer(raw[b*144+2:b*144+4], dtype=np.float16)[0] for b in range(n_blocks)], dtype=np.float64)

    print(f"\n=== Statistics across all {n_blocks} Q4_K blocks ===")
    print(f"  d     range: [{all_d.min():.6f}, {all_d.max():.6f}]  mean={all_d.mean():.6f}")
    print(f"  dmin  range: [{all_dmin.min():.6f}, {all_dmin.max():.6f}]  mean={all_dmin.mean():.6f}")
    print(f"\nIf d values are small (~0.001-0.01) → Q4_K is likely correct")
    print(f"If d values are large (>1) → Q4_K scale factor is wrong")

if __name__ == "__main__":
    main()

# ── Additional: check alignment and try different data_start offsets ──────────
def main2():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    args = p.parse_args()

    with open(args.gguf, "rb") as f:
        def rstr():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", errors="replace")
        def rval(t):
            if t==0: return struct.unpack("<B",f.read(1))[0]
            if t==4: return struct.unpack("<I",f.read(4))[0]
            if t==5: return struct.unpack("<i",f.read(4))[0]
            if t==6: return struct.unpack("<f",f.read(4))[0]
            if t==7: return struct.unpack("<?",f.read(1))[0]
            if t==8: return rstr()
            if t==9:
                et,=struct.unpack("<I",f.read(4)); cnt,=struct.unpack("<Q",f.read(8))
                return [rval(et) for _ in range(cnt)]
            if t==10: return struct.unpack("<Q",f.read(8))[0]
            if t==11: return struct.unpack("<q",f.read(8))[0]
            return None

        assert f.read(4) == b"GGUF"
        struct.unpack("<I", f.read(4))
        nt, = struct.unpack("<Q", f.read(8))
        nk, = struct.unpack("<Q", f.read(8))

        meta = {}
        for _ in range(nk):
            k = rstr(); vt, = struct.unpack("<I", f.read(4)); meta[k] = rval(vt)

        alignment = meta.get("general.alignment", 32)
        print(f"\ngeneral.alignment = {alignment}")
        print(f"All alignment-related metadata: { {k:v for k,v in meta.items() if 'align' in k.lower()} }")

        tinfo = {}
        for _ in range(nt):
            name = rstr()
            nd, = struct.unpack("<I", f.read(4))
            dims = list(struct.unpack(f"<{nd}Q", f.read(8*nd)))
            dtype, = struct.unpack("<I", f.read(4))
            offset, = struct.unpack("<Q", f.read(8))
            tinfo[name] = (dims, dtype, offset)

        pos = f.tell()
        # Try different alignments
        for align in [1, 8, 16, 32, 64, 128, 256]:
            ds = (pos + align - 1) & ~(align - 1)
            diff = ds - pos

            dims, dtype, offset = tinfo["blk.0.ffn_down.weight"]
            f.seek(ds + offset)
            block0 = f.read(144)
            d_val = np.frombuffer(block0[:2], dtype=np.float16)[0].astype(np.float64)
            dm_val = np.frombuffer(block0[2:4], dtype=np.float16)[0].astype(np.float64)
            print(f"  align={align:3d}: data_start=pos+{diff:2d}, block0 d={d_val:12.4f}  dmin={dm_val:12.6f}  {'← POSITIVE d, looks correct!' if d_val > 0 and d_val < 1 else ''}")

if __name__ == "__main__":
    main2()