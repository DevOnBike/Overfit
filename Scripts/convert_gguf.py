#!/usr/bin/env python3
"""
convert_gguf.py — Convert GGUF model (Ollama/llama.cpp) to Overfit binary format.

Supported quantizations:
  F32, F16, BF16, Q8_0, Q4_0, Q4_K (Q4_K_M / Q4_K_S), Q6_K

Usage:
  python3 Scripts/convert_gguf.py --input path/to/model.gguf --out test_fixtures/

The GGUF file can come from:
  ollama pull qwen2.5:0.5b
  # Model stored in: C:\\Users\\<user>\\.ollama\\models\\blobs\\sha256-...

Output: test_fixtures/<model_name>.bin in Overfit binary format.
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

# ── GGUF constants ────────────────────────────────────────────────────────────

GGUF_MAGIC   = b"GGUF"
GGUF_VERSION = 3

GGML_TYPE = {
    0:  "F32", 1:  "F16", 2:  "Q4_0", 3:  "Q4_1",
    6:  "Q5_0", 7:  "Q5_1", 8: "Q8_0",
    10: "Q2_K", 11: "Q3_K", 12: "Q3_K", 13: "Q3_K",
    14: "Q4_K", 15: "Q4_K", 16: "Q5_K", 17: "Q5_K",
    18: "Q6_K", 28: "BF16",
}
GGML_BLOCK_BYTES = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2),
    "Q8_0": (32, 34), "Q5_0": (32, 22), "Q5_1": (32, 24),
    "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210),
    "Q2_K": (256, 84),  "Q3_K": (256, 110),
}

# ── GGUF reader ───────────────────────────────────────────────────────────────

class GGUFReader:
    def __init__(self, path: str):
        self._f     = open(path, "rb")
        self.meta   = {}
        self.tensors = {}
        self._parse()

    def _read(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self._f.read(size))

    def _read_str(self) -> str:
        n, = self._read("<Q")
        return self._f.read(n).decode("utf-8", errors="replace")

    def _read_value(self, vtype: int):
        if vtype == 0:  return self._read("<B")[0]           # uint8
        if vtype == 1:  return self._read("<b")[0]           # int8
        if vtype == 2:  return self._read("<H")[0]           # uint16
        if vtype == 3:  return self._read("<h")[0]           # int16
        if vtype == 4:  return self._read("<I")[0]           # uint32
        if vtype == 5:  return self._read("<i")[0]           # int32
        if vtype == 6:  return self._read("<f")[0]           # float32
        if vtype == 7:  return self._read("<?")[0]           # bool
        if vtype == 8:  return self._read_str()              # string
        if vtype == 9:                                       # array
            elem_type, = self._read("<I")
            count,     = self._read("<Q")
            return [self._read_value(elem_type) for _ in range(count)]
        if vtype == 10: return self._read("<Q")[0]           # uint64
        if vtype == 11: return self._read("<q")[0]           # int64
        if vtype == 12: return self._read("<d")[0]           # float64
        raise ValueError(f"Unknown value type {vtype}")

    def _parse(self):
        magic = self._f.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic={magic!r})")

        version, = self._read("<I")
        n_tensors, = self._read("<Q")
        n_kv, = self._read("<Q")

        print(f"GGUF version {version}, {n_tensors} tensors, {n_kv} metadata entries")

        # Read metadata
        for _ in range(n_kv):
            key   = self._read_str()
            vtype, = self._read("<I")
            self.meta[key] = self._read_value(vtype)

        # Read tensor info
        tensor_info = []
        for _ in range(n_tensors):
            name    = self._read_str()
            n_dims, = self._read("<I")
            dims    = list(self._read(f"<{n_dims}Q"))
            dtype,  = self._read("<I")
            offset, = self._read("<Q")
            tensor_info.append((name, dims, dtype, offset))

        # Data section starts at next 32-byte aligned position
        pos = self._f.tell()
        data_start = (pos + 31) & ~31
        self._data_start = data_start

        # Load all tensors
        for name, dims, dtype_id, offset in tensor_info:
            type_name = GGML_TYPE.get(dtype_id, f"UNK_{dtype_id}")
            n_elements = 1
            for d in dims:
                n_elements *= d
            self._f.seek(data_start + offset)
            raw = self._f.read(self._tensor_bytes(type_name, n_elements))
            arr = self._dequantize(raw, type_name, n_elements, dims)
            self.tensors[name] = arr
            print(f"  {name}: {dims} {type_name} → f32 {arr.shape}", flush=True)

    def _tensor_bytes(self, type_name: str, n_elements: int) -> int:
        if type_name not in GGML_BLOCK_BYTES:
            raise ValueError(f"Unsupported type {type_name}")
        blk_elems, blk_bytes = GGML_BLOCK_BYTES[type_name]
        return (n_elements // blk_elems) * blk_bytes

    def _dequantize(self, raw: bytes, type_name: str, n_elements: int, dims) -> np.ndarray:
        if type_name == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(dims[::-1])
        if type_name == "F16":
            return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(dims[::-1])
        if type_name == "BF16":
            i16 = np.frombuffer(raw, dtype="<i2")
            i32 = i16.astype("<i4") << 16
            return i32.view(np.float32).reshape(dims[::-1])
        if type_name == "Q8_0":
            return self._dequant_q8_0(raw, n_elements).reshape(dims[::-1])
        if type_name == "Q4_0":
            return self._dequant_q4_0(raw, n_elements).reshape(dims[::-1])
        if type_name == "Q3_K":
            return self._dequant_q3_k(raw, n_elements).reshape(dims[::-1])
        if type_name == "Q5_0":
            return self._dequant_q5_0(raw, n_elements).reshape(dims[::-1])
        if type_name == "Q4_K":
            return self._dequant_q4_k(raw, n_elements).reshape(dims[::-1])
        if type_name == "Q6_K":
            return self._dequant_q6_k(raw, n_elements).reshape(dims[::-1])
        raise NotImplementedError(f"Dequantization not implemented for {type_name}. "
                                  f"Use F16/F32/BF16/Q8_0/Q4_0/Q4_K/Q6_K GGUF.")

    # ── Q8_0: 34 bytes per 32 elements ───────────────────────────────────────
    def _dequant_q8_0(self, raw: bytes, n: int) -> np.ndarray:
        n_blocks = n // 32
        data = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, 34)
        d = data[:, :2].view(np.float16).astype(np.float32).reshape(n_blocks)
        qs = data[:, 2:].view(np.int8).reshape(n_blocks, 32)
        return (d[:, None] * qs).reshape(n).astype(np.float32)

    # ── Q4_0: 18 bytes per 32 elements ───────────────────────────────────────
    def _dequant_q4_0(self, raw: bytes, n: int) -> np.ndarray:
        n_blocks = n // 32
        data = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, 18)
        d = data[:, :2].view(np.float16).astype(np.float32).reshape(n_blocks)
        qs = data[:, 2:].reshape(n_blocks, 16)
        lo = (qs & 0x0F).astype(np.int8) - 8
        hi = ((qs >> 4) & 0x0F).astype(np.int8) - 8
        vals = np.stack([lo, hi], axis=2).reshape(n_blocks, 32)
        return (d[:, None] * vals).reshape(n).astype(np.float32)

    # ── Q5_0: 22 bytes per 32 elements ──────────────────────────────────────────
    # d(f16,2) + qh(u8,4 — 1 high bit per value) + qs(u8,16 — 4 low bits per value)
    def _dequant_q5_0(self, raw: bytes, n: int) -> np.ndarray:
        n_blocks = n // 32
        result   = np.zeros(n, dtype=np.float32)
        for b in range(n_blocks):
            off = b * 22
            d  = np.frombuffer(raw[off:off+2], dtype=np.float16)[0].astype(np.float32)
            qh = int.from_bytes(raw[off+2:off+6], "little")
            qs = raw[off+6:off+22]
            base = b * 32
            for j in range(16):
                lo_nibble = qs[j] & 0x0F
                hi_nibble = (qs[j] >> 4) & 0x0F
                high_bit0 = (qh >> j)       & 1
                high_bit1 = (qh >> (j + 16)) & 1
                result[base + j     ] = d * (int(lo_nibble | (high_bit0 << 4)) - 16)
                result[base + j + 16] = d * (int(hi_nibble | (high_bit1 << 4)) - 16)
        return result

        # ── Q3_K: 110 bytes per 256 elements ─────────────────────────────────────────
    # hmask(32) + qs(64) + scales(12) + d(f16,2) = 110 bytes
    # 16 sub-blocks of 16 elements, 6-bit scales.
    def _dequant_q3_k(self, raw: bytes, n: int) -> np.ndarray:
        QK_K = 256
        n_blocks = n // QK_K
        result = np.zeros(n, dtype=np.float32)

        for b in range(n_blocks):
            off    = b * 110
            hmask  = raw[off:off+32]        # 1 high bit per element
            qs     = raw[off+32:off+96]     # 2 low bits per element
            sc_raw = raw[off+96:off+108]    # 12 bytes → 16 six-bit scales
            d      = np.frombuffer(raw[off+108:off+110], dtype=np.float16)[0].astype(np.float32)
            if not np.isfinite(d): d = 0.0

            # Unpack 16 six-bit scales from 12 bytes (llama.cpp dequantize_row_q3_K)
            lsc = [0] * 16
            for j in range(4):
                lsc[j]    = (sc_raw[j] & 0xF) | (((sc_raw[j+8] >> 0) & 3) << 4)
                lsc[j+4]  = (sc_raw[j+4] & 0xF) | (((sc_raw[j+8] >> 2) & 3) << 4)
                lsc[j+8]  = (sc_raw[j] >> 4)  | (((sc_raw[j+8] >> 4) & 3) << 4)
                lsc[j+12] = (sc_raw[j+4] >> 4) | (((sc_raw[j+8] >> 6) & 3) << 4)
            lsc = [s - 32 for s in lsc]  # signed 6-bit: subtract 32

            base = b * QK_K
            for i in range(QK_K):
                sub   = i // 16
                lo2   = (qs[i//4] >> ((i % 4) * 2)) & 3
                hi1   = (hmask[i//8] >> (i % 8)) & 1
                val3  = ((hi1 << 2) | lo2) - 4   # signed range -4..3
                result[base + i] = d * lsc[sub] * val3

        return result

        # ── Q4_K: 144 bytes per 256 elements ─────────────────────────────────────
    # Reference: llama.cpp ggml-quants.c dequantize_row_q4_K
    # Layout: 4 pairs of 32-byte blocks, each pair → 64 output elements:
    #   elements[s*64..s*64+31]  = d*scale[2s]   * lo(qs[s*32..s*32+31]) - dmin*min[2s]
    #   elements[s*64+32..s*64+63] = d*scale[2s+1] * hi(qs[s*32..s*32+31]) - dmin*min[2s+1]
    def _dequant_q4_k(self, raw: bytes, n: int) -> np.ndarray:
        QK_K = 256
        n_blocks = n // QK_K
        result = np.zeros(n, dtype=np.float32)

        for b in range(n_blocks):
            off   = b * 144
            d     = np.frombuffer(raw[off:off+2],   dtype=np.float16)[0].astype(np.float32)
            dmin  = np.frombuffer(raw[off+2:off+4], dtype=np.float16)[0].astype(np.float32)
            if not np.isfinite(d):    d    = 0.0
            if not np.isfinite(dmin): dmin = 0.0
            sc    = raw[off+4 : off+16]    # 12 scale bytes
            qs    = raw[off+16: off+144]   # 128 quantized bytes

            # Unpack 8 scales + 8 mins (6-bit each) from 12 bytes
            scales = np.zeros(8, np.float32)
            mins   = np.zeros(8, np.float32)
            for j in range(8):
                if j < 4:
                    scales[j] = sc[j] & 0x3F
                    mins[j]   = sc[j+4] & 0x3F
                else:
                    scales[j] = ((sc[j+4] & 0xF) | ((sc[j-4] >> 6) << 4))
                    mins[j]   = ((sc[j+4] >> 4)  | ((sc[j+0] >> 6) << 4))

            base   = b * QK_K
            qs_arr = np.frombuffer(bytes(qs), dtype=np.uint8)

            # 4 pairs × 32 bytes → 4 × 64 output elements
            for s in range(4):
                qb = qs_arr[s*32:(s+1)*32]
                lo = (qb & 0x0F).astype(np.float32)
                hi = ((qb >> 4) & 0x0F).astype(np.float32)
                result[base + s*64     : base + s*64 + 32] = d * scales[s*2]   * lo - dmin * mins[s*2]
                result[base + s*64 + 32: base + s*64 + 64] = d * scales[s*2+1] * hi - dmin * mins[s*2+1]

        return result

    # ── Q6_K: 210 bytes per 256 elements ─────────────────────────────────────
    def _dequant_q6_k(self, raw: bytes, n: int) -> np.ndarray:
        QK_K = 256
        n_blocks = n // QK_K
        result = np.zeros(n, dtype=np.float32)

        for b in range(n_blocks):
            off = b * 210
            ql  = np.frombuffer(raw[off:off+128],      dtype=np.uint8)   # 128 bytes low 4 bits
            qh  = np.frombuffer(raw[off+128:off+192],  dtype=np.uint8)   # 64 bytes high 2 bits
            sc  = np.frombuffer(raw[off+192:off+208],  dtype=np.int8)    # 16 scale bytes
            d   = np.frombuffer(raw[off+208:off+210],  dtype=np.float16)[0].astype(np.float32)

            q = np.zeros(QK_K, dtype=np.int32)
            for i in range(128):
                q[i]         = (ql[i] & 0x0F) | (((qh[i//2] >> (4*(i%2)  )) & 0x3) << 4)
                q[i + 128]   = ((ql[i] >> 4)  ) | (((qh[i//2] >> (4*(i%2)+2)) & 0x3) << 4)

            q = q.astype(np.float32) - 32.0

            base = b * QK_K
            for s in range(16):
                scale = d * sc[s]
                result[base + s*16 : base + s*16 + 16] = scale * q[s*16:s*16+16]

        return result

    def close(self):
        self._f.close()


# ── GGUF → Overfit name mapping ───────────────────────────────────────────────

def map_tensor_names(tensors: dict, n_layers: int) -> dict:
    """Map GGUF tensor names to HuggingFace names (same as convert_llama.py expects)."""
    mapped = {}

    name_map = {
        "token_embd.weight":    "model.embed_tokens.weight",
        "output_norm.weight":   "model.norm.weight",
        "output.weight":        "lm_head.weight",
    }
    for l in range(n_layers):
        p = f"blk.{l}"
        h = f"model.layers.{l}"
        name_map.update({
            f"{p}.attn_norm.weight":    f"{h}.input_layernorm.weight",
            f"{p}.ffn_norm.weight":     f"{h}.post_attention_layernorm.weight",
            f"{p}.attn_q.weight":       f"{h}.self_attn.q_proj.weight",
            f"{p}.attn_k.weight":       f"{h}.self_attn.k_proj.weight",
            f"{p}.attn_v.weight":       f"{h}.self_attn.v_proj.weight",
            f"{p}.attn_output.weight":  f"{h}.self_attn.o_proj.weight",
            f"{p}.ffn_gate.weight":     f"{h}.mlp.gate_proj.weight",
            f"{p}.ffn_up.weight":       f"{h}.mlp.up_proj.weight",
            f"{p}.ffn_down.weight":     f"{h}.mlp.down_proj.weight",
            # biases (Qwen has none, but map anyway)
            f"{p}.attn_q.bias":         f"{h}.self_attn.q_proj.bias",
            f"{p}.attn_k.bias":         f"{h}.self_attn.k_proj.bias",
            f"{p}.attn_v.bias":         f"{h}.self_attn.v_proj.bias",
        })

    for gguf_name, tensor in tensors.items():
        hf_name = name_map.get(gguf_name, gguf_name)
        mapped[hf_name] = tensor

    return mapped


# ── Conversion (reuse logic from convert_llama.py) ───────────────────────────

MAGIC   = 0x4F565246
VERSION = 2
FFN_SWIGLU = 3


def zeros(shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def write_f32(f, arr: np.ndarray):
    f.write(arr.astype(np.float32).flatten().tobytes())


def convert(tensors: dict, meta: dict, out_path: str):
    arch = meta.get("general.architecture", "qwen2")

    n_layers   = meta.get(f"{arch}.block_count",               24)
    d_model    = meta.get(f"{arch}.embedding_length",          896)
    n_heads    = meta.get(f"{arch}.attention.head_count",      14)
    n_kv_heads = meta.get(f"{arch}.attention.head_count_kv",   2)
    d_ff       = meta.get(f"{arch}.feed_forward_length",       4864)
    ctx        = meta.get(f"{arch}.context_length",            4096)
    rope_theta = float(meta.get(f"{arch}.rope.freq_base",      1_000_000.0))
    vocab_size = meta.get("tokenizer.ggml.tokens",             [None] * 151936)
    if isinstance(vocab_size, list):
        vocab_size = len(vocab_size)
    tie        = int(meta.get("general.tie_embeddings", True))

    head_dim = d_model // n_heads

    print(f"\nConfig:")
    print(f"  layers={n_layers}, d={d_model}, heads={n_heads}, kv={n_kv_heads}")
    print(f"  vocab={vocab_size}, ctx={min(ctx,8192)}, d_ff={d_ff}, rope_theta={rope_theta:.0f}")

    # GGUF stores weights transposed vs HuggingFace
    # (GGUF: [out, in] already in row-major = same as HF .weight)
    # But dimensions are reversed in GGUF tensor dims — already handled in _dequantize reshape

    def get(name, fallback=None):
        if name in tensors:
            return tensors[name]
        if fallback is not None:
            return fallback
        raise KeyError(f"Tensor not found: {name}. Available: {list(tensors)[:10]}")

    ctx = min(ctx, 8192)

    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<i", n_layers))
        f.write(struct.pack("<i", d_model))
        f.write(struct.pack("<i", n_heads))
        f.write(struct.pack("<i", n_kv_heads))
        f.write(struct.pack("<i", vocab_size))
        f.write(struct.pack("<i", ctx))
        f.write(struct.pack("<i", d_ff))
        f.write(struct.pack("<i", 1))               # use_rope
        f.write(struct.pack("<f", rope_theta))
        f.write(struct.pack("<i", FFN_SWIGLU))
        f.write(struct.pack("<i", tie))

        emb = get("model.embed_tokens.weight")
        write_f32(f, emb)
        print(f"  embed_tokens: {emb.shape}")

        for l in range(n_layers):
            p = f"model.layers.{l}"
            print(f"  layer {l}/{n_layers} …", end="\r", flush=True)

            # Attention norm (RMSNorm — no beta)
            write_f32(f, get(f"{p}.input_layernorm.weight"))
            write_f32(f, zeros(d_model))

            # Q heads
            wq_full = get(f"{p}.self_attn.q_proj.weight")
            bq_full = tensors.get(f"{p}.self_attn.q_proj.bias", zeros(n_heads * head_dim))
            # GGUF weights are [out, in] — same as HF, but might need transpose check
            for h in range(n_heads):
                write_f32(f, wq_full[h*head_dim:(h+1)*head_dim, :].T)
                write_f32(f, bq_full[h*head_dim:(h+1)*head_dim].astype(np.float32))

            # KV heads
            wk_full = get(f"{p}.self_attn.k_proj.weight")
            wv_full = get(f"{p}.self_attn.v_proj.weight")
            bk_full = tensors.get(f"{p}.self_attn.k_proj.bias", zeros(n_kv_heads * head_dim))
            bv_full = tensors.get(f"{p}.self_attn.v_proj.bias", zeros(n_kv_heads * head_dim))
            for kv in range(n_kv_heads):
                write_f32(f, wk_full[kv*head_dim:(kv+1)*head_dim, :].T)
                write_f32(f, bk_full[kv*head_dim:(kv+1)*head_dim].astype(np.float32))
                write_f32(f, wv_full[kv*head_dim:(kv+1)*head_dim, :].T)
                write_f32(f, bv_full[kv*head_dim:(kv+1)*head_dim].astype(np.float32))

            # O heads
            wo_full = get(f"{p}.self_attn.o_proj.weight")
            bo_full = tensors.get(f"{p}.self_attn.o_proj.bias", zeros(d_model))
            for h in range(n_heads):
                write_f32(f, wo_full[:, h*head_dim:(h+1)*head_dim].T)
                write_f32(f, bo_full.astype(np.float32))

            # FFN norm
            write_f32(f, get(f"{p}.post_attention_layernorm.weight"))
            write_f32(f, zeros(d_model))

            # SwiGLU
            write_f32(f, get(f"{p}.mlp.gate_proj.weight").T)
            write_f32(f, get(f"{p}.mlp.up_proj.weight").T)
            write_f32(f, get(f"{p}.mlp.down_proj.weight").T)

        print(f"\n  All {n_layers} layers written.")

        write_f32(f, get("model.norm.weight"))
        write_f32(f, zeros(d_model))

        try:
            lm = get("lm_head.weight")
        except KeyError:
            lm = emb
        write_f32(f, lm)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nWritten: {out_path}  ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Convert GGUF model to Overfit format.")
    parser.add_argument("--input", required=True, help="Path to .gguf file")
    parser.add_argument("--out", default="test_fixtures/", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"File not found: {args.input}")

    print(f"Reading {args.input} …")
    reader = GGUFReader(args.input)

    n_layers = reader.meta.get(
        f"{reader.meta.get('general.architecture','qwen2')}.block_count", 24)

    tensors = map_tensor_names(reader.tensors, n_layers)
    reader.close()

    name = os.path.splitext(os.path.basename(args.input))[0].lower().replace("-", "_")
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{name}.bin")

    convert(tensors, reader.meta, out_path)


if __name__ == "__main__":
    main()