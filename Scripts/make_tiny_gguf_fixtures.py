"""Generate tiny (kilobyte) GGUF fixtures for the RoPE-convention regression test (task #95).
Values are irrelevant — only the architecture + metadata + tensor shapes matter, so the loader
can build a GPT1Config and we assert RopeSplitHalf (qwen2 → true, llama → false)."""
import numpy as np
from gguf import GGUFWriter

OUT = r"D:\Overfit\Tests\test_fixtures\gguf"
import os
os.makedirs(OUT, exist_ok=True)

def build(arch, path, with_bias):
    d, nh, nkv, hd, dff, vocab, nl, ctx = 32, 2, 1, 16, 64, 16, 1, 128
    w = GGUFWriter(path, arch)
    w.add_uint32(f"{arch}.block_count", nl)
    w.add_uint32(f"{arch}.embedding_length", d)
    w.add_uint32(f"{arch}.attention.head_count", nh)
    w.add_uint32(f"{arch}.attention.head_count_kv", nkv)
    w.add_uint32(f"{arch}.feed_forward_length", dff)
    w.add_uint32(f"{arch}.context_length", ctx)
    w.add_uint32(f"{arch}.vocab_size", vocab)
    w.add_float32(f"{arch}.rope.freq_base", 1_000_000.0)

    def t(name, shape):
        w.add_tensor(name, np.zeros(shape, dtype=np.float32))

    t("token_embd.weight", (vocab, d))
    t("output_norm.weight", (d,))
    for l in range(nl):
        t(f"blk.{l}.attn_norm.weight", (d,))
        t(f"blk.{l}.attn_q.weight", (nh * hd, d))
        t(f"blk.{l}.attn_k.weight", (nkv * hd, d))
        t(f"blk.{l}.attn_v.weight", (nkv * hd, d))
        t(f"blk.{l}.attn_output.weight", (d, nh * hd))
        if with_bias:
            t(f"blk.{l}.attn_q.bias", (nh * hd,))
            t(f"blk.{l}.attn_k.bias", (nkv * hd,))
            t(f"blk.{l}.attn_v.bias", (nkv * hd,))
        t(f"blk.{l}.ffn_norm.weight", (d,))
        t(f"blk.{l}.ffn_gate.weight", (dff, d))
        t(f"blk.{l}.ffn_up.weight", (dff, d))
        t(f"blk.{l}.ffn_down.weight", (d, dff))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote", path)

build("qwen2", os.path.join(OUT, "tiny-qwen2.gguf"), with_bias=True)
build("llama", os.path.join(OUT, "tiny-llama.gguf"), with_bias=False)
