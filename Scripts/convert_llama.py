#!/usr/bin/env python3
"""
convert_llama.py — Convert HuggingFace Llama / Mistral / Qwen weights to Overfit binary format.

Supported models:
  meta-llama/Llama-3.2-1B        → llama32_1b.bin
  meta-llama/Llama-3.2-3B        → llama32_3b.bin
  microsoft/Phi-3-mini-4k-instruct → phi3_mini.bin
  Qwen/Qwen2.5-0.5B              → qwen25_0_5b.bin
  Qwen/Qwen2.5-1.5B              → qwen25_1_5b.bin

Requirements:
  pip install huggingface_hub safetensors torch numpy

Usage:
  python3 Scripts/convert_llama.py --model meta-llama/Llama-3.2-1B --out test_fixtures/
  python3 Scripts/convert_llama.py --model Qwen/Qwen2.5-0.5B --out test_fixtures/

HuggingFace token required for gated models (Llama):
  huggingface-cli login
  or set HF_TOKEN environment variable.

Output format (same as convert_gpt2.py):
  [int32 magic = 0x4F565246]  "OVRF"
  [int32 version = 2]
  [int32 n_layers]
  [int32 d_model]
  [int32 n_heads]
  [int32 n_kv_heads]
  [int32 vocab_size]
  [int32 context_length]
  [int32 d_ff]
  [int32 use_rope = 1]
  [float32 rope_theta]
  [int32 ffn_activation = 3]  // SwiGLU
  [int32 tie_weights]
  --- weights (all float32, no padding) ---
  token_embedding [vocab_size, d_model]
  for each layer:
    attn_norm_gamma  [d_model]
    attn_norm_beta   [d_model]
    for each q_head (n_heads):
      wq [d_model, head_dim]
      bq [head_dim]  (zeros if model has no bias)
    for each kv_head (n_kv_heads):
      wk [d_model, head_dim]
      bk [head_dim]
      wv [d_model, head_dim]
      bv [head_dim]
    for each q_head (n_heads):
      wo [head_dim, d_model]
      bo [d_model]  (zeros if model has no bias)
    ffn_norm_gamma [d_model]
    ffn_norm_beta  [d_model]
    ffn_gate       [d_model, d_ff]  (Wgate for SwiGLU)
    ffn_up         [d_model, d_ff]  (Wup for SwiGLU)
    ffn_down       [d_ff, d_model]  (Wdown for SwiGLU)
  final_norm_gamma [d_model]
  final_norm_beta  [d_model]
  lm_head [vocab_size, d_model]  (written even if tie_weights=1, Overfit resolves at load)
"""

import argparse
import os
import struct
import sys
import numpy as np

MAGIC   = 0x4F565246  # "OVRF"
VERSION = 2
FFN_SWIGLU = 3


def load_safetensors(model_dir: str) -> dict:
    """Load all safetensors shards from model_dir into a flat dict."""
    try:
        from safetensors import safe_open
    except ImportError:
        sys.exit("Install safetensors: pip install safetensors")

    tensors = {}
    shards  = sorted(f for f in os.listdir(model_dir)
                     if f.endswith(".safetensors"))

    if not shards:
        sys.exit(f"No .safetensors files found in {model_dir}")

    for shard in shards:
        path = os.path.join(model_dir, shard)
        print(f"  Loading {shard} …")
        with safe_open(path, framework="numpy", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    return tensors


def get_tensor(tensors: dict, *candidates) -> np.ndarray:
    """Return first matching tensor key."""
    for key in candidates:
        if key in tensors:
            return tensors[key].astype(np.float32)
    raise KeyError(f"None of {candidates} found in tensors. Available: {sorted(tensors)[:20]}")


def zeros(shape) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def write_f32(f, arr: np.ndarray):
    f.write(arr.astype(np.float32).flatten().tobytes())


def detect_config(tensors: dict, model_id: str) -> dict:
    """Infer model config from tensor shapes and model_id."""
    try:
        import json
        config_path = os.path.join(args.cache_dir or "", "config.json")
        if os.path.exists(config_path):
            with open(config_path) as fp:
                cfg = json.load(fp)
        else:
            cfg = {}
    except Exception:
        cfg = {}

    # Detect from tensor shapes
    emb = get_tensor(tensors, "model.embed_tokens.weight", "embed_tokens.weight")
    vocab_size, d_model = emb.shape

    # Layer 0 Q weight
    q0 = get_tensor(tensors,
        "model.layers.0.self_attn.q_proj.weight",
        "layers.0.attention.wq.weight")
    n_heads_x_head_dim, _ = q0.shape
    head_dim = cfg.get("head_dim", d_model // cfg.get("num_attention_heads", 32))
    n_heads  = cfg.get("num_attention_heads",  n_heads_x_head_dim // head_dim)

    k0 = get_tensor(tensors,
        "model.layers.0.self_attn.k_proj.weight",
        "layers.0.attention.wk.weight")
    n_kv_heads = cfg.get("num_key_value_heads",  k0.shape[0] // head_dim)

    gate0 = None
    try:
        gate0 = get_tensor(tensors, "model.layers.0.mlp.gate_proj.weight")
    except KeyError:
        pass
    d_ff = cfg.get("intermediate_size",
                   gate0.shape[0] if gate0 is not None else 4 * d_model)

    n_layers  = cfg.get("num_hidden_layers",
                         sum(1 for k in tensors if k.endswith(".self_attn.q_proj.weight")))
    ctx       = cfg.get("max_position_embeddings", 4096)
    rope_theta= float(cfg.get("rope_theta", 10_000.0))
    tie       = int(cfg.get("tie_word_embeddings", True))

    return dict(n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                n_kv_heads=n_kv_heads, vocab_size=vocab_size,
                context_length=min(ctx, 8192), d_ff=d_ff,
                rope_theta=rope_theta, tie_weights=tie, head_dim=head_dim)


def convert(tensors: dict, cfg: dict, out_path: str):
    n_layers    = cfg["n_layers"]
    d_model     = cfg["d_model"]
    n_heads     = cfg["n_heads"]
    n_kv_heads  = cfg["n_kv_heads"]
    vocab_size  = cfg["vocab_size"]
    ctx         = cfg["context_length"]
    d_ff        = cfg["d_ff"]
    rope_theta  = cfg["rope_theta"]
    tie         = cfg["tie_weights"]
    head_dim    = cfg["head_dim"]

    print(f"\nConfig: layers={n_layers}, d={d_model}, heads={n_heads}, kv={n_kv_heads}, "
          f"vocab={vocab_size}, ctx={ctx}, d_ff={d_ff}, rope_theta={rope_theta:.0f}")

    with open(out_path, "wb") as f:
        # Header
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<i", n_layers))
        f.write(struct.pack("<i", d_model))
        f.write(struct.pack("<i", n_heads))
        f.write(struct.pack("<i", n_kv_heads))
        f.write(struct.pack("<i", vocab_size))
        f.write(struct.pack("<i", ctx))
        f.write(struct.pack("<i", d_ff))
        f.write(struct.pack("<i", 1))             # use_rope = true
        f.write(struct.pack("<f", rope_theta))
        f.write(struct.pack("<i", FFN_SWIGLU))
        f.write(struct.pack("<i", tie))

        # Token embedding
        emb = get_tensor(tensors, "model.embed_tokens.weight", "embed_tokens.weight")
        write_f32(f, emb)
        print(f"  embed_tokens: {emb.shape}")

        # Layers
        for l in range(n_layers):
            p = f"model.layers.{l}"
            print(f"  layer {l}/{n_layers} …", end="\r", flush=True)

            # Attention norm
            write_f32(f, get_tensor(tensors, f"{p}.input_layernorm.weight"))
            write_f32(f, zeros(d_model))  # Llama RMSNorm has no beta

            # Q heads [n_heads, d_model, head_dim]
            wq_full = get_tensor(tensors, f"{p}.self_attn.q_proj.weight")
            bq_full = tensors.get(f"{p}.self_attn.q_proj.bias", zeros(n_heads * head_dim))
            for h in range(n_heads):
                write_f32(f, wq_full[h*head_dim:(h+1)*head_dim, :].T)  # [d_model, head_dim]
                write_f32(f, bq_full[h*head_dim:(h+1)*head_dim].astype(np.float32))

            # KV heads [n_kv_heads, d_model, head_dim]
            wk_full = get_tensor(tensors, f"{p}.self_attn.k_proj.weight")
            wv_full = get_tensor(tensors, f"{p}.self_attn.v_proj.weight")
            bk_full = tensors.get(f"{p}.self_attn.k_proj.bias", zeros(n_kv_heads * head_dim))
            bv_full = tensors.get(f"{p}.self_attn.v_proj.bias", zeros(n_kv_heads * head_dim))
            for kv in range(n_kv_heads):
                write_f32(f, wk_full[kv*head_dim:(kv+1)*head_dim, :].T)
                write_f32(f, bk_full[kv*head_dim:(kv+1)*head_dim].astype(np.float32))
                write_f32(f, wv_full[kv*head_dim:(kv+1)*head_dim, :].T)
                write_f32(f, bv_full[kv*head_dim:(kv+1)*head_dim].astype(np.float32))

            # O heads [n_heads, head_dim, d_model]
            wo_full = get_tensor(tensors, f"{p}.self_attn.o_proj.weight")
            bo_full = tensors.get(f"{p}.self_attn.o_proj.bias", zeros(d_model))
            for h in range(n_heads):
                write_f32(f, wo_full[:, h*head_dim:(h+1)*head_dim].T)   # [head_dim, d_model]
                write_f32(f, bo_full.astype(np.float32))

            # FFN norm
            write_f32(f, get_tensor(tensors, f"{p}.post_attention_layernorm.weight"))
            write_f32(f, zeros(d_model))

            # SwiGLU weights
            write_f32(f, get_tensor(tensors, f"{p}.mlp.gate_proj.weight").T)  # [d_model, d_ff]
            write_f32(f, get_tensor(tensors, f"{p}.mlp.up_proj.weight"  ).T)  # [d_model, d_ff]
            write_f32(f, get_tensor(tensors, f"{p}.mlp.down_proj.weight").T)  # [d_ff, d_model]

        print(f"\n  All {n_layers} layers written.")

        # Final norm
        write_f32(f, get_tensor(tensors,
            "model.norm.weight", "norm.weight"))
        write_f32(f, zeros(d_model))

        # LM head
        try:
            lm = get_tensor(tensors, "lm_head.weight")
        except KeyError:
            lm = emb  # tied weights
        write_f32(f, lm)  # [vocab_size, d_model]

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nWritten: {out_path}  ({size_mb:.1f} MB)")


def main():
    global args
    parser = argparse.ArgumentParser(description="Convert HuggingFace model to Overfit format.")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID, e.g. meta-llama/Llama-3.2-1B")
    parser.add_argument("--out", default="test_fixtures/",
                        help="Output directory (default: test_fixtures/)")
    parser.add_argument("--cache-dir", default=None,
                        help="HuggingFace cache directory override")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("Install huggingface_hub: pip install huggingface_hub")

    print(f"Downloading {args.model} …")
    token = os.environ.get("HF_TOKEN")
    model_dir = snapshot_download(
        args.model,
        cache_dir=args.cache_dir,
        token=token,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*",
                         "pytorch_model.bin.index.json"],
    )

    # Copy config.json path for detect_config
    args.cache_dir = model_dir

    print(f"Model dir: {model_dir}")
    print("Loading tensors …")
    tensors = load_safetensors(model_dir)

    cfg = detect_config(tensors, args.model)

    name = args.model.split("/")[-1].lower().replace("-", "_")
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{name}.bin")

    convert(tensors, cfg, out_path)


if __name__ == "__main__":
    main()
