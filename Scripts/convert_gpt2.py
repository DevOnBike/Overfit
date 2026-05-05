#!/usr/bin/env python3
"""
Converts GPT-2 weights (HuggingFace format) to Overfit binary format.

Downloads GPT-2 Small from HuggingFace and writes:
  gpt2_small.bin   — model weights in Overfit format
  vocab.json       — BPE vocabulary
  merges.txt       — BPE merge rules

Usage:
  pip install numpy requests
  python3 convert_gpt2.py --size small --out .
  python3 convert_gpt2.py --size medium --out .

GPT-2 sizes:
  small  — 117M params, 12L, 768d,  12H
  medium — 345M params, 24L, 1024d, 16H
  large  — 762M params, 36L, 1280d, 20H
  xl     — 1.5B params, 48L, 1600d, 25H

Overfit weight layout (matches GPT1Model.Save order):
  TokenEmbedding.Weight     [vocab, d]        wte
  PositionEmbedding.Weight  [ctx, d]          wpe
  Per block (12x):
    Norm1.Gamma   [d]     ln_1.weight
    Norm1.Beta    [d]     ln_1.bias
    MHA per head (12x):
      Wq_h  [d, d/nH]   split from c_attn.weight rows 0..d
      Wk_h  [d, d/nH]   split from c_attn.weight rows d..2d
      Wv_h  [d, d/nH]   split from c_attn.weight rows 2d..3d
      Wo_h  [d/nH, d]   split from c_proj.weight cols
    MHA.Bo  [d]          c_proj.bias
    Norm2.Gamma   [d]     ln_2.weight
    Norm2.Beta    [d]     ln_2.bias
    FFN.W1  [d, 4d]      mlp.c_fc.weight  (transposed)
    FFN.B1  [4d]         mlp.c_fc.bias
    FFN.W2  [4d, d]      mlp.c_proj.weight (transposed)
    FFN.B2  [d]          mlp.c_proj.bias
  FinalNorm.Gamma  [d]    ln_f.weight
  FinalNorm.Beta   [d]    ln_f.bias
  LMHead  [vocab, d]      wte (weight-tied in GPT-2)
"""

import argparse
import json
import os
import struct
import urllib.request
import numpy as np

SIZES = {
    "small":  {"n_layer": 12, "n_head": 12, "n_embd": 768,  "n_ctx": 1024, "n_vocab": 50257},
    "medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024, "n_ctx": 1024, "n_vocab": 50257},
    "large":  {"n_layer": 36, "n_head": 20, "n_embd": 1280, "n_ctx": 1024, "n_vocab": 50257},
    "xl":     {"n_layer": 48, "n_head": 25, "n_embd": 1600, "n_ctx": 1024, "n_vocab": 50257},
}

HF_BASE = "https://huggingface.co/openai-community/gpt2/resolve/main"

def download(url, path):
    if os.path.exists(path):
        print(f"  Already exists: {path}")
        return
    print(f"  Downloading: {url}")
    urllib.request.urlretrieve(url, path)
    print(f"  Saved: {path} ({os.path.getsize(path) / 1e6:.1f}MB)")

def load_pytorch_weights(model_path):
    """Load weights from PyTorch .bin file (pickle format)."""
    import pickle

    with open(model_path, "rb") as f:
        # PyTorch .bin files are pickled dicts of numpy arrays
        data = pickle.load(f, encoding="latin1")

    # Handle both flat dict and nested dict formats
    weights = {}
    for k, v in data.items():
        if hasattr(v, "numpy"):
            weights[k] = v.numpy().astype(np.float32)
        elif isinstance(v, np.ndarray):
            weights[k] = v.astype(np.float32)

    return weights

def write_parameter(f, data: np.ndarray):
    """Write parameter in Overfit format: int32 length + float32 data."""
    flat = data.flatten().astype(np.float32)
    f.write(struct.pack("<i", len(flat)))
    f.write(flat.tobytes())

def convert(weights, cfg, out_path):
    d      = cfg["n_embd"]
    n_head = cfg["n_head"]
    n_ctx  = cfg["n_ctx"]
    n_vocab= cfg["n_vocab"]
    n_layer= cfg["n_layer"]
    d_head = d // n_head

    with open(out_path, "wb") as f:

        # TokenEmbedding.Weight [vocab, d]
        print("  Writing: TokenEmbedding")
        write_parameter(f, weights["wte"])

        # PositionEmbedding.Weight [ctx, d]
        print("  Writing: PositionEmbedding")
        write_parameter(f, weights["wpe"])

        # Transformer blocks
        for layer in range(n_layer):
            print(f"  Writing: Block {layer}/{n_layer-1}")
            pfx = f"h.{layer}"

            # LayerNorm 1
            write_parameter(f, weights[f"{pfx}.ln_1.weight"])  # Gamma
            write_parameter(f, weights[f"{pfx}.ln_1.bias"])    # Beta

            # MultiHeadAttention — factored per-head
            # c_attn.weight is [d, 3d] in PyTorch (Conv1D: input × weight)
            # c_attn.bias is [3d]
            attn_w = weights[f"{pfx}.attn.c_attn.weight"]  # [d, 3d]
            attn_b = weights[f"{pfx}.attn.c_attn.bias"]    # [3d]

            wq_full = attn_w[:, :d]           # [d, d]
            wk_full = attn_w[:, d:2*d]        # [d, d]
            wv_full = attn_w[:, 2*d:3*d]      # [d, d]

            bq = attn_b[:d]                   # [d]
            bk = attn_b[d:2*d]                # [d]
            bv = attn_b[2*d:3*d]              # [d]

            # c_proj.weight is [d, d], c_proj.bias is [d]
            proj_w = weights[f"{pfx}.attn.c_proj.weight"]  # [d, d]
            proj_b = weights[f"{pfx}.attn.c_proj.bias"]    # [d]

            # Write per-head weights — Overfit factored MHA layout
            # Each head: Wq_h [d, d_head], Wk_h [d, d_head], Wv_h [d, d_head], Wo_h [d_head, d]
            for h in range(n_head):
                start = h * d_head
                end   = start + d_head

                # Wq_h [d, d_head]
                write_parameter(f, wq_full[:, start:end])
                # Wk_h [d, d_head]
                write_parameter(f, wk_full[:, start:end])
                # Wv_h [d, d_head]
                write_parameter(f, wv_full[:, start:end])
                # Wo_h [d_head, d] — output projection row for this head
                write_parameter(f, proj_w[start:end, :])

            # Bo [d] — output projection bias
            write_parameter(f, proj_b)

            # LayerNorm 2
            write_parameter(f, weights[f"{pfx}.ln_2.weight"])  # Gamma
            write_parameter(f, weights[f"{pfx}.ln_2.bias"])    # Beta

            # FFN
            # c_fc.weight [d, 4d], c_fc.bias [4d]
            # c_proj.weight [4d, d], c_proj.bias [d]
            # GPT-2 uses Conv1D so weight is transposed vs Linear
            write_parameter(f, weights[f"{pfx}.mlp.c_fc.weight"])    # W1 [d, 4d]
            write_parameter(f, weights[f"{pfx}.mlp.c_fc.bias"])      # B1 [4d]
            write_parameter(f, weights[f"{pfx}.mlp.c_proj.weight"])  # W2 [4d, d]
            write_parameter(f, weights[f"{pfx}.mlp.c_proj.bias"])    # B2 [d]

        # FinalNorm (ln_f)
        print("  Writing: FinalNorm")
        write_parameter(f, weights["ln_f.weight"])  # Gamma
        write_parameter(f, weights["ln_f.bias"])    # Beta

        # LMHead — weight-tied with wte in GPT-2
        # Overfit stores LMHead as [vocab, d] (same as wte)
        # TieWeights=false in config but we write it anyway for compatibility
        print("  Writing: LMHead (weight-tied = wte)")
        write_parameter(f, weights["wte"])

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Saved: {out_path} ({size_mb:.0f}MB)")

def main():
    parser = argparse.ArgumentParser(description="Convert GPT-2 to Overfit format")
    parser.add_argument("--size",   default="small",
                        choices=["small", "medium", "large", "xl"])
    parser.add_argument("--out",    default=".", help="Output directory")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip download, use local pytorch_model.bin")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = SIZES[args.size]

    print(f"\n=== GPT-2 {args.size.capitalize()} → Overfit ===")
    print(f"  Config: {cfg}")

    # Download weights
    model_path = os.path.join(args.out, f"pytorch_model_{args.size}.bin")
    vocab_path = os.path.join(args.out, "vocab.json")
    merges_path = os.path.join(args.out, "merges.txt")

    if not args.no_download:
        print("\nDownloading from HuggingFace...")
        base = HF_BASE if args.size == "small" else \
               f"https://huggingface.co/openai-community/gpt2-{args.size}/resolve/main"

        download(f"{base}/pytorch_model.bin", model_path)
        download(f"{HF_BASE}/vocab.json",  vocab_path)
        download(f"{HF_BASE}/merges.txt",  merges_path)

    # Convert
    print("\nLoading PyTorch weights...")
    weights = load_pytorch_weights(model_path)
    print(f"  {len(weights)} tensors loaded")

    out_bin = os.path.join(args.out, f"gpt2_{args.size}.bin")
    print("\nConverting to Overfit format...")
    convert(weights, cfg, out_bin)

    print(f"\nDone! Files written to: {args.out}")
    print(f"  {os.path.basename(out_bin)}  — Overfit model weights")
    print(f"  vocab.json               — BPE vocabulary (50257 tokens)")
    print(f"  merges.txt               — BPE merge rules")
    print()
    print("Load in Overfit:")
    print(f"""
  var tokenizer = BytePairEncoder.Load("vocab.json", "merges.txt");
  var config    = Gpt2Config.Small; // or Medium/Large/XL
  using var model = new GPT1Model(config);
  model.Load(new BinaryReader(File.OpenRead("{os.path.basename(out_bin)}")));
  model.Eval();

  var prompt    = tokenizer.Encode("The future of C# is");
  var generated = model.Generate(prompt, maxNewTokens: 100);
  Console.WriteLine(tokenizer.Decode(generated));
""")

if __name__ == "__main__":
    main()
