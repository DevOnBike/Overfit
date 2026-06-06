r"""Convert the SNAC 24 kHz codec to Overfit's native format (offline; the only Python step — runtime is Python-free).

Exports just the DECODE path (the C# decoder needs codebooks + out_proj + the decoder stack, not the encoder):
  - <out>/snac_24khz.safetensors  effective (weight_norm folded) weights, canonical names matching the C# loader
  - <out>/config.json             the architecture dims the C# side reads
  - <out>/codes.bin               a deterministic set of real codes (int32 LE): [n][len0..n-1] then concatenated codes
  - <out>/reference_noiseoff.f32  the reference decode of those codes with the stochastic NoiseBlock disabled
                                  (float32 LE) — the deterministic target the C# parity gate compares against.

Usage:  python Scripts/convert_snac.py --out C:\snac
"""
import argparse
import json
import os
import struct

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file

from snac import SNAC
from snac.layers import NoiseBlock

REPO = "hubertsiuzdak/snac_24khz"


def w(module):
    """Effective (weight_norm-folded) weight of a conv, contiguous float32."""
    return module.weight.detach().to(torch.float32).contiguous()


def b(module):
    return module.bias.detach().to(torch.float32).contiguous()


def alpha(snake):
    return snake.alpha.detach().to(torch.float32).reshape(-1).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"C:\snac")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = SNAC.from_pretrained(REPO).eval()
    print(f"loaded SNAC: latent={model.latent_dim} hop={model.hop_length} vq_strides={model.vq_strides}")

    t = {}

    # ── quantizer decode path: codebook + out_proj per level ──
    for i, q in enumerate(model.quantizer.quantizers):
        t[f"q.{i}.codebook"] = q.codebook.weight.detach().to(torch.float32).contiguous()  # [size, dim]
        t[f"q.{i}.out_proj.weight"] = w(q.out_proj)  # [input_dim, codebook_dim, 1]
        t[f"q.{i}.out_proj.bias"] = b(q.out_proj)    # [input_dim]

    dec = model.decoder.model  # nn.Sequential

    # dec[0] depthwise input conv, dec[1] pointwise to channels
    t["dec.in_dw.weight"] = w(dec[0])   # [768,1,7] groups=768
    t["dec.in_dw.bias"] = b(dec[0])
    t["dec.in_pw.weight"] = w(dec[1])   # [1024,768,1]
    t["dec.in_pw.bias"] = b(dec[1])

    # dec[2..5] DecoderBlocks
    n_blocks = len(model.decoder_rates)
    for bi in range(n_blocks):
        blk = dec[2 + bi].block  # Sequential: [0]snake [1]convt [2]NoiseBlock [3,4,5]ResidualUnit
        t[f"dec.block.{bi}.snake_in.alpha"] = alpha(blk[0])
        t[f"dec.block.{bi}.convt.weight"] = w(blk[1])  # [in,out,k]
        t[f"dec.block.{bi}.convt.bias"] = b(blk[1])
        t[f"dec.block.{bi}.noise.weight"] = w(blk[2].linear)  # [out,out,1] bias=False
        for ri in range(3):
            res = blk[3 + ri].block  # [0]snake [1]conv(dw) [2]snake [3]conv(pw)
            t[f"dec.block.{bi}.res.{ri}.snake1.alpha"] = alpha(res[0])
            t[f"dec.block.{bi}.res.{ri}.conv1.weight"] = w(res[1])  # [dim,1,7] groups=dim
            t[f"dec.block.{bi}.res.{ri}.conv1.bias"] = b(res[1])
            t[f"dec.block.{bi}.res.{ri}.snake2.alpha"] = alpha(res[2])
            t[f"dec.block.{bi}.res.{ri}.conv2.weight"] = w(res[3])  # [dim,dim,1]
            t[f"dec.block.{bi}.res.{ri}.conv2.bias"] = b(res[3])

    # dec[6] snake_out, dec[7] out_conv, dec[8] tanh
    t["dec.snake_out.alpha"] = alpha(dec[6])
    t["dec.out_conv.weight"] = w(dec[7])  # [1,64,7]
    t["dec.out_conv.bias"] = b(dec[7])

    st_path = os.path.join(args.out, "snac_24khz.safetensors")
    save_file(t, st_path)
    print(f"wrote {st_path}  ({len(t)} tensors, {sum(v.numel() for v in t.values())/1e6:.2f}M params)")

    cfg = {
        "sampling_rate": model.sampling_rate,
        "latent_dim": model.latent_dim,
        "decoder_dim": model.decoder_dim,
        "decoder_rates": model.decoder_rates,
        "codebook_size": model.codebook_size,
        "codebook_dim": model.codebook_dim,
        "vq_strides": model.vq_strides,
        "noise": True,
        "depthwise": True,
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ── reference fixture: deterministic audio → codes → NOISE-OFF decode ──
    sr = model.sampling_rate
    n = 12000
    tt = np.arange(n, dtype=np.float64) / sr
    sig = (0.4 * np.sin(2 * np.pi * 220 * tt)
           + 0.2 * np.sin(2 * np.pi * 440 * tt)
           + 0.1 * np.sin(2 * np.pi * 660 * tt * (1 + tt)))  # mild chirp
    audio = torch.tensor(sig, dtype=torch.float32).reshape(1, 1, -1)

    with torch.no_grad():
        codes = model.encode(audio)  # list of [1, T_i] int

    # disable stochastic noise so the decode is reproducible (the parity gate runs the deterministic path)
    NoiseBlock.forward = lambda self, x: x
    with torch.no_grad():
        ref = model.decode(codes).reshape(-1).to(torch.float32).cpu().numpy()

    codes_np = [c.reshape(-1).to(torch.int32).cpu().numpy() for c in codes]
    print("codes lengths:", [len(c) for c in codes_np], "| reference samples:", len(ref))

    with open(os.path.join(args.out, "codes.bin"), "wb") as f:
        f.write(struct.pack("<i", len(codes_np)))
        for c in codes_np:
            f.write(struct.pack("<i", len(c)))
        for c in codes_np:
            f.write(c.tobytes())

    ref.tofile(os.path.join(args.out, "reference_noiseoff.f32"))
    print("wrote codes.bin + reference_noiseoff.f32")


if __name__ == "__main__":
    main()
