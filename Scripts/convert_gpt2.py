#!/usr/bin/env python3
"""
Convert HuggingFace GPT-2 weights to the Overfit GPT1Model binary format.

Usage:

    pip install numpy torch
    python3 Scripts/convert_gpt2.py --size small --out Tests/test_fixtures/

Outputs:

    gpt2_small.bin             Overfit weights
    pytorch_model_small.bin    downloaded HuggingFace PyTorch checkpoint
    vocab.json                 GPT-2 BPE vocabulary
    merges.txt                 GPT-2 BPE merges
"""

from __future__ import annotations

import argparse
import struct
import sys
import urllib.request
from collections.abc import Mapping
from pathlib import Path

import numpy as np


SIZES: dict[str, dict[str, int | str]] = {
    "small": {
        "repo": "openai-community/gpt2",
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "n_ctx": 1024,
        "n_vocab": 50257,
    },
    "medium": {
        "repo": "openai-community/gpt2-medium",
        "n_layer": 24,
        "n_head": 16,
        "n_embd": 1024,
        "n_ctx": 1024,
        "n_vocab": 50257,
    },
    "large": {
        "repo": "openai-community/gpt2-large",
        "n_layer": 36,
        "n_head": 20,
        "n_embd": 1280,
        "n_ctx": 1024,
        "n_vocab": 50257,
    },
    "xl": {
        "repo": "openai-community/gpt2-xl",
        "n_layer": 48,
        "n_head": 25,
        "n_embd": 1600,
        "n_ctx": 1024,
        "n_vocab": 50257,
    },
}


def hf_url(repo: str, filename: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def download(
    url: str,
    path: Path,
    force: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not force:
        print(f"  Already exists: {path}")
        return

    if path.exists() and force:
        path.unlink()

    print(f"  Downloading: {url}")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Overfit-GPT2-Converter/1.0"},
    )

    with urllib.request.urlopen(request) as response:
        with path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)

                if not chunk:
                    break

                f.write(chunk)

    print(f"  Saved: {path} ({path.stat().st_size / 1e6:.1f} MB)")


def load_pytorch_weights(
    model_path: Path,
) -> dict[str, np.ndarray]:
    """
    Load HuggingFace GPT-2 PyTorch state_dict.

    HuggingFace GPT-2 checkpoints may expose either:

        wte.weight
        h.0.attn.c_attn.weight
        ln_f.weight

    or prefixed names such as:

        transformer.wte.weight
        transformer.h.0.attn.c_attn.weight
        transformer.ln_f.weight

    The writer below accepts both.
    """
    try:
        import torch
    except ImportError as ex:
        raise RuntimeError(
            "PyTorch is required to read pytorch_model.bin. "
            "Install it with: pip install torch"
        ) from ex

    print("Loading PyTorch weights with torch.load(...)...")

    try:
        data = torch.load(
            str(model_path),
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        data = torch.load(
            str(model_path),
            map_location="cpu",
        )

    if isinstance(data, Mapping) and "state_dict" in data and isinstance(data["state_dict"], Mapping):
        data = data["state_dict"]

    if isinstance(data, Mapping) and "model" in data and isinstance(data["model"], Mapping):
        data = data["model"]

    if not isinstance(data, Mapping):
        raise TypeError(
            f"Expected torch.load(...) to return a mapping/state_dict, got {type(data)!r}."
        )

    weights: dict[str, np.ndarray] = {}

    for key, value in data.items():
        if hasattr(value, "detach"):
            array = value.detach().cpu().numpy()
        elif hasattr(value, "cpu") and hasattr(value.cpu(), "numpy"):
            array = value.cpu().numpy()
        elif isinstance(value, np.ndarray):
            array = value
        else:
            continue

        weights[str(key)] = np.asarray(
            array,
            dtype=np.float32,
        )

    if not weights:
        raise ValueError(f"No tensor weights were loaded from {model_path}.")

    return weights


def resolve_weight_name(
    weights: dict[str, np.ndarray],
    name: str,
) -> str:
    if name in weights:
        return name

    if name.startswith("transformer."):
        stripped = name[len("transformer."):]

        if stripped in weights:
            return stripped

    prefixed = f"transformer.{name}"

    if prefixed in weights:
        return prefixed

    if name.startswith("module."):
        stripped_module = name[len("module."):]

        if stripped_module in weights:
            return stripped_module

        if stripped_module.startswith("transformer."):
            stripped_transformer = stripped_module[len("transformer."):]

            if stripped_transformer in weights:
                return stripped_transformer

    module_name = f"module.{name}"

    if module_name in weights:
        return module_name

    module_transformer_name = f"module.transformer.{name}"

    if module_transformer_name in weights:
        return module_transformer_name

    raise KeyError(name)


def get_weight(
    weights: dict[str, np.ndarray],
    name: str,
) -> np.ndarray:
    try:
        resolved = resolve_weight_name(
            weights,
            name,
        )

        return weights[resolved]
    except KeyError as ex:
        available = "\n".join(sorted(weights.keys())[:80])
        raise KeyError(
            f"Missing weight: {name}\nFirst available keys:\n{available}"
        ) from ex


def expect_shape(
    name: str,
    array: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    if array.shape == shape:
        return array

    raise ValueError(
        f"Weight {name} has shape {array.shape}, expected {shape}."
    )


def conv1d_weight(
    weights: dict[str, np.ndarray],
    name: str,
    in_dim: int,
    out_dim: int,
) -> np.ndarray:
    """
    HuggingFace GPT-2 Conv1D weights are normally stored as [in_dim, out_dim].
    Some conversion paths may produce [out_dim, in_dim]. Accept both and return
    [in_dim, out_dim], which matches Overfit Linear weight layout.
    """
    array = get_weight(
        weights,
        name,
    )

    if array.shape == (in_dim, out_dim):
        return array

    if array.shape == (out_dim, in_dim):
        return array.T.copy()

    raise ValueError(
        f"Weight {name} has shape {array.shape}, expected {(in_dim, out_dim)} "
        f"or {(out_dim, in_dim)}."
    )


def write_parameter(
    f,
    array: np.ndarray,
) -> None:
    flat = np.asarray(
        array,
        dtype=np.float32,
    ).reshape(-1)

    f.write(struct.pack("<i", int(flat.size)))
    f.write(flat.astype("<f4", copy=False).tobytes(order="C"))


def split_qkv_heads(
    c_attn_weight: np.ndarray,
    n_embd: int,
    n_head: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    d_head = n_embd // n_head

    q = c_attn_weight[:, 0:n_embd]
    k = c_attn_weight[:, n_embd:2 * n_embd]
    v = c_attn_weight[:, 2 * n_embd:3 * n_embd]

    q_heads: list[np.ndarray] = []
    k_heads: list[np.ndarray] = []
    v_heads: list[np.ndarray] = []

    for h in range(n_head):
        start = h * d_head
        end = start + d_head

        q_heads.append(q[:, start:end].copy())
        k_heads.append(k[:, start:end].copy())
        v_heads.append(v[:, start:end].copy())

    return q_heads, k_heads, v_heads


def split_output_heads(
    c_proj_weight: np.ndarray,
    n_embd: int,
    n_head: int,
) -> list[np.ndarray]:
    """
    c_proj_weight is [d, d]. Overfit stores one [dHead, d] output projection
    matrix per head and sums projected head outputs.
    """
    d_head = n_embd // n_head
    wo_heads: list[np.ndarray] = []

    for h in range(n_head):
        start = h * d_head
        end = start + d_head

        wo_heads.append(c_proj_weight[start:end, :].copy())

    return wo_heads


def print_key_diagnostics(
    weights: dict[str, np.ndarray],
) -> None:
    print("  Key style diagnostics:")

    for candidate in (
        "wte.weight",
        "transformer.wte.weight",
        "h.0.attn.c_attn.weight",
        "transformer.h.0.attn.c_attn.weight",
        "ln_f.weight",
        "transformer.ln_f.weight",
    ):
        if candidate in weights:
            print(f"    found: {candidate}")


def write_overfit_checkpoint(
    output_path: Path,
    weights: dict[str, np.ndarray],
    config: dict[str, int | str],
) -> None:
    n_layer = int(config["n_layer"])
    n_head = int(config["n_head"])
    n_embd = int(config["n_embd"])
    n_ctx = int(config["n_ctx"])
    n_vocab = int(config["n_vocab"])
    d_ff = 4 * n_embd
    d_head = n_embd // n_head

    print("")
    print("Writing Overfit checkpoint...")
    print(f"  Output: {output_path}")
    print(f"  Layers: {n_layer}")
    print(f"  Heads: {n_head}")
    print(f"  d_model: {n_embd}")
    print(f"  d_head: {d_head}")
    print(f"  d_ff: {d_ff}")
    print(f"  vocab: {n_vocab}")
    print(f"  ctx: {n_ctx}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        wte = expect_shape(
            "wte.weight",
            get_weight(weights, "transformer.wte.weight"),
            (n_vocab, n_embd),
        )

        write_parameter(f, wte)

        wpe = expect_shape(
            "wpe.weight",
            get_weight(weights, "transformer.wpe.weight"),
            (n_ctx, n_embd),
        )

        write_parameter(f, wpe)

        for layer in range(n_layer):
            prefix = f"transformer.h.{layer}"

            print(f"  Layer {layer + 1}/{n_layer}")

            write_parameter(
                f,
                expect_shape(
                    f"{prefix}.ln_1.weight",
                    get_weight(weights, f"{prefix}.ln_1.weight"),
                    (n_embd,),
                ),
            )

            write_parameter(
                f,
                expect_shape(
                    f"{prefix}.ln_1.bias",
                    get_weight(weights, f"{prefix}.ln_1.bias"),
                    (n_embd,),
                ),
            )

            c_attn = conv1d_weight(
                weights,
                f"{prefix}.attn.c_attn.weight",
                n_embd,
                3 * n_embd,
            )

            q_heads, k_heads, v_heads = split_qkv_heads(
                c_attn,
                n_embd,
                n_head,
            )

            c_proj = conv1d_weight(
                weights,
                f"{prefix}.attn.c_proj.weight",
                n_embd,
                n_embd,
            )

            wo_heads = split_output_heads(
                c_proj,
                n_embd,
                n_head,
            )

            for h in range(n_head):
                write_parameter(f, q_heads[h])
                write_parameter(f, k_heads[h])
                write_parameter(f, v_heads[h])
                write_parameter(f, wo_heads[h])

            # Overfit's current attention layer has no Q/K/V bias parameters.
            # GPT-2 c_attn.bias is therefore intentionally ignored.
            write_parameter(
                f,
                expect_shape(
                    f"{prefix}.attn.c_proj.bias",
                    get_weight(weights, f"{prefix}.attn.c_proj.bias"),
                    (n_embd,),
                ),
            )

            write_parameter(
                f,
                expect_shape(
                    f"{prefix}.ln_2.weight",
                    get_weight(weights, f"{prefix}.ln_2.weight"),
                    (n_embd,),
                ),
            )

            write_parameter(
                f,
                expect_shape(
                    f"{prefix}.ln_2.bias",
                    get_weight(weights, f"{prefix}.ln_2.bias"),
                    (n_embd,),
                ),
            )

            write_parameter(
                f,
                conv1d_weight(
                    weights,
                    f"{prefix}.mlp.c_fc.weight",
                    n_embd,
                    d_ff,
                ),
            )

            write_parameter(
                f,
                expect_shape(
                    f"{prefix}.mlp.c_fc.bias",
                    get_weight(weights, f"{prefix}.mlp.c_fc.bias"),
                    (d_ff,),
                ),
            )

            write_parameter(
                f,
                conv1d_weight(
                    weights,
                    f"{prefix}.mlp.c_proj.weight",
                    d_ff,
                    n_embd,
                ),
            )

            write_parameter(
                f,
                expect_shape(
                    f"{prefix}.mlp.c_proj.bias",
                    get_weight(weights, f"{prefix}.mlp.c_proj.bias"),
                    (n_embd,),
                ),
            )

        write_parameter(
            f,
            expect_shape(
                "ln_f.weight",
                get_weight(weights, "transformer.ln_f.weight"),
                (n_embd,),
            ),
        )

        write_parameter(
            f,
            expect_shape(
                "ln_f.bias",
                get_weight(weights, "transformer.ln_f.bias"),
                (n_embd,),
            ),
        )

        # GPT-2 ties LM head to token embedding. Overfit GPT-2 configs use
        # TieWeights=false for this imported checkpoint path, so write LMHead
        # explicitly as [d_model, vocab].
        write_parameter(f, wte.T.copy())

    print(f"  Done: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")


def parse_args(
    argv: list[str],
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace GPT-2 weights to Overfit binary format."
    )

    parser.add_argument(
        "--size",
        choices=sorted(SIZES.keys()),
        default="small",
        help="GPT-2 model size.",
    )

    parser.add_argument(
        "--out",
        default=".",
        help="Output directory.",
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload HuggingFace files even when they already exist.",
    )

    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
) -> int:
    args = parse_args(
        sys.argv[1:] if argv is None else argv
    )

    config = SIZES[args.size]
    repo = str(config["repo"])
    out_dir = Path(args.out)

    print(f"=== GPT-2 {args.size.capitalize()} \u2192 Overfit ===")
    print(f"  Repo: {repo}")
    print(f"  Config: {config}")
    print("")

    model_path = out_dir / f"pytorch_model_{args.size}.bin"
    vocab_path = out_dir / "vocab.json"
    merges_path = out_dir / "merges.txt"
    overfit_path = out_dir / f"gpt2_{args.size}.bin"

    print("Downloading from HuggingFace...")

    download(
        hf_url(repo, "pytorch_model.bin"),
        model_path,
        force=args.force_download,
    )

    download(
        hf_url(repo, "vocab.json"),
        vocab_path,
        force=args.force_download,
    )

    download(
        hf_url(repo, "merges.txt"),
        merges_path,
        force=args.force_download,
    )

    print("")
    print("Loading PyTorch weights...")

    weights = load_pytorch_weights(model_path)

    print(f"  Loaded tensors: {len(weights)}")
    print_key_diagnostics(weights)

    write_overfit_checkpoint(
        overfit_path,
        weights,
        config,
    )

    print("")
    print("Done.")
    print(f"  Weights: {overfit_path}")
    print(f"  Vocab:   {vocab_path}")
    print(f"  Merges:  {merges_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
