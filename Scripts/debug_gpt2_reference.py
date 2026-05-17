#!/usr/bin/env python3
"""
Export PyTorch GPT-2 reference stages and attention internals for Overfit import diagnostics.

Install:

    pip install torch transformers numpy

Generate/download fixtures first:

    python3 Scripts/convert_gpt2.py --size small --out Tests/test_fixtures/

Then export reference data:

    python3 Scripts/debug_gpt2_reference.py \
        --size small \
        --fixtures Tests/test_fixtures \
        --prompt "The future of software development is" \
        --out Tests/test_fixtures/gpt2_reference_small.json

This script exports:
- tokenizer ids,
- embedding output,
- block 0 LN/attention/MLP stages,
- block 0 Q/K/V projections with and without GPT-2 c_attn.bias,
- block 0 attention context before c_proj,
- final norm,
- final logits.

The key diagnostic question is:

    Does Overfit Q/K/V match PyTorch Q/K/V without bias?

If yes, the converter split/orientation is probably right and the missing GPT-2
Q/K/V bias is the first major incompatibility. If no, the bug is earlier: QKV
weight orientation or Q/K/V/head split.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel


SIZES: dict[str, dict[str, int]] = {
    "small": {
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "n_ctx": 1024,
        "n_vocab": 50257,
    },
    "medium": {
        "n_layer": 24,
        "n_head": 16,
        "n_embd": 1024,
        "n_ctx": 1024,
        "n_vocab": 50257,
    },
    "large": {
        "n_layer": 36,
        "n_head": 20,
        "n_embd": 1280,
        "n_ctx": 1024,
        "n_vocab": 50257,
    },
    "xl": {
        "n_layer": 48,
        "n_head": 25,
        "n_embd": 1600,
        "n_ctx": 1024,
        "n_vocab": 50257,
    },
}


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))

    cs = bs[:]
    n = 0

    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    return dict(zip(bs, [chr(n) for n in cs]))


class Gpt2BpeTokenizer:
    _pattern = re.compile(
        r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
    )

    def __init__(
        self,
        vocab_path: Path,
        merges_path: Path,
    ) -> None:
        with vocab_path.open("r", encoding="utf-8") as f:
            self.encoder: dict[str, int] = json.load(f)

        self.decoder: dict[int, str] = {
            int(v): str(k)
            for k, v in self.encoder.items()
        }

        byte_encoder = bytes_to_unicode()
        self.byte_encoder: dict[int, str] = byte_encoder
        self.byte_decoder: dict[str, int] = {
            v: k
            for k, v in byte_encoder.items()
        }

        self.bpe_ranks: dict[tuple[str, str], int] = {}

        with merges_path.open("r", encoding="utf-8") as f:
            rank = 0

            for raw in f:
                line = raw.strip()

                if not line or line.startswith("#"):
                    continue

                parts = line.split()

                if len(parts) != 2:
                    continue

                self.bpe_ranks[(parts[0], parts[1])] = rank
                rank += 1

        self.cache: dict[str, list[str]] = {}

    @property
    def vocab_size(self) -> int:
        return max(self.decoder.keys()) + 1

    def encode(
        self,
        text: str,
    ) -> list[int]:
        result: list[int] = []

        for match in self._pattern.finditer(text):
            token_text = match.group(0)

            if not token_text:
                continue

            byte_encoded = "".join(
                self.byte_encoder[b]
                for b in token_text.encode("utf-8")
            )

            for token in self.bpe(byte_encoded):
                token_id = self.encoder.get(token)

                if token_id is None:
                    token_id = self.encoder.get("<|endoftext|>", 50256)

                result.append(int(token_id))

        return result

    def decode(
        self,
        token_ids: list[int] | tuple[int, ...],
    ) -> str:
        text = "".join(
            self.decoder.get(int(token_id), "")
            for token_id in token_ids
        )

        byte_values = bytearray()

        for ch in text:
            value = self.byte_decoder.get(ch)

            if value is not None:
                byte_values.append(value)
                continue

            byte_values.extend(ch.encode("utf-8", errors="replace"))

        return byte_values.decode("utf-8", errors="replace")

    def bpe(
        self,
        token: str,
    ) -> list[str]:
        cached = self.cache.get(token)

        if cached is not None:
            return cached

        word = list(token)

        if len(word) <= 1:
            self.cache[token] = word
            return word

        while True:
            best_rank = sys.maxsize
            best_index = -1

            for i in range(len(word) - 1):
                rank = self.bpe_ranks.get((word[i], word[i + 1]))

                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_index = i

            if best_index < 0:
                break

            word[best_index] = word[best_index] + word[best_index + 1]
            del word[best_index + 1]

            if len(word) <= 1:
                break

        self.cache[token] = word
        return word


def first_tensor(
    value: Any,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value

    if isinstance(value, (tuple, list)):
        if not value:
            raise RuntimeError("Module output tuple/list is empty.")

        first = value[0]

        if not isinstance(first, torch.Tensor):
            raise TypeError(f"First module output is not a tensor: {type(first)!r}")

        return first

    if hasattr(value, "last_hidden_state"):
        result = value.last_hidden_state

        if isinstance(result, torch.Tensor):
            return result

    raise TypeError(f"Unsupported module output type: {type(value)!r}")


def require_rank3(
    name: str,
    tensor: torch.Tensor,
) -> torch.Tensor:
    if tensor.dim() != 3:
        raise RuntimeError(
            f"{name} expected rank-3 tensor [batch, seq, hidden], got shape {tuple(tensor.shape)}."
        )

    return tensor


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(
        str(path),
        map_location="cpu",
    )

    if isinstance(state, Mapping) and "state_dict" in state and isinstance(state["state_dict"], Mapping):
        state = state["state_dict"]

    if isinstance(state, Mapping) and "model" in state and isinstance(state["model"], Mapping):
        state = state["model"]

    if not isinstance(state, Mapping):
        raise TypeError(f"Expected mapping/state_dict from {path}, got {type(state)!r}.")

    normalized: dict[str, torch.Tensor] = {}

    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue

        target_key = str(key)

        if not target_key.startswith("transformer.") and not target_key.startswith("lm_head."):
            if (
                target_key.startswith("wte.")
                or target_key.startswith("wpe.")
                or target_key.startswith("h.")
                or target_key.startswith("ln_f.")
            ):
                target_key = f"transformer.{target_key}"

        normalized[target_key] = value

    if "lm_head.weight" not in normalized and "transformer.wte.weight" in normalized:
        normalized["lm_head.weight"] = normalized["transformer.wte.weight"]

    return normalized


def build_model(
    size: str,
    state_dict: dict[str, torch.Tensor],
) -> GPT2LMHeadModel:
    cfg = SIZES[size]

    config = GPT2Config(
        vocab_size=cfg["n_vocab"],
        n_positions=cfg["n_ctx"],
        n_ctx=cfg["n_ctx"],
        n_embd=cfg["n_embd"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
    )

    model = GPT2LMHeadModel(config)

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if missing:
        print("Missing keys:")
        for key in missing[:40]:
            print(f"  {key}")

    if unexpected:
        print("Unexpected keys:")
        for key in unexpected[:40]:
            print(f"  {key}")

    model.eval()
    return model


def flat_tensor(
    tensor: torch.Tensor,
) -> list[float]:
    return tensor.detach().cpu().contiguous().view(-1).numpy().astype(np.float32).tolist()


def topk(
    logits: np.ndarray,
    k: int,
) -> list[dict[str, float | int]]:
    indexes = np.argsort(logits)[::-1][:k]

    return [
        {
            "token": int(index),
            "logit": float(logits[index]),
        }
        for index in indexes
    ]


def split_heads(
    x: torch.Tensor,
    n_head: int,
) -> torch.Tensor:
    batch, seq, hidden = x.shape
    d_head = hidden // n_head

    return x.view(batch, seq, n_head, d_head).permute(0, 2, 1, 3).contiguous()


def merge_heads(
    x: torch.Tensor,
) -> torch.Tensor:
    batch, n_head, seq, d_head = x.shape

    return x.permute(0, 2, 1, 3).contiguous().view(batch, seq, n_head * d_head)


def causal_attention_context(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    n_head: int,
) -> torch.Tensor:
    q = split_heads(q_raw, n_head)
    k = split_heads(k_raw, n_head)
    v = split_heads(v_raw, n_head)

    d_head = q.shape[-1]
    seq = q.shape[-2]

    scores = torch.matmul(
        q,
        k.transpose(-1, -2),
    ) / math.sqrt(d_head)

    mask = torch.tril(
        torch.ones(
            (seq, seq),
            dtype=torch.bool,
            device=scores.device,
        )
    )

    scores = scores.masked_fill(
        ~mask.view(1, 1, seq, seq),
        torch.finfo(scores.dtype).min,
    )

    probs = torch.softmax(
        scores,
        dim=-1,
    )

    context = torch.matmul(
        probs,
        v,
    )

    return merge_heads(context)


def compute_reference_stages(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
) -> tuple[dict[str, list[float]], dict[str, list[float]], np.ndarray]:
    transformer = model.transformer
    block0 = transformer.h[0]
    n_head = int(model.config.n_head)
    n_embd = int(model.config.n_embd)

    position_ids = torch.arange(
        0,
        input_ids.shape[1],
        dtype=torch.long,
        device=input_ids.device,
    ).unsqueeze(0)

    with torch.no_grad():
        token_emb = transformer.wte(input_ids)
        position_emb = transformer.wpe(position_ids)
        hidden = transformer.drop(token_emb + position_emb)
        hidden = require_rank3("embedding", hidden)

        stages: dict[str, list[float]] = {
            "embedding": flat_tensor(hidden),
        }

        block0_ln1 = require_rank3(
            "block0_ln1",
            block0.ln_1(hidden),
        )
        stages["block0_ln1"] = flat_tensor(block0_ln1)

        qkv_without_bias = torch.matmul(
            block0_ln1,
            block0.attn.c_attn.weight,
        )

        qkv_with_bias = block0.attn.c_attn(block0_ln1)

        q_nb, k_nb, v_nb = qkv_without_bias.split(n_embd, dim=2)
        q_wb, k_wb, v_wb = qkv_with_bias.split(n_embd, dim=2)

        context_with_bias = causal_attention_context(
            q_wb,
            k_wb,
            v_wb,
            n_head,
        )

        context_without_bias = causal_attention_context(
            q_nb,
            k_nb,
            v_nb,
            n_head,
        )

        attn_manual_cproj_with_bias = block0.attn.c_proj(context_with_bias)
        attn_manual_cproj_without_qkv_bias = block0.attn.c_proj(context_without_bias)

        attention: dict[str, list[float]] = {
            "block0_qkv_without_bias": flat_tensor(qkv_without_bias),
            "block0_qkv_with_bias": flat_tensor(qkv_with_bias),
            "block0_q_without_bias": flat_tensor(q_nb),
            "block0_k_without_bias": flat_tensor(k_nb),
            "block0_v_without_bias": flat_tensor(v_nb),
            "block0_q_with_bias": flat_tensor(q_wb),
            "block0_k_with_bias": flat_tensor(k_wb),
            "block0_v_with_bias": flat_tensor(v_wb),
            "block0_context_without_qkv_bias": flat_tensor(context_without_bias),
            "block0_context_with_bias": flat_tensor(context_with_bias),
            "block0_attn_manual_cproj_without_qkv_bias": flat_tensor(attn_manual_cproj_without_qkv_bias),
            "block0_attn_manual_cproj_with_bias": flat_tensor(attn_manual_cproj_with_bias),
        }

        block0_attn_result = block0.attn(
            block0_ln1,
            use_cache=False,
            output_attentions=False,
        )

        block0_attn = require_rank3(
            "block0_attn",
            first_tensor(block0_attn_result),
        )
        stages["block0_attn"] = flat_tensor(block0_attn)

        block0_after_attn = require_rank3(
            "block0_after_attn_residual",
            hidden + block0_attn,
        )
        stages["block0_after_attn_residual"] = flat_tensor(block0_after_attn)

        block0_ln2 = require_rank3(
            "block0_ln2",
            block0.ln_2(block0_after_attn),
        )
        stages["block0_ln2"] = flat_tensor(block0_ln2)

        block0_mlp = require_rank3(
            "block0_mlp",
            block0.mlp(block0_ln2),
        )
        stages["block0_mlp"] = flat_tensor(block0_mlp)

        block0_output = require_rank3(
            "block0_output",
            block0_after_attn + block0_mlp,
        )
        stages["block0_output"] = flat_tensor(block0_output)

        hidden = block0_output

        for layer_index, block in enumerate(transformer.h[1:], start=1):
            block_result = block(
                hidden,
                use_cache=False,
                output_attentions=False,
            )

            hidden = require_rank3(
                f"block{layer_index}_output",
                first_tensor(block_result),
            )

        final_norm = require_rank3(
            "final_norm",
            transformer.ln_f(hidden),
        )
        stages["final_norm"] = flat_tensor(final_norm)

        final_logits = model.lm_head(final_norm)[0, -1, :].detach().cpu().numpy().astype(np.float32)

    return stages, attention, final_logits


def export_reference(
    size: str,
    fixtures: Path,
    prompt: str,
    out_path: Path,
    top_k: int,
) -> None:
    model_path = fixtures / f"pytorch_model_{size}.bin"
    vocab_path = fixtures / "vocab.json"
    merges_path = fixtures / "merges.txt"

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if not vocab_path.exists():
        raise FileNotFoundError(vocab_path)

    if not merges_path.exists():
        raise FileNotFoundError(merges_path)

    print("Loading tokenizer...")
    tokenizer = Gpt2BpeTokenizer(
        vocab_path,
        merges_path,
    )

    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    if tokenizer.vocab_size != SIZES[size]["n_vocab"]:
        raise RuntimeError(
            f"Tokenizer vocab size {tokenizer.vocab_size} does not match expected {SIZES[size]['n_vocab']}."
        )

    print("Loading PyTorch checkpoint...")
    state_dict = load_state_dict(model_path)

    print(f"Loaded tensors: {len(state_dict)}")
    print("Building GPT-2 reference model...")
    model = build_model(size, state_dict)

    token_ids = tokenizer.encode(prompt)

    if not token_ids:
        raise RuntimeError(
            f"Tokenizer produced zero tokens for prompt {prompt!r}."
        )

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
    )

    print(f"Prompt: {prompt!r}")
    print(f"Token ids: {token_ids}")

    stages, attention, final_logits = compute_reference_stages(
        model,
        input_ids,
    )

    next_token = int(np.argmax(final_logits))
    decoded_next = tokenizer.decode([next_token])

    payload: dict[str, Any] = {
        "size": size,
        "prompt": prompt,
        "tokens": token_ids,
        "vocab_size": int(final_logits.shape[0]),
        "stage_shape": [1, len(token_ids), int(SIZES[size]["n_embd"])],
        "next_token": next_token,
        "next_token_text": decoded_next,
        "top_logits": topk(final_logits, top_k),
        "logits": final_logits.tolist(),
        "stages": stages,
        "attention": attention,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("")
    print(f"Reference written: {out_path}")
    print(f"Next token: {next_token} {decoded_next!r}")
    print("Top logits:")

    for item in payload["top_logits"]:
        print(f"  {item['token']:>6} {item['logit']:>12.6f} {tokenizer.decode([int(item['token'])])!r}")

    print("")
    print("Exported stages:")
    for name, values in stages.items():
        print(f"  {name}: {len(values)} floats")

    print("")
    print("Exported attention internals:")
    for name, values in attention.items():
        print(f"  {name}: {len(values)} floats")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GPT-2 PyTorch reference stages/logits for Overfit import diagnostics."
    )

    parser.add_argument(
        "--size",
        choices=sorted(SIZES.keys()),
        default="small",
    )

    parser.add_argument(
        "--fixtures",
        default="Tests/test_fixtures",
    )

    parser.add_argument(
        "--prompt",
        default="The future of software development is",
    )

    parser.add_argument(
        "--out",
        default="Tests/test_fixtures/gpt2_reference_small.json",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    export_reference(
        size=args.size,
        fixtures=Path(args.fixtures),
        prompt=args.prompt,
        out_path=Path(args.out),
        top_k=args.top_k,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
