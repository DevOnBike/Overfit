"""Produce the committed llama.cpp reference vectors for the Qwen3-Embedding parity test.

Writes Tests/test_fixtures/qwen3_embedding_llamacpp_reference.json: for a fixed set of texts, the
embedding llama.cpp's `llama-embedding` produces from the SAME quantised GGUF bytes Overfit reads,
under both `--pooling last` and `--pooling mean`, plus the token count llama.cpp reports for each
text on its own.

Why a committed artefact rather than a number in a report: the reference is the whole oracle, and a
cosine quoted in a chat log cannot be re-checked by anybody. Both engines read the identical file, so
any disagreement is one engine's arithmetic and not quantisation.

The model file is NOT reproducible from this repository (639 MB, downloaded). This script therefore
records its size and the SHA-256 of its first mebibyte, and the test refuses a file that does not
match — a reference produced from a different quantisation would be a silently wrong oracle.

Usage (llama.cpp's llama-embedding must be built with LLAMA_BUILD_EXAMPLES=ON):

    python Scripts/qwen3_embedding_reference.py \
        --gguf C:\\qwen3-embed\\Qwen3-Embedding-0.6B-Q8_0.gguf \
        --llama-embedding D:\\llamacpp-tmp\\build-avx2\\bin\\Release\\llama-embedding.exe
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Four texts, chosen so a failure says something specific:
#   0  plain English, the shortest useful sequence
#   1  Polish written in ASCII      — the multilingual claim without a non-ASCII byte in it
#   2  the same Polish with 'z'     — a two-byte UTF-8 character through byte-level BPE
#   3  an instruction-prefixed query, the shape Qwen3-Embedding's query side actually uses
TEXTS = [
    "The capital of France is Paris.",
    "Stolica Francji to Paryz.",
    "Stolica Francji to Pary\u017C.",
    "Instruct: Given a web search query, retrieve relevant passages that answer the query "
    "Query: What is the capital of France?",
]


def fingerprint(path):
    with open(path, "rb") as f:
        head = f.read(1024 * 1024)
    return os.path.getsize(path), hashlib.sha256(head).hexdigest()


def embed(exe, gguf, texts, pooling, threads):
    """Returns (vectors, n_tokens_total, stderr) for one llama-embedding invocation."""
    fd, probe = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(probe, "wb") as f:
            f.write("\n".join(texts).encode("utf-8"))
        cmd = [exe, "-m", gguf, "-f", probe, "--pooling", pooling,
               "--embd-normalize", "2", "--embd-output-format", "json",
               "-t", str(threads)]
        r = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise SystemExit(f"llama-embedding failed ({r.returncode}):\n{r.stderr[-2000:]}")
        data = json.loads(r.stdout)["data"]
        vectors = [item["embedding"] for item in data]
        if len(vectors) != len(texts):
            raise SystemExit(f"expected {len(texts)} vectors, got {len(vectors)}")
        total = None
        for line in r.stderr.splitlines():
            m = re.search(r"n_tokens\s*=\s*(\d+)", line)
            if m:
                total = int(m.group(1))
        return vectors, total, r.stderr
    finally:
        os.unlink(probe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--llama-embedding", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Tests", "test_fixtures", "qwen3_embedding_llamacpp_reference.json"))
    args = ap.parse_args()

    size, digest = fingerprint(args.gguf)
    print(f"model: {os.path.basename(args.gguf)}  {size} bytes  sha256(first 1 MiB)={digest[:16]}")

    # Per-text token counts. One invocation per text, because llama.cpp reports the BATCH total and a
    # per-text count derived from our own tokenizer would make the assertion circular.
    counts = []
    for text in TEXTS:
        _, total, _ = embed(args.llama_embedding, args.gguf, [text], "last", args.threads)
        counts.append(total)
        print(f"  n_tokens={total}  {text[:56]!r}")

    last, last_total, _stderr = embed(args.llama_embedding, args.gguf, TEXTS, "last", args.threads)
    mean, mean_total, _ = embed(args.llama_embedding, args.gguf, TEXTS, "mean", args.threads)
    print(f"batch n_tokens: last={last_total} mean={mean_total} sum(per-text)={sum(counts)}")

    # Provenance is the point of this file, so the producer's own version string goes in it verbatim.
    # `--version` writes to STDERR and exits 0 — reading only stdout returns an empty string, which
    # would have written a reference with a blank provenance field. Both streams, then.
    ver = subprocess.run([args.llama_embedding, "--version"], capture_output=True,
                         encoding="utf-8", errors="replace")
    build = " | ".join(line.strip()
                       for line in ((ver.stdout or "") + (ver.stderr or "")).splitlines()
                       if line.strip())
    if not build:
        raise SystemExit("could not read llama-embedding --version; the reference would have no provenance")
    print(f"producer build: {build}")

    payload = {
        "// what this is":
            "llama.cpp reference embeddings for Qwen3-Embedding-0.6B, read from the same quantised "
            "GGUF bytes Overfit reads. Regenerate with Scripts/qwen3_embedding_reference.py. Both "
            "engines read one file, so a disagreement is arithmetic, not quantisation.",
        "model_file": os.path.basename(args.gguf),
        "model_size_bytes": size,
        "model_sha256_first_1mib": digest,
        "producer": os.path.basename(args.llama_embedding),
        "producer_build": build,
        "producer_args": "--pooling {last|mean} --embd-normalize 2 --embd-output-format json "
                         f"-t {args.threads}",
        "texts": TEXTS,
        "token_counts": counts,
        "last": [[round(x, 8) for x in v] for v in last],
        "mean": [[round(x, 8) for x in v] for v in mean],
    }

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=1)
        f.write("\n")
    print(f"wrote {args.out}  ({os.path.getsize(args.out)} bytes)")

    size2, digest2 = fingerprint(args.gguf)
    if (size, digest) != (size2, digest2):
        raise SystemExit("the model file changed during this run")
    print("model file unchanged")


main()
