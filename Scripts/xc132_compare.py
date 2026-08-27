"""XC-132: where do the two `quantize` arms diverge, and which one is llama.cpp closer to?

WHAT IT COMPARES. Three arms of per-token hidden states for the same four texts and the same GGUF bytes:

    ours(quantize:true)   from Tests/bin/.../xc132-per-position.json
    ours(quantize:false)  from the same file
    llama.cpp             from `llama-embedding --pooling none --embd-normalize -1`

The capture file is written by
`Tests/LanguageModels/Diagnostics/Qwen3EmbeddingPerPositionCaptureTests` — run it with
OVERFIT_RUN_LONG=1 first.

WHY --embd-normalize -1 AND NOT THE DEFAULT. llama-embedding defaults to `2`, which L2-normalises each
per-token row. The mean of normalised rows is not the normalised mean, so with the default the rows
rebuild `--pooling mean` at cosine 0.999502 instead of 1.000000 and every per-position number is
slightly wrong in a way that looks like a finding. Measured here on 2026-08-27.

WHAT THE THIRD COLUMN IS FOR, AND IT IS THE POINT. `XC-132` is stated as "quantize:false disagrees with
llama.cpp at position 0". But llama.cpp IS the quantised arithmetic, so that comparison asks whether F32
matches Q8 through a block the model uses to cancel a massive activation. Through catastrophic
cancellation the two genuinely differ and NEITHER is automatically the wrong one. Reporting ours-vs-ours
beside both ours-vs-llama.cpp columns is what separates "our arms differ" from "we are wrong".

ALIGNMENT IS CHECKED, NOT ASSUMED. A row-count mismatch between the two tokenizers would silently shift
every comparison by one position and produce exactly the shape of a position-0 finding. The script exits
non-zero on a mismatch rather than reporting numbers.

    python Scripts/xc132_compare.py
    python Scripts/xc132_compare.py --capture <path> --llama-embedding <exe> --gguf <file>
"""

import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = ROOT / "Tests" / "bin" / "Release" / "net10.0" / "xc132-per-position.json"
DEFAULT_EXE = r"D:\llamacpp-tmp\build-avx2\bin\Release\llama-embedding.exe"
DEFAULT_GGUF = r"C:\qwen3-embed\Qwen3-Embedding-0.6B-Q8_0.gguf"


def cosine(a, b):
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        raise SystemExit("a zero vector reached the cosine; that is a capture bug, not a result")
    return dot / (math.sqrt(na) * math.sqrt(nb))


def relative_norm(a, b):
    """|a - b| / |b| — magnitude disagreement, which cosine deliberately cannot see."""
    diff = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    nb = math.sqrt(sum(y * y for y in b))
    return diff / nb


def llama_rows(exe, gguf, text, threads=8):
    fd, probe = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        pathlib.Path(probe).write_bytes(text.encode("utf-8"))
        cmd = [exe, "-m", gguf, "-f", probe, "--pooling", "none",
               "--embd-normalize", "-1", "--embd-output-format", "json", "-t", str(threads)]
        r = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise SystemExit("llama-embedding failed (%d):\n%s" % (r.returncode, r.stderr[-1500:]))
        rows = [d["embedding"] for d in json.loads(r.stdout)["data"]]
        if not rows:
            raise SystemExit("llama-embedding returned an EMPTY row set for %r" % text)
        return rows
    finally:
        os.unlink(probe)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", default=str(DEFAULT_CAPTURE))
    parser.add_argument("--llama-embedding", default=DEFAULT_EXE)
    parser.add_argument("--gguf", default=DEFAULT_GGUF)
    arguments = parser.parse_args()

    capture_path = pathlib.Path(arguments.capture)
    if not capture_path.exists():
        raise SystemExit(
            "no capture at %s — run Qwen3EmbeddingPerPositionCaptureTests with OVERFIT_RUN_LONG=1"
            % capture_path)

    capture = json.loads(capture_path.read_text(encoding="utf-8-sig"))
    texts = capture["texts"]
    token_ids = capture["token_ids"]
    arm_true = capture["quantize_true"]
    arm_false = capture["quantize_false"]

    print("capture : %s" % capture_path)
    print("gguf    : %s" % arguments.gguf)
    print()

    worst = []

    for t, text in enumerate(texts):
        ours_t, ours_f, ids = arm_true[t], arm_false[t], token_ids[t]
        reference = llama_rows(arguments.llama_embedding, arguments.gguf, text)

        # The EOS this engine appends is not in llama.cpp's row set unless llama.cpp appends it too.
        # Report the counts rather than trimming silently: a silent trim is how an off-by-one survives.
        print("text %d  %r" % (t, text))
        print("  rows: ours %d, llama.cpp %d, token ids %d" % (len(ours_t), len(reference), len(ids)))
        if len(ours_t) != len(reference):
            print("  !! ROW COUNT MISMATCH — every position below would be shifted. Not comparing.")
            print()
            continue

        print("  %-4s %-9s %-11s %-11s %-11s %s"
              % ("pos", "token", "cos(T,ref)", "cos(F,ref)", "cos(T,F)", "|F-T|/|T|"))
        for k in range(len(reference)):
            ct = cosine(ours_t[k], reference[k])
            cf = cosine(ours_f[k], reference[k])
            ctf = cosine(ours_t[k], ours_f[k])
            rel = relative_norm(ours_f[k], ours_t[k])
            flag = "  <<<" if min(ct, cf, ctf) < 0.999 else ""
            print("  %-4d %-9d %-11.6f %-11.6f %-11.6f %.4f%s" % (k, ids[k], ct, cf, ctf, rel, flag))
            worst.append((min(ct, cf, ctf), t, k, ids[k], ct, cf, ctf))
        print()

    print("=" * 78)
    print("the ten worst positions across all texts, by the weakest of the three cosines")
    print("  %-6s %-4s %-9s %-11s %-11s %s" % ("text", "pos", "token", "cos(T,ref)", "cos(F,ref)", "cos(T,F)"))
    for _, t, k, tok, ct, cf, ctf in sorted(worst)[:10]:
        print("  %-6d %-4d %-9d %-11.6f %-11.6f %.6f" % (t, k, tok, ct, cf, ctf))


if __name__ == "__main__":
    main()
