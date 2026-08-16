#!/usr/bin/env python3
"""Re-records the 2-token Python oracle for QwenLayer0CompareTests (XC-55).

WHAT THIS PRODUCES
    Tests/bin/xc55-oracle.json, whose contents are then copied over the committed fixture
    Tests/test_fixtures/qwen3b_l0_twotoken_hidden.json. That fixture is the reference for
    QwenLayer0CompareTests.L0_TwoToken_HiddenStateVsPython: the final hidden state BEFORE the final
    RMSNorm at the last of 2 tokens [BOS=151643, im_start=151644], plus logit[198].

WHEN TO RUN IT
    Whenever C:\\qwen3b\\qwen.bin is re-converted. The test asserts the model file's byte length and
    mtime against the header this script writes, so a re-conversion fails by NAME ("the oracle was
    recorded against a different model file") instead of as a mysterious cosine miss. That is the
    defect this whole task existed to close: the previous constants were recorded 2026-05-17 and
    silently described the file that the 2026-08-07 re-conversion replaced.

    Run it as:  python Scripts/oracle_twotoken.py       (~7 min: the .bin parse dominates)

WHY IT DUPLICATES THE PARSE, AND WHY THE SELF-CHECK IS LOAD-BEARING
    The arithmetic that decides the answer -- adjacent-pair (NEOX) RoPE, grouped GQA, rms_norm --
    is NOT duplicated: it is `forward_sequence` imported verbatim from Scripts/forward_multitoken.py.
    What is duplicated is that module's header/weight parse (13 header reads, per-head Q/K/V/O
    ordering, fg2, lm_head), because its main() also runs three long tests we do not need here.

    A duplicated parse is exactly the thing that rots, so TEST 1 below is not decoration: one token
    at position 0 must reproduce the oracle that the PASSING C# test L0_LogitsAfterReset_NotAfterGenerate
    already pins (top-1 = [33975] 15.5608). If the SELF-CHECK line prints FAIL, the parse is wrong and
    NOTHING else this script prints or writes may be used -- do not copy the output to the fixture.
    (Position 0 is the identity rotation, which is why that 1-token oracle survived the re-conversion
    and only position >= 1 diverged; it is a check on the parse, not on the RoPE convention.)
"""
import json, os, struct, sys, time, datetime

import numpy as np

sys.path.insert(0, r"D:\Overfit\Scripts")
import forward_multitoken as fm

BIN = r"C:\qwen3b\qwen.bin"
OUT = r"D:\Overfit\Tests\bin\xc55-oracle.json"


def main():
    st = os.stat(BIN)
    print(f"model: {BIN}")
    print(f"  size  {st.st_size} bytes")
    print(f"  mtime {datetime.datetime.fromtimestamp(st.st_mtime)}")

    t0 = time.time()
    with open(BIN, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        _ = struct.unpack("<I", f.read(4))[0]
        n_layers = struct.unpack("<i", f.read(4))[0]
        d_model = struct.unpack("<i", f.read(4))[0]
        n_heads = struct.unpack("<i", f.read(4))[0]
        n_kv_heads = struct.unpack("<i", f.read(4))[0]
        vocab_size = struct.unpack("<i", f.read(4))[0]
        _ctx = struct.unpack("<i", f.read(4))[0]
        d_ff = struct.unpack("<i", f.read(4))[0]
        _ = struct.unpack("<i", f.read(4))[0]
        rope_theta = struct.unpack("<f", f.read(4))[0]
        _ = struct.unpack("<i", f.read(4))[0]
        _ = struct.unpack("<i", f.read(4))[0]
        head_dim = d_model // n_heads
        print(f"  magic 0x{magic:08X} {n_layers}L d={d_model} h={n_heads}/{n_kv_heads} "
              f"ff={d_ff} vocab={vocab_size} head_dim={head_dim} rope_theta={rope_theta}")

        def rf(n):
            return np.frombuffer(f.read(n * 4), dtype=np.float32).copy()

        emb = rf(vocab_size * d_model).reshape(vocab_size, d_model)
        layers = []
        for _l in range(n_layers):
            ln1g = rf(d_model); _ln1b = rf(d_model)
            wq, bq = [], []
            for _h in range(n_heads):
                wq.append(rf(d_model * head_dim).reshape(d_model, head_dim))
                bq.append(rf(head_dim))
            wk, bk, wv, bv = [], [], [], []
            for _kv in range(n_kv_heads):
                wk.append(rf(d_model * head_dim).reshape(d_model, head_dim))
                bk.append(rf(head_dim))
                wv.append(rf(d_model * head_dim).reshape(d_model, head_dim))
                bv.append(rf(head_dim))
            wo, bo = [], []
            for _h in range(n_heads):
                wo.append(rf(head_dim * d_model).reshape(head_dim, d_model))
                bo.append(rf(d_model))
            ln2g = rf(d_model); _ = rf(d_model)
            fg = rf(d_model * d_ff).reshape(d_model, d_ff)
            fu = rf(d_model * d_ff).reshape(d_model, d_ff)
            fd = rf(d_ff * d_model).reshape(d_ff, d_model)
            layers.append((ln1g, wq, bq, wk, bk, wv, bv, wo, bo, ln2g, fg, fu, fd))
        fg2 = rf(d_model); _ = rf(d_model)
        lm_head = rf(vocab_size * d_model).reshape(vocab_size, d_model)
        tail = f.read(16)
    print(f"  loaded in {time.time() - t0:.1f}s; trailing bytes after lm_head = {len(tail)} (expect 0)")

    # ---- SELF-CHECK: 1 token, position 0 -------------------------------------------------
    t = time.time()
    h1, lg1 = fm.forward_sequence(emb, layers, fg2, lm_head, [151643],
                                  head_dim, n_heads, n_kv_heads, rope_theta)
    top1_1 = int(np.argmax(lg1))
    print(f"\nTEST 1 [BOS] ({time.time() - t:.1f}s): top-1 = [{top1_1}] {float(lg1[top1_1]):.4f}")
    print("  expected by the PASSING C# test L0_LogitsAfterReset: [33975] 15.5608")
    ok1 = (top1_1 == 33975) and abs(float(lg1[top1_1]) - 15.5608) < 0.1
    print(f"  SELF-CHECK: {'PASS' if ok1 else 'FAIL — parse is wrong, discard everything below'}")

    # ---- TEST 2: 2 tokens, the oracle being re-recorded ------------------------------------
    t = time.time()
    h2, lg2 = fm.forward_sequence(emb, layers, fg2, lm_head, [151643, 151644],
                                  head_dim, n_heads, n_kv_heads, rope_theta)
    order = np.argsort(lg2)[-5:][::-1]
    print(f"\nTEST 2 [BOS, im_start] ({time.time() - t:.1f}s):")
    for tok in order:
        print(f"    [{int(tok):7d}] {float(lg2[tok]):9.4f}")
    print(f"  logit[198] = {float(lg2[198]):.4f}   (docstring from May said 12.3511)")
    print(f"  hidden[:4] = {[round(float(v), 5) for v in h2[:4]]}")
    print("    (docstring from May said [0.14059, 0.84549, 1.01591, -1.83366])")
    print(f"  |hidden| = {float(np.linalg.norm(h2)):.4f}")

    mtime_utc = datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc)
    doc = {
        "// what this is": ("Python oracle for QwenLayer0CompareTests.L0_TwoToken_HiddenStateVsPython: "
                            "final hidden state BEFORE the final RMSNorm, at the last of 2 tokens "
                            "[BOS=151643, im_start=151644]."),
        "recorded": "2026-08-15",
        "task": "XC-55",
        "script": "Scripts/forward_multitoken.py (forward_sequence), driven by Scripts/oracle_twotoken.py",
        "model_file": BIN,
        "model_bytes": st.st_size,
        "model_mtime_utc": mtime_utc.replace(tzinfo=None).isoformat() + "Z",
        "convention": ("adjacent-pair (GPT-NeoX / llama.cpp NEOX) RoPE; the .bin was re-converted "
                       "2026-08-07 by Scripts/convert_llama.py AFTER commit 265fd77 added "
                       "permute_rope_rows, so the previous constants (2026-05-17) describe the file "
                       "this one replaced. Position 0 is the identity rotation, which is why the "
                       "1-token oracle survived the fixture change and only position >= 1 diverged."),
        "tokens": [151643, 151644],
        "d_model": int(d_model),
        "self_check_1token_top1_id": top1_1,
        "self_check_1token_top1_logit": float(lg1[top1_1]),
        "self_check_passed": bool(ok1),
        "top1_id": int(order[0]),
        "top1_logit": float(lg2[int(order[0])]),
        "logit_198": float(lg2[198]),
        "hidden_first4": [float(v) for v in h2[:4]],
        "hidden": [float(v) for v in h2],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    print(f"\nwrote {OUT}")
    print("  copy it over Tests/test_fixtures/qwen3b_l0_twotoken_hidden.json ONLY if SELF-CHECK printed PASS")


if __name__ == "__main__":
    main()
